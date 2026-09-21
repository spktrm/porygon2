import { TrainablePlayerAI } from "../server/runner";
import { DecisionSource } from "../agents/decision";
import { JevClient } from "./client";
import { serialiseBattle } from "./serialise_battle";

export type JevDecode = "argmax" | "sample";

export interface JevMeta {
    asked: boolean;
    confidence?: number;
    chosenProbability?: number;
    latencyMs?: number;
    inputTokens?: number;
    payloadChars?: number;
    truncatedTurns: number;
    droppedChatLines: number;
}

export interface JevSourceOptions {
    decode: JevDecode;
    contextBudgetTokens: number;
    random: () => number;
}

// Measured 1.68 characters per jev input token on this payload (2026-09-21,
// 53 calls); held below that so the guard drops a turn too early rather than
// overrunning jev's context.
const CHARS_PER_TOKEN = 1.6;
const CELL_KEY_PREFIX = "cell_";
// Opponent-authored text on a ladder: never part of the game state, and the
// one untrusted input that would otherwise reach the model.
const CHAT_LINE_PREFIXES = [
    "|c|",
    "|c:|",
    "|chat|",
    "|raw|",
    "|html|",
    "|uhtml|",
    "|uhtmlchange|",
];

interface RequestView {
    active?: { moves: { move: string }[] }[];
    side?: { id: string; pokemon: { details: string }[] };
}

export function dropChatLines(log: readonly string[]): {
    lines: string[];
    dropped: number;
} {
    const lines = log.filter(
        (line) => !CHAT_LINE_PREFIXES.some((prefix) => line.startsWith(prefix)),
    );
    return { lines, dropped: log.length - lines.length };
}

/**
 * Drops whole turns from the OLDEST end until the log fits, keeping the
 * preamble before the first `|turn|` (the leads) and every recent turn.
 */
export function fitLogToBudget(
    log: readonly string[],
    budgetChars: number,
): { lines: string[]; truncatedTurns: number } {
    const turnStarts: number[] = [];
    log.forEach((line, index) => {
        if (line.startsWith("|turn|")) {
            turnStarts.push(index);
        }
    });
    const lineChars = (lines: readonly string[]) =>
        lines.reduce((sum, line) => sum + line.length + 1, 0);
    let truncatedTurns = 0;
    let lines = [...log];
    while (
        lineChars(lines) > budgetChars &&
        truncatedTurns < turnStarts.length - 1
    ) {
        truncatedTurns += 1;
        lines = [
            ...log.slice(0, turnStarts[0]),
            ...log.slice(turnStarts[truncatedTurns]),
        ];
    }
    return { lines, truncatedTurns };
}

/**
 * The Showdown choice string plus the name it points at, so a label reads
 * `move 2 - Earthquake` rather than a bare slot number.
 */
export function labelChoice(choice: string, request: RequestView): string {
    const [kind, slot] = choice.split(" ");
    const index = Number(slot) - 1;
    let name: string | undefined;
    if (kind === "move") {
        name = request.active?.[0]?.moves[index]?.move;
    } else if (kind === "switch") {
        name = request.side?.pokemon[index]?.details;
    }
    if (name === undefined) {
        return choice;
    }
    return `${choice} - ${name}`;
}

export function decodeChoice(
    choice: string,
    probabilities: Record<string, number>,
    legalKeys: string[],
    options: Pick<JevSourceOptions, "decode" | "random">,
): string {
    if (options.decode === "argmax") {
        return choice;
    }
    const total = legalKeys.reduce(
        (sum, key) => sum + (probabilities[key] ?? 0),
        0,
    );
    if (total <= 0) {
        return choice;
    }
    let remaining = options.random() * total;
    for (const key of legalKeys) {
        remaining -= probabilities[key] ?? 0;
        if (remaining < 0) {
            return key;
        }
    }
    return choice;
}

/**
 * Throws on a jev failure or an answer outside the legal cells. What a
 * failure costs is the caller's decision: the benchmark voids the game, a
 * ladder client plays a fallback.
 */
export function jevSource(
    client: JevClient,
    options: JevSourceOptions,
): DecisionSource<JevMeta> {
    return async (player: TrainablePlayerAI) => {
        const legalCells = [...player.legalChoiceByCell.keys()];
        if (legalCells.length <= 1) {
            return {
                cell: legalCells[0] ?? 0,
                meta: { asked: false, truncatedTurns: 0, droppedChatLines: 0 },
            };
        }

        const request = player.privateBattle.request as RequestView;
        const criteria: Record<string, string> = {};
        for (const [cell, choice] of player.legalChoiceByCell) {
            criteria[`${CELL_KEY_PREFIX}${cell}`] = labelChoice(
                choice,
                request,
            );
        }

        const chat = dropChatLines(player.log);
        const withoutLog = serialiseBattle(player.privateBattle, []);
        const budgetChars =
            options.contextBudgetTokens * CHARS_PER_TOKEN -
            JSON.stringify(withoutLog).length -
            JSON.stringify(criteria).length;
        const fitted = fitLogToBudget(chat.lines, budgetChars);
        const state = { ...withoutLog, log: fitted.lines };

        const sideId = request.side?.id ?? "p1";
        const result = await client.choose(
            state,
            `You are player ${sideId} in a Pokemon Showdown ${player.privateBattle.tier || "gen9randombattle"} battle. ` +
                "`request` is your current decision request, `sides` is both teams as far as you know them, " +
                "`log` is the battle protocol so far. Choose the action that maximises your probability of winning the battle.",
            criteria,
        );

        const key = decodeChoice(
            result.choice,
            result.probabilities,
            Object.keys(criteria),
            options,
        );
        const cell = Number(key.slice(CELL_KEY_PREFIX.length));
        if (
            !key.startsWith(CELL_KEY_PREFIX) ||
            !player.legalChoiceByCell.has(cell)
        ) {
            throw new Error(`jev chose '${key}', which is not a legal cell`);
        }
        return {
            cell,
            meta: {
                asked: true,
                confidence: result.confidence,
                chosenProbability: result.probabilities[key],
                latencyMs: result.latencyMs,
                inputTokens: result.inputTokens,
                payloadChars: JSON.stringify(state).length,
                truncatedTurns: fitted.truncatedTurns,
                droppedChatLines: chat.dropped,
            },
        };
    };
}
