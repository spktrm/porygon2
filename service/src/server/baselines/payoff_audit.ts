/** Read-only audit hooks. Only own requests and resolved public protocol enter
 * records. Opponent choices/private requests and the live simulator are absent.
 */
import { Dex } from "@pkmn/sim";
import type { TrainablePlayerAI } from "../runner";
import {
    captureObservation,
    ObservedBattle,
    ObservedPokemon,
} from "./mcts_observation";
import { evaluatePosition, PotentialPokemon } from "./position_potential";
import { normaliseChoice } from "./mcts_simulator";

export function observedPotential(observation: ObservedBattle): number {
    const convert = (mon: ObservedPokemon): PotentialPokemon => {
        const offensiveTypes = [...Dex.species.get(mon.species).types];
        let defensiveTypes = [...offensiveTypes];
        if (mon.terastallized && mon.terastallized !== "Stellar") {
            if (!offensiveTypes.includes(mon.terastallized))
                offensiveTypes.push(mon.terastallized);
            defensiveTypes = [mon.terastallized];
        }
        return {
            hp: mon.hp,
            active: mon.active,
            offensiveTypes,
            defensiveTypes,
        };
    };
    const opposing = observation.opponent.map(convert);
    while (opposing.length < observation.opponentSize)
        opposing.push({
            hp: 1,
            active: false,
            offensiveTypes: [],
            defensiveTypes: [],
        });
    return evaluatePosition(observation.own.map(convert), opposing);
}

export interface PublicReply {
    kind: "move" | "switch";
    identity: string;
    tera: boolean;
}
export interface PayoffRecord {
    pair: number;
    game: number;
    turn: number;
    side: number;
    observation: ObservedBattle;
    ownAction: string;
    legal: string[];
    rootPotential: number;
    reply?: PublicReply;
    actualPotential?: number;
    next?: ObservedBattle;
    trace?: string[];
    exclusion?: string;
}

export function resolvedReply(
    trace: string[],
    ownSide: number,
): PublicReply | undefined {
    const opposingPrefix = `p${2 - ownSide}a:`;
    const tera = trace.some((line) =>
        line.startsWith(`|-terastallize|${opposingPrefix}`),
    );
    for (const line of trace) {
        const fields = line.split("|");
        if (!fields[2]?.startsWith(opposingPrefix)) continue;
        if (fields[1] === "cant" || fields[1] === "faint") return undefined;
        if (fields[1] === "move") {
            // Called moves are not the submitted action (e.g. Sleep Talk).
            if (line.includes("[from]")) return undefined;
            return { kind: "move", identity: Dex.toID(fields[3]), tera };
        }
        if (fields[1] === "switch")
            return {
                kind: "switch",
                identity: Dex.toID(fields[3].split(",")[0]),
                tera: false,
            };
    }
    return undefined;
}

export class PayoffRecorder {
    private pending?: { record: PayoffRecord; logIndex: number };
    private attempts = 0;
    private nextTurn = 2;
    constructor(
        private emit: (record: PayoffRecord) => void,
        private identity: { pair: number; game: number },
    ) {}
    begin(player: TrainablePlayerAI, choice: string): void {
        if (this.attempts >= 4 || player.publicBattle.turn < this.nextTurn)
            return;
        let observation: ObservedBattle;
        try {
            observation = captureObservation(player);
        } catch {
            return;
        }
        const record = {
            ...this.identity,
            turn: observation.turn,
            side: player.getPlayerIndex(),
            observation,
            ownAction: normaliseChoice(choice),
            legal: [
                ...new Set(
                    [...player.legalChoiceByCell.values()].map(normaliseChoice),
                ),
            ],
            rootPotential: observedPotential(observation),
        };
        this.pending = { record, logIndex: player.log.length };
        this.attempts++;
        this.nextTurn = observation.turn + 3;
    }
    resolve(player: TrainablePlayerAI): void {
        if (!this.pending) return;
        const { record, logIndex } = this.pending;
        this.pending = undefined;
        record.trace = player.log.slice(logIndex);
        record.reply = resolvedReply(record.trace, record.side);
        try {
            record.next = captureObservation(player);
            if (record.next.turn !== record.turn + 1)
                record.exclusion = "not_one_complete_turn";
            else if (record.trace.some((line) => line.startsWith("|faint|")))
                record.exclusion = "replacement_boundary";
            else if (!record.reply)
                record.exclusion = "opponent_action_not_public";
            else if (
                record.trace.filter((line) => line.startsWith("|switch|"))
                    .length >
                Number(record.ownAction.startsWith("switch ")) +
                    Number(record.reply.kind === "switch")
            )
                record.exclusion = "extra_replacement_or_pivot";
            else record.actualPotential = observedPotential(record.next);
        } catch (error) {
            record.exclusion = String(error);
        }
        this.emit(record);
    }
    finish(): void {
        if (!this.pending) return;
        this.pending.record.exclusion = "terminal_or_truncated_boundary";
        this.emit(this.pending.record);
        this.pending = undefined;
    }
}
