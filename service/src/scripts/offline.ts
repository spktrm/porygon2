import * as fs from "fs";
import * as path from "path";
import { Battle } from "@pkmn/client";
import { Generations } from "@pkmn/data";
import { Dex } from "@pkmn/dex";
import {
    Args,
    BattleArgKWArgs,
    BattleArgs,
    KWArgs,
    Protocol,
} from "@pkmn/protocol";
import {
    calculate,
    Generations as CalcGenerations,
    Pokemon as CalcPokemon,
    Move as CalcMove,
} from "@smogon/calc";

/**
 * Matches your JSON structure
 */
interface ReplayFile {
    id: string;
    format: string;
    players: string[];
    log: string;
    uploadtime: number;
    views: number;
    formatid: string;
    rating?: number;
}

// 1. Define the interfaces representing the JSON structure
export interface SmogonStatsInfo {
    metagame: string;
    cutoff: number;
    "cutoff deviation"?: number;
    "team type"?: number;
    "number of battles": number;
}

export interface PokemonUsageStats {
    usage: number; // Weighted usage fraction
    Raw: number; // Raw usage count
    viability?: [number, number, number, number]; // [ceiling, floor, ..., ...]
    Abilities: Record<string, number>; // Ability name -> usage fraction
    Items: Record<string, number>; // Item name -> usage fraction
    Moves: Record<string, number>; // Move name -> usage fraction
    Spreads?: Record<string, number>; // Nature:EVs -> usage fraction
    Happiness?: Record<string, number>;
    Teammates: Record<string, number>; // Teammate name -> usage fraction
    ChecksAndCounters: Record<string, [number, number, number]>; // [KOed, Switched out, Score]

    // Note: Depending on the specific @pkmn/stats parser version,
    // these keys might be lowercase (e.g., 'abilities', 'items').
}

export interface SmogonStatsJSON {
    info: SmogonStatsInfo;
    data: Record<string, PokemonUsageStats>;
}

function getUserTargetMove(
    lines: string[],
    index: number,
    calcGen: ReturnType<typeof CalcGenerations.get>,
    stats: SmogonStatsJSON,
): { users: CalcPokemon[]; targets: CalcPokemon[]; move: CalcMove } {
    const { args, kwArgs } = Protocol.parseBattleLine(lines[index]) as {
        args: Args["|move|"];
        kwArgs: KWArgs["|move|"];
    };
    const damageLines = [];

    const users: never[] = [];
    const targets: never[] = [];
    const [_, __, moveId] = args;
    const move = new CalcMove(calcGen, moveId);

    return { users, targets, move };
}

/**
 * Reconstructs the state of the battle from the raw protocol log.
 */
function parseReplayPerspective(replay: ReplayFile, stats: SmogonStatsJSON) {
    const battle = new Battle(new Generations(Dex));
    const calcGen = CalcGenerations.get(battle.gen.num);

    // Stream the log into the client state machine
    const lines = replay.log.split("\n");
    for (const line of lines) {
        battle.add(line);
    }

    for (const [index, line] of lines.entries()) {
        if (line.startsWith("|move|")) {
            const { users, targets, move } = getUserTargetMove(
                lines,
                index,
                calcGen,
                stats,
            );
            for (const user of users) {
                for (const target of targets) {
                    const result = calculate(calcGen, user, target, move);
                }
            }
        }
    }

    return {
        id: replay.id,
        turn: battle.turn,
        p1: {
            name: battle.p1.name,
            team: battle.p1.team.map((p) => p.species.name),
        },
        p2: {
            name: battle.p2.name,
            team: battle.p2.team.map((p) => p.species.name),
        },
        weather: battle.field.weather,
    };
}

export async function fetchStats(
    generation: number,
    smogonFormat: string,
): Promise<SmogonStatsJSON> {
    const url = `https://raw.githubusercontent.com/pkmn/smogon/refs/heads/main/data/stats/gen${generation}${smogonFormat}.json`;

    try {
        const response = await fetch(url);

        if (!response.ok) {
            throw new Error(
                `Failed to fetch stats: ${response.status} ${response.statusText}`,
            );
        }

        // Parse and cast the JSON response to our defined types
        const statsData = (await response.json()) as SmogonStatsJSON;
        return statsData;
    } catch (error) {
        console.error("Error fetching Gen 9 OU stats:", error);
        throw error;
    }
}

/**
 * Main execution logic
 */
async function runOfflineProcessing() {
    // Path: service/src/scripts/../../.. => Project Root
    // Then: replays/data/gen9ou/
    const REPLAY_DIR = path.resolve(__dirname, "../../../replays/data/gen9ou");
    const stats = await fetchStats(9, "ou");

    if (!fs.existsSync(REPLAY_DIR)) {
        console.error(
            `Error: Could not find replay directory at ${REPLAY_DIR}`,
        );
        return;
    }

    const files = fs.readdirSync(REPLAY_DIR).filter((f) => f.endsWith(".json"));
    console.log(`Found ${files.length} replays in ${REPLAY_DIR}\n`);

    const results = [];

    for (const file of files) {
        try {
            const filePath = path.join(REPLAY_DIR, file);
            const content = fs.readFileSync(filePath, "utf-8");
            const json: ReplayFile = JSON.parse(content);

            const battleState = parseReplayPerspective(json, stats);
            results.push(battleState);

            console.log(
                `Processed: ${battleState.id} | Turns: ${battleState.turn}`,
            );
        } catch (err) {
            console.error(`Failed to parse ${file}:`, err);
        }
    }

    console.log("\nProcessing complete.");
    // You could now save 'results' to a database or a summary JSON file
}

runOfflineProcessing();
