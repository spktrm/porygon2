import * as fs from "fs";
import * as path from "path";
import { Battle } from "@pkmn/client";
import { Generations } from "@pkmn/data";
import { Dex } from "@pkmn/dex";
import { Protocol } from "@pkmn/protocol";

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

/**
 * Reconstructs the state of the battle from the raw protocol log.
 */
function parseReplayPerspective(replay: ReplayFile) {
    const battle = new Battle(new Generations(Dex));

    // Stream the log into the client state machine
    const lines = replay.log.split("\n");
    for (const line of lines) {
        battle.add(line);
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

/**
 * Main execution logic
 */
async function runOfflineProcessing() {
    // Path: service/src/scripts/../../.. => Project Root
    // Then: replays/data/gen9ou/
    const REPLAY_DIR = path.resolve(__dirname, "../../../replays/data/gen9ou");

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

            const battleState = parseReplayPerspective(json);
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
