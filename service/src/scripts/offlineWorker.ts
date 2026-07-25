/**
 * Worker-thread body for the offline replay exporter.
 *
 * Receives a list of replay JSON files, replays each spectator protocol log
 * through the SAME state encoder used in live self-play (TrainablePlayerAI +
 * StateHandler) from both players' perspectives, and appends one
 * EnvironmentTrajectory record per (replay, perspective) to its own shard
 * file as [uint32-LE length][serialized proto bytes].
 *
 * Spectator logs carry no |request| lines, so:
 *  - playerIndex is pinned manually per perspective,
 *  - private_team / my_moveset encode as all-unspecified (public-view only),
 *  - the action mask is all-ones (StateHandler already supports a null
 *    request),
 *  - states are emitted at each |turn| boundary plus one terminal state.
 *
 * Must run with CWD=service/ (data.ts loads ../constants and ../data
 * relative to the working directory).
 */

import * as fs from "fs";
import { parentPort, workerData } from "worker_threads";

import { TrainablePlayerAI } from "../server/runner";
import {
    EnvironmentState,
    EnvironmentTrajectory,
} from "../../protos/service_pb";

interface ReplayFile {
    id: string;
    players: string[];
    log: string;
    rating?: number;
}

interface OfflineWorkerData {
    files: string[];
    shardPath: string;
    minRating: number;
    progressEvery: number;
}

export interface OfflineWorkerStats {
    processed: number;
    trajectories: number;
    states: number;
    skippedRating: number;
    failed: number;
}

// The player stream is only consumed by BattlePlayer.start(), which the
// offline path never calls.
const noopStream = {
    write: async () => {},
    read: async () => null,
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
} as any;

function encodePerspective(
    replay: ReplayFile,
    lines: string[],
    playerIndex: 0 | 1,
): { record: Uint8Array; states: number } | null {
    const player = new TrainablePlayerAI(
        replay.players[playerIndex],
        noopStream,
        {},
    );
    player.playerIndex = playerIndex;

    const states: EnvironmentState[] = [];
    for (const line of lines) {
        if (!line.startsWith("|")) {
            continue;
        }
        const cmd = line.slice(1).split("|")[0];
        // getWinReward scans player.log for the |win| line, comparing the
        // winner name against player.userName.
        player.log.push(line);
        if (cmd === "win" || cmd === "tie") {
            player.done = true;
        }
        player.addLine(cmd, line);
        if (cmd === "turn" && !player.done) {
            states.push(player.createGameState());
            player.requestCount += 1;
        }
    }

    if (!player.done) {
        // No decided outcome in the log — useless as a critic target.
        return null;
    }
    states.push(player.createGameState());
    if (states.length < 2) {
        return null;
    }

    const trajectory = new EnvironmentTrajectory();
    trajectory.setStatesList(states);
    return { record: trajectory.serializeBinary(), states: states.length };
}

async function run() {
    const { files, shardPath, minRating, progressEvery } =
        workerData as OfflineWorkerData;

    const stats: OfflineWorkerStats = {
        processed: 0,
        trajectories: 0,
        states: 0,
        skippedRating: 0,
        failed: 0,
    };

    const out = fs.createWriteStream(shardPath);
    const lengthPrefix = Buffer.alloc(4);
    const write = (record: Uint8Array) =>
        new Promise<void>((resolve, reject) => {
            lengthPrefix.writeUInt32LE(record.length, 0);
            out.write(Buffer.from(lengthPrefix));
            // Respect backpressure so shard bytes never pile up in memory.
            if (out.write(Buffer.from(record))) {
                resolve();
            } else {
                out.once("drain", resolve);
                out.once("error", reject);
            }
        });

    for (const file of files) {
        try {
            const replay: ReplayFile = JSON.parse(
                fs.readFileSync(file, "utf-8"),
            );
            if ((replay.rating ?? 0) < minRating) {
                stats.skippedRating += 1;
                continue;
            }
            const lines = replay.log.split("\n");
            for (const playerIndex of [0, 1] as const) {
                const encoded = encodePerspective(replay, lines, playerIndex);
                if (encoded !== null) {
                    await write(encoded.record);
                    stats.trajectories += 1;
                    stats.states += encoded.states;
                }
            }
        } catch (err) {
            stats.failed += 1;
            parentPort?.postMessage({
                type: "error",
                file,
                message: err instanceof Error ? err.message : String(err),
            });
        } finally {
            stats.processed += 1;
            if (stats.processed % progressEvery === 0) {
                parentPort?.postMessage({ type: "progress", stats });
            }
        }
    }

    await new Promise<void>((resolve, reject) => {
        out.end(() => resolve());
        out.once("error", reject);
    });
    parentPort?.postMessage({ type: "done", stats });
}

run().catch((err) => {
    parentPort?.postMessage({
        type: "fatal",
        message: err instanceof Error ? (err.stack ?? err.message) : String(err),
    });
    process.exit(1);
});