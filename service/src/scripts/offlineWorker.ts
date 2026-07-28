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
import { AnyObject } from "@pkmn/sim";

import { TrainablePlayerAI } from "../server/runner";
import {
    EnvironmentBatch,
    EnvironmentState,
    EnvironmentTrajectory,
} from "../../protos/service_pb";

export interface ReplayFile {
    id: string;
    players: string[];
    log: string;
    formatid?: string;
    rating?: number;
}

interface OfflineWorkerData {
    files: string[];
    shardPath: string;
    formatId: string;
    minRating: number;
    minTurns: number;
    progressEvery: number;
}

export interface OfflineWorkerStats {
    processed: number;
    trajectories: number;
    states: number;
    skippedRating: number;
    skippedShort: number;
    skippedFormat: number;
    failed: number;
}

// The player stream is only consumed by BattlePlayer.start(), which the
// offline path never calls.
const noopStream = {
    write: async () => {},
    read: async () => null,
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
} as any;

class OfflinePlayerAI extends TrainablePlayerAI {
    override getRequest(): AnyObject {
        // Spectator logs carry no |request| lines, so battle.request stays
        // undefined forever. StateHandler distinguishes null ("legitimately
        // absent — offline replay") from undefined ("live-path invariant
        // violation"), so normalize to null here.
        return this.privateBattle.request ?? (null as unknown as AnyObject);
    }
}

export function encodePerspective(
    replay: ReplayFile,
    lines: string[],
    playerIndex: 0 | 1,
): { trajectory: EnvironmentTrajectory; states: number } | null {
    const player = new OfflinePlayerAI(
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
            // History caches are shared per trajectory (RL Trajectory
            // convention): only the terminal state carries them, so
            // non-terminal states skip the O(history) snapshot and records
            // stay O(T) instead of O(T^2).
            states.push(player.createGameState(false));
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
    return { trajectory, states: states.length };
}

async function run() {
    const { files, shardPath, formatId, minRating, minTurns, progressEvery } =
        workerData as OfflineWorkerData;

    const stats: OfflineWorkerStats = {
        processed: 0,
        trajectories: 0,
        states: 0,
        skippedRating: 0,
        skippedShort: 0,
        skippedFormat: 0,
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
            // A stray file from another format would silently poison the
            // dataset with a different observation distribution.
            if (replay.formatid !== undefined && replay.formatid !== formatId) {
                stats.skippedFormat += 1;
                continue;
            }
            if ((replay.rating ?? 0) < minRating) {
                stats.skippedRating += 1;
                continue;
            }
            const lines = replay.log.split("\n");
            // Very short games are where the outcome stops correlating with
            // position quality (early forfeits, disconnects), so they make
            // poor critic targets. Turn count is perspective-independent —
            // check it once before paying for any encoding.
            const numTurns = lines.filter((l) => l.startsWith("|turn|")).length;
            if (numTurns < minTurns) {
                stats.skippedShort += 1;
                continue;
            }
            const batch = new EnvironmentBatch();
            let maxLength = 0;
            for (const playerIndex of [0, 1] as const) {
                const encoded = encodePerspective(replay, lines, playerIndex);
                if (encoded !== null) {
                    batch.addTrajectories(encoded.trajectory);
                    maxLength = Math.max(maxLength, encoded.states);
                    stats.trajectories += 1;
                    stats.states += encoded.states;
                }
            }
            if (batch.getTrajectoriesList().length > 0) {
                // One record per replay: both perspectives travel together
                // so the trainer's train/eval split is by game, never
                // separating a game from its mirrored (label-flipped) twin.
                batch.setMaxTrajectoryLength(maxLength);
                await write(batch.serializeBinary());
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

// Only run when loaded as a worker thread — the encode helpers above are
// also imported directly by one-shot tools (scripts/exportReplay.ts).
if (parentPort) {
    run().catch((err) => {
        parentPort?.postMessage({
            type: "fatal",
            message:
                err instanceof Error ? (err.stack ?? err.message) : String(err),
        });
        process.exit(1);
    });
}
