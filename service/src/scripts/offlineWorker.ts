/**
 * Worker-thread body for the offline replay exporter.
 *
 * Receives a list of replay JSON files, replays each spectator protocol log
 * through the SAME state encoder used in live self-play (TrainablePlayerAI +
 * StateHandler) from both players' perspectives, and appends one
 * EnvironmentTrajectory per (replay, perspective), both in one
 * EnvironmentBatch record per replay, to its own shard file as
 * [uint32-LE length][serialized proto bytes].
 *
 * Spectator logs carry no |request| lines, so:
 *  - playerIndex is pinned manually per perspective,
 *  - private_team / my_moveset encode as all-unspecified (public-view only),
 *  - the action mask is all-ones (StateHandler already supports a null
 *    request),
 *  - states are emitted at every committed history edge (one per major
 *    arg -- move, switch/drag/replace, cant, faint -- taken BEFORE the line
 *    that opens the next edge, so a slice never carries the next event's
 *    announcement), at each |turn| boundary, plus one terminal state.
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
    verbose: boolean;
}

export interface OfflineWorkerStats {
    processed: number;
    trajectories: number;
    states: number;
    skippedRating: number;
    skippedShort: number;
    skippedFormat: number;
    failed: number;
    // console.log/warn/error calls swallowed from the sim/state encoder
    // while verbose was off — replay logs routinely trip protocol warnings.
    warnings: number;
}

// The player stream is only consumed by BattlePlayer.start(), which the
// offline path never calls.
const noopStream = {
    write: async () => {},
    read: async () => null,
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
} as any;

// The lines whose handlers commit the pending edge (state.ts addEdge): the
// major args, and the bare "|" block separator the protocol parser reports
// as |done| -- it follows every action's effect lines, so the state before
// it is the post-effect state of that edge. |turn| commits too and is
// handled in the loop.
const EDGE_OPENING_CMDS = new Set([
    "move",
    "switch",
    "drag",
    "replace",
    "cant",
    "faint",
    "",
]);

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

    // Pre-game ladder ratings from the |player| lines
    // (|player|p1|name|avatar|rating). MUST come from the log, not replay
    // metadata: anything recorded at upload time can be post-game, and
    // post-game ratings leak the result. Unrated games leave 0 (unknown).
    for (const line of lines) {
        const match = /^\|player\|p([12])\|[^|]*\|[^|]*\|(\d+)/.exec(line);
        if (match) {
            player.ratings[Number(match[1]) - 1] = Number(match[2]);
        }
    }

    const states: EnvironmentState[] = [];
    // The stream position (last edge held) of each pushed slice: one slice
    // per position, so a boundary slice replaces a same-position slice and
    // the terminal state stands for the last edge.
    const positions: number[] = [];
    const edgeBuffer = player.eventHandler.edgeBuffer;
    for (const line of lines) {
        if (!line.startsWith("|")) {
            continue;
        }
        const cmd = line.slice(1).split("|")[0];
        // A major arg commits the pending edge inside its own handler, so
        // the state of the committed edge is the one BEFORE this line;
        // taken speculatively, kept only if a commit follows (the first
        // major arg after |turn| finds an empty edge and commits nothing).
        // History caches are shared per trajectory (RL Trajectory
        // convention): only the terminal state carries them, so
        // non-terminal states skip the O(history) snapshot and records
        // stay O(T) instead of O(T^2).
        let edgeState: EnvironmentState | undefined;
        const edgesBefore = edgeBuffer.numEdges;
        if (EDGE_OPENING_CMDS.has(cmd) && !player.done) {
            edgeState = player.createGameState(false, edgesBefore + 1);
        }
        // getWinReward scans player.log for the |win| line, comparing the
        // winner name against player.userName.
        player.log.push(line);
        if (cmd === "win" || cmd === "tie") {
            player.done = true;
        }
        player.addLine(cmd, line);
        if (edgeState !== undefined && edgeBuffer.numEdges > edgesBefore) {
            states.push(edgeState);
            positions.push(edgesBefore + 1);
        }
        if (cmd === "turn" && !player.done) {
            // The boundary slice is taken AFTER the line, as the live
            // request is (turn number advanced, the pending edge
            // committed); when the separator already committed it, the
            // boundary slice takes that position over. The request count
            // advances once per turn so the edge features keep the live
            // distribution.
            if (positions.at(-1) === edgeBuffer.numEdges) {
                states.pop();
                positions.pop();
            }
            states.push(player.createGameState(false));
            positions.push(edgeBuffer.numEdges);
            player.requestCount += 1;
        }
    }

    if (!player.done) {
        // No decided outcome in the log — useless as a critic target.
        return null;
    }
    if (positions.at(-1) === edgeBuffer.numEdges) {
        states.pop();
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
    const {
        files,
        shardPath,
        formatId,
        minRating,
        minTurns,
        progressEvery,
        verbose,
    } = workerData as OfflineWorkerData;

    const stats: OfflineWorkerStats = {
        processed: 0,
        trajectories: 0,
        states: 0,
        skippedRating: 0,
        skippedShort: 0,
        skippedFormat: 0,
        failed: 0,
        warnings: 0,
    };

    if (!verbose) {
        // The sim/state encoder warns liberally while replaying spectator
        // logs (unknown idents, stream errors, ...). Swallow-and-count so
        // the orchestrator's progress line stays intact; --verbose lets
        // everything through. Worker-scoped: never affects the live server.
        const swallow = () => {
            stats.warnings += 1;
        };
        console.log = swallow;
        console.warn = swallow;
        console.error = swallow;
        console.debug = swallow;
    }

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
