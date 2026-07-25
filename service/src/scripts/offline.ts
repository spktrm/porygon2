/**
 * Offline replay exporter — orchestrator.
 *
 * Fans replay JSONs (downloaded by replays/main.py into replays/data/) out
 * across a worker_threads pool. Each worker replays the logs through the
 * live state encoder and writes one shard of length-prefixed
 * EnvironmentTrajectory protos. The Python offline trainer
 * (rl/offline/dataset.py) consumes the shards directly.
 *
 * Usage (from service/, after tsc):
 *   node dist/scripts/offline.js gen9randombattle \
 *     [--workers N] [--min-rating R] [--limit K] [--out DIR]
 */

import * as fs from "fs";
import * as os from "os";
import * as path from "path";
import { Worker } from "worker_threads";

import type { OfflineWorkerStats } from "./offlineWorker";

const WORKER_PATH = path.resolve(__dirname, "offlineWorker.js");
const PROJECT_ROOT = path.resolve(__dirname, "../../..");

interface Args {
    formatId: string;
    workers: number;
    minRating: number;
    limit: number;
    outDir: string;
}

function parseArgs(argv: string[]): Args {
    const positional: string[] = [];
    const flags: Record<string, string> = {};
    for (let i = 0; i < argv.length; i++) {
        const arg = argv[i];
        if (arg.startsWith("--")) {
            flags[arg.slice(2)] = argv[++i];
        } else {
            positional.push(arg);
        }
    }
    const formatId = positional[0] ?? "gen9randombattle";
    return {
        formatId,
        workers:
            parseInt(flags["workers"] ?? "") ||
            Math.max(1, os.availableParallelism() - 1),
        minRating: parseInt(flags["min-rating"] ?? "") || 0,
        limit: parseInt(flags["limit"] ?? "") || Infinity,
        outDir:
            flags["out"] ?? path.join(PROJECT_ROOT, "replays", "shards"),
    };
}

function emptyStats(): OfflineWorkerStats {
    return {
        processed: 0,
        trajectories: 0,
        states: 0,
        skippedRating: 0,
        failed: 0,
    };
}

async function main() {
    const args = parseArgs(process.argv.slice(2));

    const replayDir = path.join(
        PROJECT_ROOT,
        "replays",
        "data",
        args.formatId,
    );
    if (!fs.existsSync(replayDir)) {
        console.error(`No replay directory at ${replayDir}`);
        process.exit(1);
    }
    let files = fs
        .readdirSync(replayDir)
        .filter((f) => f.endsWith(".json"))
        .map((f) => path.join(replayDir, f));
    if (files.length > args.limit) {
        files = files.slice(0, args.limit);
    }

    const shardDir = path.join(args.outDir, args.formatId);
    fs.mkdirSync(shardDir, { recursive: true });

    const numWorkers = Math.min(args.workers, files.length) || 1;
    console.log(
        `Exporting ${files.length} replays from ${replayDir} ` +
            `across ${numWorkers} workers -> ${shardDir}`,
    );

    const perWorker: OfflineWorkerStats[] = [];
    const startTime = Date.now();

    const logProgress = () => {
        const total = perWorker.reduce(
            (acc, s) => ({
                processed: acc.processed + s.processed,
                trajectories: acc.trajectories + s.trajectories,
                states: acc.states + s.states,
                skippedRating: acc.skippedRating + s.skippedRating,
                failed: acc.failed + s.failed,
            }),
            emptyStats(),
        );
        const elapsed = (Date.now() - startTime) / 1000;
        const rate = total.processed / Math.max(elapsed, 1e-6);
        process.stdout.write(
            `\r${total.processed}/${files.length} replays | ` +
                `${total.trajectories} trajectories | ${total.states} states | ` +
                `${total.failed} failed | ${rate.toFixed(1)} replays/s   `,
        );
        return total;
    };

    const runs = Array.from({ length: numWorkers }, (_, workerIndex) => {
        const workerFiles = files.filter(
            (_, i) => i % numWorkers === workerIndex,
        );
        const shardPath = path.join(
            shardDir,
            `shard-${String(workerIndex).padStart(3, "0")}.bin`,
        );
        perWorker[workerIndex] = emptyStats();

        return new Promise<void>((resolve, reject) => {
            const worker = new Worker(WORKER_PATH, {
                workerData: {
                    files: workerFiles,
                    shardPath,
                    minRating: args.minRating,
                    progressEvery: 25,
                },
            });
            worker.on("message", (msg) => {
                if (msg.type === "progress" || msg.type === "done") {
                    perWorker[workerIndex] = msg.stats;
                    logProgress();
                } else if (msg.type === "error") {
                    console.error(`\n[worker ${workerIndex}] ${msg.file}: ${msg.message}`);
                } else if (msg.type === "fatal") {
                    console.error(`\n[worker ${workerIndex}] fatal: ${msg.message}`);
                }
            });
            worker.on("error", reject);
            worker.on("exit", (code) =>
                code === 0
                    ? resolve()
                    : reject(new Error(`worker ${workerIndex} exited ${code}`)),
            );
        });
    });

    await Promise.all(runs);
    const total = logProgress();

    const manifest = {
        format_id: args.formatId,
        source: path.relative(PROJECT_ROOT, replayDir),
        min_rating: args.minRating,
        num_replays: total.processed,
        num_trajectories: total.trajectories,
        num_states: total.states,
        num_failed: total.failed,
        num_skipped_rating: total.skippedRating,
        shards: numWorkers,
        created_at: new Date().toISOString(),
        record_format:
            "repeated [uint32-LE length][EnvironmentTrajectory proto bytes]",
        perspective: "both players; public-view only (no private info)",
    };
    fs.writeFileSync(
        path.join(shardDir, "manifest.json"),
        JSON.stringify(manifest, null, 2),
    );
    console.log(`\nWrote manifest to ${path.join(shardDir, "manifest.json")}`);
}

main();