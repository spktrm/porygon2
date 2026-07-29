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
 *     [--workers N] [--min-rating R] [--min-turns T] [--limit K] [--out DIR] \
 *     [--verbose]
 *
 * Sim warnings and per-file errors are counted but silenced by default so
 * the progress line stays readable; --verbose prints them all.
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
    minTurns: number;
    limit: number;
    outDir: string;
    verbose: boolean;
}

function parseArgs(argv: string[]): Args {
    const positional: string[] = [];
    const flags: Record<string, string> = {};
    for (let i = 0; i < argv.length; i++) {
        const arg = argv[i];
        if (arg.startsWith("--")) {
            // Boolean flags: no value token follows (end of argv or
            // another --flag).
            const next = argv[i + 1];
            if (next === undefined || next.startsWith("--")) {
                flags[arg.slice(2)] = "true";
            } else {
                flags[arg.slice(2)] = argv[++i];
            }
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
        minTurns: parseInt(flags["min-turns"] ?? "") || 5,
        limit: parseInt(flags["limit"] ?? "") || Infinity,
        outDir: flags["out"] ?? path.join(PROJECT_ROOT, "replays", "shards"),
        verbose: flags["verbose"] === "true",
    };
}

function emptyStats(): OfflineWorkerStats {
    return {
        processed: 0,
        trajectories: 0,
        states: 0,
        skippedRating: 0,
        skippedShort: 0,
        skippedFormat: 0,
        failed: 0,
        warnings: 0,
    };
}

function formatEta(seconds: number): string {
    if (!isFinite(seconds)) return "—";
    const s = Math.round(seconds);
    if (s < 90) return `${s}s`;
    const m = Math.floor(s / 60);
    if (m < 90) return `${m}m${String(s % 60).padStart(2, "0")}s`;
    return `${Math.floor(m / 60)}h${String(m % 60).padStart(2, "0")}m`;
}

async function main() {
    const args = parseArgs(process.argv.slice(2));

    const replayDir = path.join(PROJECT_ROOT, "replays", "data", args.formatId);
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
                skippedShort: acc.skippedShort + s.skippedShort,
                skippedFormat: acc.skippedFormat + s.skippedFormat,
                failed: acc.failed + s.failed,
                warnings: acc.warnings + s.warnings,
            }),
            emptyStats(),
        );
        const elapsed = (Date.now() - startTime) / 1000;
        const rate = total.processed / Math.max(elapsed, 1e-6);
        const pct = ((100 * total.processed) / files.length).toFixed(0);
        const eta = (files.length - total.processed) / Math.max(rate, 1e-6);
        process.stdout.write(
            `\r${total.processed}/${files.length} (${pct}%) | ` +
                `${total.trajectories} trajectories | ${total.states} states | ` +
                `${total.failed} failed | ${rate.toFixed(1)}/s | ` +
                `ETA ${formatEta(eta)}   `,
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
                    formatId: args.formatId,
                    minRating: args.minRating,
                    minTurns: args.minTurns,
                    progressEvery: 25,
                    verbose: args.verbose,
                },
            });
            worker.on("message", (msg) => {
                if (msg.type === "progress" || msg.type === "done") {
                    perWorker[workerIndex] = msg.stats;
                    logProgress();
                } else if (msg.type === "error") {
                    // Counted in stats.failed either way; only verbose runs
                    // spell each file out (they shred the progress line).
                    if (args.verbose) {
                        console.error(
                            `\n[worker ${workerIndex}] ${msg.file}: ${msg.message}`,
                        );
                    }
                } else if (msg.type === "fatal") {
                    console.error(
                        `\n[worker ${workerIndex}] fatal: ${msg.message}`,
                    );
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
        min_turns: args.minTurns,
        num_replays: total.processed,
        num_trajectories: total.trajectories,
        num_states: total.states,
        num_failed: total.failed,
        num_skipped_rating: total.skippedRating,
        num_skipped_short: total.skippedShort,
        num_skipped_format: total.skippedFormat,
        num_sim_warnings: total.warnings,
        shards: numWorkers,
        created_at: new Date().toISOString(),
        record_format:
            "repeated [uint32-LE length][EnvironmentBatch proto bytes] — " +
            "one record per replay holding both perspectives",
        history: "terminal state only — shared per trajectory (RL convention)",
        perspective: "both players; public-view only (no private info)",
    };
    fs.writeFileSync(
        path.join(shardDir, "manifest.json"),
        JSON.stringify(manifest, null, 2),
    );
    console.log(`\nWrote manifest to ${path.join(shardDir, "manifest.json")}`);
    if (!args.verbose && (total.warnings > 0 || total.failed > 0)) {
        console.log(
            `${total.warnings} sim warnings and ${total.failed} per-file ` +
                `errors were suppressed — rerun with --verbose to see them.`,
        );
    }
}

main();
