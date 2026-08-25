import { WebSocketServer, WebSocket } from "ws";
import { Worker } from "worker_threads";
import path from "path";
import fs from "fs";
import http from "http";
import {
    ClientRequest,
    ErrorResponse,
    ResetRequest,
    StepRequest,
    WorkerRequest,
    WorkerResponse,
} from "../../protos/service_pb";
import pino from "pino";
import { isEvalUser, TaskQueueSystem } from "./utils";
import { availableParallelism } from "node:os";

const WORKER_PATH = path.resolve(__dirname, "../server/worker.js");

// Best-effort bridge to the learner's wandb log: the learner's own
// _log_memory_diagnostics (main-only, main.py-side) reads this file and
// folds it into the same periodic diag_* wandb row it already logs for
// the python process, so node RSS shows up next to diag_rss_mb without
// giving this service a wandb dependency of its own.
const MEMORY_STATS_DIR = path.resolve(__dirname, "../../../runtime");
// Every deployment knob below is overridable from the environment so a
// second instance can be stood up beside the training one for offline
// work (PORT=8081 MAX_WORKERS=1 MEMORY_STATS_PATH=/tmp/x.json node
// dist/server/index.js) without editing or copying this file.
const MEMORY_STATS_PATH =
    process.env.MEMORY_STATS_PATH ??
    path.join(MEMORY_STATS_DIR, "service_memory.json");
const MEMORY_STATS_INTERVAL_MS = 10_000;

interface WorkerInfo {
    worker: Worker;
    id: number;
    // Task ids in flight on this worker — failed back to their clients
    // as ErrorResponses if the worker dies (see spawnWorker).
    pending: Set<number>;
}

export class WorkerPool {
    private tasks: TaskQueueSystem<WorkerResponse>;
    private readonly workerInfos: WorkerInfo[] = [];
    private rr = 0; // round-robin counter for training actors
    private er = 0; // round-robin counter for eval actors

    private readonly sessionToWorkerIndex = new Map<string, number>();
    private closing = false;

    // Per-worker self-reported process.memoryUsage() (worker.ts's
    // reportMemoryStats) — the only way to see each isolate's own heap,
    // since the coordinator's own process.memoryUsage() can't see it.
    private readonly workerMemoryStats = new Map<
        number,
        { heapUsedMb: number; heapTotalMb: number; externalMb: number }
    >();

    constructor(
        private readonly logger: pino.Logger,
        private readonly numWorkers = 4,
    ) {
        this.tasks = new TaskQueueSystem();
        this.spawnWorkers();
    }

    private spawnWorkers(): void {
        for (let i = 0; i < this.numWorkers; i++) {
            this.workerInfos.push(this.spawnWorker(i));
        }
    }

    // One worker isolate at routing index i. A worker that throws
    // (2026-08-23: a TypeError in sendFinalState after a mid-battle
    // destroy) used to be logged and left in the routing table, so every
    // gameId hashing to it — 1/numWorkers of all games — waited on a dead
    // isolate forever. Now its in-flight tasks are failed back to their
    // clients and a fresh isolate takes its slot; sessions that lived on
    // the dead one get "No player found" on their next step, which the
    // python side turns into a BattleError and a new game.
    private spawnWorker(i: number): WorkerInfo {
        const worker = new Worker(WORKER_PATH);
        const info: WorkerInfo = { worker, id: i, pending: new Set() };
        worker.on(
            "message",
            (
                data:
                    | Buffer
                    | Uint8Array
                    | {
                          type: string;
                          heapUsedMb: number;
                          heapTotalMb: number;
                          externalMb: number;
                      },
            ) => {
                // Same posture as worker.ts's own incoming handler:
                // protobuf responses arrive as Uint8Array (Buffer is a
                // subclass, so this covers both), typed messages as a
                // plain tagged object.
                if (data instanceof Uint8Array) {
                    this.onWorkerMsg(Buffer.from(data));
                } else if (data?.type === "memory_stats") {
                    this.workerMemoryStats.set(info.id, {
                        heapUsedMb: data.heapUsedMb,
                        heapTotalMb: data.heapTotalMb,
                        externalMb: data.externalMb,
                    });
                }
            },
        );
        worker.on("error", (err) => console.error(`[worker ${i}] error`, err));
        worker.on("exit", (code) => {
            console.error(
                `[worker ${i}] exited with code ${code}; failing ` +
                    `${info.pending.size} in-flight task(s)` +
                    (this.closing ? "" : " and respawning"),
            );
            this.failPending(info, `worker ${i} exited with code ${code}`);
            if (this.closing) return;
            this.workerInfos[i] = this.spawnWorker(i);
        });
        return info;
    }

    private failPending(info: WorkerInfo, reason: string): void {
        for (const taskId of info.pending) {
            const errorResponse = new ErrorResponse();
            errorResponse.setTrace(reason);
            const workerResponse = new WorkerResponse();
            workerResponse.setErrorResponse(errorResponse);
            try {
                this.tasks.submitResult(taskId, workerResponse);
            } catch (err) {
                console.error("failed to fail pending task", taskId, err);
            }
        }
        info.pending.clear();
    }

    /** Sum of the latest self-reported heap stats across all workers that
     * have reported at least once — a fresher/live worker that hasn't hit
     * its first 10s tick yet is simply absent, not zero, so totals grow in
     * as workers report rather than under-counting from t=0. */
    workerMemoryTotals(): {
        heap_used_mb: number;
        heap_total_mb: number;
        external_mb: number;
        workers_reported: number;
    } {
        let heapUsedMb = 0;
        let heapTotalMb = 0;
        let externalMb = 0;
        for (const stats of this.workerMemoryStats.values()) {
            heapUsedMb += stats.heapUsedMb;
            heapTotalMb += stats.heapTotalMb;
            externalMb += stats.externalMb;
        }
        return {
            heap_used_mb: heapUsedMb,
            heap_total_mb: heapTotalMb,
            external_mb: externalMb,
            workers_reported: this.workerMemoryStats.size,
        };
    }

    /** Deterministic worker assignment for a training pairing: both sides
     * of a game (player and opponent) compute and send the IDENTICAL
     * gameId string before either issues its reset() — see
     * rl/online/main.py's run_training_actor_pair and
     * rl/environment/env.py's SinglePlayerSyncEnvironment.reset(), which
     * puts it on the ResetRequest itself, not just a connection-time
     * header. Hashing that shared string always lands both sides on the
     * SAME worker, regardless of arrival order.
     *
     * Replaces the previous stateful round-robin (`rr += 0.5`, "every 2
     * consecutive reset() calls share a worker"): that invariant only
     * holds if resets arrive in strict, globally-serialized pairs, which
     * concurrent self-play threads never guarantee — each pair's two
     * reset() calls are dispatched via a shared thread pool with no
     * ordering relative to OTHER pairs' calls, so with more than one pair
     * (or more than one worker) in flight, an interleaved arrival could
     * route a player and its own opponent to DIFFERENT workers.
     * pendingGames (worker.ts) is a per-worker map, so two sides of one
     * game on different workers would each wait forever for a match that
     * can never arrive on either — a silent, un-erroring hang identical
     * in shape to a genuine deadlock. This was latent even before the
     * three-population redesign (docs/exploiter-phase-plan.md) — it just
     * needed enough concurrent pairs (or workers) actually in flight to
     * surface, which multiple simultaneously-live populations makes far
     * more likely than the one-population-at-a-time design ever hit. */
    private hashGameId(gameId: string): number {
        // FNV-1a — cheap, well-distributed, no external dependency.
        let hash = 2166136261;
        for (let i = 0; i < gameId.length; i++) {
            hash ^= gameId.charCodeAt(i);
            hash = Math.imul(hash, 16777619);
        }
        return Math.abs(hash) % this.workerInfos.length;
    }

    private trainingWorkerForGameId(
        sessionId: string,
        gameId: string,
    ): WorkerInfo {
        const workerIndex = this.hashGameId(gameId);
        const workerInfo = this.workerInfos[workerIndex];
        this.sessionToWorkerIndex.set(sessionId, workerIndex);
        return workerInfo;
    }

    private nextTrainingWorker(sessionId: string): WorkerInfo {
        // Fallback only — used when a training reset somehow has no
        // gameId (shouldn't happen; env.py always sets one before
        // calling reset()). Kept as the old round-robin rather than
        // always routing to worker 0, so a missing gameId degrades to
        // "no pairing guarantee" instead of "guaranteed wrong worker."
        const workerIndex = Math.floor(this.rr);
        const workerInfo = this.workerInfos[workerIndex];
        this.sessionToWorkerIndex.set(sessionId, workerIndex);
        this.rr = (this.rr + 0.5) % this.workerInfos.length;
        return workerInfo;
    }

    private nextEvalWorker(sessionId: string): WorkerInfo {
        // Eval games never pair with another live actor (worker.ts's
        // resetPlayerFromEvalUserName creates the battle immediately
        // against a scripted baseline), so plain round-robin load
        // distribution is fine here — no shared-gameId invariant to
        // preserve.
        const workerIndex = Math.floor(this.er);
        const workerInfo = this.workerInfos[workerIndex];
        this.sessionToWorkerIndex.set(sessionId, workerIndex);
        this.er = (this.er + 1) % this.workerInfos.length;
        return workerInfo;
    }

    private routedWorker(key: string): WorkerInfo {
        const workerIndex = this.sessionToWorkerIndex.get(key);
        if (workerIndex === undefined) {
            throw new Error("No worker found");
        }
        const info = this.workerInfos[workerIndex];
        if (info === undefined) {
            throw new Error("No worker found");
        }
        return info;
    }

    private onWorkerMsg(buf: Buffer): void {
        const msg = WorkerResponse.deserializeBinary(buf);
        const taskId = msg.getTaskId();

        try {
            this.tasks.submitResult(taskId, msg);
        } catch (err) {
            console.error("failed to handle worker message", err);
        }
    }

    private async send(
        workerInfo: WorkerInfo,
        workerRequest: WorkerRequest,
    ): Promise<WorkerResponse> {
        const taskId = this.tasks.createJob();
        workerRequest.setTaskId(taskId);
        const binaryMessage = workerRequest.serializeBinary();
        workerInfo.pending.add(taskId);
        try {
            workerInfo.worker.postMessage(binaryMessage, [
                binaryMessage.buffer as ArrayBuffer,
            ]);
            const workerResponse = await this.tasks.getResult(taskId);
            workerResponse.setTaskId(taskId);
            return workerResponse;
        } finally {
            workerInfo.pending.delete(taskId);
        }
    }

    async step(stepRequest: StepRequest): Promise<WorkerResponse> {
        const userName = stepRequest.getUsername();
        if (!userName) {
            throw new Error("Username must be provided in step request");
        }
        const info = this.routedWorker(userName);
        const workerRequest = new WorkerRequest();
        workerRequest.setStepRequest(stepRequest);
        return await this.send(info, workerRequest);
    }

    nextWorker(userName: string, gameId?: string): WorkerInfo {
        if (isEvalUser(userName)) {
            return this.nextEvalWorker(userName);
        }
        // Deterministic gameId-hash routing whenever a gameId is present
        // (the normal case — see trainingWorkerForGameId's docstring for
        // why this replaces the old round-robin). Falls back to the
        // round-robin only if gameId is missing/empty.
        if (gameId) {
            return this.trainingWorkerForGameId(userName, gameId);
        }
        return this.nextTrainingWorker(userName);
    }

    async reset(resetRequest: ResetRequest): Promise<WorkerResponse> {
        const userName = resetRequest.getUsername();
        if (!userName) {
            throw new Error("Username must be provided in reset request");
        }
        const info = this.nextWorker(userName, resetRequest.getGameId());
        const workerRequest = new WorkerRequest();
        workerRequest.setResetRequest(resetRequest);
        return await this.send(info, workerRequest);
    }

    /** Graceful shutdown */
    shutdown(): void {
        this.closing = true;
        for (const { worker } of this.workerInfos) worker.terminate();
    }

    workerCount(): number {
        return this.numWorkers;
    }

    /** Fire-and-forget cleanup: tell the worker this session was last routed
     * to drop any pendingGames entry for gameId. A no-op if that session was
     * never routed, or if the entry was already consumed by a successful
     * pairing — used from the connection's close/error handlers, where a
     * disconnect while waiting for an opponent would otherwise leave the
     * entry in worker.ts's pendingGames Map forever. */
    evictPendingGame(userName: string, gameId: string): void {
        const workerIndex = this.sessionToWorkerIndex.get(userName);
        if (workerIndex === undefined) {
            return;
        }
        const info = this.workerInfos[workerIndex];
        if (info === undefined) {
            return;
        }
        info.worker.postMessage({ type: "evict_pending_game", gameId });
    }
}

export class GameServer {
    private wss: WebSocketServer;
    private memoryStatsInterval?: NodeJS.Timeout;
    private logger: pino.Logger;
    private pool: WorkerPool;

    constructor(
        port = 8080,
        options: {
            maxWorkers?: number;
            loggingLevel?: string;
        } = {},
    ) {
        const {
            maxWorkers,
            loggingLevel = "info",
        } = options;

        this.logger = pino({ level: loggingLevel });
        this.wss = new WebSocketServer({ port });
        const safeCpuFrac = 0.8;
        const safeCpuAmount = safeCpuFrac * availableParallelism();
        this.pool = new WorkerPool(this.logger, maxWorkers ?? safeCpuAmount);

        this.wss.on("connection", (ws: WebSocket, req: http.IncomingMessage) =>
            this.handleConnection(ws, req),
        );

        this.logger.info(`Game server started on port ${port}`);

        fs.mkdirSync(MEMORY_STATS_DIR, { recursive: true });
        this.memoryStatsInterval = setInterval(
            () => this.writeMemoryStats(),
            MEMORY_STATS_INTERVAL_MS,
        );
    }

    // Best-effort — a failed write here should never take the game server
    // down. Atomic (tmp + rename) so a concurrent read from the learner
    // side never sees a partial file, same posture as checkpoint writes.
    private writeMemoryStats(): void {
        try {
            const mem = process.memoryUsage();
            const workerMem = this.pool.workerMemoryTotals();
            const tmpPath = `${MEMORY_STATS_PATH}.tmp.${process.pid}`;
            fs.writeFileSync(
                tmpPath,
                JSON.stringify({
                    rss_mb: mem.rss / 2 ** 20,
                    heap_used_mb: mem.heapUsed / 2 ** 20,
                    heap_total_mb: mem.heapTotal / 2 ** 20,
                    external_mb: mem.external / 2 ** 20,
                    array_buffers_mb: mem.arrayBuffers / 2 ** 20,
                    num_workers: this.pool.workerCount(),
                    // Coordinator-thread heap only (see workerMemoryStats'
                    // doc comment) — worker_* below is the actual dex/sim
                    // data, summed across every worker that's reported.
                    worker_heap_used_mb: workerMem.heap_used_mb,
                    worker_heap_total_mb: workerMem.heap_total_mb,
                    worker_external_mb: workerMem.external_mb,
                    workers_reported: workerMem.workers_reported,
                    ts: Date.now() / 1000,
                }),
            );
            fs.renameSync(tmpPath, MEMORY_STATS_PATH);
        } catch (err) {
            this.logger.warn(`Failed to write memory stats: ${err}`);
        }
    }

    private handleConnection(ws: WebSocket, req: http.IncomingMessage): void {
        this.logger.info(`Username ${req.headers.username} connected`);
        const userName = String(req.headers.username ?? "");
        // Most recent gameId this connection asked to be matched on. Used to
        // evict a stale pendingGames entry on disconnect — see
        // WorkerPool.evictPendingGame. Harmless to re-send after a
        // successful pairing (the entry is already gone by then).
        let lastResetGameId: string | undefined;

        ws.on("message", async (clientRequestData: Buffer) => {
            try {
                const clientRequest =
                    ClientRequest.deserializeBinary(clientRequestData);
                const messageType = clientRequest.getMessageTypeCase();

                switch (messageType) {
                    case ClientRequest.MessageTypeCase.STEP: {
                        const stepRequest = clientRequest.getStep();
                        if (stepRequest !== undefined) {
                            const workerResponse =
                                await this.pool.step(stepRequest);
                            ws.send(workerResponse.serializeBinary());
                        } else {
                            throw new Error("StepRequest not defined");
                        }
                        break;
                    }
                    case ClientRequest.MessageTypeCase.RESET: {
                        const resetRequest = clientRequest.getReset();
                        if (resetRequest !== undefined) {
                            lastResetGameId = resetRequest.getGameId();
                            const workerResponse =
                                await this.pool.reset(resetRequest);
                            ws.send(workerResponse.serializeBinary());
                        } else {
                            throw new Error("StepRequest not defined");
                        }
                        break;
                    }
                }
            } catch (err) {
                console.error(`Error handling message from ${userName}:`, err);
            }
        });

        ws.on("error", (err) => {
            this.logger.error(err);
            if (lastResetGameId) {
                this.pool.evictPendingGame(userName, lastResetGameId);
            }
        });

        ws.on("close", () => {
            this.logger.info(`Username ${req.headers.username} disconnected`);
            if (lastResetGameId) {
                this.pool.evictPendingGame(userName, lastResetGameId);
            }
        });
    }

    public close(): void {
        if (this.memoryStatsInterval) {
            clearInterval(this.memoryStatsInterval);
        }
        this.wss.close();
        this.logger.info("Game server closed");
    }
}

// Initialize the server
new GameServer(Number(process.env.PORT ?? 8080), {
    // Each worker is a full V8 isolate with its own dex/sim data
    // (~150MB baseline before any battles). Under the learner's
    // block-sequential actor gating at most ONE population's pool plays
    // at a time (~6 self-play games + 3 eval), and a worker steps many
    // concurrent battles fine — sim stepping is far cheaper than the
    // python side's inference. 6 halves the idle baseline vs the old 12.
    maxWorkers: Number(process.env.MAX_WORKERS ?? 6),
    loggingLevel: process.env.LOG_LEVEL ?? "info", // 'debug' for more
});
