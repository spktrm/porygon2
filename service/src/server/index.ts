import { WebSocketServer, WebSocket } from "ws";
import { Worker } from "worker_threads";
import path from "path";
import http from "http";
import {
    ClientRequest,
    EnvironmentResponse,
    ResetRequest,
    StepRequest,
    WorkerRequest,
    WorkerResponse,
} from "../../protos/service_pb";
import pino from "pino";
import { isEvalUser, TaskQueueSystem } from "./utils";
import { availableParallelism } from "node:os";

const WORKER_PATH = path.resolve(__dirname, "../server/worker.js");

interface WorkerInfo {
    worker: Worker;
    id: number;
}

export class WorkerPool {
    private tasks: TaskQueueSystem<WorkerResponse>;
    private readonly workerInfos: WorkerInfo[] = [];
    private rr = 0; // round-robin counter for training actors
    private er = 0; // round-robin counter for eval actors

    private readonly sessionToWorkerIndex = new Map<string, number>();

    constructor(
        private readonly logger: pino.Logger,
        private readonly numWorkers = 4,
    ) {
        this.tasks = new TaskQueueSystem();
        this.spawnWorkers();
    }

    private spawnWorkers(): void {
        for (let i = 0; i < this.numWorkers; i++) {
            const worker = new Worker(WORKER_PATH);
            const info: WorkerInfo = { worker, id: i };

            worker.on("message", (buf: Buffer) => this.onWorkerMsg(buf));
            worker.on("error", (err) =>
                console.error(`[worker ${i}] error`, err),
            );
            worker.on("exit", (code) =>
                console.log(`[worker ${i}] exited`, code),
            );

            this.workerInfos.push(info);
        }
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
        workerInfo.worker.postMessage(binaryMessage, [
            binaryMessage.buffer as ArrayBuffer,
        ]);
        const workerResponse = await this.tasks.getResult(taskId);
        workerResponse.setTaskId(taskId);
        return workerResponse;
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
        for (const { worker } of this.workerInfos) worker.terminate();
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
    private actionCount: number;
    private throughputIntervalMs: number;
    private throughputInterval?: NodeJS.Timeout;
    private logger: pino.Logger;
    private pool: WorkerPool;

    constructor(
        port = 8080,
        options: {
            maxGamesPerWorker?: number;
            maxWorkers?: number;
            loggingLevel?: string;
            logThroughput?: boolean;
            throughputIntervalMs?: number;
        } = {},
    ) {
        const {
            maxWorkers,
            loggingLevel = "info",
            logThroughput = false,
            throughputIntervalMs = 5000,
        } = options;

        this.logger = pino({ level: loggingLevel });
        this.wss = new WebSocketServer({ port });
        this.actionCount = 0;
        const safeCpuFrac = 0.8;
        const safeCpuAmount = Math.min(safeCpuFrac * availableParallelism());
        this.pool = new WorkerPool(this.logger, maxWorkers ?? safeCpuAmount);

        this.wss.on("connection", (ws: WebSocket, req: http.IncomingMessage) =>
            this.handleConnection(ws, req),
        );

        this.logger.info(`Game server started on port ${port}`);

        this.throughputIntervalMs = throughputIntervalMs;
        if (logThroughput) {
            this.throughputInterval = setInterval(
                () => this.logThroughput(),
                throughputIntervalMs,
            );
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

    private logThroughput(): void {
        this.logger.info(
            `Throughput: ${
                (1000 * this.actionCount) / this.throughputIntervalMs
            } actions/sec`,
        );
        this.actionCount = 0;
    }

    public close(): void {
        if (this.throughputInterval) {
            clearInterval(this.throughputInterval);
        }
        this.wss.close();
        this.logger.info("Game server closed");
    }
}

// Initialize the server
new GameServer(8080, {
    maxGamesPerWorker: 50,
    maxWorkers: 12,
    loggingLevel: "info", // Set to 'debug' for more verbose logging
    logThroughput: false,
    throughputIntervalMs: 1000,
});
