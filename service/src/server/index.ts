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

    private nextTrainingWorker(sessionId: string): WorkerInfo {
        // Increment by 0.5 so as to allocate groups of 2 players to a worker
        const workerIndex = Math.floor(this.rr);
        const workerInfo = this.workerInfos[workerIndex];
        this.sessionToWorkerIndex.set(sessionId, workerIndex);
        this.rr = (this.rr + 0.5) % this.workerInfos.length;
        return workerInfo;
    }

    private nextEvalWorker(sessionId: string): WorkerInfo {
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

    nextWorker(userName: string): WorkerInfo {
        if (isEvalUser(userName)) {
            return this.nextEvalWorker(userName);
        } else {
            return this.nextTrainingWorker(userName);
        }
    }

    async reset(resetRequest: ResetRequest): Promise<WorkerResponse> {
        const userName = resetRequest.getUsername();
        if (!userName) {
            throw new Error("Username must be provided in reset request");
        }
        const info = this.nextWorker(userName);
        const workerRequest = new WorkerRequest();
        workerRequest.setResetRequest(resetRequest);
        return await this.send(info, workerRequest);
    }

    /** Graceful shutdown */
    shutdown(): void {
        for (const { worker } of this.workerInfos) worker.terminate();
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
        this.pool = new WorkerPool(
            this.logger,
            Math.min(maxWorkers ?? 1, availableParallelism()),
        );

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

        ws.on("message", async (clientRequestData: Buffer) => {
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
                        const workerResponse =
                            await this.pool.reset(resetRequest);
                        ws.send(workerResponse.serializeBinary());
                    } else {
                        throw new Error("StepRequest not defined");
                    }
                    break;
                }
            }
        });

        ws.on("error", (err) => {
            this.logger.error(err);
        });

        ws.on("close", () => {
            this.logger.info(`Username ${req.headers.username} disconnected`);
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
    // maxWorkers: 16,
    loggingLevel: "info", // Set to 'debug' for more verbose logging
    logThroughput: false,
    throughputIntervalMs: 1000,
});
