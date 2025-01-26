import { WebSocketServer, WebSocket } from "ws";
import { Worker } from "worker_threads";
import path from "path";
import {
    ClientMessage,
    GameState,
    ServerMessage,
} from "../../protos/service_pb";
import pino from "pino";

class GameWorker {
    worker: Worker;
    workerIndex: number;
    playerCount: number;
    players: Map<number, WebSocket>;
    logger: pino.Logger;

    constructor(workerIndex: number, logger: pino.Logger) {
        const workerPath = path.resolve(__dirname, "../server/worker.js");

        this.players = new Map();

        this.worker = new Worker(workerPath, {
            workerData: { workerIndex },
        });

        this.logger = logger.child({ workerIndex });

        this.worker.on("message", (stateBuffer: Buffer) => {
            const gameState = GameState.deserializeBinary(stateBuffer);
            const playerId = gameState.getPlayerId();
            const player = this.players.get(playerId)!;
            this.logger.debug(`Sending game state to player ${playerId}`);
            const serverMessage = new ServerMessage();
            serverMessage.setGameState(gameState);
            player.send(serverMessage.serializeBinary());
        });

        this.worker.on("error", (err) => {
            this.logger.error(err.stack?.toString());
        });

        this.worker.on("exit", (code: number) => {
            this.logger.info(`Worker exited with code ${code}`);
        });

        this.workerIndex = workerIndex;
        this.playerCount = 0;
    }
}

export class GameServer {
    private wss: WebSocketServer;
    private gameWorkerMap: Map<number, GameWorker>;
    private socketWorkerMap: Map<WebSocket, GameWorker>;
    private workers: GameWorker[];
    private maxGamesPerWorker: number;
    private maxWorkers: number;
    private actionCount: number;
    private throughputIntervalMs: number;
    private throughputInterval?: NodeJS.Timeout;
    private logger: pino.Logger;

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
            maxGamesPerWorker = 5,
            maxWorkers = 10,
            loggingLevel = "info",
            logThroughput = false,
            throughputIntervalMs = 5000,
        } = options;

        this.logger = pino({ level: loggingLevel });
        this.wss = new WebSocketServer({ port });
        this.gameWorkerMap = new Map();
        this.socketWorkerMap = new Map();
        this.workers = [];
        this.maxGamesPerWorker = maxGamesPerWorker;
        this.maxWorkers = maxWorkers;
        this.actionCount = 0;

        this.wss.on("connection", (ws: WebSocket) => this.handleConnection(ws));

        this.logger.info(`Game server started on port ${port}`);

        this.throughputIntervalMs = throughputIntervalMs;
        if (logThroughput) {
            this.throughputInterval = setInterval(
                () => this.logThroughput(),
                throughputIntervalMs,
            );
        }
    }

    private handleConnection(ws: WebSocket): void {
        ws.on("message", async (clientMessageData: Buffer) => {
            const clientMessage =
                ClientMessage.deserializeBinary(clientMessageData);
            const messageType = clientMessage.getMessageTypeCase();
            const playerId = clientMessage.getPlayerId();

            switch (messageType) {
                case ClientMessage.MessageTypeCase.CONNECT: {
                    const gameId = clientMessage.getGameId();
                    this.assignPlayerToGame(ws, playerId, gameId);
                    break;
                }

                case ClientMessage.MessageTypeCase.STEP:
                case ClientMessage.MessageTypeCase.RESET:
                    this.handlePlayerMessage(ws, clientMessageData);
                    break;
            }
        });

        ws.on("error", (err) => {
            this.logger.error(err);
        });

        ws.on("close", () => {
            this.logger.info("Player disconnected");
        });
    }

    private assignPlayerToGame(
        ws: WebSocket,
        playerId: number,
        gameId: number,
    ): void {
        const gameWorker = this.findAvailableWorker(gameId);
        if (gameWorker) {
            gameWorker.players.set(playerId, ws);
            this.socketWorkerMap.set(ws, gameWorker);
            this.logger.info(
                `Assigned player ${playerId} to game ${gameId} on worker ${gameWorker.workerIndex}`,
            );
        } else {
            this.logger.error(
                `Failed to assign player ${playerId} to game ${gameId}`,
            );
        }
    }

    private handlePlayerMessage(
        ws: WebSocket,
        clientMessageData: Uint8Array,
    ): void {
        const gameWorker = this.socketWorkerMap.get(ws);
        if (gameWorker) {
            this.logger.debug(
                `Worker Index ${gameWorker.workerIndex} recieved message`,
            );
            gameWorker.worker.postMessage(clientMessageData);
            this.actionCount += 1;
        } else {
            this.logger.warn(
                "Received message from a socket not assigned to any worker",
            );
        }
    }

    private findAvailableWorker(gameId: number): GameWorker | null {
        if (this.gameWorkerMap.has(gameId)) {
            return this.gameWorkerMap.get(gameId)!;
        }
        if (this.workers.length < this.maxWorkers) {
            const newWorker = this.createNewWorker();
            this.workers.push(newWorker);
            this.logger.info(`Created new worker ${newWorker.workerIndex}`);
            this.gameWorkerMap.set(gameId, newWorker);
            newWorker.playerCount += 1;
            return newWorker;
        } else {
            this.workers.sort((a, b) => a.playerCount - b.playerCount);
            const leastLoadedWorker = this.workers[0];

            if (leastLoadedWorker.playerCount < this.maxGamesPerWorker) {
                this.logger.info(
                    `Assigning game to existing worker ${leastLoadedWorker.workerIndex}`,
                );
                this.gameWorkerMap.set(gameId, leastLoadedWorker);
                leastLoadedWorker.playerCount += 1;
                return leastLoadedWorker;
            }

            this.logger.warn(
                "All workers are at full capacity and maximum number of workers reached",
            );
            return null;
        }
    }

    private createNewWorker(): GameWorker {
        const newWorkerIndex = this.workers.length;
        this.logger.info(`Creating new worker ${newWorkerIndex}`);
        const worker = new GameWorker(newWorkerIndex, this.logger);
        return worker;
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
    maxWorkers: 16,
    loggingLevel: "info", // Set to 'debug' for more verbose logging
    logThroughput: false,
    throughputIntervalMs: 1000,
});
