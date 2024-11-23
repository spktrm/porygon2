import { MessagePort, parentPort, workerData } from "worker_threads";
import { Game } from "./game";
import { ClientMessage, Action } from "../../protos/servicev2_pb";

export class GameWorker {
    private port: MessagePort;
    private games: Map<number, Game>;
    private workerIndex: number;

    constructor(workerIndex: number, port: MessagePort) {
        this.port = port;
        this.games = new Map();
        this.workerIndex = workerIndex;

        // Setup message listener for incoming client messages
        this.port.on("message", async (clientMessageData: Buffer) => {
            this.handleMessage(clientMessageData);
        });
    }

    async handleMessage(clientMessageData: Buffer) {
        const clientMessage =
            ClientMessage.deserializeBinary(clientMessageData);
        const messageType = clientMessage.getMessageTypeCase();
        const gameId = clientMessage.getGameId();

        switch (messageType) {
            case ClientMessage.MessageTypeCase.CONNECT:
                // Handle CONNECT case if needed
                break;

            case ClientMessage.MessageTypeCase.STEP:
                const stepMessage = clientMessage.getStep()!;
                this.handleStep(gameId, stepMessage.getAction()!);
                break;

            case ClientMessage.MessageTypeCase.RESET:
                const playerId = clientMessage.getPlayerId();
                this.handleReset(gameId, playerId);
                break;
        }
    }

    private handleReset(gameId: number, playerId: number) {
        if (!this.games.has(gameId)) {
            const game = new Game(gameId, this.workerIndex, this.port);
            this.games.set(gameId, game);
            game.addPlayerId(playerId);
            game.reset();
        } else {
            const game = this.games.get(gameId)!;
            game.addPlayerId(playerId);
            if (game === undefined) {
                console.error(`GameId: ${gameId} does not exist`);
            } else {
                game.reset();
            }
        }
    }

    private handleStep(gameId: number, action: Action) {
        const game = this.games.get(gameId);
        if (!game) {
            console.error(`GameId: ${gameId} does not exist`);
            return;
        }
        game.step(action);
    }
}

// Instantiate GameWorker to initialize and listen for messages
try {
    new GameWorker(workerData.workerIndex, parentPort!);
} catch (err) {
    console.log(err);
}
