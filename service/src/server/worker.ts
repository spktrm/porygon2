import { parentPort } from "worker_threads";

import {
    EnvironmentResponse,
    ResetRequest,
    StepRequest,
    WorkerRequest,
    WorkerResponse,
} from "../../protos/service_pb";
import { createBattle, TrainablePlayerAI } from "./runner";
import { isEvalUser } from "./utils";

interface WaitingPlayer {
    userName: string;
    resolve: (player: TrainablePlayerAI) => void;
}

class WorkerHandler {
    private playerMapping = new Map<string, TrainablePlayerAI>();
    private waitingPlayers = new Array<WaitingPlayer>();

    constructor() {
        this.setupMessageHandler();
    }

    private setupMessageHandler(): void {
        if (!parentPort) {
            throw new Error("Worker must be run as a worker thread");
        }

        parentPort.on("message", (data: Buffer) => {
            this.handleMessage(data);
        });
    }

    private getPlayerFromUsername(userName: string) {
        const player = this.playerMapping.get(userName);
        if (player === undefined) {
            throw new Error(`No player found for username ${userName}`);
        }
        return player;
    }

    private resetPlayerFromTrainingUserName(
        userName: string,
    ): Promise<TrainablePlayerAI> {
        const player = this.playerMapping.get(userName);
        if (player !== undefined) {
            player.destroy();
        }

        if (this.waitingPlayers.length > 0) {
            // Pair found, create battle
            const opponent = this.waitingPlayers.shift()!;
            const { p1: player1, p2: player2 } = createBattle({
                p1Name: opponent.userName,
                p2Name: userName,
            });

            this.playerMapping.set(opponent.userName, player1);
            opponent.resolve(player1);

            this.playerMapping.set(userName, player2);
            return Promise.resolve(player2);
        } else {
            // No pair, wait
            return new Promise<TrainablePlayerAI>((resolve) => {
                this.waitingPlayers.push({ userName, resolve });
            });
        }
    }

    private resetPlayerFromEvalUserName(userName: string) {
        const player = this.playerMapping.get(userName);
        if (player !== undefined) {
            player.destroy();
        }
        const { p1: player1 } = createBattle({
            p1Name: userName,
            p2Name: `baseline-${userName}`,
        });
        this.playerMapping.set(userName, player1);
        return player1;
    }

    private async handleMessage(data: Buffer): Promise<void> {
        try {
            const workerRequest: WorkerRequest =
                WorkerRequest.deserializeBinary(data); // keep this if using protobufjs or similar
            const taskId = workerRequest.getTaskId();

            switch (workerRequest.getRequestCase()) {
                case WorkerRequest.RequestCase.STEP_REQUEST: {
                    const stepRequest = workerRequest.getStepRequest();
                    if (stepRequest !== undefined) {
                        await this.handleStepRequest(taskId, stepRequest);
                    } else {
                        throw new Error(
                            `stepRequest must not be undefined to use`,
                        );
                    }
                    break;
                }
                case WorkerRequest.RequestCase.RESET_REQUEST: {
                    const resetRequest = workerRequest.getResetRequest();
                    if (resetRequest !== undefined) {
                        await this.handleResetRequest(taskId, resetRequest);
                    } else {
                        throw new Error(
                            `resetRequest must not be undefined to use`,
                        );
                    }
                    break;
                }
                default:
                    throw new Error(
                        "Must set either stepRequest or resetRequest",
                    );
            }
        } catch (error) {
            console.error("Error handling message in worker:", error);
        }
    }

    private createResponseFromRequest(
        request: StepRequest | ResetRequest,
    ): EnvironmentResponse {
        const environmentResponse = new EnvironmentResponse();
        environmentResponse.setUsername(request.getUsername());
        return environmentResponse;
    }

    private async handleStepRequest(
        taskId: number,
        stepRequest: StepRequest,
    ): Promise<void> {
        const userName = stepRequest.getUsername();
        const player = this.getPlayerFromUsername(userName);

        player.submitStepRequest(stepRequest);

        const state = await player.recieveEnvironmentState();

        const environmentResponse = this.createResponseFromRequest(stepRequest);
        environmentResponse.setState(state);

        const workerResponse = new WorkerResponse();
        workerResponse.setEnvironmentResponse(environmentResponse);

        this.sendMessage(taskId, workerResponse);
    }

    private resetPlayerFromUserName(
        userName: string,
    ): Promise<TrainablePlayerAI> {
        if (isEvalUser(userName)) {
            return Promise.resolve(this.resetPlayerFromEvalUserName(userName));
        } else {
            return this.resetPlayerFromTrainingUserName(userName);
        }
    }

    private async handleResetRequest(
        taskId: number,
        resetRequest: ResetRequest,
    ): Promise<void> {
        const userName = resetRequest.getUsername();

        const player = await this.resetPlayerFromUserName(userName);
        const state = await player.recieveEnvironmentState();

        const environmentResponse =
            this.createResponseFromRequest(resetRequest);
        environmentResponse.setState(state);

        const workerResponse = new WorkerResponse();
        workerResponse.setEnvironmentResponse(environmentResponse);

        this.sendMessage(taskId, workerResponse);
    }

    private sendMessage(taskId: number, workerResponse: WorkerResponse): void {
        if (!parentPort) {
            throw new Error("Parent port not available");
        }
        workerResponse.setTaskId(taskId);
        const messageBuffer = workerResponse.serializeBinary();
        parentPort.postMessage(messageBuffer, [
            messageBuffer.buffer as ArrayBuffer,
        ]);
    }
}

// Initialize the worker handler
new WorkerHandler();
