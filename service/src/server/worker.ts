import { MessagePort, parentPort } from "worker_threads";

import {
    EnvironmentResponse,
    ResetRequest,
    StepRequest,
    WorkerRequest,
    WorkerResponse,
} from "../../protos/service_pb";
import { createBattle, TrainablePlayerAI } from "./runner";
import { isEvalUser } from "./utils";
import { generateTeamFromIndices } from "./state";

interface WaitingPlayer {
    userName: string;
    team: string | null;
    smogonFormat: string;
    resolve: (player: TrainablePlayerAI) => void;
}

export class WorkerHandler {
    private port: MessagePort | null | undefined = parentPort;
    private playerMapping = new Map<string, TrainablePlayerAI>();
    private waitingPlayers = new Array<WaitingPlayer>();

    constructor(port: MessagePort | null | undefined) {
        this.port = port;
        this.setupMessageHandler();
    }

    private setupMessageHandler(): void {
        if (!this.port) {
            throw new Error("Worker must be run as a worker thread");
        }

        this.port.on("message", (data: Buffer) => {
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
        teamString: string | null,
        smogonFormat: string,
    ): Promise<TrainablePlayerAI> {
        const player = this.playerMapping.get(userName);
        if (player !== undefined) {
            player.destroy();
        }

        if (this.waitingPlayers.length > 0) {
            // Pair found, create battle
            const opponent = this.waitingPlayers.shift()!;
            if (opponent.smogonFormat !== smogonFormat) {
                throw new Error(
                    `Mismatched formats: ${opponent.smogonFormat} vs ${smogonFormat}`,
                );
            }
            const { p1: player1, p2: player2 } = createBattle({
                p1Name: opponent.userName,
                p2Name: userName,
                p1team: opponent.team,
                p2team: teamString,
                smogonFormat,
            });

            this.playerMapping.set(opponent.userName, player1);
            opponent.resolve(player1);

            this.playerMapping.set(userName, player2);
            return Promise.resolve(player2);
        } else {
            // No pair, wait
            return new Promise<TrainablePlayerAI>((resolve) => {
                this.waitingPlayers.push({
                    userName,
                    team: teamString,
                    smogonFormat,
                    resolve,
                });
            });
        }
    }

    private resetPlayerFromEvalUserName(
        userName: string,
        teamString: string | null,
        smogonFormat: string,
    ) {
        const player = this.playerMapping.get(userName);
        if (player !== undefined) {
            player.destroy();
        }
        const { p1: player1 } = createBattle({
            p1Name: userName,
            p1team: teamString,
            p2Name: `baseline-${userName}`,
            p2team: teamString,
            smogonFormat,
        });
        this.playerMapping.set(userName, player1);
        return player1;
    }

    private async handleMessage(data: Buffer): Promise<void> {
        const workerRequest = WorkerRequest.deserializeBinary(data);
        const taskId = workerRequest.getTaskId();
        try {
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
            console.error(
                "Error handling message in worker:",
                workerRequest.toObject(),
                error,
            );
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

        const state = await player.receiveEnvironmentState();

        const environmentResponse = this.createResponseFromRequest(stepRequest);
        environmentResponse.setState(state);

        const workerResponse = new WorkerResponse();
        workerResponse.setEnvironmentResponse(environmentResponse);

        this.sendMessage(taskId, workerResponse);
    }

    private resetPlayerFromUserName(
        userName: string,
        smogonFormat: string,
        packedSetIndices?: number[],
    ): Promise<TrainablePlayerAI> {
        const teamString = generateTeamFromIndices(
            smogonFormat,
            packedSetIndices,
        );
        if (isEvalUser(userName)) {
            return Promise.resolve(
                this.resetPlayerFromEvalUserName(
                    userName,
                    teamString,
                    smogonFormat,
                ),
            );
        } else {
            return this.resetPlayerFromTrainingUserName(
                userName,
                teamString,
                smogonFormat,
            );
        }
    }

    private async handleResetRequest(
        taskId: number,
        resetRequest: ResetRequest,
    ): Promise<void> {
        const userName = resetRequest.getUsername();
        const packedSetIndices = resetRequest.getPackedSetIndicesList();
        const smogonFormat = resetRequest.getSmogonFormat();

        const player = await this.resetPlayerFromUserName(
            userName,
            smogonFormat,
            packedSetIndices,
        );
        const state = await player.receiveEnvironmentState();

        const environmentResponse =
            this.createResponseFromRequest(resetRequest);
        environmentResponse.setState(state);

        const workerResponse = new WorkerResponse();
        workerResponse.setEnvironmentResponse(environmentResponse);

        this.sendMessage(taskId, workerResponse);
    }

    private sendMessage(taskId: number, workerResponse: WorkerResponse): void {
        if (!this.port) {
            throw new Error("Parent port not available");
        }
        workerResponse.setTaskId(taskId);
        const messageBuffer = workerResponse.serializeBinary();
        this.port.postMessage(messageBuffer, [
            messageBuffer.buffer as ArrayBuffer,
        ]);
    }
}

// Initialize the worker handler
new WorkerHandler(parentPort);
