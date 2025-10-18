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
import {
    generateTeamFromFormat,
    generateTeamFromIndices,
    getSampleTeam,
} from "./state";

interface PlayerDetails {
    userName: string;
    smogonFormat: string;
    speciesIndices?: number[];
    packedSetIndices?: number[];
}

interface WaitingPlayerResolveArgs {
    player: TrainablePlayerAI;
    opponentDetails: PlayerDetails | null;
}

interface WaitingPlayer {
    playerDetails: PlayerDetails;
    resolve: (args: WaitingPlayerResolveArgs) => void;
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

    private async cycleWaitingPlayers(
        lookingForUsernamePrefix: string,
    ): Promise<WaitingPlayer> {
        while (true) {
            const opponent = this.waitingPlayers.shift();
            if (opponent !== undefined) {
                if (
                    opponent.playerDetails.userName.startsWith(
                        lookingForUsernamePrefix,
                    )
                ) {
                    return opponent;
                } else {
                    this.waitingPlayers.push(opponent);
                }
            }

            // Yield control to avoid blocking (especially in long loops)
            await new Promise((resolve) => setImmediate(resolve));
        }
    }

    private async resetPlayerFromTrainingUserName(
        userName: string,
        smogonFormat: string,
        speciesIndices?: number[],
        packedSetIndices?: number[],
    ): Promise<WaitingPlayerResolveArgs> {
        const player = this.playerMapping.get(userName);
        if (player !== undefined) {
            player.destroy();
        }
        const details = {
            userName,
            smogonFormat,
            speciesIndices,
            packedSetIndices,
        };

        if (this.waitingPlayers.length > 0) {
            // Pair found, create battle
            const oppositePrefix = {
                challenger: "leader",
                leader: "challenger",
            }[userName.split("-")[0]];
            if (oppositePrefix === undefined) {
                throw new Error(
                    `Invalid training userName prefix: ${userName}`,
                );
            }
            const opponent = await this.cycleWaitingPlayers(oppositePrefix);

            console.log(
                `Pairing ${userName} vs ${opponent.playerDetails.userName}`,
            );

            if (opponent.playerDetails.smogonFormat !== smogonFormat) {
                throw new Error(
                    `Mismatched formats: ${opponent.playerDetails.smogonFormat} vs ${smogonFormat}`,
                );
            }
            const { p1: player1, p2: player2 } = createBattle({
                p1Name: opponent.playerDetails.userName,
                p2Name: userName,
                p1team: generateTeamFromIndices(
                    smogonFormat,
                    opponent.playerDetails.speciesIndices,
                    opponent.playerDetails.packedSetIndices,
                ),
                p2team: generateTeamFromIndices(
                    smogonFormat,
                    speciesIndices,
                    packedSetIndices,
                ),
                smogonFormat,
            });

            this.playerMapping.set(opponent.playerDetails.userName, player1);
            opponent.resolve({
                player: player1,
                opponentDetails: details,
            });

            this.playerMapping.set(userName, player2);
            return Promise.resolve({
                player: player2,
                opponentDetails: opponent.playerDetails,
            });
        } else {
            // No pair, wait
            return new Promise((resolve) => {
                this.waitingPlayers.push({ playerDetails: details, resolve });
            });
        }
    }

    private resetPlayerFromEvalUserName(
        userName: string,
        smogonFormat: string,
        speciesIndices?: number[],
        packedSetIndices?: number[],
    ) {
        const teamString = generateTeamFromIndices(
            smogonFormat,
            speciesIndices,
            packedSetIndices,
        );
        const player = this.playerMapping.get(userName);
        if (player !== undefined) {
            player.destroy();
        }
        const { p1: player1 } = createBattle({
            p1Name: userName,
            p1team: teamString,
            p2Name: `baseline-${userName}`,
            p2team: getSampleTeam(smogonFormat),
            smogonFormat,
        });
        this.playerMapping.set(userName, player1);
        return { player: player1, opponentDetails: null };
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
        speciesIndices?: number[],
        packedSetIndices?: number[],
    ): Promise<WaitingPlayerResolveArgs> {
        if (isEvalUser(userName)) {
            return Promise.resolve(
                this.resetPlayerFromEvalUserName(
                    userName,
                    smogonFormat,
                    speciesIndices,
                    packedSetIndices,
                ),
            );
        } else {
            return this.resetPlayerFromTrainingUserName(
                userName,
                smogonFormat,
                speciesIndices,
                packedSetIndices,
            );
        }
    }

    private async handleResetRequest(
        taskId: number,
        resetRequest: ResetRequest,
    ): Promise<void> {
        const userName = resetRequest.getUsername();
        const speciesIndices = resetRequest.getSpeciesIndicesList();
        const packedSetIndices = resetRequest.getPackedSetIndicesList();
        const smogonFormat = resetRequest.getSmogonFormat();

        const { player, opponentDetails } = await this.resetPlayerFromUserName(
            userName,
            smogonFormat,
            speciesIndices,
            packedSetIndices,
        );
        const state = await player.receiveEnvironmentState();

        const environmentResponse =
            this.createResponseFromRequest(resetRequest);
        environmentResponse.setState(state);

        const workerResponse = new WorkerResponse();
        workerResponse.setEnvironmentResponse(environmentResponse);
        if (opponentDetails) {
            const opponentResetRequest = new ResetRequest();
            opponentResetRequest.setUsername(opponentDetails.userName);
            opponentResetRequest.setSmogonFormat(opponentDetails.smogonFormat);
            if (opponentDetails.speciesIndices) {
                opponentResetRequest.setSpeciesIndicesList(
                    opponentDetails.speciesIndices,
                );
            }
            if (opponentDetails.packedSetIndices) {
                opponentResetRequest.setPackedSetIndicesList(
                    opponentDetails.packedSetIndices,
                );
            }
            workerResponse.setOpponentResetRequest(opponentResetRequest);
        }

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
