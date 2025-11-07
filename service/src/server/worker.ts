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
import { generateTeamFromIndices, getSampleTeam } from "./state";

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
    private waitingQueues = new Map<number, WaitingPlayer[]>();

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

    private async resetPlayerFromTrainingUserName(
        userName: string,
        smogonFormat: string,
        speciesIndices?: number[],
        packedSetIndices?: number[],
        currentCkpt?: number,
        opponentCkpt?: number,
    ): Promise<WaitingPlayerResolveArgs> {
        // Destroy old player if one exists
        const player = this.playerMapping.get(userName);
        if (player !== undefined) {
            player.destroy();
        }
        if (currentCkpt === undefined) {
            throw new Error(`Invalid training currentCkpt: ${currentCkpt}`);
        }
        if (opponentCkpt === undefined) {
            throw new Error(`Invalid training opponentCkpt: ${opponentCkpt}`);
        }

        const details: PlayerDetails = {
            userName,
            smogonFormat,
            speciesIndices,
            packedSetIndices,
        };

        const opponentQueue = this.waitingQueues.get(opponentCkpt);
        const opponent = opponentQueue?.shift(); // Get the first waiting opponent, if any

        if (opponent !== undefined) {
            // --- CASE 1: Opponent was found ---
            console.log(
                `Pairing ${userName} vs ${opponent.playerDetails.userName}`,
            );

            // Check for format mismatch (same logic as your original code)
            if (opponent.playerDetails.smogonFormat !== smogonFormat) {
                // Note: This is a critical error in matchmaking logic.
                // You might want to put the opponent back in the queue instead of throwing.
                // But for this refactor, we will replicate the original's behavior.
                throw new Error(
                    `Mismatched formats: ${opponent.playerDetails.smogonFormat} vs ${smogonFormat}`,
                );
            }

            // 5. Create the battle
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

            // 6. Register players in the map
            this.playerMapping.set(opponent.playerDetails.userName, player1);
            this.playerMapping.set(userName, player2);

            // 7. "Wake up" the waiting opponent by resolving their promise
            opponent.resolve({
                player: player1,
                opponentDetails: details,
            });

            // 8. Return the args for the *current* player
            return Promise.resolve({
                player: player2,
                opponentDetails: opponent.playerDetails,
            });
        } else {
            // --- CASE 2: No opponent found, this player must wait ---
            console.log(`No opponent found for ${userName}. Waiting...`);

            // 5. Add this player to their *own* queue and return a promise
            return new Promise((resolve) => {
                // Find or create the queue for this player's prefix
                let myQueue = this.waitingQueues.get(currentCkpt);
                if (myQueue === undefined) {
                    myQueue = [];
                    this.waitingQueues.set(currentCkpt, myQueue);
                }
                // Add this player (and their "wake-up" function) to the queue
                myQueue.push({ playerDetails: details, resolve });
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
        currentCkpt?: number,
        opponentCkpt?: number,
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
                currentCkpt,
                opponentCkpt,
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
        const currentCkpt = resetRequest.getCurrentCkpt();
        const opponentCkpt = resetRequest.getOpponentCkpt();

        const { player, opponentDetails } = await this.resetPlayerFromUserName(
            userName,
            smogonFormat,
            speciesIndices,
            packedSetIndices,
            currentCkpt,
            opponentCkpt,
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
