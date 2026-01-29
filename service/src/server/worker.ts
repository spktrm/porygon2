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

    // Changed: We now map a specific GameID to a single Waiting Player.
    // Logic: First player to arrive sits here. Second player triggers the match.
    private pendingGames = new Map<string, WaitingPlayer>();

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
        gameId: string, // Added gameId param
        smogonFormat: string,
        speciesIndices?: number[],
        packedSetIndices?: number[],
    ): Promise<WaitingPlayerResolveArgs> {
        // Destroy old player if one exists
        const player = this.playerMapping.get(userName);
        if (player !== undefined) {
            player.destroy();
        }

        if (!gameId) {
            throw new Error("gameId is required for matchmaking.");
        }

        const details: PlayerDetails = {
            userName,
            smogonFormat,
            speciesIndices,
            packedSetIndices,
        };

        // Check if someone is already waiting for this Game ID
        const opponent = this.pendingGames.get(gameId);

        if (opponent !== undefined) {
            // --- CASE 1: Opponent is already waiting (We are the 2nd player) ---

            // Safety check: prevent self-matching if unique usernames are required
            if (opponent.playerDetails.userName === userName) {
                throw new Error(
                    `User ${userName} attempted to match with themselves on gameId ${gameId}`,
                );
            }

            console.log(
                `Pairing ${userName} vs ${opponent.playerDetails.userName} (GameID: ${gameId})`,
            );

            if (opponent.playerDetails.smogonFormat !== smogonFormat) {
                // If formats mismatch, we cannot start the game.
                // We remove the waiting player to prevent the queue from being "clogged" by a bad match,
                // or you could choose to throw and leave them waiting.
                // Here we throw, but the pending game remains (waiting for a valid match).
                throw new Error(
                    `Mismatched formats for GameID ${gameId}: ${opponent.playerDetails.smogonFormat} vs ${smogonFormat}`,
                );
            }

            // 1. Create the battle
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

            // 2. Register players in the map
            this.playerMapping.set(opponent.playerDetails.userName, player1);
            this.playerMapping.set(userName, player2);

            // 3. Remove the game from pending now that it has started
            this.pendingGames.delete(gameId);

            // 4. "Wake up" the waiting opponent
            opponent.resolve({
                player: player1,
                opponentDetails: details,
            });

            // 5. Return the args for the *current* player
            return Promise.resolve({
                player: player2,
                opponentDetails: opponent.playerDetails,
            });
        } else {
            // --- CASE 2: No one is here yet (We are the 1st player) ---
            console.log(
                `Waiting for opponent on GameID ${gameId} (User: ${userName})`,
            );

            return new Promise((resolve) => {
                // Store this player as the waiting party for this GameID
                this.pendingGames.set(gameId, {
                    playerDetails: details,
                    resolve,
                });
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
        gameId: string, // Added param
        smogonFormat: string,
        speciesIndices?: number[],
        packedSetIndices?: number[],
        // We no longer need ckpt params for matchmaking logic
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
                gameId,
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
        const gameId = resetRequest.getGameId();

        const { player, opponentDetails } = await this.resetPlayerFromUserName(
            userName,
            gameId,
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
