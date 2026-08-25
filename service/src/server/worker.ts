import { MessagePort, parentPort } from "worker_threads";

import {
    EnvironmentResponse,
    EnvironmentState,
    ErrorResponse,
    ResetRequest,
    StepRequest,
    WorkerRequest,
    WorkerResponse,
} from "../../protos/service_pb";
import { createBattle, TrainablePlayerAI } from "./runner";
import { isEvalUser } from "./utils";
import { generateTeamFromArray, randomSampleTeam } from "./state";

import { Teams } from "@pkmn/sim";
import { TeamGenerators } from "@pkmn/randoms";

Teams.setGeneratorFactory(TeamGenerators);

// Battle-lifecycle guarantees (2026-08-23). Before these, a battle that
// stopped producing states — a swallowed choose() error, an RQID
// mismatch thrown out of the stream loop, a partner that never reset —
// parked the client's receive forever: the worker's catch only logged,
// nothing ever went back over the socket, and the python side polls its
// receive only against training shutdown. Both actor slots of that game
// were then silently lost for the rest of the run (~2% of games in the
// 2026-08-23 offline sweep). Every await on a battle is now bounded and
// every failure reaches the client as an ErrorResponse.
// 10 min, not 2: a step legitimately stalls for the learner's lattice
// precompile (~9 min at launch, actors blocked behind the GPU lock) — the
// 2026-08-23 first launch aborted every battle in flight and restarted
// them. The watchdog is for battles that NEVER resolve.
const STEP_TIMEOUT_MS = Number(process.env.STEP_TIMEOUT_MS ?? 600_000);
const RESET_TIMEOUT_MS = Number(process.env.RESET_TIMEOUT_MS ?? 300_000);

function withTimeout<T>(
    promise: Promise<T>,
    ms: number,
    what: () => string,
): Promise<T> {
    let timer: NodeJS.Timeout | undefined;
    const timeout = new Promise<never>((_, reject) => {
        timer = setTimeout(
            () =>
                reject(new Error(`${what()} did not complete within ${ms}ms`)),
            ms,
        );
    });
    return Promise.race([promise, timeout]).finally(() => clearTimeout(timer));
}

interface PlayerDetails {
    userName: string;
    smogonFormat: string;
    packedTeam: number[] | undefined;
}

interface WaitingPlayerResolveArgs {
    player: TrainablePlayerAI;
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
        // Each worker is its own V8 isolate: process.memoryUsage() called
        // from GameServer only ever sees the coordinator thread's own
        // heap, never this one (a Node quirk, not a bug — heap stats are
        // per-isolate, RSS is process-wide). Self-report periodically so
        // the coordinator can attribute the ~150MB/worker dex-data
        // baseline instead of only seeing it as opaque process RSS.
        setInterval(() => this.reportMemoryStats(), 10_000);
    }

    private reportMemoryStats(): void {
        if (!this.port) {
            return;
        }
        const mem = process.memoryUsage();
        this.port.postMessage({
            type: "memory_stats",
            heapUsedMb: mem.heapUsed / 2 ** 20,
            heapTotalMb: mem.heapTotal / 2 ** 20,
            externalMb: mem.external / 2 ** 20,
        });
    }

    private setupMessageHandler(): void {
        if (!this.port) {
            throw new Error("Worker must be run as a worker thread");
        }

        this.port.on(
            "message",
            (data: Buffer | Uint8Array | { type: string; gameId: string }) => {
                // postMessage() transfers protobuf payloads as plain
                // Uint8Array (serializeBinary()'s return type), NOT a Node
                // Buffer — Buffer.isBuffer() is false for those, which
                // silently dropped every reset/step request here (neither
                // branch matched, no log, no error). Buffer is itself a
                // Uint8Array subclass, so this covers both.
                if (data instanceof Uint8Array) {
                    this.handleMessage(Buffer.from(data));
                } else if (data?.type === "evict_pending_game") {
                    // Fire-and-forget cleanup from index.ts's disconnect
                    // handler (WorkerPool.evictPendingGame). No-op if the
                    // entry was already consumed by a successful pairing.
                    this.pendingGames.delete(data.gameId);
                }
            },
        );
    }

    private getPlayerFromUsername(userName: string) {
        const player = this.playerMapping.get(userName);
        if (player === undefined) {
            throw new Error(`No player found for username ${userName}`);
        }
        return player;
    }

    private generateTeam(args: {
        packedTeam: number[] | undefined;
        smogonFormat: string;
    }): string {
        const { packedTeam, smogonFormat } = args;
        if (smogonFormat.includes("randombattle")) {
            return Teams.pack(Teams.generate(smogonFormat));
        }
        if (packedTeam !== undefined) {
            return generateTeamFromArray(packedTeam);
        } else {
            return randomSampleTeam(smogonFormat);
        }
    }

    private async resetPlayerFromTrainingUserName(args: {
        userName: string;
        gameId: string; // Added gameId param
        smogonFormat: string;
        packedTeam: number[] | undefined;
    }): Promise<WaitingPlayerResolveArgs> {
        const { userName, gameId, smogonFormat, packedTeam } = args;
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
            packedTeam,
        };

        // Check if someone is already waiting for this Game ID
        const opponent = this.pendingGames.get(gameId);

        if (opponent !== undefined) {
            // Remove the pairing immediately: any of the checks below can
            // throw, and a throw here must not leave this gameId
            // permanently stuck in pendingGames (it would never be reused,
            // and the waiting opponent's promise would never resolve).
            this.pendingGames.delete(gameId);

            if (opponent.playerDetails.userName === userName) {
                throw new Error(
                    `User ${userName} attempted to match with themselves on gameId ${gameId}`,
                );
            }

            console.log(
                `Pairing ${userName} vs ${opponent.playerDetails.userName} (GameID: ${gameId})`,
            );

            if (opponent.playerDetails.smogonFormat !== smogonFormat) {
                throw new Error(
                    `Mismatched formats for GameID ${gameId}: ${opponent.playerDetails.smogonFormat} vs ${smogonFormat}`,
                );
            }

            // 1. Create the battle
            const { p1: player1, p2: player2 } = createBattle({
                p1Name: opponent.playerDetails.userName,
                p2Name: userName,
                p1team: this.generateTeam({
                    packedTeam: opponent.playerDetails.packedTeam,
                    smogonFormat: opponent.playerDetails.smogonFormat,
                }),
                p2team: this.generateTeam({ packedTeam, smogonFormat }),
                smogonFormat,
            });

            // 2. Register players in the map
            this.playerMapping.set(opponent.playerDetails.userName, player1);
            this.playerMapping.set(userName, player2);

            // 3. "Wake up" the waiting opponent
            opponent.resolve({
                player: player1,
            });

            // 4. Return the args for the *current* player
            return Promise.resolve({
                player: player2,
            });
        } else {
            // --- CASE 2: No one is here yet (We are the 1st player) ---
            console.log(
                `Waiting for opponent on GameID ${gameId} (User: ${userName})`,
            );

            return new Promise((resolve, reject) => {
                // Store this player as the waiting party for this GameID,
                // bounded: a partner that never resets must not park this
                // request (and its actor thread) forever.
                const timer = setTimeout(() => {
                    if (
                        this.pendingGames.get(gameId)?.playerDetails === details
                    ) {
                        this.pendingGames.delete(gameId);
                    }
                    reject(
                        new Error(
                            `No opponent arrived for GameID ${gameId} ` +
                                `(User: ${userName}) within ${RESET_TIMEOUT_MS}ms`,
                        ),
                    );
                }, RESET_TIMEOUT_MS);
                this.pendingGames.set(gameId, {
                    playerDetails: details,
                    resolve: (args) => {
                        clearTimeout(timer);
                        resolve(args);
                    },
                });
            });
        }
    }

    private resetPlayerFromEvalUserName(args: {
        userName: string;
        smogonFormat: string;
        packedTeam: number[] | undefined;
    }) {
        const { userName, smogonFormat, packedTeam } = args;
        const teamString = this.generateTeam({ packedTeam, smogonFormat });
        const player = this.playerMapping.get(userName);
        if (player !== undefined) {
            player.destroy();
        }
        const { p1: player1 } = createBattle({
            p1Name: userName,
            p1team: teamString,
            p2Name: `baseline-${userName}`,
            p2team: this.generateTeam({ packedTeam: undefined, smogonFormat }),
            smogonFormat,
        });
        this.playerMapping.set(userName, player1);
        return { player: player1 };
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
            this.sendError(taskId, error);
        }
    }

    private sendError(taskId: number, error: unknown): void {
        const errorResponse = new ErrorResponse();
        const trace =
            error instanceof Error
                ? (error.stack ?? error.message)
                : String(error);
        errorResponse.setTrace(trace);
        const workerResponse = new WorkerResponse();
        workerResponse.setErrorResponse(errorResponse);
        try {
            this.sendMessage(taskId, workerResponse);
        } catch (sendErr) {
            console.error("Failed to send error response:", sendErr);
        }
    }

    private describe(player: TrainablePlayerAI): string {
        const opp = player.opponent;
        return (
            `${player.userName} turn=${player.privateBattle.turn} ` +
            `requests=${player.requestCount} done=${player.done} ` +
            `queued=${player.outgoingQueue.size()}` +
            (opp
                ? ` | opponent ${opp.userName} requests=${opp.requestCount} ` +
                  `done=${opp.done} queued=${opp.outgoingQueue.size()}`
                : "")
        );
    }

    // Tears down both sides of a wedged battle so the opponent's pending
    // await resolves (its queue is cleared) and a later reset for either
    // username starts clean.
    private abortBattle(player: TrainablePlayerAI, reason: string): void {
        console.error(`Aborting battle: ${reason} [${this.describe(player)}]`);
        for (const p of [player, player.opponent]) {
            if (p === undefined) continue;
            try {
                p.outgoingQueue.clear();
                p.destroy();
            } catch (err) {
                console.error("Error during abort:", err);
            }
            this.playerMapping.delete(p.userName);
        }
    }

    private async awaitState(
        player: TrainablePlayerAI,
        what: string,
    ): Promise<EnvironmentState> {
        let state: EnvironmentState | undefined;
        try {
            state = await withTimeout(
                player.receiveEnvironmentState(),
                STEP_TIMEOUT_MS,
                () => `${what} for ${player.userName}`,
            );
        } catch (err) {
            this.abortBattle(player, String(err));
            throw err;
        }
        if (state === undefined) {
            // The queue was cleared by an abort from the other side.
            throw new Error(`${what} for ${player.userName}: battle aborted`);
        }
        return state;
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
        const state = await this.awaitState(player, "step");

        const environmentResponse = this.createResponseFromRequest(stepRequest);
        environmentResponse.setState(state);

        const workerResponse = new WorkerResponse();
        workerResponse.setEnvironmentResponse(environmentResponse);

        this.sendMessage(taskId, workerResponse);

        // Eager cleanup: this was the game's final state, so nothing will
        // ever be requested from this player again — the python actor's
        // next contact is a reset, and gameId-hash routing (index.ts)
        // usually lands that reset on a DIFFERENT worker, so the old
        // "destroy on next reset of the same username" path here left one
        // finished battle (two client Battles + the full sim Battle via
        // the stream refs) retained per username per worker indefinitely.
        // destroy() also ends the BattleStream, which is what actually
        // frees the sim Battle on the early-finish path (no `end` line
        // ever arrives there).
        if (player.done) {
            player.destroy();
            this.playerMapping.delete(userName);
        }
    }

    private resetPlayerFromUserName(
        userName: string,
        gameId: string, // Added param
        smogonFormat: string,
        packedTeam: number[] | undefined,
    ): Promise<WaitingPlayerResolveArgs> {
        if (isEvalUser(userName)) {
            return Promise.resolve(
                this.resetPlayerFromEvalUserName({
                    userName,
                    smogonFormat,
                    packedTeam,
                }),
            );
        } else {
            return this.resetPlayerFromTrainingUserName({
                userName,
                gameId,
                smogonFormat,
                packedTeam,
            });
        }
    }

    private async handleResetRequest(
        taskId: number,
        resetRequest: ResetRequest,
    ): Promise<void> {
        const userName = resetRequest.getUsername();
        const smogonFormat = resetRequest.getSmogonFormat();
        const gameId = resetRequest.getGameId();
        const packedTeam = resetRequest.getPackedTeamsList();

        const { player } = await this.resetPlayerFromUserName(
            userName,
            gameId,
            smogonFormat,
            packedTeam,
        );
        const state = await this.awaitState(player, "reset");

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
