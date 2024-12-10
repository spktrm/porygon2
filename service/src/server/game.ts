import { TeamGenerators } from "@pkmn/randoms";
import { BattleStreams, Teams } from "@pkmn/sim";
import { Player, Tracker } from "./player";
import { TaskQueueSystem } from "./utils";
import { recvFnType, sendFnType } from "./types";
import { Action, GameState } from "../../protos/servicev2_pb";
import { MessagePort } from "worker_threads";
import { EVAL_GAME_ID_OFFSET } from "./data";
import { getEvalAction } from "./eval";

const formatId = "gen3randombattle";
const generator = TeamGenerators.getTeamGenerator(formatId);

export const DRAW_TURNS = 150;

export class Game {
    gameId: number;
    workerIndex: number;

    players: [Player, Player] | undefined;
    tasks: TaskQueueSystem<Action>;
    port: MessagePort;

    resetCount: number;
    playerIds: number[];
    maxPlayers: number;

    tracker: Tracker;

    constructor(gameId: number, workerIndex: number, port: MessagePort) {
        this.gameId = gameId;
        this.workerIndex = workerIndex;

        this.tasks = new TaskQueueSystem();
        this.port = port;

        this.maxPlayers = this.gameId < EVAL_GAME_ID_OFFSET ? 2 : 1;
        this.resetCount = 0;
        this.playerIds = [];

        this.tracker = new Tracker();
    }

    addPlayerId(playerId: number) {
        if (this.playerIds.length < this.maxPlayers)
            this.playerIds.push(playerId);
    }

    canReset() {
        if (this.playerIds.length === this.maxPlayers) {
            if (this.gameId >= EVAL_GAME_ID_OFFSET) {
                return this.resetCount > 0;
            } else {
                return this.resetCount === 2;
            }
        }
        return false;
    }

    reset(options?: { seed: number[] }) {
        this.resetCount += 1;
        if (this.canReset()) {
            if (this.playerIds.length < this.maxPlayers) {
                console.error("No players have been added");
            }
            this.resetCount = 0;
            this.tracker.reset();
            this.tasks.reset();
            this._reset(options);
        }
    }

    _drawGame() {
        if (this.players !== undefined) {
            for (const player of this.players) {
                player.draw = true;
            }
        }
    }

    async _waitAllDone() {
        await new Promise((resolve) => {
            const interval = setInterval(() => {
                if (this.tasks.allDone()) {
                    clearInterval(interval);
                    resolve(true);
                }
            }, 1);
        });
    }

    async _reset(options?: { seed: number[] }) {
        const stream = new BattleStreams.BattleStream();
        const streams = BattleStreams.getPlayerStreams(stream);
        const spec = { formatid: formatId, ...options };

        void streams.omniscient.write(`>start ${JSON.stringify(spec)}
`);
        // await this._waitAllDone();
        if (!this.tasks.allDone()) {
            throw new Error("Not all tasks are finished");
        }

        const p1spec = {
            name: `${this.workerIndex}-${this.gameId}-1`,
            team: Teams.pack(generator.getTeam()),
        };
        const p2spec = {
            name: `${this.workerIndex}-${this.gameId}-2`,
            team: Teams.pack(generator.getTeam()),
        };

        void streams.omniscient.write(`>player p1 ${JSON.stringify(p1spec)}
>player p2 ${JSON.stringify(p2spec)}`);

        const sendFn: sendFnType = async (player) => {
            const gameState = new GameState();
            const { faintedReward, hpReward } = this.tracker.update2(
                stream.battle!,
            );
            const state = player.createState();

            const rewards = state.getInfo()!.getRewards()!;
            rewards.setHpreward(hpReward);
            rewards.setFaintedreward(faintedReward);

            gameState.setState(state.serializeBinary());
            const playerId =
                this.playerIds[+state.getInfo()!.getPlayerindex()] ?? 1;
            let rqid = -1;
            if (!state.getInfo()!.getDone()) {
                rqid = this.tasks.createJob();
            }
            gameState.setRqid(rqid);
            gameState.setPlayerId(playerId);

            if (this.gameId >= EVAL_GAME_ID_OFFSET && playerId === 1) {
                const action = getEvalAction(player);
                action.setRqid(rqid);
                this.step(action);
            } else {
                const stateBuffer = gameState.serializeBinary();
                this.port.postMessage(stateBuffer);
            }
            return rqid;
        };

        const recvFn: recvFnType = async (rqid) => {
            return rqid >= 0 ? this.tasks.getResult(rqid) : undefined;
        };

        this.players = [
            new Player(
                this.workerIndex,
                this.gameId,
                streams.p1,
                this.playerIds[0],
                sendFn,
                recvFn,
                stream,
            ),
            new Player(
                this.workerIndex,
                this.gameId,
                streams.p2,
                this.playerIds[1],
                sendFn,
                recvFn,
                stream,
            ),
        ];

        for (const player of this.players) {
            player.start();
        }

        if (this.gameId < EVAL_GAME_ID_OFFSET) {
            for await (const chunk of streams.omniscient) {
                for (const line of chunk.split("\n")) {
                    if (line.startsWith("|turn")) {
                        const turnValue = parseInt(line.split("|")[2]);
                        if (turnValue >= DRAW_TURNS) {
                            this._drawGame();
                        }
                    }
                }
                if (stream.battle !== null) {
                    const numConsecutiveSwitches = 20;
                    const lastTenMoves = stream.battle.inputLog.slice(
                        -numConsecutiveSwitches,
                    );
                    const switchCount = lastTenMoves.reduce((prev, curr) => {
                        const isSwitch =
                            curr.split(" ")[1].toString() === "switch";
                        return prev + +isSwitch;
                    }, 0);
                    if (switchCount === numConsecutiveSwitches) {
                        this._drawGame();
                    }
                }
            }
        }
    }

    step(action: Action) {
        const rqid = action.getRqid();
        if (rqid >= 0) this.tasks.submitResult(rqid, action);
    }
}
