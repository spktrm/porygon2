import { Battle as World } from "@pkmn/sim";
import { TeamGenerators } from "@pkmn/randoms";
import { BattleStreams, Teams } from "@pkmn/sim";
import { ObjectReadWriteStream } from "@pkmn/streams";

import { MessagePort } from "worker_threads";
import { StreamHandler } from "../logic/handler";
import { actionIndexMapping, AllValidActions, MAX_TS } from "../logic/data";
import { TaskQueueSystem } from "../utils";
import { Action } from "../../protos/action_pb";
import { State } from "../../protos/state_pb";

const formatId = "gen3randombattle";
const generator = TeamGenerators.getTeamGenerator(formatId);

export class Game {
    port: MessagePort | null;
    done: boolean;
    gameId: number;
    handlers: {
        [k: string]: StreamHandler;
    };
    queueSystem: TaskQueueSystem<Action>;
    world: World | null;
    isTraining: boolean;
    ts: number;
    tied: boolean;

    constructor(args: {
        port: MessagePort | null;
        gameId: number;
        isTraining: boolean;
    }) {
        const { gameId, port, isTraining } = args;
        this.port = port;
        this.done = false;
        this.gameId = gameId;
        this.world = null;
        this.isTraining = isTraining;
        this.ts = 0;
        this.tied = false;

        this.queueSystem = new TaskQueueSystem<Action>();
        this.handlers = Object.fromEntries(
            ["p1", "p2"].map((sideId) => [
                sideId,
                new StreamHandler({
                    gameId,
                    isTraining: sideId === "p2" && isTraining,
                    sendFn: async (state) => {
                        const jobKey = this.queueSystem.createJob();
                        state.setKey(jobKey);
                        this.sendState(state);
                        return jobKey;
                    },
                    recvFn: async (key: string) => {
                        return await this.queueSystem.getResult(key);
                    },
                }),
            ]),
        );
    }

    getWinner() {
        if (this.world && this.done) {
            return this.world.winner;
        }
    }

    getRewardFromHpDiff(playerIndex: number): [number, number] {
        const omniscientHandler = this.handlers[playerIndex ? "p2" : "p1"];
        const publicBattle = omniscientHandler.publicBattle;
        const sideHpSums = [
            publicBattle.sides[playerIndex],
            publicBattle.sides[1 - playerIndex],
        ].map((side) =>
            side.team
                .map((pokemon) => pokemon.hp / pokemon.maxhp)
                .reduce((a, b) => a + b),
        );
        if (sideHpSums[0] === sideHpSums[1]) {
            this.tied = true;
            return [0, 0];
        }
        return [
            sideHpSums[0] > sideHpSums[1] ? 1 : -1,
            sideHpSums[1] > sideHpSums[0] ? 1 : -1,
        ];
    }

    getRewardFromFinish(playerIndex: number): [number, number] {
        const winner = this.getWinner();
        const omniscientHandler = this.handlers[playerIndex ? "p2" : "p1"];
        const publicBattle = omniscientHandler.publicBattle;
        if (winner) {
            const p1Reward = publicBattle.p1.name === winner ? 1 : -1;
            const p2Reward = publicBattle.p2.name === winner ? 1 : -1;
            return [p1Reward, p2Reward];
        } else {
            this.tied = true;
            return [0, 0];
        }
    }

    getRewards(playerIndex: number): [number, number] {
        return this.getRewardFromHpDiff(playerIndex);
        // return this.getRewardFromFinish(sideId);
    }

    sendState(state: State) {
        const isDone = this.done;
        let info = state.getInfo();
        if (isDone && info) {
            info.setDone(isDone);
            const playerIndex = info.getPlayerindex();
            const [r1, r2] = this.getRewards(playerIndex ? 1 : 0);
            info.setPlayeronereward(r1);
            info.setPlayertworeward(r2);
            state.setInfo(info);
            state.setLegalactions(AllValidActions);
        }
        let stateArr: Uint8Array;
        stateArr = state.serializeBinary();
        this.ts += 1;
        return this.port?.postMessage(stateArr, [stateArr.buffer]);
    }

    handleAction(stream: ObjectReadWriteStream<string>, action: Action) {
        const actionIndex = action.getIndex();
        if (actionIndex < 0) {
            const actiontext = action.getText();
            stream.write(actiontext);
        } else {
            const action =
                actionIndexMapping[
                    actionIndex as keyof typeof actionIndexMapping
                ];
            stream.write(action);
        }
    }

    async runPlayer(args: {
        id: "p1" | "p2";
        stream: ObjectReadWriteStream<string>;
    }) {
        const { id, stream } = args;
        const handler = this.handlers[id];
        for await (const chunk of stream) {
            const action = await handler.ingestChunk(chunk);
            if (action !== undefined) {
                this.handleAction(stream, action);
            }
            if (this.ts > MAX_TS) {
                break;
            }
        }
    }

    async run(options?: { seed: number[]; [k: string]: any }) {
        const stream = new BattleStreams.BattleStream();
        const streams = BattleStreams.getPlayerStreams(stream);
        const spec = { formatid: formatId, ...options };

        const players = Promise.all([
            this.runPlayer({
                id: "p1",
                stream: streams.p1,
            }),
            this.runPlayer({
                id: "p2",
                stream: streams.p2,
            }),
        ]);

        const p1spec = {
            name: `Bot${this.gameId}1`,
            team: Teams.pack(generator.getTeam()),
        };
        const p2spec = {
            name: `Bot${this.gameId}2`,
            team: Teams.pack(generator.getTeam()),
        };

        void streams.omniscient.write(`>start ${JSON.stringify(spec)}
>player p1 ${JSON.stringify(p1spec)}
>player p2 ${JSON.stringify(p2spec)}`);
        this.world = stream.battle;

        await players;

        this.done = true;
        const state = this.handlers.p1.getState();
        this.sendState(state);
    }

    reset() {
        for (const handlerId of Object.keys(this.handlers)) {
            this.handlers[handlerId].reset();
        }
        this.done = false;
        this.world = null;
        this.ts = 0;
        this.tied = false;
    }
}
