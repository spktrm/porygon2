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
        [k: number]: StreamHandler;
    };
    queueSystem: TaskQueueSystem<Action>;
    world: World | null;
    ts: number;
    tied: boolean;
    earlyFinish: boolean;
    numFainted: number[][];

    constructor(args: { port: MessagePort | null; gameId: number }) {
        const { gameId, port } = args;
        this.port = port;
        this.done = false;
        this.gameId = gameId;
        this.world = null;
        this.queueSystem = new TaskQueueSystem<Action>();
        this.ts = 0;
        this.tied = false;
        this.earlyFinish = false;
        this.numFainted = [[0, 0]];

        this.handlers = Object.fromEntries(
            [0, 1].map((playerIndex) => [
                playerIndex,
                new StreamHandler({
                    gameId,
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

    getRewardFromHpDiff(): number {
        const sideHpSums = this.world?.sides.map((side) =>
            side.pokemon.reduce(
                (total, member) => total + member.hp / member.maxhp,
                0,
            ),
        ) ?? [6, 6];

        if (sideHpSums[0] === sideHpSums[1]) {
            if (this.done) {
                this.tied = true;
            }
            return 0;
        }
        return sideHpSums[0] > sideHpSums[1] ? 1 : -1;
    }

    getRewardFromFinish(): number {
        if (this.done) {
            const winner = this.getWinner();
            if (winner === this.world?.p1.name) {
                return 1;
            } else if (winner === this.world?.p2.name) {
                return -1;
            }
        }
        return 0;
    }

    getRewardFromPrevHpDiff(): number {
        const prevFainted = this.numFainted.at(-1) ?? [0, 0];

        const numFainted = this.world?.sides.map((side) =>
            side.pokemon.reduce((total, member) => total + +member.fainted, 0),
        ) ?? [0, 0];
        this.numFainted.push(numFainted);

        const fOfX = (value: number) => {
            return 2 ** Math.floor(value - 7);
        };

        const calcScore = (playerIndex: number) => {
            const diff = prevFainted[playerIndex] - numFainted[playerIndex];
            return diff * fOfX(numFainted[playerIndex]);
        };

        const score = calcScore(0) - calcScore(1);
        if (isNaN(score)) {
            throw Error("Score is nan");
        }

        return score;
    }

    sendState(state: State) {
        const isDone = this.done;
        let info = state.getInfo();
        if (info) {
            const winReward = this.getRewardFromFinish();
            const hpReward = this.getRewardFromPrevHpDiff();

            const p1SwitchReward = this.handlers[0].getSwitchReward();
            const p2SwitchReward = this.handlers[1].getSwitchReward();
            const switchReward =
                0.5 *
                Math.sign(p1SwitchReward - p2SwitchReward) *
                Math.abs(hpReward);

            info.setSwitchreward(switchReward);
            info.setWinreward(winReward);
            info.setHpreward(hpReward);

            if (isDone) {
                info.setDone(isDone);
                state.setLegalactions(AllValidActions);
            }
            state.setInfo(info);
        }
        const stateArr = state.serializeBinary();
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
        let handler = this.handlers[{ p1: 0, p2: 1 }[id]];
        handler.world = this.world;
        for await (const chunk of stream) {
            const action = await handler.ingestChunk(chunk);
            if (action !== undefined) {
                this.handleAction(stream, action);
            }
            const playerIndex = handler.getPlayerIndex();
            if (playerIndex) {
                handler = this.handlers[playerIndex];
            }
            if (this.ts > MAX_TS) {
                this.earlyFinish = true;
                break;
            }
        }
    }

    async run(options?: { seed: number[]; [k: string]: any }) {
        const stream = new BattleStreams.BattleStream();
        const streams = BattleStreams.getPlayerStreams(stream);
        const spec = { formatid: formatId, ...options };

        void streams.omniscient.write(`>start ${JSON.stringify(spec)}
`);

        this.world = stream.battle;

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

        void streams.omniscient.write(`>player p1 ${JSON.stringify(p1spec)}
>player p2 ${JSON.stringify(p2spec)}`);

        await players;

        return new Promise(async (resolve, reject) => {
            this.done = true;
            const state = await this.handlers[0].getState();
            this.sendState(state);
            resolve(true);
        });
    }

    reset() {
        for (const playerIndex of [0, 1]) {
            this.handlers[playerIndex].reset();
        }
        this.done = false;
        this.world = null;
        this.ts = 0;
        this.tied = false;
        this.numFainted = [[0, 0]];
        this.earlyFinish = false;
    }
}
