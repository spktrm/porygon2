import { Battle } from "@pkmn/client";
import { Battle as World } from "@pkmn/sim";
import { Generations } from "@pkmn/data";
import { Dex } from "@pkmn/dex";
import { TeamGenerators } from "@pkmn/randoms";
import { AnyObject, BattleStreams, Teams } from "@pkmn/sim";
import { ObjectReadWriteStream } from "@pkmn/streams";
import { Protocol } from "@pkmn/protocol";

import { AsyncQueue } from "./utils";
import { AllValidActions, EventHandler, StateHandler } from "./state";
import { MessagePort } from "worker_threads";

import { State } from "../protos/state_pb";
import { Action } from "../protos/action_pb";
import { rejects } from "assert";
import { resolve } from "path";

const formatId = "gen3randombattle";
const generator = TeamGenerators.getTeamGenerator(formatId);
const generations = new Generations(Dex);

type sendFnType = (state: State) => void;
type recvFnType = () => Promise<Action | undefined>;
type requestType = Protocol.MoveRequest;

export class StreamHandler {
    sendFn: sendFnType;
    recvFn: recvFnType;

    ts: number;
    log: string[];
    gameId: number;

    publicBattle: Battle;
    privatebattle: Battle;
    eventHandler: EventHandler;

    rqid: string | undefined;
    queue: AsyncQueue<Action>;
    playerIndex: number | undefined;

    constructor(args: {
        gameId: number;
        sendFn: sendFnType;
        recvFn: recvFnType;
    }) {
        const { gameId, sendFn, recvFn } = args;

        this.ts = Date.now();
        this.gameId = gameId;
        this.log = [];

        this.publicBattle = new Battle(generations);
        this.privatebattle = new Battle(generations);
        this.eventHandler = new EventHandler(this);

        this.rqid = undefined;
        this.playerIndex = undefined;
        this.queue = new AsyncQueue();

        this.sendFn = sendFn;
        this.recvFn = recvFn;
    }

    ingestLine(line: string) {
        this.publicBattle.add(line);
        this.privatebattle.add(line);
        this.log.push(line);
        this.ingestEvent(line);
    }

    ingestEvent(line: string) {
        const { args, kwArgs } = Protocol.parseBattleLine(line);
        const key = Protocol.key(args);
        if (!key) return;
        if (key in this.eventHandler) {
            (this.eventHandler as any)[key](args, kwArgs);
        }
        if (line.startsWith("|-")) {
            this.eventHandler.handleHyphenLine(args, kwArgs);
        }
    }

    isActionRequired(chunk: string): boolean {
        const request = (this.privatebattle.request ?? {}) as AnyObject;
        this.rqid = request.rqid;
        if (request === undefined) {
            return false;
        }
        if (request.teamPreview) {
            return true;
        }
        if (!!request.wait) {
            return false;
        }
        if (chunk.includes("|turn")) {
            return true;
        }
        if (!chunk.includes("|request")) {
            return !!(request.forceSwitch ?? [])[0];
        }
        return false;
    }

    getRequest(): requestType {
        return this.privatebattle.request as requestType;
    }

    getState(): State {
        return new StateHandler(this).getState();
    }

    getPlayerIndex(): number | undefined {
        const request = this.privatebattle.request;
        if (request) {
            if (this.playerIndex) {
                return this.playerIndex;
            }
            this.playerIndex =
                parseInt(request.side?.id.toString().slice(1) ?? "") - 1;
            return this.playerIndex;
        }
    }

    async stateActionStep(): Promise<Action | undefined> {
        const state = this.getState();
        const legalActions = state.getLegalactions();
        if (legalActions) {
            const legalObj = legalActions.toObject();

            const allMoves = Object.keys(legalObj);
            const validMoves = Object.fromEntries(
                Object.entries(legalObj).filter(([i, v]) => v)
            );

            if (Object.values(validMoves).length === 1) {
                const action = new Action();
                action.setGameid(this.gameId);
                action.setIndex(
                    allMoves.indexOf(Object.keys(validMoves).at(0) ?? "")
                );
                return action;
            }
        }
        this.sendFn(state);
        return await this.recvFn();
    }

    async ingestChunk(chunk: string) {
        if (chunk) {
            if (chunk.startsWith("|error")) {
                console.log(chunk);
            }
            for (const line of chunk.split("\n")) {
                this.ingestLine(line);
            }
            if (this.isActionRequired(chunk)) {
                return await this.stateActionStep();
            }
        }
    }

    reset() {
        this.log = [];

        this.publicBattle = new Battle(generations);
        this.privatebattle = new Battle(generations);
        this.eventHandler.reset();

        this.rqid = undefined;
        this.playerIndex = undefined;
    }
}

const actionIndexMapping: { [k: number]: string } = {
    0: "move 1",
    1: "move 2",
    2: "move 3",
    3: "move 4",
    4: "switch 1",
    5: "switch 2",
    6: "switch 3",
    7: "switch 4",
    8: "switch 5",
    9: "switch 6",
};

const MAX_TS = 100;

export class Game {
    port: MessagePort | null;
    done: boolean;
    gameId: number;
    handlers: {
        [k: string]: StreamHandler;
    };
    queues: {
        [k: string]: AsyncQueue<Action>;
    };
    world: World | null;
    ts: number;

    constructor(args: { port: MessagePort | null; gameId: number }) {
        const { gameId, port } = args;
        this.port = port;
        this.done = false;
        this.gameId = gameId;
        this.world = null;
        this.ts = 0;

        this.queues = { p1: new AsyncQueue(), p2: new AsyncQueue() };
        this.handlers = Object.fromEntries(
            ["p1", "p2"].map((x) => [
                x,
                new StreamHandler({
                    gameId,
                    sendFn: (state) => {
                        this.sendState(state);
                    },
                    recvFn: async () => {
                        return await this.queues[x].dequeue();
                    },
                }),
            ])
        );
    }

    getWinner() {
        if (this.world && this.done) {
            return this.world.winner;
        }
    }

    getRewardFromHpDiff(sideId: "p1" | "p2"): [number, number] {
        const omniscientHandler = this.handlers[sideId];
        const publicBattle = omniscientHandler.publicBattle;
        const sideHpSums = publicBattle.sides.map((side) =>
            side.team
                .map((pokemon) => pokemon.hp / pokemon.maxhp)
                .reduce((a, b) => a + b)
        );
        return [
            sideHpSums[0] > sideHpSums[1] ? 1 : -1,
            sideHpSums[1] > sideHpSums[0] ? 1 : -1,
        ];
    }

    getRewardFromFinish(sideId: "p1" | "p2"): [number, number] {
        const winner = this.getWinner();
        const omniscientHandler = this.handlers[sideId];
        const publicBattle = omniscientHandler.publicBattle;
        if (winner) {
            const p1Reward = publicBattle.p1.name === winner ? 1 : -1;
            const p2Reward = publicBattle.p2.name === winner ? 1 : -1;
            return [p1Reward, p2Reward];
        } else {
            return [0, 0];
        }
    }

    getRewards(sideId: "p1" | "p2"): [number, number] {
        return this.getRewardFromHpDiff(sideId);
        return this.getRewardFromFinish(sideId);
    }

    sendState(state: State) {
        const isDone = this.done;
        let info = state.getInfo();
        if (isDone && info) {
            info.setDone(isDone);
            const sideId = info.getPlayerindex() ? "p2" : "p1";
            const [r1, r2] = this.getRewards(sideId);
            info.setPlayeronereward(r1);
            info.setPlayertworeward(r2);
            state.setInfo(info);
            state.setLegalactions(AllValidActions);
        }
        const stateArr = state.serializeBinary();
        this.ts += 1;
        return this.port?.postMessage(stateArr, [stateArr.buffer]);
    }

    handleAction(stream: ObjectReadWriteStream<string>, action: Action) {
        const actionIndex = action.getIndex();
        stream.write(actionIndexMapping[actionIndex] ?? "default");
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
    }
}
