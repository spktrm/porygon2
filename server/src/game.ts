import { Battle } from "@pkmn/client";
import { Battle as World } from "@pkmn/sim";
import { Generations } from "@pkmn/data";
import { Dex } from "@pkmn/dex";
import { TeamGenerators } from "@pkmn/randoms";
import { AnyObject, BattleStreams, Teams } from "@pkmn/sim";
import { ObjectReadWriteStream } from "@pkmn/streams";

import { AsyncQueue } from "./utils";
import { Protocol } from "@pkmn/protocol";
import { MessagePort } from "worker_threads";

import { State } from "../protos/state_pb";
import { Action } from "../protos/action_pb";
import { AllValidActions, StateHandler } from "./state";

const formatId = "gen3randombattle";
const generator = TeamGenerators.getTeamGenerator(formatId);
const generations = new Generations(Dex);

type sendFnType = (state: State) => void;
type recvFnType = (playerIndex: number) => Promise<Action | undefined>;
type requestType = Protocol.MoveRequest;

export class StreamHandler {
    sendFn: sendFnType;
    recvFn: recvFnType;

    gameId: number;
    battle: Battle;
    rqid: string | undefined;
    queue: AsyncQueue<Action>;
    playerIndex: number | undefined;

    constructor(args: {
        gameId: number;
        sendFn: sendFnType;
        recvFn: recvFnType;
    }) {
        const { gameId, sendFn, recvFn } = args;

        this.gameId = gameId;
        this.battle = new Battle(generations);
        this.rqid = undefined;
        this.queue = new AsyncQueue();
        this.playerIndex = undefined;

        this.sendFn = sendFn;
        this.recvFn = recvFn;
    }

    ingestLine(line: string) {
        this.battle.add(line);
    }

    isActionRequired(chunk: string): boolean {
        const request = (this.battle.request ?? {}) as AnyObject;
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
        return this.battle.request as requestType;
    }

    getState(): State {
        return new StateHandler(this).getState();
    }

    getPlayerIndex(): number | undefined {
        const request = this.battle.request;
        if (request) {
            if (this.playerIndex) {
                return this.playerIndex;
            }
            this.playerIndex =
                parseInt(request.side?.id.toString().slice(1) ?? "") - 1;
            return this.playerIndex;
        }
    }

    async stateActionStep(done: boolean = false): Promise<Action | undefined> {
        const state = this.getState();
        this.sendFn(state);
        const playerIndex = this.getPlayerIndex();
        if (playerIndex !== undefined && !done) {
            return await this.recvFn(playerIndex);
        }
    }

    async ingestChunk(chunk: string) {
        if (chunk) {
            for (const line of chunk.split("\n")) {
                this.ingestLine(line);
            }
        }
        if (this.isActionRequired(chunk)) {
            return await this.stateActionStep();
        }
    }

    reset() {
        this.rqid = undefined;
        this.battle = new Battle(generations);
        return;
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

export class Game {
    port: MessagePort | null;
    gameId: number;
    dones: number;
    handlers: {
        [k: string]: StreamHandler;
    };
    queues: {
        [k: string]: AsyncQueue<Action>;
    };
    world: World | null;

    constructor(args: { port: MessagePort | null; gameId: number }) {
        const { gameId, port } = args;
        this.port = port;
        this.gameId = gameId;
        this.dones = 0;
        this.world = null;

        this.queues = { p1: new AsyncQueue(), p2: new AsyncQueue() };
        this.handlers = Object.fromEntries(
            ["p1", "p2", "omniscient"].map((x) => [
                x,
                new StreamHandler({
                    gameId,
                    sendFn: (state) => {
                        if (x !== "omniscient") this.sendState(state);
                    },
                    recvFn: async () => {
                        return x !== "omniscient"
                            ? await this.queues[x].dequeue()
                            : undefined;
                    },
                }),
            ])
        );
    }

    getWinner() {
        if (this.world && this.isDone()) {
            return this.world.winner;
        }
    }

    getRewards(): [number, number] {
        const winner = this.getWinner();
        const omniscientHandler = this.handlers["omniscient"];
        const publicBattle = omniscientHandler.battle;
        if (winner) {
            const p1Reward = publicBattle.p1.name === winner ? 1 : -1;
            const p2Reward = publicBattle.p2.name === winner ? 1 : -1;
            return [p1Reward, p2Reward];
        } else {
            return [0, 0];
        }
    }

    isDone() {
        return this.dones === 2;
    }

    sendState(state: State) {
        const isDone = this.isDone();
        let info = state.getInfo();
        if (isDone && info) {
            info.setDone(this.dones === 2);
            const [r1, r2] = this.getRewards();
            info.setPlayeronereward(r1);
            info.setPlayertworeward(r2);
            state.setInfo(info);
            state.setLegalactions(AllValidActions);
        }
        const stateArr = state.serializeBinary();
        return this.port?.postMessage(stateArr, [stateArr.buffer]);
    }

    handleAction(stream: ObjectReadWriteStream<string>, action: Action) {
        const actionIndex = action.getIndex();
        stream.write(actionIndexMapping[actionIndex] ?? "default");
    }

    async runPlayer(args: {
        id: "omniscient" | "p1" | "p2";
        stream: ObjectReadWriteStream<string>;
    }) {
        const { id, stream } = args;
        const handler = this.handlers[id];
        for await (const chunk of stream) {
            const action = await handler.ingestChunk(chunk);
            if (action !== undefined) {
                this.handleAction(stream, action);
            }
        }
        this.dones += 1;
        const isDone = this.isDone();
        if (isDone) {
            await handler.stateActionStep(isDone);
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
            this.runPlayer({
                id: "omniscient",
                stream: streams.omniscient,
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

        return await players;
    }

    reset() {
        for (const handlerId of Object.keys(this.handlers)) {
            this.handlers[handlerId].reset();
        }
        this.dones = 0;
        return;
    }
}
