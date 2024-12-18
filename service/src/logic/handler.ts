import { Battle } from "@pkmn/client";
import { Generations } from "@pkmn/data";
import { Dex } from "@pkmn/dex";
import { AnyObject } from "@pkmn/sim";
import { Protocol } from "@pkmn/protocol";
import { Battle as World } from "@pkmn/sim";

import { EventHandler as EventHandler, StateHandler, Tracker } from "./state";

import { State } from "../../protos/state_pb";
import { Action } from "../../protos/action_pb";
import { getEvalAction } from "./eval";
import { OneDBoolean } from "./arr";
import { ObjectReadWriteStream } from "@pkmn/sim/build/cjs/lib/streams";
import { exit } from "process";

const generations = new Generations(Dex);

type keyType = string;
type sendFnType = (state: State) => Promise<keyType>;
type recvFnType = (key: keyType) => Promise<Action | undefined>;
type requestType = Protocol.MoveRequest;

export class StreamHandler {
    sendFn: sendFnType;
    recvFn: recvFnType;

    log: string[];
    actionLog: (Action | undefined)[];
    gameId: number;

    publicBattle: Battle;
    privateBattle: Battle;

    eventHandler: EventHandler;

    world: World | null;
    rqid: string | undefined;
    playerIndex: 0 | 1 | undefined;

    constructor(args: {
        gameId: number;
        sendFn: sendFnType;
        recvFn: recvFnType;
        playerIndex?: 0 | 1;
    }) {
        const { gameId, sendFn, recvFn, playerIndex } = args;

        this.gameId = gameId;
        this.log = [];
        this.actionLog = [];

        this.publicBattle = new Battle(generations);
        this.privateBattle = new Battle(generations);

        this.eventHandler = new EventHandler(this);

        this.world = null;
        this.rqid = undefined;
        this.playerIndex = playerIndex;

        this.sendFn = sendFn;
        this.recvFn = recvFn;
    }

    async ingestLine(line: string) {
        try {
            this.privateBattle.add(line);
        } catch (err) {
            console.log(err);
        }
        if (!line.startsWith("|request|")) {
            this.publicBattle.add(line);
        } else {
            this.getPlayerIndex();
        }
        this.log.push(line);
        this.ingestEvent(line);
        if (line.startsWith("|start")) {
            this.eventHandler.reset();
        }
    }

    ingestEvent(line: string) {
        const { args, kwArgs } = Protocol.parseBattleLine(line);
        const key = Protocol.key(args);
        if (!key) return;
        if (key in this.eventHandler) {
            (this.eventHandler as any)[key](args, kwArgs);
        }
    }

    isActionRequired(
        chunk: string,
        stream?: ObjectReadWriteStream<string>,
    ): boolean {
        const request = this.getRequest()! as AnyObject;
        if (request === undefined) {
            return false;
        }
        this.rqid = request.rqid;
        if (request.teamPreview) {
            return true;
        }
        if (!!request.wait) {
            return false;
        }
        if (stream === undefined) {
            if (chunk.includes("|turn")) {
                return true;
            }
            if (!chunk.includes("|request")) {
                return !!(request.forceSwitch ?? [])[0];
            }
            return false;
        }
        return true;
    }

    getRequest(): requestType {
        return this.privateBattle.request as requestType;
    }

    async getState(numHistory?: number): Promise<State> {
        const handler = new StateHandler(this);
        return await handler.getState(numHistory);
    }

    getPlayerIndex(): number | undefined {
        const request = this.privateBattle.request;
        if (this.playerIndex !== undefined) {
            return this.playerIndex;
        }
        if (request) {
            this.playerIndex = (parseInt(
                request.side?.id.toString().slice(1) ?? "",
            ) - 1) as 0 | 1;
            return this.playerIndex;
        }
    }

    async stateActionStep(): Promise<Action | undefined> {
        const state = await this.getState();
        // const legalActions = state.getLegalactions_asU8();
        // const request = this.privateBattle.request;
        // if (request !== undefined) {
        //     const oneDBool = new OneDBoolean(10);
        //     oneDBool.buffer.set(legalActions);
        //     const numValidMoves = oneDBool
        //         .toBinaryVector()
        //         .reduce((prev, curr) => prev + curr);
        //     if (numValidMoves <= 1) {
        //         const action = new Action();
        //         action.setIndex(-1);
        //         action.setText("default");
        //         return action;
        //     }
        // }
        const key = await this.sendFn(state);
        const action = await this.recvFn(key);
        this.actionLog.push(action);
        this.privateBattle.request = undefined;
        return action;
    }

    ensureRequestApplied() {
        if (this.privateBattle.request) {
            while (this.privateBattle.requestStatus !== "applied") {
                this.privateBattle.update();
            }
            if (this.privateBattle.turn === 1) {
                this.privateBattle.requestStatus =
                    "received" as Battle["requestStatus"];
                while (this.privateBattle.requestStatus !== "applied") {
                    this.privateBattle.update();
                }
            }
        }
    }

    async ingestChunk(chunk: string, stream?: ObjectReadWriteStream<string>) {
        if (chunk) {
            if (chunk.startsWith("|error")) {
                console.log(chunk);
            }
            for (const line of chunk.split("\n")) {
                await this.ingestLine(line);
            }
            const numMessages = stream?.buf?.length ?? 0;
            if (numMessages === 0 && this.isActionRequired(chunk, stream)) {
                this.ensureRequestApplied();
                return await this.stateActionStep();
            }
        }
    }

    reset() {
        this.log = [];
        this.actionLog = [];

        this.publicBattle = new Battle(generations);
        this.privateBattle = new Battle(generations);

        this.rqid = undefined;
        this.playerIndex = undefined;
    }
}
