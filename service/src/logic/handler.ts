import { Battle } from "@pkmn/client";
import { Generations } from "@pkmn/data";
import { Dex } from "@pkmn/dex";
import { AnyObject } from "@pkmn/sim";
import { Protocol } from "@pkmn/protocol";
import { Battle as World } from "@pkmn/sim";

import { EventHandler2, StateHandler } from "./state";

import { State } from "../../protos/state_pb";
import { Action } from "../../protos/action_pb";
import { getEvalAction } from "./eval";

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
    eventHandler2: EventHandler2;
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
        this.eventHandler2 = new EventHandler2(this);

        this.world = null;
        this.rqid = undefined;
        this.playerIndex = playerIndex;

        this.sendFn = sendFn;
        this.recvFn = recvFn;
    }

    ingestLine(line: string) {
        this.privateBattle.add(line);
        if (!line.startsWith("|request|")) {
            this.publicBattle.add(line);
        } else {
            this.getPlayerIndex();
        }
        this.log.push(line);
        this.ingestEvent(line);
    }

    ingestEvent(line: string) {
        const { args, kwArgs } = Protocol.parseBattleLine(line);
        const key = Protocol.key(args);
        if (!key) return;
        if (key in this.eventHandler2) {
            (this.eventHandler2 as any)[key](args, kwArgs);
        }
    }

    getSwitchReward(maxLogLength: number = 5) {
        let count = 0;
        if (this.actionLog.length === 0) {
            return count;
        }
        const actionLog = this.actionLog.slice(-maxLogLength);
        let numActions = 0;
        for (const action of [...actionLog].reverse()) {
            if (action && action.getIndex() !== -1) {
                if (action.getIndex() >= 4) {
                    break;
                }
                numActions += 1;
            }
        }
        return (maxLogLength - numActions) / maxLogLength;
    }

    isActionRequired(chunk: string): boolean {
        const request = (this.privateBattle.request ?? {}) as AnyObject;
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
        return this.privateBattle.request as requestType;
    }

    async getState(): Promise<State> {
        const handler = new StateHandler(this);
        return await handler.getState();
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
        const legalActions = state.getLegalactions();
        const request = this.privateBattle.request;
        if (request && legalActions) {
            const legalObj = legalActions.toObject();
            const numValidMoves = Object.values(legalObj)
                .map((x) => (x ? 1 : 0) as number)
                .reduce((a, b) => a + b);
            if (numValidMoves <= 1) {
                const action = new Action();
                action.setIndex(-1);
                action.setText("default");
                return action;
            }
        }
        const key = await this.sendFn(state);
        const action = await this.recvFn(key);
        this.actionLog.push(action);
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

    async ingestChunk(chunk: string) {
        if (chunk) {
            if (chunk.startsWith("|error")) {
                console.log(chunk);
            }
            for (const line of chunk.split("\n")) {
                this.ingestLine(line);
            }
            if (this.isActionRequired(chunk)) {
                this.ensureRequestApplied();
                return await this.stateActionStep();
            }
        }
    }

    reset() {
        this.eventHandler2.reset();

        this.log = [];
        this.actionLog = [];

        this.publicBattle = new Battle(generations);
        this.privateBattle = new Battle(generations);

        this.rqid = undefined;
        this.playerIndex = undefined;
    }
}
