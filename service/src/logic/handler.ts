import { Battle } from "@pkmn/client";
import { Generations } from "@pkmn/data";
import { Dex } from "@pkmn/dex";
import { AnyObject } from "@pkmn/sim";
import { Protocol } from "@pkmn/protocol";

import { EventHandler, StateHandler } from "./state";

import { State } from "../../protos/state_pb";
import { Action } from "../../protos/action_pb";
import { getEvalAction } from "./baselines";

const generations = new Generations(Dex);

type keyType = string;
type sendFnType = (state: State) => Promise<keyType>;
type recvFnType = (key: keyType) => Promise<Action | undefined>;
type requestType = Protocol.MoveRequest;

export class StreamHandler {
    sendFn: sendFnType;
    recvFn: recvFnType;

    ts: number;
    log: string[];
    gameId: number;
    isTraining: boolean;

    publicBattle: Battle;
    privatebattle: Battle;
    eventHandler: EventHandler;

    rqid: string | undefined;
    isEvalAction: boolean | undefined;
    playerIndex: 0 | 1 | undefined;

    constructor(args: {
        gameId: number;
        isTraining: boolean;
        sendFn: sendFnType;
        recvFn: recvFnType;
    }) {
        const { gameId, isTraining, sendFn, recvFn } = args;

        this.ts = Date.now();
        this.gameId = gameId;
        this.isTraining = isTraining;
        this.log = [];

        this.publicBattle = new Battle(generations);
        this.privatebattle = new Battle(generations);
        this.eventHandler = new EventHandler(this);

        this.rqid = undefined;
        this.playerIndex = undefined;
        this.isEvalAction = undefined;

        this.sendFn = sendFn;
        this.recvFn = recvFn;
    }

    ingestLine(line: string) {
        this.privatebattle.add(line);
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
            this.playerIndex = (parseInt(
                request.side?.id.toString().slice(1) ?? "",
            ) - 1) as 0 | 1;
            return this.playerIndex;
        }
    }

    getIsEvalAction() {
        if (this.isEvalAction === undefined) {
            this.isEvalAction = !this.isTraining && this.playerIndex === 1;
        }
        return this.isEvalAction;
    }

    async stateActionStep(): Promise<Action | undefined> {
        if (this.getIsEvalAction()) {
            return getEvalAction(this);
        } else {
            const state = this.getState();
            const legalActions = state.getLegalactions();
            if (legalActions) {
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
            return await this.recvFn(key);
        }
    }

    ensureRequestApplied() {
        if (this.privatebattle.request) {
            while (this.privatebattle.requestStatus !== "applied") {
                this.privatebattle.update();
            }
            if (this.privatebattle.turn === 1) {
                this.privatebattle.requestStatus =
                    "received" as Battle["requestStatus"];
                while (this.privatebattle.requestStatus !== "applied") {
                    this.privatebattle.update();
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
        this.log = [];

        this.publicBattle = new Battle(generations);
        this.privatebattle = new Battle(generations);
        this.eventHandler.reset();

        this.rqid = undefined;
        this.playerIndex = undefined;
    }
}
