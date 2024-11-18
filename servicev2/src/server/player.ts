import { Battle } from "@pkmn/client";
import { Battle as World, AnyObject } from "@pkmn/sim";
import { Generations } from "@pkmn/data";
import { Dex } from "@pkmn/dex";
import { ObjectReadWriteStream } from "@pkmn/streams";
import { BattleStreams } from "@pkmn/sim";
import { recvFnType, sendFnType } from "./types";
import { EventHandler, StateHandler } from "./state";
import { Protocol } from "@pkmn/protocol";
import { Action } from "../../protos/servicev2_pb";

const generations = new Generations(Dex);

const actionStrings = [
    "move 1",
    "move 2",
    "move 3",
    "move 4",
    "switch 1",
    "switch 2",
    "switch 3",
    "switch 4",
    "switch 5",
    "switch 6",
];

export class Player extends BattleStreams.BattlePlayer {
    publicBattle: Battle;
    privateBattle: Battle;
    eventHandler: EventHandler;
    actionLog: Action[];

    send: sendFnType;
    recv: recvFnType;

    workerIndex: number;
    gameId: number;
    playerId: number | undefined;
    playerIndex: number | undefined;
    rqid: string | undefined;
    done: boolean;

    worldStream: ObjectReadWriteStream<string> | null;

    constructor(
        workerIndex: number,
        gameId: number,
        playerStream: ObjectReadWriteStream<string>,
        playerId: number,
        send: sendFnType,
        recv: recvFnType,
        worldStream: ObjectReadWriteStream<string> | null,
    ) {
        super(playerStream);

        this.send = send;
        this.recv = recv;

        this.publicBattle = new Battle(generations);
        this.privateBattle = new Battle(generations);
        this.eventHandler = new EventHandler(this);
        this.actionLog = [];

        this.workerIndex = workerIndex;
        this.gameId = gameId;
        this.playerId = playerId;
        this.playerIndex = undefined;
        this.rqid = undefined;
        this.done = false;
        this.worldStream = worldStream;
    }

    getPlayerIndex(): number | undefined {
        if (this.playerIndex !== undefined) {
            return this.playerIndex;
        }
        const request = this.privateBattle.request;
        if (!!request) {
            this.playerIndex = (parseInt(
                request.side?.id.toString().slice(1) ?? "",
            ) - 1) as 0 | 1;
            return this.playerIndex;
        }
    }

    getRequest() {
        return this.privateBattle.request!;
    }

    addLine(line: string) {
        this.privateBattle.add(line);
        this.publicBattle.add(line);
        this.getPlayerIndex();
        if (line.startsWith("|start")) {
            this.eventHandler.reset();
        }
        this.ingestEvent(line);
    }

    ingestEvent(line: string) {
        const { args, kwArgs } = Protocol.parseBattleLine(line);
        const key = Protocol.key(args);
        if (!key) return;
        if (key in this.eventHandler) {
            (this.eventHandler as any)[key](args, kwArgs);
        }
    }

    isActionRequired(chunk: string): boolean {
        const request = this.getRequest()! as AnyObject;
        if (!!!request) {
            return false;
        }
        this.rqid = request.rqid;
        if (request.teamPreview) {
            return true;
        }
        if (!!request.wait) {
            return false;
        }
        if (this.worldStream === null) {
            if (chunk.includes("|turn")) {
                return true;
            }
            if (!chunk.includes("|request")) {
                return !!(request.forceSwitch ?? [])[0];
            }
            return false;
        } else {
        }
        return true;
    }

    isWorldReady() {
        if (this.worldStream !== null) {
            return this.worldStream!.buf.length === 0;
        }
        return true;
    }

    async start() {
        for await (const chunk of this.stream) {
            if (this.done) {
                break;
            }

            this.receive(chunk);

            for (const line of chunk.split("\n")) {
                this.addLine(line);
            }
            this.privateBattle.update();

            if (
                this.isWorldReady() &&
                this.stream.buf.length === 0 &&
                this.isActionRequired(chunk)
            ) {
                const key = await this.send(this);
                this.privateBattle.request = undefined;
                const action = await this.recv(key!);
                if (action !== undefined) {
                    this.actionLog.push(action);
                    const actionValue = action.getValue();
                    this.choose(actionStrings[actionValue] ?? "default");
                }
            }
        }

        this.done = true;
        await this.send(this);
    }

    receiveRequest(request: AnyObject): void {}

    createState() {
        const handler = new StateHandler(this);
        return handler.getState();
    }
}
