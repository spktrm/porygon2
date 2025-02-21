import { Battle } from "@pkmn/client";
import { Battle as World } from "@pkmn/sim";
import { AnyObject } from "@pkmn/sim";
import { Generations } from "@pkmn/data";
import { Dex } from "@pkmn/dex";
import { ObjectReadWriteStream } from "@pkmn/streams";
import { BattleStreams } from "@pkmn/sim";
import { recvFnType, sendFnType } from "./types";
import { EventHandler, StateHandler } from "./state";
import { Protocol } from "@pkmn/protocol";
import { Action } from "../../protos/service_pb";

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

interface TeamTrackerDatum {
    hpTotal: number;
    aliveTotal: number;
    faintedTotal: number;
    numPokemon: number;
    damageTotal: number;
    playerIndex: number;
}

export class Tracker {
    battle: World | undefined;
    data: TeamTrackerDatum[][];
    f_int: (x: number) => number;

    constructor() {
        this.data = [];
        this.battle = undefined;

        this.f_int = (x) => {
            return (1 / 216) * x ** 3;
        };
    }

    setBattle(world: World) {
        this.battle = world;
    }

    update(playerIndex: number) {
        if (this.battle === undefined) {
            throw new Error();
        }

        const sides = [];
        for (const side of this.battle.sides) {
            let hpTotal = 0;
            let aliveTotal = 0;
            for (const member of side.pokemon) {
                hpTotal += member.hp / member.maxhp;
                aliveTotal += +(member.hp > 0);
            }
            const numPokemon = side.pokemon.length;
            sides.push({
                hpTotal,
                aliveTotal,
                numPokemon,
                faintedTotal: numPokemon - aliveTotal,
                damageTotal: numPokemon - hpTotal,
                playerIndex,
            });
        }
        this.data.push(sides);
    }

    getFaintedReward() {
        const tm1 = this.data.at(-1) ?? [];
        const tm2 = this.data.at(-2) ?? [];

        const p1tm1 = tm1.at(0)?.aliveTotal ?? 6;
        const p1tm2 = tm2.at(0)?.aliveTotal ?? 6;

        const p2tm1 = tm1.at(1)?.aliveTotal ?? 6;
        const p2tm2 = tm2.at(1)?.aliveTotal ?? 6;

        const reward = p1tm1 - p1tm2 - (p2tm1 - p2tm2);
        return reward;
    }

    getHpReward() {
        const tm1 = this.data.at(-1) ?? [];
        const tm2 = this.data.at(-2) ?? [];

        const p1tm1 = tm1.at(0)?.hpTotal ?? 6;
        const p1tm2 = tm2.at(0)?.hpTotal ?? 6;

        const p2tm1 = tm1.at(1)?.hpTotal ?? 6;
        const p2tm2 = tm2.at(1)?.hpTotal ?? 6;

        const reward = p1tm1 - p1tm2 - (p2tm1 - p2tm2);
        return reward;
    }

    getScaledFaintedReward() {
        const tm1 = this.data.at(-1) ?? [];
        const tm2 = this.data.at(-2) ?? [];

        const f1tm1 = tm1.at(0)?.faintedTotal ?? 0;
        const f1tm2 = tm2.at(0)?.faintedTotal ?? 0;

        const f2tm1 = tm1.at(1)?.faintedTotal ?? 0;
        const f2tm2 = tm2.at(1)?.faintedTotal ?? 0;

        const p1Diff = this.f_int(f2tm2) - this.f_int(f2tm1);
        const p2Diff = this.f_int(f1tm2) - this.f_int(f1tm1);

        const reward = p2Diff - p1Diff;
        return reward;
    }

    getScaledHpReward() {
        const tm1 = this.data.at(-1) ?? [];
        const tm2 = this.data.at(-2) ?? [];

        const d1tm1 = tm1.at(0)?.damageTotal ?? 0;
        const d1tm2 = tm2.at(0)?.damageTotal ?? 0;

        const d2tm1 = tm1.at(1)?.damageTotal ?? 0;
        const d2tm2 = tm2.at(1)?.damageTotal ?? 0;

        const p1Diff = this.f_int(d2tm2) - this.f_int(d2tm1);
        const p2Diff = this.f_int(d1tm2) - this.f_int(d1tm1);

        const reward = p2Diff - p1Diff;
        return reward;
    }

    getWinReward() {
        const tm1 = this.data.at(-1) ?? [];

        const f1tm1 = tm1.at(0)?.faintedTotal ?? 0;
        const n1tm1 = tm1.at(0)?.numPokemon ?? 6;

        const f2tm1 = tm1.at(1)?.faintedTotal ?? 0;
        const n2tm1 = tm1.at(1)?.numPokemon ?? 6;

        const p1Diff = +(f1tm1 === n1tm1);
        const p2Diff = +(f2tm1 === n2tm1);

        const reward = p2Diff - p1Diff;
        return reward;
    }

    getReward() {
        const faintedReward = this.getFaintedReward();
        const hpReward = this.getHpReward();
        const scaledHpReward = this.getScaledHpReward();
        const scaledFaintedReward = this.getScaledFaintedReward();
        const winReward = this.getWinReward();

        return {
            faintedReward,
            hpReward,
            scaledFaintedReward,
            scaledHpReward,
            winReward,
        };
    }

    reset() {
        this.data = [];
    }
}

export class Player extends BattleStreams.BattlePlayer {
    publicBattle: Battle;
    privateBattle: Battle;
    eventHandler: EventHandler;
    tracker: Tracker;

    actionLog: Action[];

    send: sendFnType;
    recv: recvFnType;

    workerIndex: number;
    gameId: number;
    playerId: number | undefined;
    playerIndex: number | undefined;
    rqid: string | undefined;
    requestCount: number;

    done: boolean;
    draw: boolean;

    worldStream: BattleStreams.BattleStream | null;
    offline: boolean;
    hasRequest: boolean;

    constructor(
        workerIndex: number,
        gameId: number,
        playerStream: ObjectReadWriteStream<string>,
        playerId: number,
        send: sendFnType,
        recv: recvFnType,
        worldStream: BattleStreams.BattleStream | null,
        choose?: (action: string) => void,
        offline: boolean = false,
        playerIndex: number | undefined = undefined,
    ) {
        super(playerStream);

        this.send = send;
        this.recv = recv;

        if (choose !== undefined) {
            this.choose = choose;
        }

        this.publicBattle = new Battle(generations);
        this.privateBattle = new Battle(generations);
        this.eventHandler = new EventHandler(this);

        this.tracker = new Tracker();

        this.actionLog = [];

        this.workerIndex = workerIndex;
        this.gameId = gameId;
        this.playerId = playerId;
        this.playerIndex = playerIndex;
        this.rqid = undefined;
        this.requestCount = 0;

        this.done = false;
        this.draw = false;

        this.worldStream = worldStream;
        this.offline = offline;
        this.hasRequest = false;
    }

    getPlayerIndex(): number | undefined {
        if (this.playerIndex !== undefined) {
            return this.playerIndex;
        }
        const request = this.privateBattle.request;
        if (request) {
            this.playerIndex = (parseInt(
                request.side?.id.toString().slice(1) ?? "",
            ) - 1) as 0 | 1;
            return this.playerIndex;
        }
    }

    getRequest(): Protocol.Request {
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
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            (this.eventHandler as any)[key](args, kwArgs);
        }
    }

    isActionRequired(chunk: string): boolean {
        const request = this.getRequest()! as AnyObject;
        if (this.offline) {
            return true;
        }
        if (!this.offline && !request) {
            return false;
        }
        this.rqid = request?.rqid;
        if (request?.teamPreview) {
            return true;
        }
        if (request?.wait) {
            return false;
        }
        if (this.worldStream === null) {
            if (chunk.includes("|turn")) {
                return true;
            }
            if (!chunk.includes("|request")) {
                return !!(request?.forceSwitch ?? [])[0];
            }
            return false;
        } else {
            /* empty */
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
        const backup: string[] = [];

        for await (const chunk of this.stream) {
            if (this.done || this.draw) {
                // Early finish
                break;
            }

            try {
                this.receive(chunk);
            } catch (err) {
                console.log(err);
            }

            if (!this.hasRequest) {
                if (chunk.includes("|request")) {
                    this.hasRequest = true;
                }
            }

            if (this.hasRequest) {
                for (const line of chunk.split("\n")) {
                    this.addLine(line);
                }
                while (backup.length > 0) {
                    const line = backup.shift();
                    if (line) this.addLine(line);
                }
            } else {
                for (const line of chunk.split("\n")) {
                    backup.push(line);
                }
            }
            this.privateBattle.update();

            if (
                this.isWorldReady() &&
                (this.offline || this.stream.buf.length === 0) &&
                this.isActionRequired(chunk)
            ) {
                this.log.push("---request---");
                this.requestCount += 1;
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

    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    receiveRequest(request: AnyObject): void {}

    createState() {
        const handler = new StateHandler(this);
        return handler.getState();
    }
}
