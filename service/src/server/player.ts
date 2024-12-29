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

export class Tracker {
    hpDiffs: number[];
    faintedDiffs: number[];
    turnsWithoutChange: number;

    constructor() {
        this.hpDiffs = [0];
        this.faintedDiffs = [0];
        this.turnsWithoutChange = 0;
    }

    update1(battle: Battle) {
        const [aliveTotal1, aliveTotal2] = battle.sides.map((side) => {
            const aliveInTeam = side.team.reduce(
                (count, pokemon) => count + (pokemon.fainted ? 0 : 1),
                0,
            );
            const remainingPokemon = side.totalPokemon - side.team.length;
            return aliveInTeam + remainingPokemon;
        });
        const aliveDiff = aliveTotal1 - aliveTotal2;
        this.faintedDiffs.push(aliveDiff);
        const faintedReward =
            (this.faintedDiffs.at(-1) ?? 0) - (this.faintedDiffs.at(-2) ?? 0);

        // Calculate HP totals for both sides
        const [hpTotal1, hpTotal2] = battle.sides.map((side) => {
            const hpInTeam = side.team.reduce(
                (sum, pokemon) => sum + pokemon.hp / pokemon.maxhp,
                0,
            );
            const remainingPokemon = side.totalPokemon - side.team.length;
            return hpInTeam + remainingPokemon;
        });

        const hpDiff = hpTotal1 - hpTotal2;
        const hpReward = hpDiff - (this.hpDiffs.at(-1) ?? 0);
        this.hpDiffs.push(hpDiff);

        return { faintedReward, hpReward };
    }

    update2(battle: World) {
        const [aliveTotal1, aliveTotal2] = battle.sides.map((side) => {
            const aliveInTeam = side.pokemon.reduce(
                (count, pokemon) => count + +(pokemon.hp > 0),
                0,
            );
            return aliveInTeam;
        });
        const aliveDiff = aliveTotal1 - aliveTotal2;
        this.faintedDiffs.push(aliveDiff);
        const faintedReward =
            (this.faintedDiffs.at(-1) ?? 0) - (this.faintedDiffs.at(-2) ?? 0);

        // // Calculate HP totals for both sides
        const [hpTotal1, hpTotal2] = battle.sides.map((side) => {
            const hpInTeam = side.pokemon.reduce(
                (sum, pokemon) => sum + pokemon.hp / pokemon.maxhp,
                0,
            );
            return hpInTeam;
        });

        const hpDiff = hpTotal1 - hpTotal2;
        const hpReward = hpDiff - (this.hpDiffs.at(-1) ?? 0);
        if (hpReward === 0) {
            this.turnsWithoutChange += 1;
        } else {
            this.turnsWithoutChange = 0;
        }
        this.hpDiffs.push(hpDiff);

        return { faintedReward, hpReward };
    }

    reset() {
        this.hpDiffs = [0];
        this.faintedDiffs = [0];
        this.turnsWithoutChange = 0;
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
