import {
    AnyObject,
    BattleStreams,
    PRNG,
    PRNGSeed,
    RandomPlayerAI,
    Teams,
} from "@pkmn/sim";
import { TeamGenerators } from "@pkmn/randoms";
import { Battle } from "@pkmn/client";
import { Generations } from "@pkmn/data";
import { Dex } from "@pkmn/dex";
import { ChoiceRequest } from "@pkmn/sim/build/cjs/sim/side";
import { ObjectReadWriteStream } from "@pkmn/sim/build/cjs/lib/streams";
import { EventHandler, RewardTracker, StateHandler } from "./state";
import { numActionFeatures } from "./data";
import { Protocol } from "@pkmn/protocol";
import { Action, EnvironmentState, StepRequest } from "../../protos/service_pb";
import { evalActionMapping, numEvals } from "./eval";
import { isBaselineUser, TaskQueueSystem } from "./utils";

Teams.setGeneratorFactory(TeamGenerators);

function splitFirst(str: string, delimiter: string, limit = 1) {
    const splitStr = [];
    while (splitStr.length < limit) {
        const delimiterIndex = str.indexOf(delimiter);
        if (delimiterIndex >= 0) {
            splitStr.push(str.slice(0, delimiterIndex));
            str = str.slice(delimiterIndex + delimiter.length);
        } else {
            splitStr.push(str);
            str = "";
        }
    }
    splitStr.push(str);
    return splitStr;
}

async function withTimeoutWarning<T>(
    fn: () => Promise<T>,
    thresholdMs: number,
    fnName: string,
): Promise<T> {
    const start = Date.now();

    try {
        const result = await fn();
        const duration = Date.now() - start;

        if (duration > thresholdMs) {
            console.warn(
                `${fnName} took ${duration}ms (threshold: ${thresholdMs}ms)`,
            );
        }

        return result;
    } catch (err) {
        const duration = Date.now() - start;
        console.error(`${fnName} failed after ${duration}ms:`, err);
        throw err;
    }
}

interface Queue<T> {
    enqueue(item: T): void;
    dequeue(): T | undefined;
    isEmpty(): boolean;
    size(): number;
}

export class AsyncQueue<T> implements Queue<T> {
    private items: T[] = [];
    private waitingResolvers: ((value: T) => void)[] = [];
    private maxSize: number;

    constructor(maxSize: number = Infinity) {
        this.maxSize = maxSize;
    }

    enqueue(item: T): void {
        if (this.items.length >= this.maxSize) {
            throw new Error(`Queue is full. Maximum size: ${this.maxSize}`);
        }

        this.items.push(item);

        // If there are waiting resolvers, immediately resolve the oldest one
        if (this.waitingResolvers.length > 0) {
            const resolver = this.waitingResolvers.shift()!;
            resolver(this.items.shift()!);
        }
    }

    dequeue(): T | undefined {
        return this.items.shift();
    }

    // Async version that waits for items to be available
    async dequeueAsync(): Promise<T> {
        // If items are available, return immediately
        if (this.items.length > 0) {
            return this.items.shift()!;
        }

        // Otherwise, wait for an item to be enqueued
        return new Promise<T>((resolve) => {
            this.waitingResolvers.push(resolve);
        });
    }

    isEmpty(): boolean {
        return this.items.length === 0;
    }

    size(): number {
        return this.items.length;
    }

    clear(): void {
        this.items = [];
        // Reject all waiting promises
        this.waitingResolvers.forEach((resolver) => {
            // You might want to reject with an error instead
            resolver(undefined as never);
        });
        this.waitingResolvers = [];
    }

    peek(): T | undefined {
        return this.items[0];
    }

    // Get a copy of all items without removing them
    getItems(): T[] {
        return [...this.items];
    }
}

const globalGens = new Generations(Dex);

export class TrainablePlayerAI extends RandomPlayerAI {
    userName: string;
    privateBattle: Battle;
    publicBattle: Battle;
    eventHandler: EventHandler;

    tasks: TaskQueueSystem<StepRequest>;
    outgoingQueue: AsyncQueue<EnvironmentState>;
    rewardTracker: RewardTracker;

    done: boolean;

    finishedEarly: boolean;
    playerIndex: number | undefined;
    // Pre-game ladder ratings [p1, p2], parsed from |player| lines by the
    // offline exporter; [0, 0] (unknown) in live self-play.
    ratings: [number, number] = [0, 0];
    requestCount: number;
    rqid: number;
    choices: string[];
    actions: Action[];
    // (src * numActionFeatures + tgt) -> the Showdown choice string that cell
    // means, rebuilt by StateHandler.getActionMask on every request (and, in
    // doubles, every sub-decision). choiceFromAction is a lookup in it, so the
    // mask and the decoder cannot disagree about what a cell means. Empty
    // until the first state is built, and on requests that carry no choice.
    legalChoiceByCell: Map<number, string> = new Map();
    // How many choices this battle's sim rejected outright. Should be 0.
    invalidChoiceCount: number = 0;

    isBaseline: boolean;
    baselineIndex: number;

    firstRequest: AnyObject | undefined;
    // Training self-play only: the other TrainablePlayerAI in this battle,
    // wired by createBattle after both players exist. Deploy-time players
    // have no opponent object, so everything reading this must tolerate
    // undefined (the opponent's private info simply does not exist there).
    opponent: TrainablePlayerAI | undefined;
    constructor(
        userName: string,
        playerStream: ObjectReadWriteStream<string>,
        options: {
            move?: number;
            mega?: number;
            seed?: PRNG | PRNGSeed | null;
        } = {},
        debug: boolean = false,
    ) {
        super(playerStream, options, debug);

        this.userName = userName;

        this.privateBattle = new Battle(globalGens, null);
        this.publicBattle = new Battle(globalGens, null);
        this.eventHandler = new EventHandler(this);
        this.done = false;
        this.choices = [];
        this.actions = [];

        this.outgoingQueue = new AsyncQueue<EnvironmentState>();
        this.tasks = new TaskQueueSystem();
        this.rewardTracker = new RewardTracker();

        this.playerIndex = undefined;
        this.requestCount = 0;
        this.finishedEarly = false;
        this.rqid = -1;
        this.firstRequest = undefined;
        this.opponent = undefined;

        const isBaseline = isBaselineUser(userName);
        this.isBaseline = isBaseline;
        if (isBaseline) {
            const baselineIndex = parseInt(userName.split(":").at(-1) ?? "");
            this.baselineIndex = baselineIndex;
        } else {
            this.baselineIndex = -1;
        }
    }

    finishEarly() {
        this.finishedEarly = true;
    }

    getPlayerIndex() {
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
        throw new Error(
            "Player index is undefined and request is not available",
        );
    }

    public submitStepRequest(stepRequest: StepRequest) {
        const rqid = stepRequest.getRqid();
        this.tasks.submitResult(rqid, stepRequest);
    }

    public async receiveEnvironmentState() {
        return await this.outgoingQueue.dequeueAsync();
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

    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    override receiveRequest(request: ChoiceRequest) {}

    createGameState(includeHistory: boolean = true): EnvironmentState {
        const stateHandler = new StateHandler(this);
        return stateHandler.build(includeHistory);
    }

    getRequest(): AnyObject {
        // Guard the snapshot: battle.request is undefined until the first
        // |request| line, and JSON.stringify(undefined) is not parseable.
        if (this.firstRequest === undefined && this.privateBattle.request) {
            this.firstRequest = JSON.parse(
                JSON.stringify(this.privateBattle.request),
            );
        }
        return this.privateBattle.request as AnyObject;
    }

    isActionRequired(): boolean {
        const request = this.getRequest()! as AnyObject;
        if (!request) {
            return false;
        }
        if (request?.wait) {
            return false;
        }
        return true;
    }

    /**
     * The chosen cell -> the Showdown choice string, by lookup in the map
     * StateHandler.getActionMask built when it legalised that cell.
     *
     * This used to be a hand-written decoder that re-derived the string from
     * (src, tgt) independently of the mask, and the two drifted: it appended
     * " terastallize" to every wildcard, it re-resolved move targets through a
     * different move list than the mask indexed under Dynamax, and it ignored
     * the target on team preview entirely. One table, built where the facts
     * are, makes all three unrepresentable.
     */
    choiceFromAction(action: Action): string {
        const cell = action.getSrc() * numActionFeatures + action.getTgt();
        const choice = this.legalChoiceByCell.get(cell);
        if (choice !== undefined) {
            return choice;
        }
        // An empty map means the current request carries no choice (a wait, or
        // no request at all). getActionMask lights every cell in that case so
        // masked averages downstream never see an empty row, and "default" is
        // the only thing any of those cells can mean.
        if (this.legalChoiceByCell.size === 0) {
            return "default";
        }
        throw new Error(
            `Action (src ${action.getSrc()}, tgt ${action.getTgt()}) is not a legal cell`,
        );
    }

    addLine(cmd: string, line: string) {
        this.ingestEvent(line);
        try {
            this.privateBattle.add(line);
        } catch (err) {
            console.log(err);
            this.privateBattle.add(line);
        }
        if (cmd !== "request") {
            this.publicBattle.add(line);
        }
    }

    async generateStepRequest(
        gameState: EnvironmentState,
    ): Promise<StepRequest> {
        const future = this.tasks.createJob();
        gameState.setRqid(future);
        this.outgoingQueue.enqueue(gameState);
        return await this.tasks.getResult(future);
    }

    private async getTrainingActorChoice() {
        // Create game state and put it in outgoing queue
        const gameState = this.createGameState();

        // Wait for action from incoming queue
        const stepRequest = await this.generateStepRequest(gameState);

        if (stepRequest.getRqid() !== gameState.getRqid()) {
            throw new Error(
                `RQID mismatch: ${stepRequest.getRqid()} !== ${gameState.getRqid()}`,
            );
        }

        const action = stepRequest.getAction()!;
        this.actions.push(action);

        return this.choiceFromAction(action);
    }

    private getEvalActorChoice() {
        if (
            this.baselineIndex < 0 ||
            this.baselineIndex >= evalActionMapping.length
        ) {
            throw new Error(
                `Invalid eval index: ${
                    this.baselineIndex
                }. Must be between 0 and ${numEvals - 1}.`,
            );
        }
        const evalFn = evalActionMapping[this.baselineIndex];
        if (evalFn === undefined) {
            throw new Error(
                `No eval function found for username: ${this.userName}`,
            );
        }
        const actions = evalFn({
            player: this,
        });
        return this.choiceFromAction(actions);
    }

    async getChoice(): Promise<string> {
        if (this.isBaseline && this.playerIndex === 1) {
            return Promise.resolve(this.getEvalActorChoice());
        } else {
            return await this.getTrainingActorChoice();
        }
    }

    sendFinalState() {
        if (!this.isBaseline) {
            const gameState = this.createGameState();
            this.outgoingQueue.enqueue(gameState);
        }
    }

    receiveLine(line: string) {
        if (this.debug) console.log(line);
        if (!line.startsWith("|")) return;
        const [cmd, rest] = splitFirst(line.slice(1), "|");
        if (cmd === "request") {
            return this.receiveRequest(JSON.parse(rest));
        }
        if (cmd === "error") {
            return this.receiveError(new Error(rest));
        }
        this.log.push(line);
    }

    async getChoices() {
        let choicesNeeded = 1;
        if (this.privateBattle.gameType === "doubles") {
            choicesNeeded = 2;
        } else if (this.privateBattle.gameType === "triples") {
            choicesNeeded = 3;
        }

        const request = this.getRequest();
        choicesNeeded = request.maxChosenTeamSize ?? choicesNeeded;

        while (this.choices.length < choicesNeeded) {
            const choice = await withTimeoutWarning(
                () => this.getChoice(),
                1000,
                "getChoice",
            );
            this.choices.push(choice);
        }

        const getTeamPreviewChoice = () => {
            const order = [1, 2, 3, 4, 5, 6];
            for (const [toIdx, choice] of this.choices.entries()) {
                const fromIdx = parseInt(choice.split(" ")[1]) - 1;
                [order[toIdx], order[fromIdx]] = [order[fromIdx], order[toIdx]];
            }
            const slicedOrder = order.slice(0, request.maxChosenTeamSize ?? 6);
            return "team " + slicedOrder.join("");
        };

        const choice =
            (request.teamPreview ?? false)
                ? getTeamPreviewChoice()
                : this.choices.join(",");

        this.choices = [];
        this.actions = [];

        return choice;
    }

    override async start() {
        const choices = [];
        for await (const chunk of this.stream) {
            if (chunk.startsWith("|error|")) {
                if (chunk.includes("Invalid choice")) {
                    // Counted, not merely logged: a mask cell whose choice
                    // string the sim rejects is exactly the drift the
                    // cell -> choice map exists to prevent, and a console line
                    // is invisible to the test suite. The harness asserts this
                    // stays at zero.
                    this.invalidChoiceCount += 1;
                    console.error(`Invalid choice error in stream: ${chunk}`);
                } else if (chunk.includes("Unavailable choice")) {
                    console.log(`Unavailable move error in stream: ${chunk}`);
                } else {
                    console.error(`Error in stream: ${chunk}`);
                }
            }

            if (this.done || this.finishedEarly) {
                break;
            }

            try {
                this.receive(chunk);
            } catch (err) {
                console.log(err);
            }

            for (const line of chunk.split("\n")) {
                if (line) {
                    const [cmd] = line.slice(1).split("|");
                    if (cmd === "tie" || cmd === "win") {
                        this.done = true;
                    }
                    this.addLine(cmd, line);

                    if (cmd === "request" && this.isActionRequired()) {
                        this.rqid = this.getRequest().rqid;

                        const choice = await this.getChoices();

                        choices.push(choice);

                        // Process the received action
                        try {
                            this.choose(choice);
                        } catch (err) {
                            console.error(
                                `Error choosing action ${choice}:`,
                                err,
                            );
                        }

                        // Increment internal counters
                        this.requestCount += 1;
                    }
                }
            }
        }

        this.done = true;

        this.sendFinalState();
    }

    // One-shot battle-stream teardown shared by BOTH players of a game,
    // attached by createBattle — lets destroy() end the underlying
    // BattleStream so the sim Battle (whose `log`/`inputLog`/Pokemon
    // graph dwarfs the client-side views) is freed too. Without this, an
    // early-finished game's BattleStream never sees an `end` and retains
    // the full sim Battle for as long as either player object is
    // reachable. MUST be once-only and swallow async errors:
    // BattleStream._writeEnd unconditionally re-runs battle.destroy(),
    // which throws on a second call (already-nulled internals), and
    // writeEnd() is async so a plain try/catch around it catches
    // nothing — the rejection killed workers as an unhandled 'error'
    // (see the 2026-08-13 service crash).
    endBattleStream: (() => void) | undefined;

    destroy() {
        this.privateBattle.destroy();
        this.publicBattle.destroy();
        this.endBattleStream?.();
    }
}

function hpDiff(battle: Battle): number {
    let p1Total = 0;
    let p2Total = 0;

    for (let i = 0; i < 2; i++) {
        const side = battle.sides[i];
        let knownHp = 0;

        // Use a standard for-loop instead of .reduce
        for (let j = 0; j < side.team.length; j++) {
            const pkmn = side.team[j];
            if (pkmn.fainted) continue;

            if (pkmn.maxhp === 0) {
                knownHp += 1;
            } else {
                knownHp += pkmn.hp / pkmn.maxhp;
            }
        }

        const unknownHp = side.totalPokemon - side.team.length;
        const total = knownHp + unknownHp;

        if (i === 0) p1Total = total;
        else p2Total = total;
    }

    return p1Total - p2Total;
}

export function createBattle(
    options: {
        p1Name: string;
        p2Name: string;
        p1team: string | null;
        p2team: string | null;
        smogonFormat: string;
    },
    debug: boolean = false,
) {
    const { p1Name, p2Name, p1team, p2team } = options;
    const smogonFormat = options.smogonFormat.replace("_ou_all_formats", "ou");

    const streams = BattleStreams.getPlayerStreams(
        new BattleStreams.BattleStream(),
    );
    const spec = { formatid: smogonFormat };

    const p1Sets =
        p1team === null ? Teams.generate(smogonFormat) : Teams.unpack(p1team);
    const p2Sets =
        p2team === null ? Teams.generate(smogonFormat) : Teams.unpack(p2team);

    if (p1Sets === null || p2Sets === null) {
        throw new Error(`Invalid team format for p1: ${p1team}, p2: ${p2team}`);
    }

    const p1spec = {
        name: p1Name,
        team: Teams.pack(p1Sets),
    };
    const p2spec = {
        name: p2Name,
        team: Teams.pack(p2Sets),
    };

    const p1 = new TrainablePlayerAI(p1spec.name, streams.p1, {}, debug);
    const p2 = new TrainablePlayerAI(p2spec.name, streams.p2, {}, debug);
    p1.opponent = p2;
    p2.opponent = p1;
    // Shared once-guard: whichever player's cleanup runs first ends the
    // stream (freeing the sim Battle); the second call is a no-op. The
    // .catch swallows the async double-destroy TypeError class — by this
    // point the battle is torn down either way, so there is nothing
    // actionable in the rejection.
    let battleStreamEnded = false;
    const endBattleStream = () => {
        if (battleStreamEnded) return;
        battleStreamEnded = true;
        streams.omniscient.writeEnd().catch(() => {});
    };
    p1.endBattleStream = endBattleStream;
    p2.endBattleStream = endBattleStream;

    p1.start();
    p2.start();

    void streams.omniscient.write(`>start ${JSON.stringify(spec)}
>player p1 ${JSON.stringify(p1spec)}
>player p2 ${JSON.stringify(p2spec)}`);

    (async () => {
        const spectator = new Battle(globalGens, null);

        // Replace your tracking variables with this:
        const windowSize = 40;
        const maxChange = 0.01;

        const hpHistory = new Float32Array(windowSize);
        let historyIndex = 0;
        let turnsLogged = 0;

        for await (const chunk of streams.omniscient) {
            for (const line of chunk.split("\n")) {
                spectator.add(line);

                if (line.startsWith("|turn")) {
                    const currentHpDiff = hpDiff(spectator);

                    // Log the current HP diff into our circular buffer
                    hpHistory[historyIndex] = currentHpDiff;
                    historyIndex = (historyIndex + 1) % windowSize;
                    turnsLogged++;

                    // Only check for stagnation if we've filled the window
                    if (turnsLogged >= windowSize) {
                        let maxDiff = -Infinity;
                        let minDiff = Infinity;

                        // Find the highest and lowest HP diffs in the last 40 turns
                        for (let i = 0; i < windowSize; i++) {
                            if (hpHistory[i] > maxDiff) maxDiff = hpHistory[i];
                            if (hpHistory[i] < minDiff) minDiff = hpHistory[i];
                        }

                        // If the total fluctuation over 40 turns is tiny, it's a stall
                        if (maxDiff - minDiff <= maxChange) {
                            p1.finishEarly();
                            p2.finishEarly();
                            break;
                        }
                    }
                }
            }
        }
        spectator.destroy();
    })();

    return { p1, p2 };
}
