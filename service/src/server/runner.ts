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
import { Generations, PokemonSet } from "@pkmn/data";
import { Dex } from "@pkmn/dex";
import { ChoiceRequest } from "@pkmn/sim/build/cjs/sim/side";
import { ObjectReadWriteStream } from "@pkmn/sim/build/cjs/lib/streams";
import { EventHandler, RewardTracker, StateHandler } from "./state";
import { Protocol } from "@pkmn/protocol";
import { Action, EnvironmentState, StepRequest } from "../../protos/service_pb";
import { evalActionMapping, numEvals } from "./eval";
import { isBaselineUser, TaskQueueSystem } from "./utils";
import { ActionMaskFeature, ActionType } from "../../protos/features_pb";

Teams.setGeneratorFactory(TeamGenerators);

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

const ACTION_TYPES = new Map<number, string>();
ACTION_TYPES.set(ActionType.ACTION_TYPE__MOVE, "move");
ACTION_TYPES.set(ActionType.ACTION_TYPE__SWITCH, "switch");
ACTION_TYPES.set(ActionType.ACTION_TYPE__TEAMPREVIEW, "team");
ACTION_TYPES.set(ActionType.ACTION_TYPE__DEFAULT, "default");

export class TrainablePlayerAI extends RandomPlayerAI {
    userName: string;
    privateBattle: Battle;
    publicBattle: Battle;
    eventHandler: EventHandler;

    tasks: TaskQueueSystem<StepRequest>;
    outgoingQueue: AsyncQueue<EnvironmentState>;
    rewardTracker: RewardTracker;

    currentRequest: ChoiceRequest | null = null;
    done: boolean;

    finishedEarly: boolean;
    playerIndex: number | undefined;
    requestCount: number;
    rqid: number;

    isBaseline: boolean;
    baselineIndex: number;
    earliestTeraTurn: number;

    constructor(
        userName: string,
        playerStream: ObjectReadWriteStream<string>,
        options: {
            move?: number;
            mega?: number;
            seed?: PRNG | PRNGSeed | null;
        } = {},
        debug: boolean = false,
        sets?: PokemonSet[],
        earliestTeraTurn: number = 0,
    ) {
        super(playerStream, options, debug);

        this.userName = userName;

        this.eventHandler = new EventHandler(this);
        this.privateBattle = new Battle(new Generations(Dex), null, sets);
        this.publicBattle = new Battle(new Generations(Dex), null);
        this.done = false;

        this.outgoingQueue = new AsyncQueue<EnvironmentState>();
        this.tasks = new TaskQueueSystem();
        this.rewardTracker = new RewardTracker();

        this.playerIndex = undefined;
        this.requestCount = 0;
        this.finishedEarly = false;
        this.rqid = -1;
        this.earliestTeraTurn = earliestTeraTurn;

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

    createGameState(): EnvironmentState {
        const stateHandler = new StateHandler(this);
        return stateHandler.build();
    }

    getRequest(): AnyObject {
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

    choiceFromAction(action: Action) {
        const actionTypeIndex = action.getActionType();
        const moveSlotIndex = action.getMoveSlot()! + 1;
        const switchSlotIndex = action.getSwitchSlot()! + 1;
        const wildCardSlot = action.getWildcardSlot();

        if (actionTypeIndex === undefined) {
            throw new Error(
                "Action type index is undefined. Ensure the action is properly defined.",
            );
        }
        const actionType = ACTION_TYPES.get(actionTypeIndex);
        if (actionType === undefined) {
            throw new Error(
                `Invalid action type index: ${actionTypeIndex}. Must be one of ${Array.from(
                    ACTION_TYPES.keys(),
                )}.`,
            );
        }
        if (actionType === "default") {
            return "default";
        }
        if (actionType === "move" && moveSlotIndex !== undefined) {
            let moveString = `${actionType} ${moveSlotIndex}`;
            if (
                wildCardSlot ===
                ActionMaskFeature.ACTION_MASK_FEATURE__CAN_MEGA -
                    ActionMaskFeature.ACTION_MASK_FEATURE__CAN_NORMAL
            ) {
                moveString += " mega";
            }
            if (
                wildCardSlot ===
                ActionMaskFeature.ACTION_MASK_FEATURE__CAN_ZMOVE -
                    ActionMaskFeature.ACTION_MASK_FEATURE__CAN_NORMAL
            ) {
                moveString += " zmove";
            }
            if (
                wildCardSlot ===
                ActionMaskFeature.ACTION_MASK_FEATURE__CAN_MAX -
                    ActionMaskFeature.ACTION_MASK_FEATURE__CAN_NORMAL
            ) {
                moveString += " dynamax";
            }
            if (
                wildCardSlot ===
                ActionMaskFeature.ACTION_MASK_FEATURE__CAN_TERA -
                    ActionMaskFeature.ACTION_MASK_FEATURE__CAN_NORMAL
            ) {
                moveString += " terastallize";
            }
            return moveString;
        } else if (actionType === "switch" && switchSlotIndex !== undefined) {
            return `${actionType} ${switchSlotIndex}`;
        } else if (actionType === "team" && switchSlotIndex !== undefined) {
            const rest = [1, 2, 3, 4, 5, 6];
            rest.splice(rest.indexOf(switchSlotIndex), 1);
            return `${actionType} ${switchSlotIndex}${rest.join("")}`;
        } else {
            throw new Error(
                `Invalid action: ${actionType} with moveSlot: ${moveSlotIndex} and switchSlot: ${switchSlotIndex}.`,
            );
        }
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
        const action = evalFn({
            player: this,
        });
        return this.choiceFromAction(action);
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

    serialize() {
        return {
            log: this.log,
        };
    }

    override async start() {
        for await (const chunk of this.stream) {
            if (chunk.includes("error|")) {
                console.log(`Error in stream: ${chunk}`);
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

                        const choice = await withTimeoutWarning(
                            () => this.getChoice(),
                            1000,
                            "getChoice",
                        );

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

    destroy() {
        this.privateBattle.destroy();
        this.publicBattle.destroy();
    }
}

let totalBattles = 0;

function sampleEarliestTeraTurn(
    minTurn: number = 2,
    maxTurn: number = 20,
    scale: number = 500,
): number {
    if (minTurn > maxTurn) {
        throw new Error("minTurn must be ≤ maxTurn");
    }

    // Compute a shape parameter. As totalBattles grows, alpha grows and
    // the distribution becomes more skewed toward smaller values.
    const alpha = 1 + totalBattles / Math.max(1, scale);

    // Draw a uniform random number U in (0, 1)
    const u = Math.random();

    // Transform U to follow a Beta(1, alpha) distribution.
    // Beta(1, alpha) is skewed toward 0 for alpha > 1.
    const x = 1 - Math.pow(u, 1 / alpha);

    // Map x ∈ [0, 1) to an integer turn in [minTurn, maxTurn].
    const range = maxTurn - minTurn + 1;
    const turn = minTurn + Math.floor(x * range);

    return turn;
}

export function createBattle(
    options: {
        p1Name: string;
        p2Name: string;
        p1team: string | null;
        p2team: string | null;
        maxRequestCount?: number;
        smogonFormat: string;
    },
    debug: boolean = false,
) {
    const { p1Name, p2Name, p1team, p2team } = options;
    const maxRequestCount = options.maxRequestCount ?? 60;
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

    totalBattles += 1;

    const p1 = new TrainablePlayerAI(
        p1spec.name,
        streams.p1,
        {},
        debug,
        p1Sets,
        sampleEarliestTeraTurn(),
    );
    const p2 = new TrainablePlayerAI(
        p2spec.name,
        streams.p2,
        {},
        debug,
        p2Sets,
        sampleEarliestTeraTurn(),
    );

    p1.start();
    p2.start();

    void streams.omniscient.write(`>start ${JSON.stringify(spec)}
>player p1 ${JSON.stringify(p1spec)}
>player p2 ${JSON.stringify(p2spec)}`);

    (async () => {
        // eslint-disable-next-line @typescript-eslint/no-unused-vars
        for await (const chunk of streams.omniscient) {
            if (
                p1.requestCount >= maxRequestCount ||
                p2.requestCount >= maxRequestCount
            ) {
                p1.finishEarly();
                p2.finishEarly();
            }
        }
    })();

    return { p1, p2 };
}
