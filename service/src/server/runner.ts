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
import {
    Action,
    ActionEnum,
    EnvironmentState,
    StepRequest,
    WildCardEnum,
} from "../../protos/service_pb";
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

    choiceFromAction(action: Action, isSingles: boolean = true): string {
        const actionIndex = action.getAction();
        if (
            ActionEnum.ACTION_ENUM__MOVE_1_TARGET_NA <= actionIndex &&
            actionIndex <= ActionEnum.ACTION_ENUM__MOVE_4_TARGET_4
        ) {
            const moveIndex = Math.floor((actionIndex - 1) / 5);
            const targetIndex = (actionIndex - 1) % 5;
            const wildCardIndex = action.getWildcard();

            let moveCommand = `move ${moveIndex + 1}`;
            if (targetIndex !== 0 && !isSingles) {
                const showdownFormat =
                    targetIndex <= 2 ? -targetIndex : targetIndex - 2;
                moveCommand += ` ${showdownFormat}`;
            }
            if (wildCardIndex !== WildCardEnum.WILD_CARD_ENUM__CAN_NORMAL) {
                switch (wildCardIndex) {
                    case WildCardEnum.WILD_CARD_ENUM__CAN_MEGA:
                        moveCommand += " mega";
                        break;
                    case WildCardEnum.WILD_CARD_ENUM__CAN_ZMOVE:
                        moveCommand += " zmove";
                        break;
                    case WildCardEnum.WILD_CARD_ENUM__CAN_MAX:
                        moveCommand += " dynamax";
                        break;
                    case WildCardEnum.WILD_CARD_ENUM__CAN_TERA:
                        moveCommand += " terastallize";
                        break;
                    default:
                        throw new Error(
                            `Invalid wildcard index: ${wildCardIndex}`,
                        );
                }
            }

            return moveCommand;
        } else if (
            ActionEnum.ACTION_ENUM__SWITCH_1 <= actionIndex &&
            actionIndex <= ActionEnum.ACTION_ENUM__SWITCH_6
        ) {
            const switchIndex = actionIndex - ActionEnum.ACTION_ENUM__SWITCH_1;
            return `switch ${switchIndex + 1}`;
        } else if (actionIndex === ActionEnum.ACTION_ENUM__DEFAULT) {
            return "default";
        } else if (actionIndex === ActionEnum.ACTION_ENUM__PASS) {
            return "pass";
        } else {
            throw new Error(`Invalid action index: ${actionIndex}`);
        }
    }

    choicesFromActions(
        actions: Action[],
        isTeamPreview: boolean = false,
    ): string {
        if (isTeamPreview) {
            const order: number[] = [];
            for (const action of actions) {
                const actionIndex = action.getAction();
                if (
                    ActionEnum.ACTION_ENUM__SWITCH_1 <= actionIndex &&
                    actionIndex <= ActionEnum.ACTION_ENUM__SWITCH_6
                ) {
                    const switchIndex =
                        actionIndex - ActionEnum.ACTION_ENUM__SWITCH_1;
                    order.push(switchIndex + 1);
                } else if (actionIndex === ActionEnum.ACTION_ENUM__DEFAULT) {
                    return "default";
                } else {
                    throw new Error(
                        `Invalid team preview action index: ${actionIndex}`,
                    );
                }
            }
            // Numbers excluding those chosen
            const rest = [1, 2, 3, 4, 5, 6]
                .filter((n) => !order.includes(n))
                .map((n) => n.toString())
                .join("");
            return `team ${order.join("")}${rest}`;
        }
        const choiceList: string[] = [];
        const isSingles = !!this.privateBattle.gameType?.includes("singles");
        if (actions.length === 1 || actions.length === 2) {
            for (const action of actions) {
                const choice = this.choiceFromAction(action, isSingles);
                choiceList.push(choice);
            }
            return choiceList.join(",");
        } else {
            throw new Error(
                `Expected actions length of 1 or 2, but got ${actions.length}`,
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

        const actions = stepRequest.getActionsList()!;
        return this.choicesFromActions(
            actions,
            !!this.getRequest()?.teamPreview,
        );
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
        return this.choicesFromActions(
            actions,
            !!this.getRequest()?.teamPreview,
        );
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

    receiveLine(line: string) {
        if (this.debug) console.log(line);
        if (!line.startsWith("|")) return;
        const [cmd, rest] = splitFirst(line.slice(1), "|");
        if (cmd === "request") return this.receiveRequest(JSON.parse(rest));
        if (cmd === "error") return this.receiveError(new Error(rest));
        this.log.push(line);
    }

    override async start() {
        const choices = [];
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

    destroy() {
        this.privateBattle.destroy();
        this.publicBattle.destroy();
    }
}

const MAX_REQUEST_COUNT = 32 * 3;

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
    const maxRequestCount = options.maxRequestCount ?? MAX_REQUEST_COUNT;
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

    const p1 = new TrainablePlayerAI(
        p1spec.name,
        streams.p1,
        {},
        debug,
        p1Sets,
    );
    const p2 = new TrainablePlayerAI(
        p2spec.name,
        streams.p2,
        {},
        debug,
        p2Sets,
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
