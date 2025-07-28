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
import { EventHandler, StateHandler } from "./state";
import { Protocol } from "@pkmn/protocol";
import { EnvironmentState, StepRequest } from "../../protos/service_pb";
import { evalActionMapping } from "./eval";
import { isBaselineUser, TaskQueueSystem } from "./utils";

Teams.setGeneratorFactory(TeamGenerators);

const formatid = "gen3ou";

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

const CHOICES = [
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

export class TrainablePlayerAI extends RandomPlayerAI {
    userName: string;
    privateBattle: Battle;
    publicBattle: Battle;
    eventHandler: EventHandler;

    tasks: TaskQueueSystem<StepRequest>;
    outgoingQueue: AsyncQueue<EnvironmentState>;

    currentRequest: ChoiceRequest | null = null;
    done: boolean;
    backup: string[];

    finishedEarly: boolean;
    hasRequest: boolean;
    started: boolean;
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
        this.publicBattle = new Battle(new Generations(Dex), null, sets);
        this.done = false;

        this.outgoingQueue = new AsyncQueue<EnvironmentState>();
        this.tasks = new TaskQueueSystem();

        this.backup = [];
        this.hasRequest = false;
        this.started = false;

        this.playerIndex = undefined;
        this.requestCount = 0;
        this.finishedEarly = false;
        this.rqid = -1;

        const isBaseline = isBaselineUser(userName);
        this.isBaseline = isBaseline;
        if (isBaseline) {
            const baselineIndex = parseInt(userName.split("-").at(-1) ?? "");
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
    }

    public submitStepRequest(stepRequest: StepRequest) {
        const rqid = stepRequest.getRqid();
        this.tasks.submitResult(rqid, stepRequest);
    }

    public async receiveEnvironmentState() {
        return await this.outgoingQueue.dequeueAsync();
    }

    updateRequest() {
        while (this.privateBattle.requestStatus !== "applied") {
            this.privateBattle.update();
        }
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

    isActionRequired(chunk: string): boolean {
        const request = this.getRequest()! as AnyObject;
        if (this.privateBattle.requestStatus !== "applied") {
            return false;
        }
        if (!request) {
            return false;
        }
        if (
            request?.teamPreview ||
            (chunk.includes("|inactive|") && chunk.includes("30 seconds left"))
        ) {
            return true;
        }
        if (request?.wait) {
            return false;
        }
        if (chunk.includes("|turn")) {
            return true;
        }
        if (!chunk.includes("|request")) {
            return !!(request?.forceSwitch ?? [])[0];
        }
        return false;
    }

    isActionRequired2(chunk: string) {
        const request = this.getRequest()! as AnyObject;
        return chunk.includes("|request") && !request?.wait;
    }

    choiceFromAction(action: number) {
        if (action < 0) {
            return "default";
        }
        return CHOICES[action];
    }

    addLine(line: string) {
        this.ingestEvent(line);
        try {
            this.privateBattle.add(line);
        } catch (err) {
            console.log(err);
            this.privateBattle.add(line);
        }
        this.publicBattle.add(line);
        this.getPlayerIndex();
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
                }. Must be between 0 and ${evalActionMapping.length - 1}`,
            );
        }
        const evalFn = evalActionMapping[this.baselineIndex];
        const { actionString, actionIndex } = evalFn({
            player: this,
        });
        if (actionString !== undefined) {
            return actionString;
        }
        if (actionIndex !== undefined) {
            return this.choiceFromAction(actionIndex);
        }
        throw new Error("actionString or actionIndex should not be undefined");
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

    override async start() {
        const backup: string[] = [];

        const shiftBackup = () => {
            while (backup.length > 0) {
                const line = backup.shift();
                if (line) this.addLine(line);
            }
        };
        const lines = [];

        for await (const chunk of this.stream) {
            lines.push(...chunk.split("\n"));

            if (this.done || this.finishedEarly) {
                break;
            }

            try {
                this.receive(chunk);
            } catch (err) {
                console.log(err);
            }

            if (!this.hasRequest) {
                if (chunk.includes("|request")) {
                    this.log.push("---request---");
                    this.hasRequest = true;
                }
            }

            for (const line of chunk.split("\n")) {
                backup.push(line);
                if (line.startsWith("|start")) {
                    this.started = true;
                }
            }

            if (this.hasRequest && this.started) {
                shiftBackup();
                // for (const line of chunk.split("\n")) {
                //     this.addLine(line);
                // }
                this.updateRequest();
                if (this.privateBattle.turn === 1) {
                    this.privateBattle.requestStatus = "received";
                    this.updateRequest();
                }
                this.hasRequest = false;
            }

            // When stream is empty, wait for action from async source
            if (
                this.stream.buf.length === 0 &&
                this.isActionRequired(chunk) &&
                // this.isActionRequired2(chunk) &&
                true
            ) {
                shiftBackup();
                this.privateBattle.requestStatus = "received";
                this.updateRequest();
                this.rqid = this.getRequest().rqid;

                const choice = await this.getChoice();
                this.privateBattle.request = undefined;

                // Process the received action
                this.choose(choice);

                // Increment internal counters
                this.requestCount += 1;
            }
        }

        this.done = true;

        shiftBackup();
        this.sendFinalState();
    }

    destroy() {
        this.privateBattle.destroy();
        this.publicBattle.destroy();
    }
}

export function createBattle(
    options: { p1Name: string; p2Name: string; maxRequestCount?: number },
    debug: boolean = false,
) {
    const { p1Name, p2Name } = options;
    const maxRequestCount = options.maxRequestCount ?? 100;

    const streams = BattleStreams.getPlayerStreams(
        new BattleStreams.BattleStream(),
    );
    const spec = { formatid };

    const p1Sets = Teams.generate("gen3randombattle");
    const p1spec = {
        name: p1Name,
        team: Teams.pack(p1Sets),
    };
    const p2Sets = Teams.generate("gen3randombattle");
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
