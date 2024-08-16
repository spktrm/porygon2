import { EvalActionFnType } from "../eval";
import { Battle } from "@pkmn/sim";
import { SideID } from "@pkmn/types";
import { createHash } from "crypto";
import { GetRandomAction } from "./random";
import { StreamHandler } from "../handler";
import { Random } from "random-js";

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

const PLAYERS: SideID[] = ["p1", "p2"];

function CopyBattle(battle: Battle): Battle {
    // const send = battle.send;
    const copy = Battle.fromJSON(JSON.stringify(battle));
    copy.restart(() => {});
    return copy;
}

class State {
    battle: Battle | null;
    playerIndex?: number;

    constructor(battle: Battle | null = null, playerIndex?: number) {
        this.battle = battle;
        this.playerIndex = playerIndex;
    }

    hash(): string {
        const stateString = JSON.stringify(this.battle);
        return `${createHash("md5").update(stateString).digest("hex")}`;
    }

    clone(): State {
        const copy = CopyBattle(this.battle!);
        return new State(copy);
    }

    returns(): number[] {
        const [totalHp1, totalHp2] = this.battle?.sides.map((side) =>
            side.pokemon.reduce((total, member) => {
                return total + member.hp / member.maxhp;
            }, 0),
        ) ?? [0, 0];
        const score = totalHp1 - totalHp2;
        return [score, -score];
    }

    isTerminal() {
        return this.battle?.ended ?? false;
    }

    private updateCurrentPlayer() {
        const isDones = this.battle!.sides.map((side) => side.isChoiceDone());
        const playerIndex = isDones.indexOf(false);
        this.playerIndex = playerIndex;
        return playerIndex;
    }

    currentPlayer() {
        if (!this.playerIndex) {
            this.updateCurrentPlayer();
        }
        return this.playerIndex as number;
    }

    legalActions(playerIndex?: number) {
        if (!playerIndex) playerIndex = this.currentPlayer();
        const legalActions: number[] = [];
        for (let actionIndex = 0; actionIndex < CHOICES.length; actionIndex++) {
            const clone = this.clone();
            const valid = clone.applyAction(actionIndex);
            if (valid) {
                legalActions.push(actionIndex);
            }
        }
        return legalActions;
    }

    applyAction(moveIndex: number): boolean {
        let moveStr = CHOICES[moveIndex];
        const currentPlayer = this.currentPlayer();
        const valid = this.battle!.choose(PLAYERS[currentPlayer], moveStr);
        this.updateCurrentPlayer();
        return valid;
    }
}

abstract class Evaluator {
    /** Abstract class representing an evaluation function for a game.
     *
     * The evaluation function takes in an intermediate state in the game and returns
     * an evaluation of that state, which should correlate with chances of winning
     * the game. It returns the evaluation from all player's perspectives.
     */
    abstract evaluate(state: State): number[];
    abstract prior(state: State): [number, number][];
}

class RandomRolloutEvaluator extends Evaluator {
    private nRollouts: number;
    private randomState: Random;
    private numSteps: number;

    constructor(nRollouts: number = 1, randomState: Random = new Random()) {
        super();
        this.nRollouts = nRollouts;
        this.randomState = randomState;
        this.numSteps = 3;
    }

    evaluate(state: State): number[] {
        let result: number[] | null = null;
        for (let i = 0; i < this.nRollouts; i++) {
            let workingState = state.clone();

            let n = 0;
            while (!workingState.isTerminal() && n <= this.numSteps) {
                let valid = false;
                const legalActions = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
                while (!valid) {
                    const legalActionIndex = Math.floor(
                        Math.random() * legalActions.length,
                    );
                    const actionIndex = legalActions[legalActionIndex];
                    valid = workingState.applyAction(actionIndex);
                    if (!valid) {
                        legalActions.splice(legalActionIndex, 1);
                    }
                    if (legalActions.length === 0) {
                        break;
                    }
                }
                n += 1;
            }
            const returns = workingState.returns();
            result = result ? result.map((r, i) => r + returns[i]) : returns;
        }
        return result!.map((r) => r / this.nRollouts);
    }

    prior(state: State): [number, number][] {
        const legalActions = state.legalActions(state.currentPlayer());
        return legalActions.map((action) => [
            action,
            1.0 / legalActions.length,
        ]);
    }
}

class SearchNode {
    action: number | null;
    player: number;
    prior: number;
    exploreCount: number;
    totalReward: number;
    outcome: number[] | null;
    children: SearchNode[];

    constructor(action: number | null, player: number, prior: number) {
        this.action = action;
        this.player = player;
        this.prior = prior;
        this.exploreCount = 0;
        this.totalReward = 0.0;
        this.outcome = null;
        this.children = [];
    }

    uctValue(parentExploreCount: number, uctC: number): number {
        if (this.outcome !== null) {
            return this.outcome[this.player];
        }
        if (this.exploreCount === 0) {
            return Infinity;
        }
        return (
            this.totalReward / this.exploreCount +
            uctC * Math.sqrt(Math.log(parentExploreCount) / this.exploreCount)
        );
    }

    puctValue(parentExploreCount: number, uctC: number): number {
        if (this.outcome !== null) {
            return this.outcome[this.player];
        }
        return (
            (this.exploreCount ? this.totalReward / this.exploreCount : 0) +
            (uctC * this.prior * Math.sqrt(parentExploreCount)) /
                (this.exploreCount + 1)
        );
    }

    sortKey(): [number, number, number] {
        return [
            this.outcome === null ? 0 : this.outcome[this.player],
            this.exploreCount,
            this.totalReward,
        ];
    }

    bestChild(): SearchNode {
        return this.children.reduce((a, b) =>
            this.compareNodes(a, b) > 0 ? a : b,
        );
    }

    compareNodes(a: SearchNode, b: SearchNode): number {
        const [outcomeA, exploreCountA, totalRewardA] = a.sortKey();
        const [outcomeB, exploreCountB, totalRewardB] = b.sortKey();
        return outcomeA !== outcomeB
            ? outcomeA - outcomeB
            : exploreCountA !== exploreCountB
            ? exploreCountA - exploreCountB
            : totalRewardA - totalRewardB;
    }
}

class MCTSBot {
    private uctC: number;
    private maxSimulations: number;
    private evaluator: Evaluator;
    private solve: boolean;
    private maxUtility: number;
    private dirichletNoise: [number, number] | null;
    private randomState: Random;
    private childSelectionFn: (
        node: SearchNode,
        parentExploreCount: number,
        uctC: number,
    ) => number;
    private dontReturnChanceNode: boolean;

    constructor(
        uctC: number,
        maxSimulations: number,
        evaluator: Evaluator,
        solve: boolean = true,
        randomState: Random = new Random(),
        childSelectionFn: (
            node: SearchNode,
            parentExploreCount: number,
            uctC: number,
        ) => number = (node, parentExploreCount, uctC) =>
            node.uctValue(parentExploreCount, uctC),
        dirichletNoise: [number, number] | null = null,
        dontReturnChanceNode: boolean = false,
    ) {
        this.uctC = uctC;
        this.maxSimulations = maxSimulations;
        this.evaluator = evaluator;
        this.solve = solve;
        this.maxUtility = 6;
        this.dirichletNoise = dirichletNoise;
        this.randomState = randomState;
        this.childSelectionFn = childSelectionFn;
        this.dontReturnChanceNode = dontReturnChanceNode;
    }

    restartAt(state: State): void {}

    stepWithPolicy(state: State): [SearchNode, number] {
        const t1 = Date.now();
        const root = this.mctsSearch(state);
        const best = root.bestChild();

        const mctsAction = best.action!;

        return [root, mctsAction];
    }

    step(state: State): [SearchNode, number] {
        return this.stepWithPolicy(state);
    }

    private gamma(shape: number, scale: number = 1): number {
        if (shape < 1) {
            // We use the method of Marsaglia and Tsang (2000) to handle shape < 1
            return (
                this.gamma(shape + 1, scale) *
                Math.pow(this.randomState.real(0, 1, false), 1 / shape)
            );
        }

        const d = shape - 1 / 3;
        const c = 1 / Math.sqrt(9 * d);

        while (true) {
            let x: number;
            let v: number;
            do {
                x = this.randomState.real(-1, 1, false);
                v = 1 + c * x;
            } while (v <= 0);

            v = v * v * v;
            const u = this.randomState.real(0, 1, false);

            if (
                u < 1 - 0.0331 * x * x * x * x ||
                Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))
            ) {
                return d * v * scale;
            }
        }
    }

    private dirichlet(alpha: number[]): number[] {
        const gammas = alpha.map((a) => this.gamma(a, 1));
        const sumGammas = gammas.reduce((a, b) => a + b, 0);
        return gammas.map((g) => g / sumGammas);
    }

    private applyTreePolicy(
        root: SearchNode,
        state: State,
    ): [SearchNode[], State] {
        const visitPath: SearchNode[] = [root];
        const workingState = state.clone();
        let currentNode = root;

        while (!workingState.isTerminal() && currentNode.exploreCount > 0) {
            if (!currentNode.children.length) {
                const legalActions = this.evaluator.prior(workingState);
                if (currentNode === root && this.dirichletNoise) {
                    const [epsilon, alpha] = this.dirichletNoise;
                    const noise = this.dirichlet(
                        Array(legalActions.length).fill(alpha),
                    );
                    legalActions.forEach((value, index) => {
                        legalActions[index][1] =
                            (1 - epsilon) * value[1] + epsilon * noise[index];
                    });
                }
                this.randomState.shuffle(legalActions);
                const player = workingState.currentPlayer();
                currentNode.children = legalActions.map(
                    ([action, prior]) => new SearchNode(action, player, prior),
                );
            }

            currentNode = currentNode.children.reduce((a, b) =>
                this.childSelectionFn(a, currentNode.exploreCount, this.uctC) >
                this.childSelectionFn(b, currentNode.exploreCount, this.uctC)
                    ? a
                    : b,
            );

            workingState.applyAction(currentNode.action!);
            visitPath.push(currentNode);
        }

        return [visitPath, workingState];
    }

    private mctsSearch(state: State): SearchNode {
        const root = new SearchNode(null, state.currentPlayer(), 1);
        for (let i = 0; i < this.maxSimulations; i++) {
            const [visitPath, workingState] = this.applyTreePolicy(root, state);
            let returns: number[];
            let solved = false;

            if (workingState.isTerminal()) {
                returns = workingState.returns();
                visitPath[visitPath.length - 1].outcome = returns;
                solved = this.solve;
            } else {
                returns = this.evaluator.evaluate(workingState);
            }

            while (visitPath.length) {
                const node = visitPath.pop()!;
                node.totalReward += returns[node.player];
                node.exploreCount += 1;

                if (solved && node.children.length) {
                    const player = node.children[0].player;

                    let best: SearchNode | null = null;
                    let allSolved = true;
                    for (const child of node.children) {
                        if (child.outcome === null) {
                            allSolved = false;
                        } else if (
                            best === null ||
                            (best.outcome !== null &&
                                child.outcome[player] > best.outcome[player])
                        ) {
                            best = child;
                        }
                    }

                    if (
                        best &&
                        best.outcome !== null &&
                        (allSolved || best.outcome[player] === this.maxUtility)
                    ) {
                        node.outcome = best.outcome;
                    } else {
                        solved = false;
                    }
                }
            }

            if (root.outcome !== null) {
                break;
            }
        }
        return root;
    }
}

export const GetSearchAction: EvalActionFnType = ({ handler }) => {
    if (handler.world === null) {
        return -1;
    }
    if (handler.world?.ended) {
        return -1;
    }
    const bot = new MCTSBot(
        config.uctC,
        config.maxSimulations,
        new RandomRolloutEvaluator(),
    );
    const state = new State(handler.world);
    try {
        const [root, mctsAction] = bot.step(state);
        return mctsAction;
    } catch (err) {
        return GetRandomAction({ handler });
    }
};

const config = {
    uctC: 2,
    maxSimulations: 50,
};

export function GetSearchDistribution(args: {
    handler: StreamHandler;
}): number[] {
    const { handler } = args;
    if (handler.world === null) {
        return new Array(10).fill(1);
    }
    if (handler.world?.ended) {
        return new Array(10).fill(1);
    }
    const bot = new MCTSBot(
        config.uctC,
        config.maxSimulations,
        new RandomRolloutEvaluator(),
    );
    const playerIndex = handler.getPlayerIndex() as number;
    const state = new State(handler.world, playerIndex);
    const [root, mctsAction] = bot.step(state);
    return root.children.map((child) => child.exploreCount);
}
