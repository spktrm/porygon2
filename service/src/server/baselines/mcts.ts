/** Simultaneous-action, open-loop Monte Carlo tree search.
 *
 * Both sides select from their own bandit before either action is applied.
 * Fresh root samples can represent hidden-state hypotheses and simulator RNG;
 * the tree indexes action histories, never an opponent's submitted live choice.
 * Decoupled UCT is a bounded baseline, not a Nash-equilibrium guarantee.
 */

import { puctProbabilities, sampleProbability } from "./puct";

export interface SearchState {
    priors?(side: 0 | 1, actions: string[]): number[];
    actions(side: 0 | 1): string[];
    advance(ownAction: string, opponentAction: string): void;
    terminalValue(): number | undefined;
    evaluate(): number;
    dispose(): void;
}

export interface SearchOptions {
    iterations: number;
    maxDepth: number;
    random: () => number;
    maxMillis?: number;
    exploration?: number;
    selection?: "uct" | "puct";
    cpuct?: number;
    now?: () => number;
}

interface Arm {
    visits: number;
    total: number;
}

interface Node {
    visits: number;
    total: number;
    arms: [Map<string, Arm>, Map<string, Arm>];
    children: Map<string, Node>;
}

export interface SearchResult {
    action: string;
    iterations: number;
    root: {
        action: string;
        visits: number;
        mean: number;
        probability?: number;
    }[];
    matrix?: {
        worlds: number;
        cells: number;
        evaluations: number;
        solverIterations: number;
        gap: number;
        ownActions: number;
        legalOwnActions: number;
    };
}

function newNode(): Node {
    return {
        visits: 0,
        total: 0,
        arms: [new Map(), new Map()],
        children: new Map(),
    };
}

function select(
    node: Node,
    side: 0 | 1,
    legal: string[],
    random: () => number,
    exploration: number,
) {
    if (!legal.length)
        throw new Error("Search state has no action (use an explicit wait)");
    const shuffled = [...new Set(legal)];
    for (let index = shuffled.length - 1; index > 0; index--) {
        const other = Math.floor(random() * (index + 1));
        [shuffled[index], shuffled[other]] = [shuffled[other], shuffled[index]];
    }
    let selected = shuffled[0];
    let best = -Infinity;
    for (const action of shuffled) {
        let arm = node.arms[side].get(action);
        if (!arm) {
            arm = { visits: 0, total: 0 };
            node.arms[side].set(action, arm);
        }
        if (!arm.visits) return action;
        let mean = arm.total / arm.visits;
        if (side === 1) mean = -mean;
        // UCB1 on rewards transformed from [-1,1] to [0,1].
        let explorationBonus: number;
        if (exploration === Math.SQRT2) {
            explorationBonus = Math.sqrt(
                (2 * Math.log(node.visits + 1)) / arm.visits,
            );
        } else {
            explorationBonus =
                exploration * Math.sqrt(Math.log(node.visits + 1) / arm.visits);
        }
        const score = (mean + 1) / 2 + explorationBonus;
        if (score > best) {
            best = score;
            selected = action;
        }
    }
    return selected;
}

function selectPuct(
    node: Node,
    side: 0 | 1,
    state: SearchState,
    random: () => number,
    coefficient: number,
): string {
    const legal = [...new Set(state.actions(side))];
    let nodeValue = 0;
    if (node.visits) nodeValue = node.total / node.visits;
    if (side === 1) nodeValue = -nodeValue;
    const values = legal.map((action) => {
        let arm = node.arms[side].get(action);
        if (!arm) {
            arm = { visits: 0, total: 0 };
            node.arms[side].set(action, arm);
        }
        // Unvisited actions start at the node value: only prior exploration.
        if (!arm.visits) return nodeValue;
        let value = arm.total / arm.visits;
        if (side === 1) value = -value;
        return value;
    });
    let priors: number[];
    if (state.priors) priors = state.priors(side, legal);
    else priors = legal.map(() => 1);
    const probabilities = puctProbabilities(
        values,
        nodeValue,
        node.visits,
        priors,
        coefficient,
    );
    return legal[sampleProbability(probabilities, random)];
}

export function searchMcts(
    sampleRoot: () => SearchState,
    options: SearchOptions,
): SearchResult {
    if (
        !Number.isInteger(options.iterations) ||
        options.iterations < 1 ||
        !Number.isInteger(options.maxDepth) ||
        options.maxDepth < 1
    ) {
        throw new Error(
            "MCTS requires positive integer iteration and depth budgets",
        );
    }
    const exploration = options.exploration ?? Math.SQRT2;
    if (!Number.isFinite(exploration) || exploration < 0)
        throw new Error("MCTS exploration must be finite and nonnegative");
    const selection = options.selection ?? "uct";
    const cpuct = options.cpuct ?? 1;
    if (selection !== "uct" && selection !== "puct")
        throw new Error("Unknown search selection");
    if (!Number.isFinite(cpuct) || cpuct <= 0)
        throw new Error("PUCT coefficient must be positive and finite");
    const root = newNode();
    const now = options.now ?? Date.now;
    const start = now();
    let completed = 0;
    for (let iteration = 0; iteration < options.iterations; iteration++) {
        // Complete at least one iteration, so a deadline always returns an action.
        if (
            completed &&
            options.maxMillis !== undefined &&
            now() - start >= options.maxMillis
        )
            break;
        const state = sampleRoot();
        const path: { node: Node; choices: [string, string] }[] = [];
        let node = root;
        try {
            let value = state.terminalValue();
            if (value !== undefined && iteration === 0)
                throw new Error("MCTS root is terminal");
            for (
                let depth = 0;
                depth < options.maxDepth && value === undefined;
                depth++
            ) {
                let ownAction: string;
                let opponentAction: string;
                if (selection === "puct") {
                    ownAction = selectPuct(
                        node,
                        0,
                        state,
                        options.random,
                        cpuct,
                    );
                    opponentAction = selectPuct(
                        node,
                        1,
                        state,
                        options.random,
                        cpuct,
                    );
                } else {
                    ownAction = select(
                        node,
                        0,
                        state.actions(0),
                        options.random,
                        exploration,
                    );
                    opponentAction = select(
                        node,
                        1,
                        state.actions(1),
                        options.random,
                        exploration,
                    );
                }
                path.push({ node, choices: [ownAction, opponentAction] });
                state.advance(ownAction, opponentAction);
                const key = JSON.stringify([ownAction, opponentAction]);
                let child = node.children.get(key);
                if (!child) {
                    child = newNode();
                    node.children.set(key, child);
                }
                node = child;
                value = state.terminalValue();
            }
            if (value === undefined) value = state.evaluate();
            if (!Number.isFinite(value) || Math.abs(value) > 1)
                throw new Error("MCTS values must be finite and in [-1,1]");
            for (const entry of path) {
                entry.node.visits++;
                entry.node.total += value;
                for (const side of [0, 1] as const) {
                    const arm = entry.node.arms[side].get(entry.choices[side])!;
                    arm.visits++;
                    arm.total += value;
                }
            }
            completed++;
        } finally {
            state.dispose();
        }
    }
    const rootStats = [...root.arms[0]].map(([action, arm]) => {
        let mean = 0;
        if (arm.visits) mean = arm.total / arm.visits;
        return { action, visits: arm.visits, mean };
    });
    rootStats.sort(
        (left, right) => right.visits - left.visits || right.mean - left.mean,
    );
    if (!rootStats.length) throw new Error("MCTS did not visit a root action");
    return {
        action: rootStats[0].action,
        iterations: completed,
        root: rootStats,
    };
}
