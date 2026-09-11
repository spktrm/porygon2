/** Cache sampled one-transition payoffs, then solve the restricted Bayesian game.
 * No missing cell is imputed as zero; an incomplete first sweep explicitly fails.
 */
import { SearchOptions, SearchResult, SearchState } from "./mcts";
import { sampleProbability } from "./puct";
import { solveWorldMatrices } from "./regret_matching";

export const MATRIX_OPTIONS = Object.freeze({
    worlds: 3,
    actions: 3,
    solverIterations: 1024,
    solveReserveMillis: 5,
});

function topActions(
    actions: string[],
    priors: number[],
    limit: number,
): string[] {
    if (
        !actions.length ||
        priors.length !== actions.length ||
        priors.some((prior) => !Number.isFinite(prior) || prior < 0)
    )
        throw new Error("Invalid matrix action priors");
    return actions
        .map((action, index) => ({ action, prior: priors[index] }))
        .sort((left, right) => right.prior - left.prior)
        .slice(0, limit)
        .map((entry) => entry.action);
}

export function searchMatrix(
    samplers: Array<() => SearchState>,
    options: SearchOptions & { actionRandom?: () => number },
): SearchResult {
    if (
        options.maxDepth !== 1 ||
        !Number.isInteger(options.iterations) ||
        options.iterations <= samplers.length ||
        !samplers.length
    )
        throw new Error(
            "Matrix search requires depth one and a budget larger than its world count",
        );
    const now = options.now ?? Date.now;
    const started = now();
    let calls = 0;
    const replies: string[][] = [];
    let rootActions: string[] | undefined;
    let ownPriors: number[] = [];
    for (const sample of samplers) {
        const state = sample();
        calls++;
        try {
            if (state.terminalValue() !== undefined)
                throw new Error("Matrix root is terminal");
            const actions = [...new Set(state.actions(0))];
            if (rootActions === undefined) {
                rootActions = actions;
                ownPriors = actions.map(() => 0);
            }
            if (JSON.stringify(actions) !== JSON.stringify(rootActions))
                throw new Error("Own root actions differ between worlds");
            let own: number[];
            let opponent: number[];
            const opposing = [...new Set(state.actions(1))];
            if (state.priors) {
                own = state.priors(0, actions);
                opponent = state.priors(1, opposing);
            } else {
                own = actions.map(() => 1 / actions.length);
                opponent = opposing.map(() => 1 / opposing.length);
            }
            for (let index = 0; index < actions.length; index++)
                ownPriors[index] += own[index] / samplers.length;
            replies.push(
                topActions(opposing, opponent, MATRIX_OPTIONS.actions),
            );
        } finally {
            state.dispose();
        }
    }
    const own = topActions(rootActions!, ownPriors, MATRIX_OPTIONS.actions);
    const totals = replies.map((actions) =>
        own.map(() => actions.map(() => 0)),
    );
    const counts = replies.map((actions) =>
        own.map(() => actions.map(() => 0)),
    );
    const cells: { world: number; row: number; column: number }[] = [];
    for (let world = 0; world < replies.length; world++) {
        for (let row = 0; row < own.length; row++) {
            for (let column = 0; column < replies[world].length; column++)
                cells.push({ world, row, column });
        }
    }
    // Random ordering avoids always undersampling the same cells at the deadline.
    for (let index = cells.length - 1; index > 0; index--) {
        const other = Math.floor(options.random() * (index + 1));
        [cells[index], cells[other]] = [cells[other], cells[index]];
    }
    let evaluations = 0;
    while (calls < options.iterations) {
        if (
            options.maxMillis !== undefined &&
            now() - started >=
                Math.max(
                    0,
                    options.maxMillis - MATRIX_OPTIONS.solveReserveMillis,
                )
        )
            break;
        const { world, row, column } = cells[evaluations % cells.length];
        const state = samplers[world]();
        calls++;
        try {
            state.advance(own[row], replies[world][column]);
            let value = state.terminalValue();
            if (value === undefined) value = state.evaluate();
            if (!Number.isFinite(value) || Math.abs(value) > 1)
                throw new Error("Invalid matrix payoff");
            totals[world][row][column] += value;
            counts[world][row][column]++;
            evaluations++;
        } finally {
            state.dispose();
        }
    }
    if (evaluations < cells.length) throw new Error("matrix_budget_incomplete");
    const matrices = totals.map((matrix, world) =>
        matrix.map((row, rowIndex) =>
            row.map((total, column) => total / counts[world][rowIndex][column]),
        ),
    );
    const solution = solveWorldMatrices(
        matrices,
        MATRIX_OPTIONS.solverIterations,
    );
    const selected = sampleProbability(
        solution.own,
        options.actionRandom ?? options.random,
    );
    return {
        action: own[selected],
        iterations: calls,
        root: own.map((action, row) => {
            let visits = 0;
            let mean = 0;
            for (let world = 0; world < matrices.length; world++) {
                for (
                    let column = 0;
                    column < matrices[world][row].length;
                    column++
                ) {
                    visits += counts[world][row][column];
                    mean +=
                        (matrices[world][row][column] *
                            solution.opponent[world][column]) /
                        matrices.length;
                }
            }
            return { action, visits, mean, probability: solution.own[row] };
        }),
        matrix: {
            worlds: samplers.length,
            cells: cells.length,
            evaluations,
            solverIterations: solution.iterations,
            gap: solution.gap,
            ownActions: own.length,
            legalOwnActions: rootActions!.length,
        },
    };
}
