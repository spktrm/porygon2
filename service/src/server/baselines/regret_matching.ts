/** One-turn Bayesian matrix regret matching, not full-game CFR.
 * One own strategy is shared across worlds; the opponent knows its sampled set.
 * Signed cumulative regrets, positive-part normalisation, uniform zero-regret
 * fallback, simultaneous updates and arithmetic average strategies.
 * Reference: Zinkevich et al. 2007, equations 3–4 and regret matching.
 */
export interface MatrixSolution {
    own: number[];
    opponent: number[][];
    value: number;
    gap: number;
    iterations: number;
}

function regretPolicy(regrets: number[]): number[] {
    const positive = regrets.map((regret) => Math.max(0, regret));
    const total = positive.reduce((sum, regret) => sum + regret, 0);
    if (total === 0) return positive.map(() => 1 / positive.length);
    return positive.map((regret) => regret / total);
}

export function solveWorldMatrices(
    matrices: number[][][],
    iterations = 1024,
): MatrixSolution {
    if (
        !Number.isInteger(iterations) ||
        iterations < 1 ||
        !matrices.length ||
        !matrices[0].length
    )
        throw new Error(
            "Regret matching needs worlds, actions and a positive iteration budget",
        );
    const ownCount = matrices[0].length;
    for (const matrix of matrices) {
        if (
            matrix.length !== ownCount ||
            !matrix[0].length ||
            matrix.some(
                (row) =>
                    row.length !== matrix[0].length ||
                    row.some(
                        (value) =>
                            !Number.isFinite(value) || Math.abs(value) > 1,
                    ),
            )
        )
            throw new Error("Invalid world payoff matrix");
    }
    const ownRegrets = Array(ownCount).fill(0) as number[];
    const opponentRegrets = matrices.map((matrix) => matrix[0].map(() => 0));
    const ownSum = ownRegrets.slice();
    const opponentSum = opponentRegrets.map((regrets) => regrets.slice());
    const worldWeight = 1 / matrices.length;
    for (let iteration = 0; iteration < iterations; iteration++) {
        const own = regretPolicy(ownRegrets);
        const opponents = opponentRegrets.map(regretPolicy);
        const ownValues = own.map(() => 0);
        let ownValue = 0;
        for (let world = 0; world < matrices.length; world++) {
            const matrix = matrices[world];
            const opponentValues = opponents[world].map(() => 0);
            for (let row = 0; row < ownCount; row++) {
                for (let column = 0; column < matrix[row].length; column++) {
                    ownValues[row] +=
                        worldWeight *
                        matrix[row][column] *
                        opponents[world][column];
                    opponentValues[column] -= matrix[row][column] * own[row];
                }
            }
            const opponentValue = opponentValues.reduce(
                (sum, value, column) => sum + value * opponents[world][column],
                0,
            );
            for (let column = 0; column < opponentValues.length; column++) {
                opponentRegrets[world][column] +=
                    opponentValues[column] - opponentValue;
                opponentSum[world][column] += opponents[world][column];
            }
        }
        for (let row = 0; row < ownCount; row++)
            ownValue += own[row] * ownValues[row];
        for (let row = 0; row < ownCount; row++) {
            ownRegrets[row] += ownValues[row] - ownValue;
            ownSum[row] += own[row];
        }
    }
    const own = ownSum.map((total) => total / iterations);
    const opponent = opponentSum.map((sums) =>
        sums.map((total) => total / iterations),
    );
    const ownValues = own.map(() => 0);
    let lower = 0;
    for (let world = 0; world < matrices.length; world++) {
        const opposingValues = opponent[world].map(() => 0);
        for (let row = 0; row < ownCount; row++) {
            for (let column = 0; column < opposingValues.length; column++) {
                ownValues[row] +=
                    worldWeight *
                    matrices[world][row][column] *
                    opponent[world][column];
                opposingValues[column] +=
                    own[row] * matrices[world][row][column];
            }
        }
        lower += worldWeight * Math.min(...opposingValues);
    }
    return {
        own,
        opponent,
        value: ownValues.reduce((sum, value, row) => sum + value * own[row], 0),
        gap: Math.max(0, Math.max(...ownValues) - lower),
        iterations,
    };
}
