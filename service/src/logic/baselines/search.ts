import { Battle, Pokemon } from "@pkmn/sim";
import { SideID } from "@pkmn/types";
import { createHash } from "crypto";
import { StreamHandler } from "../handler";
import { StateHandler } from "../state";
import { EvalActionFnType } from "../eval";

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
    const copy = Battle.fromJSON(JSON.stringify(battle));
    copy.restart((type, data) => {
        // console.log(data);
    });
    return copy;
}

function getHealthScore(pokemon: Pokemon): number {
    // Scoring based on HP percentage
    const hpPercentage = pokemon.hp / pokemon.maxhp;

    let score = hpPercentage; // Base score based on HP percentage

    // Adjust the score based on status effects
    switch (pokemon.status) {
        case "par":
        case "brn":
        case "psn":
        case "frz":
        case "slp":
        case "tox":
            score -= 0.3; // Deduct points for being inflicted with any status condition
            break;
    }

    score -= -1 * +pokemon.fainted;

    // Ensure the score is not negative
    return score;
}

class State {
    battle: Battle | null;
    playerIndex?: number;
    legalActions: number[];

    constructor(battle: Battle | null = null, playerIndex?: number) {
        this.battle = battle;
        this.playerIndex = playerIndex;
        this.legalActions = this.getLegalActions(playerIndex);
    }

    hash(): string {
        const stateString = JSON.stringify(this.battle);
        return `${createHash("md5").update(stateString).digest("hex")}`;
    }

    clone(): State {
        const copy = CopyBattle(this.battle!);
        return new State(copy);
    }

    totalFainted(): number[] {
        return (
            this.battle?.sides.map((side) =>
                side.pokemon.reduce((total, member) => {
                    return total + +member.fainted;
                }, 0),
            ) ?? [0, 0]
        );
    }

    getValue(): number {
        if (this.isTerminal()) {
            const [fainted1, fainted2] = this.totalFainted();
            if (fainted1 === this.battle!.sides[0].pokemon.length) return -1;
            if (fainted2 === this.battle!.sides[1].pokemon.length) return 1;
        }

        const [side1Score, side2Score] = this.battle!.sides.map((side) =>
            side.pokemon.reduce((total, member) => {
                return total + getHealthScore(member);
            }, 0),
        );

        const faintedDifference = side1Score - side2Score;
        return faintedDifference;
    }

    isTerminal() {
        return this.battle?.ended ?? false;
    }

    updateCurrentPlayer() {
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

    getLegalActions(playerIndex?: number) {
        if (!playerIndex) playerIndex = this.currentPlayer();

        const legalActions = StateHandler.getLegalActions(
            this.battle!.sides[playerIndex].activeRequest,
        );
        return legalActions
            .toBinaryVector()
            .flatMap((action, index) => (action ? index : []));
    }

    applyAction(moveIndex?: number): boolean {
        if (moveIndex === undefined || this.isTerminal()) {
            return true;
        }
        let moveStr = CHOICES[moveIndex];
        const currentPlayer = this.currentPlayer();
        const valid = this.battle!.choose(PLAYERS[currentPlayer], moveStr);
        this.updateCurrentPlayer();
        return valid;
    }

    currentTurn() {
        return this.battle?.turn;
    }
}

const MAX_DEPTH = 2;

interface NashEquilibrium {
    strategy1: number[]; // Player 1's mixed strategy
    strategy2: number[]; // Player 2's mixed strategy
    value: number; // Expected payoff for player 1 (player 2's payoff is -value)
}

interface PayoffMatrix {
    matrix: number[][]; // The zero-sum game payoff matrix for Player 1
}

// Simplex algorithm implementation
function solveZeroSumGame(payoffMatrix: PayoffMatrix): NashEquilibrium {
    const A = payoffMatrix.matrix;
    const numRows = A.length; // Number of strategies for Player 1
    const numCols = A[0].length; // Number of strategies for Player 2

    // Ensure all payoffs are non-negative by shifting if necessary
    let minPayoff = Infinity;
    for (const row of A) {
        for (const value of row) {
            if (value < minPayoff) minPayoff = value;
        }
    }
    const offset = minPayoff < 0 ? -minPayoff + 1 : 0;
    const adjustedA = A.map((row) => row.map((value) => value + offset));

    // Formulate the LP for Player 1 (Primal)
    // Maximize v
    // Subject to:
    // sum_i x_i * adjustedA[i][j] >= v for all j
    // sum_i x_i = 1
    // x_i >= 0

    // Convert the LP to standard form suitable for the simplex method
    // We'll convert inequalities to equalities by adding slack variables

    // Number of constraints (excluding non-negativity): numCols + 1
    // Number of variables: numRows (x_i) + numCols (slack variables) + 1 (v)

    // Initialize the simplex tableau
    const numConstraints = numCols + 1; // Payoff constraints + sum(x_i) = 1
    const numVariables = numRows + numCols + 1; // x_i + slack variables + v

    // Build the initial tableau
    const tableau: number[][] = [];

    // Objective function row (row 0)
    // We move the objective function to the left-hand side
    // Maximize v => Minimize -v (since simplex algorithm is for minimization)
    // The objective function coefficients are all zeros except for -1 for v
    const objectiveRow = new Array(numVariables + 1).fill(0);
    // Coefficient for v (we'll place v at the last variable)
    objectiveRow[numVariables - 1] = -1; // Minimize -v
    tableau.push(objectiveRow);

    // Constraints:
    // sum_i x_i * adjustedA[i][j] - v + s_j = 0 for all j
    // We'll represent them as:
    // sum_i (-adjustedA[i][j]) * x_i + v + s_j = 0
    // We need to rearrange the inequality to match the tableau format

    for (let j = 0; j < numCols; j++) {
        const row = new Array(numVariables + 1).fill(0);

        // Coefficients for x_i
        for (let i = 0; i < numRows; i++) {
            row[i] = -adjustedA[i][j];
        }

        // Coefficient for slack variable s_j
        row[numRows + j] = 1; // Coefficient for s_j

        // Coefficient for v
        row[numVariables - 1] = 1; // Coefficient for v

        // Right-hand side (b)
        row[numVariables] = 0;

        tableau.push(row);
    }

    // Constraint: sum_i x_i = 1
    const sumRow = new Array(numVariables + 1).fill(0);
    for (let i = 0; i < numRows; i++) {
        sumRow[i] = 1;
    }
    // Right-hand side (b)
    sumRow[numVariables] = 1;

    tableau.push(sumRow);

    // Now, we have the initial tableau set up
    // We need to perform the simplex algorithm to find the optimal solution

    // List of basic variables (initially, the slack variables and the sum constraint)
    const basicVariables: number[] = [];
    for (let j = 0; j < numCols; j++) {
        basicVariables.push(numRows + j); // Indices of s_j
    }
    basicVariables.push(numVariables - 1); // Index of v (we treat v as a basic variable)

    // Perform the simplex algorithm
    while (true) {
        // Identify the entering variable (most negative coefficient in the objective row)
        const objectiveCoefficients = tableau[0].slice(0, numVariables);
        let enteringVariableIndex = -1;
        let mostNegative = 0;
        for (let i = 0; i < numVariables; i++) {
            if (objectiveCoefficients[i] < mostNegative) {
                mostNegative = objectiveCoefficients[i];
                enteringVariableIndex = i;
            }
        }

        // If no negative coefficients, optimal solution is found
        if (enteringVariableIndex === -1) {
            break;
        }

        // Identify the leaving variable using the minimum ratio test
        let minimumRatio = Infinity;
        let leavingRowIndex = -1;
        for (let rowIndex = 1; rowIndex < tableau.length; rowIndex++) {
            const row = tableau[rowIndex];
            const coefficient = row[enteringVariableIndex];
            if (coefficient > 0) {
                const rhs = row[numVariables];
                const ratio = rhs / coefficient;
                if (ratio < minimumRatio) {
                    minimumRatio = ratio;
                    leavingRowIndex = rowIndex;
                }
            }
        }

        // If no valid leaving variable, the solution is unbounded
        if (leavingRowIndex === -1) {
            throw new Error("Linear program is unbounded.");
        }

        // Pivot around the entering variable and leaving variable
        pivot(tableau, leavingRowIndex, enteringVariableIndex);

        // Update the basic variables
        basicVariables[leavingRowIndex - 1] = enteringVariableIndex;
    }

    // Extract the solution
    const solution = new Array(numVariables).fill(0);
    for (let rowIndex = 1; rowIndex < tableau.length; rowIndex++) {
        const basicVarIndex = basicVariables[rowIndex - 1];
        solution[basicVarIndex] = tableau[rowIndex][numVariables];
    }

    // Extract Player 1's strategy x_i
    const strategy1 = solution.slice(0, numRows);
    const sumStrategy1 = strategy1.reduce((sum, val) => sum + val, 0);
    // Normalize in case of floating-point errors
    for (let i = 0; i < strategy1.length; i++) {
        strategy1[i] /= sumStrategy1;
    }

    // Game value v
    const value = tableau[0][numVariables] - offset;

    // To find Player 2's strategy, we solve the dual LP
    // However, since we have the final tableau, we can extract the dual variables
    // The dual variables correspond to the negative of the objective row coefficients of the non-basic variables

    // Dual variables (Player 2's strategy)
    const strategy2 = [];
    for (let j = 0; j < numCols; j++) {
        const dualVariable = tableau[0][numRows + j];
        strategy2.push(dualVariable);
    }
    const sumStrategy2 = strategy2.reduce((sum, val) => sum + val, 0);
    // Normalize
    for (let i = 0; i < strategy2.length; i++) {
        strategy2[i] /= sumStrategy2;
    }

    return { strategy1, strategy2, value };
}

// Pivot operation for the simplex method
function pivot(
    tableau: number[][],
    pivotRowIndex: number,
    pivotColIndex: number,
) {
    const numRows = tableau.length;
    const numCols = tableau[0].length;

    // Pivot element
    const pivotElement = tableau[pivotRowIndex][pivotColIndex];

    // Divide the pivot row by the pivot element
    for (let j = 0; j < numCols; j++) {
        tableau[pivotRowIndex][j] /= pivotElement;
    }

    // Subtract multiples of the pivot row from other rows to make pivot column zero
    for (let i = 0; i < numRows; i++) {
        if (i !== pivotRowIndex) {
            const factor = tableau[i][pivotColIndex];
            for (let j = 0; j < numCols; j++) {
                tableau[i][j] -= factor * tableau[pivotRowIndex][j];
            }
        }
    }
}

async function minimax(
    state: State,
    depth: number = 0,
): Promise<[number, number[], number[]]> {
    if (depth === MAX_DEPTH || state.isTerminal()) {
        return [state.getValue(), [], []];
    }

    const player1Actions = state.getLegalActions(0);
    const player2Actions = state.getLegalActions(1);

    // Handle cases where either player has no valid actions
    if (player1Actions.length === 0 || player2Actions.length === 0) {
        // If no actions are available, evaluate the current state
        return [state.getValue(), [], []];
    }

    // Create payoff matrix
    const payoffMatrix: number[][] = [];
    for (const action1 of player1Actions) {
        const row: number[] = [];
        for (const action2 of player2Actions) {
            const newState = state.clone();
            newState.applyAction(action1);
            newState.applyAction(action2);
            const [value, _, __] = await minimax(newState, depth + 1);
            row.push(value);
        }
        payoffMatrix.push(row);
    }

    // Solve for Nash equilibrium using the API
    try {
        const { value, strategy1, strategy2 } = solveZeroSumGame({
            matrix: payoffMatrix,
        });
        return [value, strategy1, strategy2];
    } catch (error) {
        console.error("Failed to solve Nash equilibrium:", error);
        // Fallback to uniform distribution if API call fails
        const uniformStrategy1 = player1Actions.map(
            () => 1 / player1Actions.length,
        );
        const uniformStrategy2 = player2Actions.map(
            () => 1 / player2Actions.length,
        );
        const uniformValue = payoffMatrix.reduce(
            (sum, row, i) =>
                sum +
                row.reduce(
                    (rowSum, cell, j) =>
                        rowSum +
                        cell * uniformStrategy1[i] * uniformStrategy2[j],
                    0,
                ),
            0,
        );
        return [uniformValue, uniformStrategy1, uniformStrategy2];
    }
}

async function getBestMove(state: State): Promise<[number[], number[]]> {
    const [_, strategy1, strategy2] = await minimax(state);
    return [strategy1, strategy2];
}

export async function GetSearchDistribution(args: {
    handler: StreamHandler;
}): Promise<Float32Array> {
    const { handler } = args;
    const dist = new Float32Array(10);

    if (handler.world === null || handler.world?.ended) {
        return dist;
    }

    const playerIndex = handler.getPlayerIndex() as number;
    const initialState = new State(handler.world, playerIndex);

    const bestMove = (await getBestMove(initialState))[playerIndex];

    for (const [
        legalActionIndex,
        moveIndex,
    ] of initialState.legalActions.entries()) {
        dist[moveIndex] = bestMove[legalActionIndex];
    }

    return dist;
}

export const GetSearchAction: EvalActionFnType = async ({ handler }) => {
    const dist = await GetSearchDistribution({ handler });
    return dist.indexOf(Math.max(...Array.from(dist)));
};
