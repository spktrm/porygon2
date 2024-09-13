import { EvalActionFnType } from "../eval";
import { Battle } from "@pkmn/sim";
import { SideID } from "@pkmn/types";
import { createHash } from "crypto";
import { StreamHandler } from "../handler";
import { StateHandler } from "../state";

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

        const [totalFainted1, totalFainted2] = this.battle!.sides.map((side) =>
            side.pokemon.reduce((total, member) => {
                return total + member.hp / member.maxhp;
            }, 0),
        );

        const faintedDifference = totalFainted1 - totalFainted2;
        const maxPossibleHpDifference = this.battle!.sides[0].pokemon.length;

        // Normalize the HP difference to a range between -1 and 1
        return faintedDifference / maxPossibleHpDifference;
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

        const legalActionMask = StateHandler.getLegalActions(
            this.battle!.sides[playerIndex].activeRequest,
        );
        const legalActions = Object.entries(legalActionMask.toObject())
            .map(([key, value], index) => {
                return { index, value };
            })
            .filter(({ value }) => value)
            .map(({ index }) => index);

        return legalActions;
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
const MAX_DEPTH = 1;

interface NashResult {
    value: number;
    strategy1: number[];
    strategy2: number[];
}
interface PayoffMatrix {
    matrix: number[][];
}

interface NashEquilibrium {
    strategy1: number[];
    strategy2: number[];
    value: number;
}

async function solveNashEquilibrium(
    payoffMatrix: PayoffMatrix,
): Promise<NashEquilibrium> {
    const url = "http://localhost:8000/solve_nash_equilibrium";

    try {
        const response = await fetch(url, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify(payoffMatrix),
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        return data as NashEquilibrium;
    } catch (error) {
        console.error("Error calling Nash equilibrium solver API:", error);
        throw error;
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
        const { value, strategy1, strategy2 } = await solveNashEquilibrium({
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
