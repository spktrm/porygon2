import { expect, test } from "vitest";
import { PRNG } from "@pkmn/sim";
import { solveWorldMatrices } from "./regret_matching";
import { searchMatrix } from "./matrix_search";
import { SearchState } from "./mcts";

test("averaged regret strategies solve an asymmetric mixed equilibrium", () => {
    const solution = solveWorldMatrices(
        [
            [
                [1, -1],
                [-1, 0.5],
            ],
        ],
        10000,
    );
    expect(solution.own[0]).toBeCloseTo(3 / 7, 1);
    expect(solution.opponent[0][0]).toBeCloseTo(3 / 7, 1);
    expect(solution.value).toBeCloseTo(-1 / 7, 2);
    expect(solution.gap).toBeLessThan(0.035);
    expect(solution.own.every((probability) => probability > 0.3)).toBe(true);
});

test("one own strategy spans hidden worlds while opponent responses condition on world", () => {
    const worlds = [
        [[1], [-1]],
        [[-1], [1]],
    ];
    expect(solveWorldMatrices(worlds).value).toBeCloseTo(0);
    // Clairvoyantly solving the worlds separately would incorrectly claim a win.
    for (const world of worlds)
        expect(solveWorldMatrices([world]).value).toBeGreaterThan(0.99);
    const informedOpponent = solveWorldMatrices([
        [
            [1, -1],
            [1, -1],
        ],
        [
            [-1, 1],
            [-1, 1],
        ],
    ]);
    expect(informedOpponent.value).toBeLessThan(-0.99);
    expect(informedOpponent.opponent[0][1]).toBeGreaterThan(0.99);
    expect(informedOpponent.opponent[1][0]).toBeGreaterThan(0.99);
});

test("dominance overrides the initial uniform strategy and malformed matrices fail", () => {
    const solution = solveWorldMatrices([
        [
            [0.5, 0.5],
            [-0.5, -0.5],
        ],
    ]);
    expect(solution.own[0]).toBeGreaterThan(0.99);
    expect(solution.gap).toBeLessThan(0.01);
    expect(() => solveWorldMatrices([[[NaN]]])).toThrow();
    expect(() => solveWorldMatrices([[[1], [1, 0]]])).toThrow();
    expect(() => solveWorldMatrices([], 100)).toThrow();
});

class CachedGame implements SearchState {
    value: number | undefined;
    constructor(private disposed: () => void) {}
    actions(): string[] {
        return ["left", "right"];
    }
    priors(): number[] {
        return [0.5, 0.5];
    }
    advance(own: string, opponent: string): void {
        if (own === opponent) this.value = 1;
        else this.value = -1;
    }
    terminalValue(): number | undefined {
        return this.value;
    }
    evaluate(): number {
        throw new Error("Terminal payoffs must bypass the potential");
    }
    dispose(): void {
        this.disposed();
    }
}

test("cached matrix search samples a mixed strategy, counts calls and disposes every clone", () => {
    let disposals = 0;
    const run = () => {
        const random = new PRNG("1,2,3,4");
        return searchMatrix(
            [
                () => new CachedGame(() => disposals++),
                () => new CachedGame(() => disposals++),
            ],
            { iterations: 42, maxDepth: 1, random: () => random.random() },
        );
    };
    const result = run();
    expect(disposals).toBe(42);
    expect(result.iterations).toBe(42);
    expect(result.matrix?.evaluations).toBe(40);
    expect(result.matrix?.cells).toBe(8);
    expect(result.matrix?.gap).toBeCloseTo(0);
    expect(result.root.map((arm) => arm.probability)).toEqual([0.5, 0.5]);
    expect(result).toEqual(run());
    expect(disposals).toBe(84);
});

test("deadline before full matrix coverage fails instead of inventing missing payoffs", () => {
    let disposals = 0;
    expect(() =>
        searchMatrix([() => new CachedGame(() => disposals++)], {
            iterations: 20,
            maxDepth: 1,
            maxMillis: 1,
            now: () => 0,
            random: () => 0.5,
        }),
    ).toThrow("matrix_budget_incomplete");
    expect(disposals).toBe(1);
    expect(() =>
        searchMatrix([() => new CachedGame(() => disposals++)], {
            iterations: 20,
            maxDepth: 2,
            random: () => 0.5,
        }),
    ).toThrow("depth one");
});

test("deployment draw is independent of simulation sampling", () => {
    const results = [0.1, 0.9].map((draw) => {
        const random = new PRNG("5,6,7,8");
        return searchMatrix([() => new CachedGame(() => {})], {
            iterations: 21,
            maxDepth: 1,
            random: () => random.random(),
            actionRandom: () => draw,
        });
    });
    expect(results[0].root).toEqual(results[1].root);
    expect(results[0].matrix).toEqual(results[1].matrix);
    expect(results[0].action).not.toEqual(results[1].action);
});
