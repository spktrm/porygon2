import { describe, expect, test } from "vitest";
import { PRNG } from "@pkmn/sim";
import { puctProbabilities } from "./puct";
import { searchMcts, SearchState } from "./mcts";

class SimultaneousGame implements SearchState {
    value: number | undefined;
    selections: number[] = [];
    constructor(
        private matrix: number[][],
        private prior: number[] = [0.5, 0.5],
    ) {}
    actions(): string[] {
        return ["left", "right"];
    }
    priors(): number[] {
        return this.prior;
    }
    advance(own: string, opponent: string): void {
        const ownIndex = this.actions().indexOf(own);
        const opponentIndex = this.actions().indexOf(opponent);
        this.value = this.matrix[ownIndex][opponentIndex];
    }
    terminalValue(): number | undefined {
        return this.value;
    }
    evaluate(): number {
        throw new Error("Expected terminal outcome");
    }
    dispose(): void {}
}

function run(matrix: number[][], iterations: number, prior = [0.5, 0.5]) {
    const generator = new PRNG("71,81,91,101");
    return searchMcts(() => new SimultaneousGame(matrix, prior), {
        iterations,
        maxDepth: 1,
        random: () => generator.random(),
        selection: "puct",
        cpuct: 1,
    });
}

describe("Jaxcalibur-style sampled PUCT", () => {
    test("initial selection follows the prior without forced one-visit coverage", () => {
        expect(puctProbabilities([0, 0], 0, 0, [0.8, 0.2], 1)).toEqual([
            0.8, 0.2,
        ]);
        const result = run(
            [
                [1, 1],
                [-1, -1],
            ],
            1,
            [1, 0],
        );
        expect(result.action).toBe("left");
        expect(result.root.find((arm) => arm.action === "right")!.visits).toBe(
            0,
        );
    });
    test("positive advantage increasingly outweighs the prior as visits grow", () => {
        const early = puctProbabilities([0.5, -0.5], 0, 1, [0.2, 0.8], 1);
        const late = puctProbabilities([0.5, -0.5], 0, 10000, [0.2, 0.8], 1);
        expect(early[0]).toBeCloseTo(0.7 / 1.5);
        expect(late[0]).toBeGreaterThan(0.98);
        expect(late[1]).toBeGreaterThan(0);
        expect(puctProbabilities([0, -0.5], 0.5, 100, [0.2, 0.8], 1)).toEqual([
            0.2, 0.8,
        ]);
    });
    test("bad prior can be overridden by terminal wins", () => {
        const result = run(
            [
                [1, 1],
                [-1, -1],
            ],
            4000,
            [0.05, 0.95],
        );
        expect(result.action).toBe("left");
        expect(
            result.root.find((arm) => arm.action === "left")!.visits,
        ).toBeGreaterThan(3000);
    });
    test("opponent sign is adversarial, with mixed play in matching pennies", () => {
        const result = run(
            [
                [1, -1],
                [-1, 1],
            ],
            12000,
        );
        for (const arm of result.root) {
            expect(arm.visits / result.iterations).toBeGreaterThan(0.35);
            expect(arm.visits / result.iterations).toBeLessThan(0.65);
            expect(Math.abs(arm.mean)).toBeLessThan(0.15);
        }
    });
    test("fixed-budget runs reproduce and malformed probabilities fail visibly", () => {
        expect(
            run(
                [
                    [1, -1],
                    [-1, 1],
                ],
                200,
            ),
        ).toEqual(
            run(
                [
                    [1, -1],
                    [-1, 1],
                ],
                200,
            ),
        );
        expect(() => puctProbabilities([0, 0], 0, 1, [0, 0], 1)).toThrow();
        expect(() => puctProbabilities([0, 0], 0, 1, [1, -1], 1)).toThrow();
        expect(() => puctProbabilities([0, 0], 0, 1, [0.5, 0.5], 0)).toThrow();
    });
});
