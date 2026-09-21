import { expect, test } from "vitest";
import { pairedBreakdown, totalScore, wilsonInterval } from "./stats";

test("wilson interval matches the hand-computed 7 of 10", () => {
    const interval = wilsonInterval(7, 10);
    expect(interval.rate).toBeCloseTo(0.7, 12);
    expect(interval.lower).toBeCloseTo(0.39678, 4);
    expect(interval.upper).toBeCloseTo(0.8922, 4);

    // Positive control: the normal approximation's bounds for the same
    // counts are what a wrong implementation would return, and they differ
    // at the precision asserted above.
    const normalHalfWidth = 1.959963984540054 * Math.sqrt((0.7 * 0.3) / 10);
    expect(Math.abs(0.7 - normalHalfWidth - interval.lower)).toBeGreaterThan(
        1e-3,
    );
    expect(Math.abs(0.7 + normalHalfWidth - interval.upper)).toBeGreaterThan(
        1e-3,
    );
});

test("wilson interval stays inside the unit interval at the extremes", () => {
    expect(wilsonInterval(0, 5).lower).toBe(0);
    expect(wilsonInterval(5, 5).upper).toBe(1);
    expect(wilsonInterval(5, 5).lower).toBeLessThan(1);
    expect(wilsonInterval(0, 0)).toEqual({
        rate: NaN,
        lower: 0,
        upper: 1,
        games: 0,
    });
});

test("a tie scores half a win", () => {
    expect(totalScore(["win", "tie", "loss", "tie"])).toBe(2);
});

test("paired breakdown tallies complete pairs and sets incomplete ones apart", () => {
    const breakdown = pairedBreakdown([
        { pair: 0, outcome: "win" },
        { pair: 0, outcome: "win" },
        { pair: 1, outcome: "win" },
        { pair: 1, outcome: "loss" },
        { pair: 2, outcome: "loss" },
        { pair: 2, outcome: "loss" },
        { pair: 3, outcome: "tie" },
        { pair: 3, outcome: "win" },
        { pair: 4, outcome: "win" },
    ]);
    expect(breakdown).toEqual({
        wonBoth: 1,
        split: 2,
        lostBoth: 1,
        incomplete: 1,
    });
});
