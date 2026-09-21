export type GameOutcome = "win" | "loss" | "tie";

export interface PairedGame {
    pair: number;
    outcome: GameOutcome;
}

export interface WilsonInterval {
    rate: number;
    lower: number;
    upper: number;
    games: number;
}

export interface PairedBreakdown {
    wonBoth: number;
    split: number;
    lostBoth: number;
    incomplete: number;
}

const NORMAL_QUANTILE_95 = 1.959963984540054;
const GAMES_PER_PAIR = 2;

const OUTCOME_SCORE: Record<GameOutcome, number> = {
    win: 1,
    tie: 0.5,
    loss: 0,
};

export function totalScore(outcomes: GameOutcome[]): number {
    return outcomes.reduce((sum, outcome) => sum + OUTCOME_SCORE[outcome], 0);
}

export function mean(values: number[]): number {
    if (values.length === 0) {
        return NaN;
    }
    return values.reduce((sum, value) => sum + value, 0) / values.length;
}

export function percentile(values: number[], fraction: number): number {
    if (values.length === 0) {
        return NaN;
    }
    const sorted = [...values].sort((left, right) => left - right);
    const rank = Math.ceil(fraction * sorted.length) - 1;
    return sorted[Math.min(sorted.length - 1, Math.max(0, rank))];
}

/**
 * Wilson rather than the normal approximation: it stays inside [0, 1] and
 * keeps its coverage at the small counts a smoke run produces. A tie scores
 * half a win, so `score` may be fractional.
 */
export function wilsonInterval(
    score: number,
    games: number,
    quantile: number = NORMAL_QUANTILE_95,
): WilsonInterval {
    if (games === 0) {
        return { rate: NaN, lower: 0, upper: 1, games };
    }
    const rate = score / games;
    const quantileSquared = quantile * quantile;
    const shrink = 1 + quantileSquared / games;
    const centre = (rate + quantileSquared / (2 * games)) / shrink;
    const halfWidth =
        (quantile *
            Math.sqrt(
                (rate * (1 - rate)) / games +
                    quantileSquared / (4 * games * games),
            )) /
        shrink;
    return {
        rate,
        lower: Math.max(0, centre - halfWidth),
        upper: Math.min(1, centre + halfWidth),
        games,
    };
}

/**
 * Both games of a pair share a battle seed and exchange rosters, so a pair
 * won twice is a win no roster explains. A pair missing a game (invalid and
 * excluded) is counted apart rather than read as a split.
 */
export function pairedBreakdown(games: PairedGame[]): PairedBreakdown {
    const outcomesByPair = new Map<number, GameOutcome[]>();
    for (const game of games) {
        const outcomes = outcomesByPair.get(game.pair) ?? [];
        outcomes.push(game.outcome);
        outcomesByPair.set(game.pair, outcomes);
    }
    const breakdown: PairedBreakdown = {
        wonBoth: 0,
        split: 0,
        lostBoth: 0,
        incomplete: 0,
    };
    for (const outcomes of outcomesByPair.values()) {
        const score = totalScore(outcomes);
        if (outcomes.length !== GAMES_PER_PAIR) {
            breakdown.incomplete += 1;
        } else if (score === GAMES_PER_PAIR) {
            breakdown.wonBoth += 1;
        } else if (score === 0) {
            breakdown.lostBoth += 1;
        } else {
            breakdown.split += 1;
        }
    }
    return breakdown;
}
