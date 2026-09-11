/** Jaxcalibur-style positive-advantage/prior sampling for simultaneous actions.
 * Reference: https://jaxcalibur.github.io/#search (public description, no code).
 * Values here are [-1,1]; cpuct is calibrated in those units. This is not the
 * usual deterministic Q + c P sqrt(N)/(1+n) argmax rule.
 */
export function puctProbabilities(
    actionValues: readonly number[],
    nodeValue: number,
    visits: number,
    priors: readonly number[],
    coefficient: number,
): number[] {
    if (!actionValues.length || actionValues.length !== priors.length)
        throw new Error("PUCT requires aligned nonempty values and priors");
    if (
        !Number.isFinite(coefficient) ||
        coefficient <= 0 ||
        !Number.isFinite(visits) ||
        visits < 0
    )
        throw new Error(
            "PUCT requires positive finite coefficient and nonnegative visits",
        );
    if (
        !Number.isFinite(nodeValue) ||
        actionValues.some((value) => !Number.isFinite(value)) ||
        priors.some((prior) => !Number.isFinite(prior) || prior < 0)
    )
        throw new Error(
            "PUCT requires finite values and nonnegative finite priors",
        );
    const priorTotal = priors.reduce((total, prior) => total + prior, 0);
    if (!(priorTotal > 0)) throw new Error("PUCT prior has no mass");
    const priorScale = coefficient / Math.sqrt(Math.max(1, visits));
    const weights = actionValues.map(
        (value, index) =>
            Math.max(value - nodeValue, 0) +
            (priorScale * priors[index]) / priorTotal,
    );
    const total = weights.reduce((sum, weight) => sum + weight, 0);
    return weights.map((weight) => weight / total);
}

export function sampleProbability(
    probabilities: readonly number[],
    random: () => number,
): number {
    const draw = random();
    let cumulative = 0;
    for (let index = 0; index < probabilities.length; index++) {
        cumulative += probabilities[index];
        if (draw < cumulative) return index;
    }
    return probabilities.length - 1;
}
