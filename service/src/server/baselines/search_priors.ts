/** Cheap heuristic priors for sampled worlds; these are not learned policies. */
import { Battle, Pokemon } from "@pkmn/sim";
import { damagePressure } from "./search_potentials";

export type SearchPriorName = "uniform" | "tactical";

export function searchPriors(
    battle: Battle,
    sideIndex: 0 | 1,
    actions: string[],
    name: SearchPriorName,
): number[] {
    if (name === "uniform") return actions.map(() => 1 / actions.length);
    if (name !== "tactical") throw new Error("Unknown search prior");
    const side = battle.sides[sideIndex];
    const active = side.active[0];
    const defender = battle.sides[1 - sideIndex].active[0];
    if (!active || !defender) return actions.map(() => 1 / actions.length);
    const matchup = (candidate: Pokemon) =>
        damagePressure(candidate, defender) -
        damagePressure(defender, candidate);
    const currentMatchup = matchup(active);
    const scores = actions.map((action) => {
        const [kind, position] = action.split(" ");
        if (kind === "switch") {
            const candidate = side.pokemon[Number(position) - 1];
            if (!candidate) return -4;
            return 2 * (matchup(candidate) - currentMatchup) - 1.2;
        }
        if (kind !== "move") return 0;
        const requested = side.activeRequest;
        if (!requested || !("active" in requested)) return 0;
        const moveRequest = requested.active[0].moves[Number(position) - 1];
        if (!moveRequest) return 0;
        const move = battle.dex.moves.get(moveRequest.id);
        if (move.category !== "Status")
            return 3 * damagePressure(active, defender, move.id);
        if (move.heal)
            return 2 * Math.max(0, 1 - active.hp / active.maxhp) - 0.3;
        if (move.boosts && damagePressure(defender, active) < 0.5) return 0.5;
        if (
            move.sideCondition &&
            !defender.side.sideConditions[move.sideCondition]
        )
            return 0.3;
        return -0.5;
    });
    const maximum = Math.max(...scores);
    const weights = scores.map((score) => Math.exp(score - maximum));
    const total = weights.reduce((sum, weight) => sum + weight, 0);
    // Every legal action retains positive prior mass, including unusual status moves.
    return weights.map(
        (weight) => (0.95 * weight) / total + 0.05 / actions.length,
    );
}
