/** Search-only leaf evaluators. The original PBRS fit is deliberately unchanged.
 * Every input belongs to a sampled hypothetical battle, never the live opponent.
 */
import { Battle, Pokemon } from "@pkmn/sim";
import { evaluatePosition } from "./position_potential";

export const SEARCH_POTENTIAL_NAMES = [
    "original",
    "material",
    "strategic",
    "tactical",
    "roster",
] as const;
export type SearchPotentialName = (typeof SEARCH_POTENTIAL_NAMES)[number];
export type SearchPotential = (battle: Battle) => number;

function original(battle: Battle): number {
    const teams = battle.sides.map((side) =>
        side.pokemon.map((mon) => {
            const offensiveTypes: string[] = [...mon.baseSpecies.types];
            let defensiveTypes: string[] = mon.getTypes();
            if (mon.terastallized && mon.terastallized !== "Stellar") {
                if (!offensiveTypes.includes(mon.terastallized))
                    offensiveTypes.push(mon.terastallized);
                defensiveTypes = [mon.terastallized];
            }
            return {
                hp: mon.hp / mon.maxhp,
                active: mon.isActive,
                offensiveTypes,
                defensiveTypes,
            };
        }),
    );
    return evaluatePosition(teams[0], teams[1]);
}

/** Deterministic damage proxy; no simulator events, RNG draws or state writes. */
export function damagePressure(
    attacker: Pokemon,
    defender: Pokemon,
    moveId?: string,
): number {
    let best = 0;
    for (const slot of attacker.moveSlots) {
        if (slot.pp <= 0 || (moveId !== undefined && slot.id !== moveId))
            continue;
        const move = attacker.battle.dex.moves.get(slot.id);
        if (move.category === "Status") continue;
        let damage = 0;
        if (typeof move.damage === "number") damage = move.damage;
        else if (move.damage === "level") damage = attacker.level;
        else {
            let attackStat = "atk" as "atk" | "spa";
            let defenceStat = "def" as "def" | "spd";
            if (move.category === "Special") {
                attackStat = "spa";
                defenceStat = "spd";
            }
            // getStat dispatches simulator events; stored stats + boosts keep this pure.
            const boosted = (
                mon: Pokemon,
                stat: "atk" | "spa" | "def" | "spd",
            ) => {
                const stage = mon.boosts[stat];
                let factor = 2 / (2 - stage);
                if (stage >= 0) factor = (2 + stage) / 2;
                return mon.storedStats[stat] * factor;
            };
            damage =
                ((((2 * attacker.level) / 5 + 2) *
                    move.basePower *
                    boosted(attacker, attackStat)) /
                    Math.max(1, boosted(defender, defenceStat)) /
                    50 +
                    2) *
                0.925;
            if (attacker.baseSpecies.types.includes(move.type)) damage *= 1.5;
            if (attacker.terastallized === move.type) damage *= 1.5;
            if (
                attacker.status === "brn" &&
                move.category === "Physical" &&
                attacker.ability !== "guts"
            )
                damage *= 0.5;
            if (attacker.item === "lifeorb") damage *= 1.3;
            if (attacker.item === "choiceband" && move.category === "Physical")
                damage *= 1.5;
            if (attacker.item === "choicespecs" && move.category === "Special")
                damage *= 1.5;
        }
        for (const defenceType of defender.getTypes()) {
            if (!attacker.battle.dex.getImmunity(move.type, defenceType))
                damage = 0;
            damage *=
                2 **
                attacker.battle.dex.getEffectiveness(move.type, defenceType);
        }
        if (move.type === "Ground" && defender.ability === "levitate")
            damage = 0;
        if (
            move.type === "Water" &&
            ["waterabsorb", "stormdrain", "dryskin"].includes(defender.ability)
        )
            damage = 0;
        if (
            move.type === "Electric" &&
            ["voltabsorb", "lightningrod", "motordrive"].includes(
                defender.ability,
            )
        )
            damage = 0;
        if (move.type === "Fire" && defender.ability === "flashfire")
            damage = 0;
        if (typeof move.accuracy === "number") damage *= move.accuracy / 100;
        best = Math.max(best, Math.min(1, damage / Math.max(1, defender.hp)));
    }
    return best;
}

function resources(
    battle: Battle,
    sideIndex: number,
    strategic: boolean,
): number {
    const side = battle.sides[sideIndex];
    let score = 0;
    for (const mon of side.pokemon) {
        if (mon.hp <= 0) continue;
        const fraction = mon.hp / mon.maxhp;
        score += 1 + 0.6 * Math.sqrt(fraction);
        if (!strategic) continue;
        if (mon.status === "brn") score -= 0.12;
        if (mon.status === "par") score -= 0.18;
        if (mon.status === "slp" || mon.status === "frz") score -= 0.25;
        if (mon.status === "psn" || mon.status === "tox") score -= 0.12;
        if (mon.isActive) {
            const stages = mon.boosts;
            score +=
                0.1 * (stages.atk + stages.spa) +
                0.06 * (stages.def + stages.spd) +
                0.08 * stages.spe;
        }
        if (mon.item === "heavydutyboots" || mon.ability === "magicguard")
            continue;
        if (side.sideConditions.stealthrock) score -= 0.08;
        if (
            side.sideConditions.spikes &&
            !mon.getTypes().includes("Flying") &&
            mon.ability !== "levitate"
        )
            score -= 0.04 * side.sideConditions.spikes.layers;
    }
    return score;
}

function tactical(battle: Battle, sideIndex: number): number {
    const own = battle.sides[sideIndex].active.filter(
        (mon) => mon && mon.hp > 0,
    );
    const opposing = battle.sides[1 - sideIndex].active.filter(
        (mon) => mon && mon.hp > 0,
    );
    let score = 0;
    for (const attacker of own) {
        for (const defender of opposing) {
            const pressure = damagePressure(attacker, defender);
            let tempo = 1;
            if (attacker.speed > defender.speed) tempo = 1.4;
            score += pressure * tempo;
        }
    }
    return score;
}

function coverage(battle: Battle, sideIndex: number): number {
    const own = battle.sides[sideIndex].pokemon.filter((mon) => mon.hp > 0);
    const opposing = battle.sides[1 - sideIndex].pokemon.filter(
        (mon) => mon.hp > 0,
    );
    let score = 0;
    for (const defender of opposing) {
        let best = 0;
        for (const attacker of own)
            best = Math.max(best, damagePressure(attacker, defender));
        score += best;
    }
    return score / Math.max(1, opposing.length);
}

export function getSearchPotential(name: SearchPotentialName): SearchPotential {
    if (name === "original") return original;
    if (!SEARCH_POTENTIAL_NAMES.includes(name))
        throw new Error(`Unknown search potential: ${name}`);
    return (battle) => {
        const strategic = name !== "material";
        let difference =
            resources(battle, 0, strategic) - resources(battle, 1, strategic);
        if (name === "tactical" || name === "roster")
            difference += 0.65 * (tactical(battle, 0) - tactical(battle, 1));
        if (name === "roster")
            difference += 0.4 * (coverage(battle, 0) - coverage(battle, 1));
        return Math.tanh(difference / 3);
    };
}
