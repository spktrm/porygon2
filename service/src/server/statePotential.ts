import { Side } from "@pkmn/client";
import { MAX_RATIO_TOKEN } from "./data";

// Per-generation heuristic positional potential, ported from poke-engine's
// evaluators:
//   gen1 -> poke-engine/src/gen1/evaluate.rs
//   gen2 -> poke-engine/src/gen2/evaluate.rs
//   gen3 -> poke-engine/src/gen3/evaluate.rs
//   gen4..gen9 -> poke-engine/src/genx/evaluate.rs   (the "genx" fallback)
//
// Each side is scored from public info only (so the differential is zero-sum /
// antisymmetric), then normalised into roughly [-1, 1] so it survives the Int16
// fixed-point packing used by the info buffer.

const POKEMON_ALIVE = 30.0;
const POKEMON_HP = 100.0;
const ITEM_BONUS = 10.0;
const USED_TERA = -75.0;

const POKEMON_ATTACK_BOOST = 30.0;
const POKEMON_DEFENSE_BOOST = 15.0;
const POKEMON_SPECIAL_ATTACK_BOOST = 30.0;
const POKEMON_SPECIAL_DEFENSE_BOOST = 15.0;
const POKEMON_SPEED_BOOST = 30.0;

// Boost-stage (-6..+6) -> multiplier table (matches the Rust constants).
const BOOST_MULTIPLIER = [
    -3.3, -3.15, -3.0, -2.5, -2.0, -1.0, 0.0, 1.0, 2.0, 2.5, 3.0, 3.15, 3.3,
];
const boostMult = (boost: number) =>
    BOOST_MULTIPLIER[Math.max(-6, Math.min(6, boost ?? 0)) + 6];

const POKEMON_FROZEN = -40.0;
const POKEMON_ASLEEP = -25.0;
const POKEMON_PARALYZED = -25.0;
const POKEMON_TOXIC = -30.0;
const POKEMON_POISONED = -10.0;
const POKEMON_BURNED = -25.0;

const LEECH_SEED = -30.0;
const SUBSTITUTE = 40.0;
const CONFUSION = -20.0;

const REFLECT = 20.0;
const LIGHT_SCREEN = 20.0;
const AURORA_VEIL = 40.0;
const SAFE_GUARD = 5.0;
const TAILWIND = 7.0;
const HEALING_WISH = 30.0;

const STEALTH_ROCK = -10.0;
const SPIKES = -7.0; // genx per-layer; gen2 per-mon-per-layer
const TOXIC_SPIKES = -7.0;
const STICKY_WEB = -25.0;

// gen3 layered-spikes values.
const GEN3_SPIKES = [0.0, -12.0, -16.0, -25.0];

// Which generations use the "genx" evaluator.
type Gen = 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9;

interface GenRules {
    hasSpecialDefenseBoost: boolean; // special split into spa/spd (gen2+)
    itemBonus: boolean; // items exist (gen2+)
    tera: boolean; // gen9
    ppPenalty: boolean; // gen3 only
    burnAbilities: Set<string>; // abilities that flip burn positive
    poisonAbilities: Set<string>; // abilities that flip poison positive
    poisonHealAbilities: Set<string>; // abilities scoring +15 under poison
    screens: "gen1" | "gen2" | "gen3" | "genx";
    hazards: "none" | "gen2" | "gen3" | "genx";
}

const NO_ABILITIES = new Set<string>();
const GEN3_BURN_ABILITIES = new Set(["guts", "marvelscale"]);
const GENX_BURN_ABILITIES = new Set(["guts", "marvelscale", "quickfeet"]);
const GENX_POISON_ABILITIES = new Set([
    "guts",
    "marvelscale",
    "quickfeet",
    "toxicboost",
    "magicguard",
]);
const GENX_POISON_HEAL = new Set(["poisonheal"]);

function rulesForGen(gen: Gen): GenRules {
    switch (gen) {
        case 1:
            return {
                hasSpecialDefenseBoost: false,
                itemBonus: false,
                tera: false,
                ppPenalty: false,
                burnAbilities: NO_ABILITIES,
                poisonAbilities: NO_ABILITIES,
                poisonHealAbilities: NO_ABILITIES,
                screens: "gen1",
                hazards: "none",
            };
        case 2:
            return {
                hasSpecialDefenseBoost: true,
                itemBonus: true,
                tera: false,
                ppPenalty: false,
                burnAbilities: NO_ABILITIES,
                poisonAbilities: NO_ABILITIES,
                poisonHealAbilities: NO_ABILITIES,
                screens: "gen2",
                hazards: "gen2",
            };
        case 3:
            return {
                hasSpecialDefenseBoost: true,
                itemBonus: true,
                tera: false,
                ppPenalty: true,
                burnAbilities: GEN3_BURN_ABILITIES,
                poisonAbilities: GEN3_BURN_ABILITIES,
                poisonHealAbilities: NO_ABILITIES,
                screens: "gen3",
                hazards: "gen3",
            };
        default: // gen4..gen9 -> genx
            return {
                hasSpecialDefenseBoost: true,
                itemBonus: true,
                tera: gen >= 9,
                ppPenalty: false,
                burnAbilities: GENX_BURN_ABILITIES,
                poisonAbilities: GENX_POISON_ABILITIES,
                poisonHealAbilities: GENX_POISON_HEAL,
                screens: "genx",
                hazards: "genx",
            };
    }
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const lc = (s: any) => (s ?? "").toString().toLowerCase();

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function isGrounded(pkmn: any): boolean {
    const types: string[] = (pkmn.types ?? []).map(lc);
    if (types.includes("flying")) return false;
    if (lc(pkmn.ability) === "levitate") return false;
    if (lc(pkmn.item) === "airballoon") return false;
    const vol = pkmn.volatiles ?? {};
    if ("magnetrise" in vol || "telekinesis" in vol) return false;
    return true;
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function evaluatePoison(pkmn: any, baseScore: number, rules: GenRules): number {
    const ability = lc(pkmn.ability);
    if (rules.poisonHealAbilities.has(ability)) return 15.0;
    if (rules.poisonAbilities.has(ability)) return 10.0;
    return baseScore;
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function evaluateBurned(pkmn: any, rules: GenRules): number {
    if (rules.burnAbilities.has(lc(pkmn.ability))) return -2.0 * POKEMON_BURNED;
    // poke-engine scales burn by the number of physical moves; with public info
    // that move set is often unknown, so we approximate by softening burn for
    // special attackers (per the same "don't punish special attackers" intent).
    const bs = pkmn.species?.baseStats ?? pkmn.baseStats;
    if (bs && bs.spa > bs.atk) return 0.5 * POKEMON_BURNED;
    return POKEMON_BURNED;
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function evaluatePokemon(pkmn: any, rules: GenRules): number {
    let score = 0.0;
    score += POKEMON_HP * (pkmn.maxhp === 0 ? 1 : pkmn.hp / pkmn.maxhp);

    switch (lc(pkmn.status)) {
        case "brn":
            score += evaluateBurned(pkmn, rules);
            break;
        case "frz":
            score += POKEMON_FROZEN;
            break;
        case "slp":
            score += POKEMON_ASLEEP;
            break;
        case "par":
            score += POKEMON_PARALYZED;
            break;
        case "tox":
            score += evaluatePoison(pkmn, POKEMON_TOXIC, rules);
            break;
        case "psn":
            score += evaluatePoison(pkmn, POKEMON_POISONED, rules);
            break;
    }

    if (rules.itemBonus && pkmn.item) score += ITEM_BONUS;

    // gen3: penalise low-PP moves (running out of PP is a real liability).
    if (rules.ppPenalty && Array.isArray(pkmn.moveSlots)) {
        for (const mv of pkmn.moveSlots) {
            const pp = typeof mv?.pp === "number" ? mv.pp : undefined;
            if (pp !== undefined && pp <= 10) score += pp * 3 - 30;
        }
    }

    // Prevent a low-hp mon from scoring negative (which would wrongly reward
    // the opponent for keeping it alive).
    if (score < 0.0) score = 0.0;

    score += POKEMON_ALIVE;
    return score;
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function evaluateHazards(pkmn: any, sc: any, rules: GenRules): number {
    let score = 0.0;
    if (rules.hazards === "gen3") {
        if (isGrounded(pkmn)) {
            const layers = sc.spikes?.level ?? (sc.spikes ? 1 : 0);
            score += GEN3_SPIKES[Math.max(0, Math.min(3, layers))];
        }
        return score;
    }
    // genx
    if (lc(pkmn.item) === "heavydutyboots") return score;
    const grounded = isGrounded(pkmn);
    if (lc(pkmn.ability) !== "magicguard") {
        if (sc.stealthrock) score += STEALTH_ROCK;
        if (grounded) {
            score += (sc.spikes?.level ?? 0) * SPIKES;
            score += (sc.toxicspikes?.level ?? 0) * TOXIC_SPIKES;
        }
    }
    if (grounded && sc.stickyweb) score += STICKY_WEB;
    return score;
}

function evaluateSide(side: Side, rules: GenRules): number {
    let score = 0.0;
    let usedTera = false;
    let aliveCount = 0;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const anySide = side as any;
    const sc = anySide.sideConditions ?? {};
    const active = anySide.active?.[0];

    for (let j = 0; j < side.team.length; j++) {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const pkmn = side.team[j] as any;
        if (rules.tera && pkmn.terastallized) usedTera = true;
        if (pkmn.fainted || pkmn.hp <= 0) continue;
        aliveCount += 1;

        score += evaluatePokemon(pkmn, rules);
        if (rules.hazards === "gen3" || rules.hazards === "genx") {
            score += evaluateHazards(pkmn, sc, rules);
        }

        if (active && pkmn === active) {
            const vol = pkmn.volatiles ?? {};
            if ("leechseed" in vol) score += LEECH_SEED;
            if ("substitute" in vol) score += SUBSTITUTE;
            if ("confusion" in vol) score += CONFUSION;

            // gen1 carried Reflect / Light Screen as active-mon volatiles.
            if (rules.screens === "gen1") {
                if ("reflect" in vol) score += REFLECT;
                if ("lightscreen" in vol) score += LIGHT_SCREEN;
            }

            const b = pkmn.boosts ?? {};
            score += boostMult(b.atk) * POKEMON_ATTACK_BOOST;
            score += boostMult(b.def) * POKEMON_DEFENSE_BOOST;
            score += boostMult(b.spa) * POKEMON_SPECIAL_ATTACK_BOOST;
            if (rules.hasSpecialDefenseBoost) {
                score += boostMult(b.spd) * POKEMON_SPECIAL_DEFENSE_BOOST;
            }
            score += boostMult(b.spe) * POKEMON_SPEED_BOOST;
        }
    }
    if (usedTera) score += USED_TERA;

    // Mons not yet revealed: assume alive at full HP (unknown status / boosts /
    // hazards), matching the previous heuristic's assumption.
    const unknown = side.totalPokemon - side.team.length;
    score += unknown * (POKEMON_ALIVE + POKEMON_HP);
    aliveCount += Math.max(0, unknown);

    // Side-wide screens / conditions.
    if (rules.screens === "gen2") {
        if (sc.reflect) score += REFLECT;
        if (sc.lightscreen) score += LIGHT_SCREEN;
        if (sc.safeguard) score += SAFE_GUARD;
    } else if (rules.screens === "gen3") {
        if (sc.reflect) score += REFLECT;
        if (sc.lightscreen) score += LIGHT_SCREEN;
    } else if (rules.screens === "genx") {
        if (sc.reflect) score += REFLECT;
        if (sc.lightscreen) score += LIGHT_SCREEN;
        if (sc.auroraveil) score += AURORA_VEIL;
        if (sc.safeguard) score += SAFE_GUARD;
        if (sc.tailwind) score += TAILWIND;
        if (sc.healingwish) score += HEALING_WISH;
    }

    // gen2 spikes: single layer, penalty applied once per (potentially)
    // switching mon -> scaled by this side's alive count.
    if (rules.hazards === "gen2") {
        const layers = sc.spikes?.level ?? (sc.spikes ? 1 : 0);
        score += layers * SPIKES * aliveCount;
    }

    return score;
}

const VALID_GENS = new Set([1, 2, 3, 4, 5, 6, 7, 8, 9]);

/**
 * Heuristic positional potential for `mySide` relative to `oppSide`, scored
 * from public information only (so the result is zero-sum / antisymmetric),
 * using the poke-engine evaluator that matches `gen`.
 *
 * Returns an Int16 fixed-point value: `floor(frac * MAX_RATIO_TOKEN)` where
 * `frac` is clamped to [-1, 1].
 */
export function getStatePotential(
    mySide: Side,
    oppSide: Side,
    gen: number,
): number {
    const g = (VALID_GENS.has(gen) ? gen : 9) as Gen;
    const rules = rulesForGen(g);
    const score = evaluateSide(mySide, rules) - evaluateSide(oppSide, rules);
    return Math.floor(
        Math.min(Math.max(score, -MAX_RATIO_TOKEN), MAX_RATIO_TOKEN),
    );
}
