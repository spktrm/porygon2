/* kaizo_plus.ts
 * Full TypeScript port of the KaizoPlus baseline.
 * ----------------------------------------------------------------- */

import { Pokemon, Battle, Side } from "@pkmn/client";
import { AnyObject } from "@pkmn/sim";
import { EvalActionFnType } from "../eval";
import { GetMoveDamage } from "./max_dmg";
import { Protocol } from "@pkmn/protocol";
import { Action } from "../../../protos/service_pb";
import { ActionType } from "../../../protos/features_pb";

/* ----------------------------------------------------------------- */
/* -------------------------- constants -----------------------------*/
/* ----------------------------------------------------------------- */

/* ——— helper RNG (Python's random.random) ——— */
const rng = (): number => Math.random();

/* ——— switch-scoring weights (copied verbatim) ——— */
const SWITCH_SCORE_KWARGS = {
    check_w: 9.0,
    def_type_disadvantage_w: 1.0,
    off_type_advantage_w: 1.0,
    speed_w: 0.001,
} as const;

/* ----- literal move-id buckets (exactly as Python) ----- */
const ALWAYS_HIT_MOVES = new Set([
    "aerialace",
    "aurasphere",
    "clearsmog",
    "falsesurrender",
    "feintattack",
    "faintattack",
    "magicalleaf",
    "magnetbomb",
    "shadowpunch",
    "shockwave",
    "swift",
    "vitalthrow",
    "trumpcard",
]);

const SPEED_LOWER_MOVES = new Set([
    "bubble",
    "bubblebeam",
    "bulldoze",
    "cottonspore",
    "drumbeating",
    "electroweb",
    "glaciate",
    "lowsweep",
    "pounce",
    "icywind",
    "mudshot",
    "rocktomb",
    "stringshot",
    "scaryface",
    "syrupbomb",
    "tarshot",
    "toxicthread",
]);

const HP_RECOVERY_MOVES = new Set([
    "milkdrink",
    "softboiled",
    "moonlight",
    "morningsun",
    "recover",
    "slackoff",
    "swallow",
    "synthesis",
    "roost",
    "healorder",
    "shoreup",
    "lunarblessing",
    "wish",
]);

const CONFUSING_MOVES = new Set([
    "confuseray",
    "supersonic",
    "dynamicpunch",
    "shadowpanic",
    "teeterdance",
    "sweetkiss",
    "flatter",
    "swagger",
    "hurricane",
]);

const PAR_MOVES = new Set(["glare", "stunspore", "thunderwave"]);

const SLEEP_MOVES = new Set([
    "grasswhistle",
    "hypnosis",
    "lovelykiss",
    "sing",
    "sleeppowder",
    "yawn",
]);

const RECHARGE_MOVES = new Set([
    "blastburn",
    "frenzyplant",
    "hydrocannon",
    "hyperbeam",
]);

const TRAP_MOVES = new Set([
    "sandtomb",
    "whirlpool",
    "wrap",
    "meanlook",
    "spiderweb",
    "anchorshot",
    "spiritshackle",
    "jawlock",
    "shadowhold",
    "thousandwaves",
    "block",
    "fairylock",
]);

/* Hazards recognised in heuristic */
const HAZARDS = ["stealthrock", "spikes", "toxicspikes", "stickyweb"] as const;

/* ----------------------------------------------------------------- */
/* -------------- helper utilities (damage, typing) -----------------*/
/* ----------------------------------------------------------------- */

/** Effectiveness multiplier (0, 0.5, 1, 2, 4, …) */
function typeAdvantageMove(moveType: string, def: Pokemon, battle: Battle) {
    return (
        def.types
            .map((t) => battle.gen.dex.getEffectiveness(moveType, t))
            .reduce((a, b) => a + b, 0) / def.types.length
    );
}

/** Effectiveness from attacker’s entire typing vs defender */
function typeAdvantageMon(att: Pokemon, def: Pokemon, battle: Battle) {
    let total = 0;
    for (const a of att.types)
        for (const d of def.types)
            total += battle.gen.dex.getEffectiveness(a, d);
    return total / (att.types.length * def.types.length);
}

/** Naïve outspeed check (ignores boosts & items for perf) */
const outspeeds = (a: Pokemon, b: Pokemon): boolean =>
    a.baseSpecies.baseStats.spe >= b.baseSpecies.baseStats.spe;

/* ----------------------------------------------------------------- */
/* ----------- score individual BOOST moves (Python port) -----------*/
/* ----------------------------------------------------------------- */
function boostMoveScores(
    battle: Battle,
    playerIndex: number,
    speed_w = 0.5,
): Map<Protocol.Request.ActivePokemon["moves"][number], number> {
    const side = battle.sides[playerIndex];
    if (side === undefined) {
        throw new Error("No side found in boostMoveScores");
    }
    const user = (side.active ?? [])[0] as Pokemon;
    if (user === null || user === undefined) {
        throw new Error("No active user in boostMoveScores");
    }
    const scores = new Map<
        Protocol.Request.ActivePokemon["moves"][number],
        number
    >();

    const request = battle.request as AnyObject;
    for (const mEntry of request?.active?.[0]?.moves ?? []) {
        const move = battle.gen.dex.moves.get(mEntry.id);
        if (!move.boosts) continue;

        const sumBoost = Object.values(move.boosts).reduce((a, b) => a + b, 0);
        if (sumBoost < 2) continue;

        let score = sumBoost;
        /* favour speed boosts slightly */
        if ("spe" in move.boosts && move.boosts.spe! > 0)
            score += speed_w * move.boosts.spe!;

        /* diminish if already boosted */
        for (const value of Object.values(move.boosts)) {
            if (value >= 3) score -= 1;
        }

        scores.set(move, score);
    }

    return scores;
}

/* ----------------------------------------------------------------- */
/* ------------------- damage helpers (expected / worst) ------------*/
/* ----------------------------------------------------------------- */
function expectedDamage(
    user: Pokemon,
    move: Protocol.Request.ActivePokemon["moves"][number],
    opp: Pokemon,
    battle: Battle,
): number {
    return GetMoveDamage({
        battle,
        attacker: user,
        defender: opp,
        moveId: move.id,
    });
}

/* Worst-case = max roll (same helper is fine) */
const worstCaseDamage = expectedDamage;

/* ----------------------------------------------------------------- */
/* ----------------------- SWITCH-SCORING ---------------------------*/
/* ----------------------------------------------------------------- */
function switchScores(
    switches: Pokemon[],
    battle: Battle,
    playerIndex: number,
): Map<Pokemon, number> {
    const { check_w, def_type_disadvantage_w, off_type_advantage_w, speed_w } =
        SWITCH_SCORE_KWARGS;

    const opp = battle.sides[playerIndex].foe.active[0] ?? undefined;

    const map = new Map<Pokemon, number>();

    for (const mon of switches) {
        if (mon.fainted || mon.isActive()) {
            map.set(mon, -Infinity);
            continue;
        }
        const hpFrac = mon.hp / mon.maxhp;
        const off =
            mon.baseSpecies.baseStats.atk + mon.baseSpecies.baseStats.spa;
        const def =
            mon.baseSpecies.baseStats.def + mon.baseSpecies.baseStats.spd;
        const spe = mon.baseSpecies.baseStats.spe;

        const offMult = opp ? typeAdvantageMon(mon, opp, battle) : 1;
        const defMult = opp ? typeAdvantageMon(opp, mon, battle) : 1;

        const score =
            10 * hpFrac +
            check_w +
            off_type_advantage_w * offMult -
            def_type_disadvantage_w * defMult +
            speed_w * spe +
            0.01 * off +
            0.005 * def;

        map.set(mon, score);
    }

    return map;
}

/* ----------------------------------------------------------------- */
/* ------------------ emergency switch heuristic --------------------*/
/* ----------------------------------------------------------------- */
function shouldEmergencySwitch(battle: Battle, playerIndex: number): boolean {
    const p = battle.sides[playerIndex].active[0] as Pokemon;
    if (p === null || p === undefined) {
        throw new Error("No active Pokemon");
    }
    const defBoost = p.boosts.def ?? 0;
    const spdBoost = p.boosts.spd ?? 0;
    const atkBoost = p.boosts.atk ?? 0;
    const spaBoost = p.boosts.spa ?? 0;
    return (
        defBoost <= -3 ||
        spdBoost <= -3 ||
        (atkBoost <= -3 &&
            p.baseSpecies.baseStats.atk >= p.baseSpecies.baseStats.spa) ||
        (spaBoost <= -3 &&
            p.baseSpecies.baseStats.spa >= p.baseSpecies.baseStats.atk)
    );
}

/* ----------------------------------------------------------------- */
/* ---------------------- MOVE-SCORING CORE -------------------------*/
/* ----------------------------------------------------------------- */
function scoreMoves(
    battle: Battle,
    playerIndex: number,
): Map<Protocol.Request.ActivePokemon["moves"][number], number> {
    const user = battle.sides[playerIndex].active[0] as Pokemon;
    const opp = (battle.sides[1 - user.side.n] as Side).active[0] as Pokemon;

    const userHP = user.hp / user.maxhp;
    const oppHP = opp.hp / opp.maxhp;
    const oppLastMoveId = opp.lastMove;

    /* --- gather hazard state --- */
    let userHasHazards = false;
    let oppHasHazards = false;
    for (const h of HAZARDS) {
        if (battle.sides[playerIndex].sideConditions[h]) userHasHazards = true;
        if (battle.sides[1 - playerIndex].sideConditions[h])
            oppHasHazards = true;
    }

    /* --- better switch availability? --- */
    const availableSwitches = battle.sides[playerIndex].team.filter(
        (m) => !m.isActive() && !m.fainted,
    );
    const switchMap = switchScores(availableSwitches, battle, playerIndex);
    const activeScore =
        switchScores([user], battle, playerIndex).get(user) ?? -Infinity;
    let hasBetterSwitch = false;
    let bestSwitch: Pokemon | undefined = undefined;
    if (switchMap.size) {
        bestSwitch = [...switchMap.entries()].reduce((a, b) =>
            b[1] > a[1] ? b : a,
        )[0];
        hasBetterSwitch =
            (switchMap.get(bestSwitch) ?? -Infinity) > activeScore;
    }

    const shouldEmergency = shouldEmergencySwitch(battle, playerIndex);

    /* --- compute highest damage (user & opponent) --- */

    const request = battle.request as AnyObject;
    const availableMovesReq: Protocol.Request.ActivePokemon["moves"] =
        request.active?.[0]?.moves ?? [];
    const availableMoves = availableMovesReq.map(({ id }) =>
        battle.gens.dex.moves.get(id),
    );
    const highestDamageMove = availableMoves.reduce((best, m) =>
        expectedDamage(user, m, opp, battle) >
        expectedDamage(user, best, opp, battle)
            ? m
            : best,
    );

    const oppMovesArr = Object.values(opp.moves);
    const oppHighestDamageMove =
        oppMovesArr.length === 0
            ? undefined
            : oppMovesArr.reduce((best, m) =>
                  worstCaseDamage(
                      opp,
                      {
                          id: m,
                      } as Protocol.Request.ActivePokemon["moves"][number],
                      user,
                      battle,
                  ) >
                  worstCaseDamage(
                      opp,
                      {
                          id: best,
                      } as Protocol.Request.ActivePokemon["moves"][number],
                      user,
                      battle,
                  )
                      ? m
                      : best,
              );

    /* --- boost moves pre-scored --- */
    const boostScores = boostMoveScores(battle, playerIndex, 0.5);
    let maxBoostMove:
        | Protocol.Request.ActivePokemon["moves"][number]
        | undefined;
    if (boostScores.size)
        maxBoostMove = [...boostScores.entries()].reduce((a, b) =>
            b[1] > a[1] ? b : a,
        )[0];

    /* -------- iterate through every available move -------- */
    const moveScores = new Map<
        Protocol.Request.ActivePokemon["moves"][number],
        number
    >();

    for (const move of availableMoves) {
        /* skip disabled via request */
        const req = availableMovesReq.find((r) => r.id === move.id);
        if (req && "disabled" in req && req.disabled) {
            moveScores.set(move, -Infinity);
            continue;
        }

        /* === port of every Python branch === */
        const typeMult = typeAdvantageMove(move.type, opp, battle);
        const ineffective = typeMult === 0;
        const resistsMove = typeMult < 1;
        const expDamage = expectedDamage(user, move, opp, battle);
        const worstDamage = worstCaseDamage(user, move, opp, battle);
        const expCanKill = expDamage >= opp.hp;
        const alwaysCanKill = worstDamage >= opp.hp;
        const selfDestruct = !!move.selfdestruct;
        const selfDestructLoses = availableSwitches.length === 0;
        const slower = !outspeeds(user, opp);
        const faster = !slower;
        const priority = move.priority > 0 && move.id !== "fakeout";

        /* ---- boost flags ---- */
        const isBoost: Record<string, boolean> = {
            atk:
                !!move.boosts &&
                Object.values(move.boosts).reduce((a, b) => a + b, 0) >= 2 &&
                !!move.boosts.atk &&
                move.boosts.atk > 0,
            def:
                !!move.boosts &&
                Object.values(move.boosts).reduce((a, b) => a + b, 0) >= 2 &&
                !!move.boosts.def &&
                move.boosts.def > 0,
            spa:
                !!move.boosts &&
                Object.values(move.boosts).reduce((a, b) => a + b, 0) >= 2 &&
                !!move.boosts.spa &&
                move.boosts.spa > 0,
            spd: !!move.boosts && !!move.boosts.spd && move.boosts.spd > 0,
        };

        /* primary score bucket */
        let score = 0;
        if (alwaysCanKill) {
            score = selfDestruct ? -1 : 8 + (priority ? 2 : 0);
            if (move === highestDamageMove) score++;
        } else if (expCanKill) {
            score = selfDestruct ? -1 : 4 + (priority ? 2 : 0);
            if (move === highestDamageMove) score++;
        } else if (move.category === "Status") {
            score = 0;
        } else if (move === highestDamageMove) {
            score = 1;
        } else if (move.basePower > 1 && move !== highestDamageMove) {
            score = -1;
        } else if (ineffective || (selfDestruct && selfDestructLoses)) {
            score = -12;
        }

        /* kill ASAP if opp is boosted */
        if (
            move === highestDamageMove &&
            (+!!opp.boosts.atk >= 3 ||
                +!!opp.boosts.spa >= 3 ||
                +!!opp.boosts.def >= 3 ||
                +!!opp.boosts.spd >= 3)
        )
            score += 2;

        /* user asleep and move unusable? */
        if (user.status === "slp" && !move.sleepUsable) score = -12;

        /* ---------------- manual scoring ---------------- */

        /* always-hit moves when accuracy tanked */
        if (
            ALWAYS_HIT_MOVES.has(move.id) &&
            +!!user.boosts.accuracy <= -3 &&
            rng() < 0.61
        )
            score += +!!user.boosts.accuracy <= -5 ? 3 : 1;

        /* --- boost-move micro-logic (atk / spa / def / spd) --- */
        if (isBoost.atk) {
            if (
                userHP === 1 &&
                +!!user.boosts.atk <= 2 &&
                move === maxBoostMove
            )
                score += rng() < 0.5 ? 4 : 3;
            else if (userHP > 0.7) {
                /* nothing */
            } else if (userHP > 0.4) {
                if (rng() < 0.84) score -= 2;
            } else score -= 2;
        }

        if (isBoost.spa) {
            if (
                userHP === 1 &&
                +!!user.boosts.spa <= 2 &&
                move === maxBoostMove
            )
                score += rng() < 0.5 ? 4 : 3;
            else if (userHP > 0.7) {
                /* nop */
            } else if (userHP > 0.4) {
                if (rng() < 0.84) score -= 2;
            } else score -= 2;
        }

        if (isBoost.def) {
            if (
                userHP === 1 &&
                +!!user.boosts.def <= 2 &&
                move === maxBoostMove
            )
                score += rng() < 0.5 ? 4 : 3;
            if (+!!user.boosts.def >= 3 && rng() < 0.61) score -= 1;
            if (userHP <= 0.4) score -= 2;
            else if (userHP < 0.7 || rng() < 0.22) {
                let finalChance = 0.7;
                if (oppLastMoveId !== undefined) {
                    const oppMove = battle.gens.dex.moves.get(oppLastMoveId);
                    if (oppMove.category === "Physical") finalChance = 0.59;
                    else if (oppMove.category === "Status") finalChance = 0.77;
                }
                if (rng() < finalChance) score -= 2;
            }
        }

        if (isBoost.spd) {
            if (
                userHP === 1 &&
                slower &&
                +!!user.boosts.spd <= 2 &&
                move === maxBoostMove
            )
                score += rng() < 0.5 ? 4 : 3;
            else if (userHP < 0.4) score -= 2;
            else if (userHP > 0.7) {
                if (rng() < 0.22) score -= 2;
            } else if (rng() < 0.7) score -= 2;
        }

        /* speed-lowering moves */
        if (SPEED_LOWER_MOVES.has(move.id)) {
            if (slower && rng() < 0.5) score += 1;
            else score -= 3;
        }

        /* recovery moves */
        if (
            HP_RECOVERY_MOVES.has(move.id) ||
            (!!move.heal && move.heal[0] / move.heal[1] > 0.2)
        ) {
            if (userHP === 1) score -= 3;
            else if (userHP > 0.5 && rng() < 0.88) score -= 3;
            else if (userHP > 0.33 && rng() < 0.5) score += 2;
            else score += 3;
        }

        /* confusion moves */
        if (CONFUSING_MOVES.has(move.id)) {
            if (["flatter", "swagger"].includes(move.id) && rng() < 0.5)
                score += 1;
            if (oppHP <= 0.7 && rng() < 0.5) score -= 1;
            if (oppHP <= 0.5 && rng() < 0.5) score -= 1;
            if (oppHP <= 0.3 && rng() < 0.5) score -= 1;
        }

        /* paralysis */
        if (PAR_MOVES.has(move.id)) {
            if (slower) {
                if (opp.status !== "par" && rng() < 0.5) score += 2;
            } else if (userHP < 0.7) score -= 1;
        }

        /* sleep moves */
        if (SLEEP_MOVES.has(move.id) && opp.status !== "slp") {
            let hasDream = false;
            for (const m of availableMoves)
                if (["dreameater", "nightmare"].includes(m.id)) hasDream = true;
            if (hasDream && rng() < 0.5) score += 2;
        }

        if (
            ["dreameater", "nightmare"].includes(move.id) &&
            opp.status === "slp"
        )
            if (rng() < 0.5) score += 2;

        /* flail / reversal logic */
        if (["flail", "reversal"].includes(move.id)) {
            if (slower) {
                if (userHP > 0.6) score -= 1;
                else if (userHP > 0.4) {
                    /* keep */
                } else score += 1;
            } else {
                if (userHP > 0.33) score -= 1;
                else if (userHP <= 0.08) score += 1;
            }
        }

        /* stealth rock */
        if (move.id === "stealthrock") {
            if (
                !opp.types.includes("Flying") &&
                !battle.sides[1 - playerIndex].sideConditions.stealthrock
            ) {
                if (rng() < 0.5) score += 2;
            } else score -= 2;
        }

        /* spikes */
        if (move.id === "spikes") {
            const oppSpikes =
                battle.sides[1 - playerIndex].sideConditions.spikes?.level ?? 0;
            if (
                !opp.types.includes("Flying") &&
                (oppSpikes === 0 || oppSpikes < 3)
            ) {
                if (rng() < 0.2) score += 2;
            } else score -= 2;
        }

        /* toxic spikes */
        if (move.id === "toxicspikes") {
            const oppTS =
                battle.sides[1 - playerIndex].sideConditions.toxicspikes
                    ?.level ?? 0;
            if (
                !opp.types.some((t) =>
                    ["Flying", "Poison", "Steel"].includes(t),
                ) &&
                oppTS < 2
            ) {
                if (rng() < 0.2) score += 2;
            } else score -= 2;
        }

        /* defog */
        if (move.id === "defog") {
            if (oppHasHazards && rng() < 0.8) score -= 1;
            if (
                userHasHazards &&
                (userHP < 0.5 || hasBetterSwitch || shouldEmergency)
            ) {
                if (rng() < 0.8) score += 2;
            }
            if (
                battle.sides[1 - playerIndex].sideConditions.lightscreen ||
                battle.sides[1 - playerIndex].sideConditions.reflect
            )
                score += 1;
        }

        /* encore logic */
        if (move.id === "encore") {
            if (slower) score -= 2;
            else {
                const encoreBadges = [
                    "attract",
                    "camouflage",
                    "charge",
                    "confuseray",
                    "conversion",
                    "conversion2",
                    "detect",
                    "dreameater",
                    "encore",
                    "endure",
                    "fakeout",
                    "followme",
                    "foresight",
                    "glare",
                    "growth",
                    "harden",
                    "haze",
                    "healbell",
                    "imprison",
                    "ingrain",
                    "knockoff",
                    "lightscreen",
                    "meanlook",
                    "mudsport",
                    "poisonpowder",
                    "protect",
                    "recycle",
                    "refresh",
                    "rest",
                    "roar",
                    "roleplay",
                    "safeguard",
                    "skillswap",
                    "stunspore",
                    "superfang",
                    "supersonic",
                    "swagger",
                    "sweetkiss",
                    "teeterdance",
                    "thief",
                    "thunderwave",
                    "toxic",
                    "watersport",
                    "willowisp",
                ];
                if (oppLastMoveId) {
                    if (encoreBadges.includes(oppLastMoveId) && rng() < 0.88)
                        score += 3;
                    else score -= 2;
                } else score -= 2;
            }
        }

        /* baton pass */
        if (move.id === "batonpass") {
            const hpThresh = slower ? 0.7 : 0.6;
            const hasBigBoost =
                +!!user.boosts.atk >= 3 ||
                +!!user.boosts.spa >= 3 ||
                +!!user.boosts.def >= 3 ||
                +!!user.boosts.spd >= 3 ||
                +!!user.boosts.evasion >= 3;
            const hasMedBoost =
                +!!user.boosts.atk >= 2 ||
                +!!user.boosts.spa >= 2 ||
                +!!user.boosts.def >= 2 ||
                +!!user.boosts.spd >= 2 ||
                +!!user.boosts.evasion >= 2;

            if (hasBigBoost && userHP < hpThresh && hasBetterSwitch) score += 3;
            else if (hasMedBoost && userHP > hpThresh) score -= 2;
            else score -= 2;
        }

        /* taunt */
        if (move.id === "taunt") {
            if (ineffective && rng() < 0.5) score -= 1;
            if (
                oppLastMoveId &&
                typeAdvantageMove(
                    battle.gens.dex.moves.get(oppLastMoveId).type,
                    user,
                    battle,
                ) > 1 &&
                rng() < 0.5
            )
                score -= 1;
        }

        /* u-turn */
        if (move.id === "uturn") {
            if (move !== highestDamageMove && !hasBetterSwitch) score -= 2;
            if (userHasHazards) score -= 1;
            if (hasBetterSwitch && slower) score += 1;
            if (shouldEmergency) score += 2;
            if (userHP < 0.4 && rng() < 0.3) score += 1;
        }

        /* rapid spin */
        if (
            move.id === "rapidspin" &&
            userHasHazards &&
            (userHP < 0.5 || hasBetterSwitch || shouldEmergency)
        )
            score += rng() < 0.5 ? 3 : 1;

        /* sleep talk / snore */
        if (move.id === "sleeptalk") score += user.status === "slp" ? 10 : -5;
        if (move.id === "snore") score += user.status === "slp" ? 8 : -5;

        /* light screen / reflect */
        if (["lightscreen", "reflect"].includes(move.id)) {
            if (userHP < 0.9) score -= 2;
            else {
                const thresh = faster ? 0.5 : 0.8;
                const oppBestCat = battle.gens.dex.moves.get(
                    oppHighestDamageMove ?? "",
                )?.category;
                if (
                    move.id === "reflect" &&
                    oppBestCat === "Physical" &&
                    !battle.sides[playerIndex].sideConditions.reflect
                ) {
                    if (rng() < thresh) score += 2;
                } else if (
                    move.id === "lightscreen" &&
                    oppBestCat === "Special" &&
                    !battle.sides[playerIndex].sideConditions.lightscreen
                ) {
                    if (rng() < thresh) score += 2;
                }
            }
        }

        /* recharge moves */
        if (RECHARGE_MOVES.has(move.id)) {
            if (resistsMove) score -= 1;
            else {
                const hpThresh = slower ? 0.6 : 0.41;
                if (userHP > hpThresh) score -= 1;
            }
        }

        /* revenge */
        if (move.id === "revenge") {
            if (
                user.status === "slp" ||
                user.volatiles.confusion ||
                rng() < 0.73
            )
                score -= 2;
            else score += 2;
        }

        /* trap */
        if (TRAP_MOVES.has(move.id)) {
            if (
                opp.status === "tox" &&
                !(
                    ["fairylock", "thousandwaves"].includes(move.id) &&
                    opp.types.includes("Ghost")
                ) &&
                rng() < 0.5
            )
                score += 1;
        }

        /* water / weather gimmicks */
        if (move.id === "watersport") {
            if (userHP > 0.5 && opp.types.includes("Fire")) score += 2;
            else score -= 1;
        }

        if (move.id === "raindance") {
            const hasSwiftSwim = availableMoves.some(
                (m) => m.id === "swiftswim",
            );
            if (hasSwiftSwim && slower) score += 1;
            if (userHP < 0.4) score -= 1;
            else if (battle.field.weather !== "Rain") score += 1;
        }

        if (move.id === "hail") {
            if (userHP < 0.4) score -= 1;
            else if (battle.field.weather !== "Hail") score += 1;
        }

        if (move.id === "sunnyday") {
            if (userHP < 0.4) score -= 1;
            else if (battle.field.weather !== "Sun") score += 1;
        }

        /* earthquake vs dig */
        if (move.id === "earthquake" && oppLastMoveId === "dig" && faster) {
            score += 1;
            if (rng() < 0.5) score += 1;
        }

        /* earth power micro */
        if (move.id === "earthpower") {
            if (oppHP === 1 && +!!opp.boosts.spd > -3 && rng() < 0.5)
                score += 1;
            if (oppHP > 0.7 && +!!opp.boosts.spd > -3 && rng() < 0.3)
                score += 1;
            if (+!!opp.boosts.spd <= -2 && rng() < 0.6) score -= 1;
        }

        /* substitute */
        if (move.id === "substitute") {
            if (userHP < 0.5 && rng() < 0.61) score -= 1;
            if (userHP < 0.7 && rng() < 0.61) score -= 1;
            if (userHP < 0.9 && rng() < 0.61) score -= 1;
        }

        /* put final score */
        moveScores.set(move, score);
    }

    return moveScores;
}

/* ----------------------------------------------------------------- */
/* ---------------------- MAIN ACTION CHOICE ----------------------- */
/* ----------------------------------------------------------------- */
export const GetKaizoPlusAction: EvalActionFnType = ({ player }) => {
    const action = new Action();
    action.setActionType(ActionType.ACTION_TYPE__DEFAULT);

    if (player.done) {
        return action;
    }

    const battle = player.privateBattle;
    const request = player.getRequest() as AnyObject;
    const activeReq = request.active ?? [];
    const idx = player.getPlayerIndex();
    if (idx === undefined) throw new Error("no index");

    const mySide = battle.sides[idx];
    const oppSide = battle.sides[1 - idx];
    const attacker = mySide.active[0] as Pokemon | undefined;
    const defender = oppSide.active[0] as Pokemon | undefined;
    const playerIndex = player.getPlayerIndex();
    if (playerIndex === undefined) {
        throw new Error("No player index found");
    }

    const mustSwitch = !!(request.forceSwitch ?? [])[0];
    const availableSwitches =
        activeReq[0]?.trapped ?? false
            ? []
            : mySide.team.filter((m) => !m.isActive() && !m.fainted);
    if (
        mustSwitch ||
        (attacker && shouldEmergencySwitch(battle, playerIndex))
    ) {
        const switchMap = switchScores(availableSwitches, battle, playerIndex);
        if (switchMap.size) {
            const best =
                [...switchMap.entries()].reduce((a, b) =>
                    b[1] > a[1] ? b : a,
                )[0] ?? undefined;
            if (best) {
                const slot = mySide.team.indexOf(best);
                if (slot >= 0) {
                    action.setActionType(ActionType.ACTION_TYPE__SWITCH);
                    action.setSwitchSlot(slot);
                    return action;
                }
            }
        }
    }

    const movesReq = activeReq[0]?.moves;
    if (movesReq && attacker && defender) {
        const moveScores = scoreMoves(battle, playerIndex);
        const ranking = movesReq
            .filter((m: { id: string; disabled: boolean }) => !m.disabled)
            .sort(
                (a: { id: string }, b: { id: string }) =>
                    (moveScores.get(battle.gens.dex.moves.get(b.id)) ??
                        -Infinity) -
                    (moveScores.get(battle.gens.dex.moves.get(a.id)) ??
                        -Infinity),
            );

        if (ranking.length) {
            const bestIdx = movesReq.findIndex(
                (m: { id: string }) => m.id === ranking[0].id,
            );
            action.setActionType(ActionType.ACTION_TYPE__MOVE);
            action.setMoveSlot(bestIdx);
            return action;
        }
    }

    return action;
};
