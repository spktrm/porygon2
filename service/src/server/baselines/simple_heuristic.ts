/* simple_heuristic.ts
 * ------------------------------------------------------------------
 * A dependency-free "harder" baseline for the flattened (src, tgt)
 * action space. Roughly equivalent to poke-env's SimpleHeuristicsPlayer:
 *
 *   - Damaging moves are scored by an estimated-damage proxy:
 *         base power x type-effectiveness x STAB x accuracy x atk/def ratio
 *     (boost stages folded into the stat ratio). The strongest is chosen.
 *   - Status / setup / recovery / hazard moves get context-aware scores so
 *     they are only picked when attacking is weak or clearly worthwhile.
 *   - When the active Pokemon is at a type disadvantage (or badly debuffed)
 *     and a reserve has a better matchup, it switches out.
 *   - Forced switches and team-preview leads pick the best matchup / lead.
 *
 * Unlike max_dmg.ts / kaizo_plus.ts this uses only @pkmn/client + @pkmn/dex
 * (no @smogon/calc), so it runs without any extra dependency, and it emits
 * actions purely by scoring the legal (src, tgt) pairs from the action mask
 * — so it is correct across singles / doubles / forceSwitch / teamPreview.
 * ------------------------------------------------------------------ */

import { Battle, Pokemon } from "@pkmn/client";
import { AnyObject } from "@pkmn/sim";
import { EvalActionFnType } from "../eval";
import { StateHandler } from "../state";
import { OneDBoolean } from "../utils";
import { Action, ActionEnum, ActionEnumMap } from "../../../protos/service_pb";

/* ----------------------------------------------------------------- */
/* ----------------------- src classification ----------------------- */
/* ----------------------------------------------------------------- */

/** ally index (0 / 1) and move slot (0-3) for a move/wildcard src, else null. */
function decodeMoveSrc(src: number): { ally: number; slot: number } | null {
    switch (src) {
        case ActionEnum.ACTION_ENUM__ALLY_1_MOVE_1:
        case ActionEnum.ACTION_ENUM__ALLY_1_MOVE_2:
        case ActionEnum.ACTION_ENUM__ALLY_1_MOVE_3:
        case ActionEnum.ACTION_ENUM__ALLY_1_MOVE_4:
            return {
                ally: 0,
                slot: src - ActionEnum.ACTION_ENUM__ALLY_1_MOVE_1,
            };
        case ActionEnum.ACTION_ENUM__ALLY_1_MOVE_1_WILDCARD:
        case ActionEnum.ACTION_ENUM__ALLY_1_MOVE_2_WILDCARD:
        case ActionEnum.ACTION_ENUM__ALLY_1_MOVE_3_WILDCARD:
        case ActionEnum.ACTION_ENUM__ALLY_1_MOVE_4_WILDCARD:
            return {
                ally: 0,
                slot: src - ActionEnum.ACTION_ENUM__ALLY_1_MOVE_1_WILDCARD,
            };
        case ActionEnum.ACTION_ENUM__ALLY_2_MOVE_1:
        case ActionEnum.ACTION_ENUM__ALLY_2_MOVE_2:
        case ActionEnum.ACTION_ENUM__ALLY_2_MOVE_3:
        case ActionEnum.ACTION_ENUM__ALLY_2_MOVE_4:
            return {
                ally: 1,
                slot: src - ActionEnum.ACTION_ENUM__ALLY_2_MOVE_1,
            };
        case ActionEnum.ACTION_ENUM__ALLY_2_MOVE_1_WILDCARD:
        case ActionEnum.ACTION_ENUM__ALLY_2_MOVE_2_WILDCARD:
        case ActionEnum.ACTION_ENUM__ALLY_2_MOVE_3_WILDCARD:
        case ActionEnum.ACTION_ENUM__ALLY_2_MOVE_4_WILDCARD:
            return {
                ally: 1,
                slot: src - ActionEnum.ACTION_ENUM__ALLY_2_MOVE_1_WILDCARD,
            };
        default:
            return null;
    }
}

function isWildcardSrc(src: number): boolean {
    return (
        (src >= ActionEnum.ACTION_ENUM__ALLY_1_MOVE_1_WILDCARD &&
            src <= ActionEnum.ACTION_ENUM__ALLY_1_MOVE_4_WILDCARD) ||
        (src >= ActionEnum.ACTION_ENUM__ALLY_2_MOVE_1_WILDCARD &&
            src <= ActionEnum.ACTION_ENUM__ALLY_2_MOVE_4_WILDCARD)
    );
}

/** team-roster index (0-5) for a reserve slot index, else null. */
function decodeReserveIndex(index: number): number | null {
    if (
        index >= ActionEnum.ACTION_ENUM__RESERVE_1_SWITCH_IN &&
        index <= ActionEnum.ACTION_ENUM__RESERVE_6_SWITCH_IN
    ) {
        return index - ActionEnum.ACTION_ENUM__RESERVE_1_SWITCH_IN;
    }
    return null;
}

/** ally slot (0/1) for a battle-switch src, else null. */
function decodeSwitchSrc(src: number): number | null {
    if (src === ActionEnum.ACTION_ENUM__ALLY_1_SWITCH) {
        return 0;
    }
    if (src === ActionEnum.ACTION_ENUM__ALLY_2_SWITCH) {
        return 1;
    }
    return null;
}

function isPassSrc(src: number): boolean {
    return (
        src === ActionEnum.ACTION_ENUM__ALLY_1_PASS ||
        src === ActionEnum.ACTION_ENUM__ALLY_2_PASS
    );
}

/* ----------------------------------------------------------------- */
/* ----------------------- battle primitives ------------------------ */
/* ----------------------------------------------------------------- */

/** Showdown boost stage (-6..6) to a stat multiplier. */
function stageMultiplier(stage: number): number {
    return stage >= 0 ? (2 + stage) / 2 : 2 / (2 - stage);
}

type StatKey = "atk" | "def" | "spa" | "spd" | "spe";

function boostedStat(mon: Pokemon, key: StatKey): number {
    const base = mon.baseSpecies.baseStats[key] ?? 1;
    const stage = (mon.boosts as Record<string, number>)?.[key] ?? 0;
    return base * stageMultiplier(stage);
}

function hpFraction(mon: Pokemon): number {
    return mon.maxhp > 0 ? mon.hp / mon.maxhp : 0;
}

/**
 * Type-effectiveness multiplier of an attacking type vs a defender's typing.
 * Returns 0 for an immunity, else the product of per-type multipliers
 * (0.25, 0.5, 1, 2, 4, ...).
 */
function typeMultiplier(
    battle: Battle,
    moveType: string,
    defenderTypes: readonly string[],
): number {
    const dex = battle.gens.dex;
    let mult = 1;
    for (const t of defenderTypes) {
        if (!dex.getImmunity(moveType, t)) {
            return 0;
        }
        mult *= Math.pow(2, dex.getEffectiveness(moveType, t));
    }
    return mult;
}

/**
 * Net matchup of `attacker` vs `defender`: best offensive STAB effectiveness
 * minus the defender's best offensive STAB effectiveness back. Positive means
 * we have the type advantage. Roughly in [-4, 4].
 */
function matchup(battle: Battle, attacker: Pokemon, defender: Pokemon): number {
    const off = Math.max(
        1,
        ...attacker.types.map((t) => typeMultiplier(battle, t, defender.types)),
    );
    const vuln = Math.max(
        1,
        ...defender.types.map((t) => typeMultiplier(battle, t, attacker.types)),
    );
    return off - vuln;
}

function severelyDebuffed(mon: Pokemon): boolean {
    const b = mon.boosts as Record<string, number>;
    return (
        (b?.def ?? 0) <= -3 ||
        (b?.spd ?? 0) <= -3 ||
        (b?.atk ?? 0) <= -3 ||
        (b?.spa ?? 0) <= -3
    );
}

/* ----------------------------------------------------------------- */
/* --------------------------- move scoring ------------------------- */
/* ----------------------------------------------------------------- */

const RECOVERY_MOVES = new Set([
    "recover",
    "roost",
    "softboiled",
    "milkdrink",
    "moonlight",
    "morningsun",
    "synthesis",
    "slackoff",
    "shoreup",
    "rest",
    "healorder",
    "wish",
]);

const HAZARD_MOVES = new Set([
    "stealthrock",
    "spikes",
    "toxicspikes",
    "stickyweb",
]);

/** Estimated-damage proxy for a damaging move (higher = better). */
function damageScore(
    battle: Battle,
    attacker: Pokemon,
    defender: Pokemon,
    move: AnyObject,
): number {
    const tm = typeMultiplier(battle, move.type, defender.types);
    if (tm === 0) {
        return -50; // immune — never pick if anything else exists
    }
    const stab = attacker.types.includes(move.type) ? 1.5 : 1;
    const acc = move.accuracy === true ? 1 : (move.accuracy ?? 100) / 100;
    const atkKey: StatKey = move.category === "Physical" ? "atk" : "spa";
    const defKey: StatKey = move.category === "Physical" ? "def" : "spd";
    const ratio = boostedStat(attacker, atkKey) / boostedStat(defender, defKey);

    let score = (move.basePower * tm * stab * acc * ratio) / 50;

    // Reward finishing with priority when we are slower and the foe is low.
    if (
        (move.priority ?? 0) > 0 &&
        boostedStat(attacker, "spe") < boostedStat(defender, "spe") &&
        hpFraction(defender) <= 0.4
    ) {
        score += 5;
    }
    return score;
}

/** Context-aware score for a non-damaging (status / setup / utility) move. */
function statusScore(
    battle: Battle,
    attacker: Pokemon,
    defender: Pokemon,
    move: AnyObject,
): number {
    const userHp = hpFraction(attacker);

    // Recovery — valuable when hurt, wasteful at full health.
    if (
        RECOVERY_MOVES.has(move.id) ||
        (move.heal && move.heal[0] / move.heal[1] >= 0.2)
    ) {
        if (userHp <= 0.5) return 4;
        if (userHp <= 0.75) return 1;
        return -1;
    }

    // Stat-boosting setup — only when safe (healthy + favourable matchup) and
    // not already heavily boosted.
    if (move.boosts && move.target === "self") {
        const total = Object.values(
            move.boosts as Record<string, number>,
        ).reduce((a, b) => a + b, 0);
        const alreadyBoosted = Object.values(
            attacker.boosts as Record<string, number>,
        ).some((v) => (v ?? 0) >= 2);
        if (
            total > 0 &&
            userHp >= 0.85 &&
            !alreadyBoosted &&
            matchup(battle, attacker, defender) >= 0
        ) {
            return 3.5;
        }
        return 0.2;
    }

    // Status infliction — good against a healthy, unstatused foe.
    if (move.status && !defender.status) {
        const tm = typeMultiplier(battle, move.type ?? "???", defender.types);
        // Respect type-based status immunities (e.g. Electric vs paralysis).
        if (tm === 0 && move.category === "Status") return 0.1;
        if (hpFraction(defender) >= 0.6) return 3;
        return 1;
    }

    // Entry hazards — set them early if not already up on the foe's side.
    if (HAZARD_MOVES.has(move.id)) {
        return 2.5;
    }

    // Generic utility: better than doing nothing, worse than any real attack.
    return 0.5;
}

function scoreMove(
    battle: Battle,
    attacker: Pokemon,
    defender: Pokemon | null,
    move: AnyObject,
): number {
    const damaging = move.category !== "Status" && (move.basePower ?? 0) > 0;
    if (damaging) {
        // No defender to hit (shouldn't happen on a normal turn) — fall back
        // to raw base power so we still favour the strongest move.
        if (!defender) return (move.basePower ?? 0) / 50;
        return damageScore(battle, attacker, defender, move);
    }
    if (!defender) return 0.5;
    return statusScore(battle, attacker, defender, move);
}

/* ----------------------------------------------------------------- */
/* ------------------------- switch scoring ------------------------- */
/* ----------------------------------------------------------------- */

/** Generic "good lead / good Pokemon" score, used at team preview. */
function leadScore(mon: Pokemon): number {
    const s = mon.baseSpecies.baseStats;
    const offense = Math.max(s.atk, s.spa);
    const bulk = s.hp + (s.def + s.spd) / 2;
    return offense + 0.5 * bulk + 0.3 * s.spe;
}

/**
 * Score a voluntary switch on a normal turn. Returns a strongly negative
 * value when we should not switch (no type disadvantage and not debuffed),
 * otherwise a score that scales with how much better the candidate's matchup
 * is, so it can outrank a weak attack but not a strong one.
 */
const SWITCH_COST = 2;
function voluntarySwitchScore(
    battle: Battle,
    current: Pokemon,
    candidate: Pokemon,
    opp: Pokemon | null,
): number {
    if (candidate.fainted || candidate.isActive()) return -Infinity;
    if (!opp) return -1000;

    const curMatch = matchup(battle, current, opp);
    const emergency = severelyDebuffed(current);
    if (curMatch >= 0 && !emergency) {
        return -1000; // current matchup is fine — stay in
    }
    const candMatch = matchup(battle, candidate, opp);
    return candMatch * 3 - SWITCH_COST + (emergency ? 4 : 0);
}

/* ----------------------------------------------------------------- */
/* --------------------------- main entry --------------------------- */
/* ----------------------------------------------------------------- */

function legalPairs(mask: OneDBoolean): {
    src: ActionEnumMap[keyof ActionEnumMap];
    tgt: ActionEnumMap[keyof ActionEnumMap];
}[] {
    const width = mask.width;
    if (width === undefined) {
        throw new Error("Action mask width is undefined");
    }
    const flat = mask.toBinaryVector();
    const pairs: {
        src: ActionEnumMap[keyof ActionEnumMap];
        tgt: ActionEnumMap[keyof ActionEnumMap];
    }[] = [];
    for (let i = 0; i < flat.length; i++) {
        if (flat[i] === 1) {
            pairs.push({
                src: Math.floor(
                    i / width,
                ) as ActionEnumMap[keyof ActionEnumMap],
                tgt: (i % width) as ActionEnumMap[keyof ActionEnumMap],
            });
        }
    }
    return pairs;
}

export const GetSimpleHeuristicAction: EvalActionFnType = ({ player }) => {
    const battle = player.privateBattle;
    const request = battle.request as AnyObject | null | undefined;

    const playerIndex = player.getPlayerIndex();
    if (playerIndex === undefined) {
        throw new Error("Player index is undefined");
    }

    const stateHandler = new StateHandler(player);
    const allyActive = player.publicBattle.sides[playerIndex].active;
    const enemyActive = player.publicBattle.sides[1 - playerIndex].active;
    const { actionMask } = stateHandler.getActionMask({
        request,
        format: battle.gameType,
        allyActive,
        enemyActive,
    });

    const mySide = battle.sides[playerIndex];
    const oppSide = battle.sides[1 - playerIndex];

    const isTeamPreview = !!request?.teamPreview;
    const activeReq = (request?.active ?? []) as AnyObject[];

    const defenderForTgt = (tgt: number): Pokemon | null => {
        if (tgt === ActionEnum.ACTION_ENUM__ENEMY_2_TARGET) {
            return oppSide.active[1] ?? oppSide.active[0] ?? null;
        }
        return oppSide.active[0] ?? oppSide.active[1] ?? null;
    };

    let best: {
        src: ActionEnumMap[keyof ActionEnumMap];
        tgt: ActionEnumMap[keyof ActionEnumMap];
        score: number;
    } | null = null;

    for (const { src, tgt } of legalPairs(actionMask)) {
        let score = -1e4; // Default fallback for unhandled pairs

        if (src === ActionEnum.ACTION_ENUM__DEFAULT || isPassSrc(src)) {
            score = -1e6;
        } else {
            const moveSrc = decodeMoveSrc(src);
            if (moveSrc !== null) {
                const attacker = mySide.active[moveSrc.ally];
                const moveReq = activeReq[moveSrc.ally]?.moves?.[moveSrc.slot];
                if (!attacker || !moveReq?.id) {
                    score = -1e5;
                } else {
                    const move = battle.gens.dex.moves.get(
                        moveReq.id,
                    ) as AnyObject;
                    score = scoreMove(
                        battle,
                        attacker,
                        defenderForTgt(tgt),
                        move,
                    );
                    if (isWildcardSrc(src)) score -= 0.01;
                }
            } else {
                // Battle switch: (ALLY_i_SWITCH, RESERVE_j); team preview
                // still encodes the chosen mon as the src.
                const switchAlly = decodeSwitchSrc(src);
                const teamIdx =
                    switchAlly !== null
                        ? decodeReserveIndex(tgt)
                        : decodeReserveIndex(src);
                if (teamIdx !== null) {
                    const candidate = mySide.team[teamIdx];
                    if (!candidate) {
                        score = -1e5;
                    } else if (isTeamPreview) {
                        score = leadScore(candidate) - 0.001 * teamIdx;
                    } else {
                        const opp = oppSide.active[0] ?? null;
                        const current = mySide.active[switchAlly ?? 0];
                        if (!current) {
                            score =
                                100 +
                                (opp ? matchup(battle, candidate, opp) : 0);
                        } else {
                            score = voluntarySwitchScore(
                                battle,
                                current,
                                candidate,
                                opp,
                            );
                        }
                    }
                }
            }
        }

        // TypeScript can track this synchronous mutation perfectly
        if (best === null || score > best.score) {
            best = { src, tgt, score };
        }
    }

    const action = new Action();
    if (best === null) {
        // No legal action found — emit DEFAULT and let the engine resolve it.
        action.setSrc(ActionEnum.ACTION_ENUM__DEFAULT);
        action.setTgt(ActionEnum.ACTION_ENUM__DEFAULT);
        return action;
    }
    action.setSrc(best.src);
    action.setTgt(best.tgt);
    return action;
};
