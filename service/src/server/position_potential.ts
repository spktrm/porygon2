/**
 * The shared human-outcome position potential (2026-09-11 fit,
 * rl/offline/position_potential_fit.json), unit scale. Search reads it as a
 * leaf value; the learner's PBRS channel multiplies it by
 * player_potential_strength (rl/online/training/targets.py). Mirrors
 * rl/offline/position_potential.py -- the shared fixture
 * (rl/offline/position_potential_fixture.json) pins the two.
 */
import { Battle, Pokemon, Side } from "@pkmn/client";
import { Dex, ModdedDex } from "@pkmn/sim";
import fs from "fs";
import path from "path";

const fit: {
    hp_weight: number;
    alive_weight: number;
    matchup_weight: number;
    reference_team_size: number;
} = JSON.parse(
    fs.readFileSync(
        path.resolve(
            __dirname,
            "../../../rl/offline/position_potential_fit.json",
        ),
        "utf8",
    ),
);

export interface PotentialPokemon {
    hp: number;
    active: boolean;
    offensiveTypes: readonly string[];
    defensiveTypes: readonly string[];
}

export interface PositionFeatures {
    hpBalance: number;
    aliveBalance: number;
    matchup: number;
}

function pressure(
    attacker: PotentialPokemon,
    defender: PotentialPokemon,
    dex: ModdedDex,
): number {
    let best = 0;
    for (const attackType of attacker.offensiveTypes) {
        let multiplier = 1;
        for (const defenceType of defender.defensiveTypes) {
            if (!dex.getImmunity(attackType, defenceType)) multiplier = 0;
            multiplier *= 2 ** dex.getEffectiveness(attackType, defenceType);
        }
        best = Math.max(best, multiplier);
    }
    return Math.log2(Math.min(4, Math.max(0.25, best)));
}

export function positionFeatures(
    own: PotentialPokemon[],
    opponent: PotentialPokemon[],
    dex: ModdedDex = Dex,
): PositionFeatures {
    function resources(team: PotentialPokemon[]) {
        if (!team.length)
            throw new Error("Potential needs a nonempty battle roster");
        let hp = 0;
        let alive = 0;
        for (const mon of team) {
            if (!Number.isFinite(mon.hp) || mon.hp < 0 || mon.hp > 1)
                throw new Error("Invalid potential HP");
            hp += mon.hp;
            if (mon.hp > 0) alive++;
        }
        return {
            hp: (hp * fit.reference_team_size) / team.length,
            alive: (alive * fit.reference_team_size) / team.length,
            active: team.filter((mon) => mon.active && mon.hp > 0),
        };
    }
    const ownResources = resources(own);
    const opponentResources = resources(opponent);
    let matchup = 0;
    if (ownResources.active.length && opponentResources.active.length) {
        function exposure(
            defenders: PotentialPokemon[],
            attackers: PotentialPokemon[],
        ) {
            return (
                defenders.reduce(
                    (total, defender) =>
                        total +
                        Math.max(
                            ...attackers.map((attacker) =>
                                pressure(attacker, defender, dex),
                            ),
                        ),
                    0,
                ) / defenders.length
            );
        }
        matchup =
            exposure(opponentResources.active, ownResources.active) -
            exposure(ownResources.active, opponentResources.active);
    }
    return {
        hpBalance: ownResources.hp - opponentResources.hp,
        aliveBalance: ownResources.alive - opponentResources.alive,
        matchup,
    };
}

/** tanh(win logit / 2): the fitted expected outcome in [-1, 1] units. */
export function potentialOf(features: PositionFeatures): number {
    return Math.tanh(
        (fit.hp_weight * features.hpBalance +
            fit.alive_weight * features.aliveBalance +
            fit.matchup_weight * features.matchup) /
            2,
    );
}

export function evaluatePosition(
    own: PotentialPokemon[],
    opponent: PotentialPokemon[],
    dex: ModdedDex = Dex,
): number {
    return potentialOf(positionFeatures(own, opponent, dex));
}

/**
 * The HP fraction the OPPONENT sees: the sim's shared health line
 * (sim/pokemon.ts getHealth, percentages from gen 7). publicBattle holds our
 * own HP exactly (the player stream carries the secret half of each split
 * line) and theirs as a percentage, on which this rule is the identity -- so
 * both perspectives read identical inputs. A revealed member at hp 0 that has
 * not fainted has no HP report yet and reads full, as in
 * StateHandler.getHpRatio.
 */
export function publicHpFraction(member: Pokemon): number {
    if (member.fainted) return 0;
    if (member.hp === 0) return 1;
    let percentage = Math.ceil((100 * member.hp) / member.maxhp);
    if (percentage === 100 && member.hp < member.maxhp) percentage = 99;
    return percentage / 100;
}

/**
 * One side's battle roster as the potential reads it. Offensive types are the
 * species' STAB plus a revealed non-Stellar tera type (the fit's convention);
 * defensive types are the client's current typing (tera, typechange, Roost)
 * plus an added type. Unrevealed members fill the roster at full HP and never
 * act, so they carry no types.
 */
export function sidePotentialTeam(side: Side): PotentialPokemon[] {
    const team: PotentialPokemon[] = side.team.map((member) => {
        const offensiveTypes: string[] = [...member.species.types];
        const teraType = member.teraType;
        if (
            member.isTerastallized &&
            teraType !== undefined &&
            teraType !== "Stellar" &&
            !offensiveTypes.includes(teraType)
        ) {
            offensiveTypes.push(teraType);
        }
        const defensiveTypes: string[] = [...member.types];
        const addedType = member.addedType;
        if (addedType !== undefined && !defensiveTypes.includes(addedType)) {
            defensiveTypes.push(addedType);
        }
        return {
            hp: publicHpFraction(member),
            active: side.active.includes(member),
            offensiveTypes,
            defensiveTypes,
        };
    });
    while (team.length < side.totalPokemon) {
        team.push({
            hp: 1,
            active: false,
            offensiveTypes: [],
            defensiveTypes: [],
        });
    }
    return team;
}

/** The position features from one player's public view, on its gen's chart. */
export function publicPositionFeatures(
    battle: Battle,
    playerIndex: number,
): PositionFeatures {
    return positionFeatures(
        sidePotentialTeam(battle.sides[playerIndex]),
        sidePotentialTeam(battle.sides[1 - playerIndex]),
        Dex.forGen(battle.gen.num),
    );
}
