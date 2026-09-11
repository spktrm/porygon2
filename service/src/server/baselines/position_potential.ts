/** The shared human-outcome potential; search uses unit scale, not PBRS eta. */
import { Dex } from "@pkmn/sim";
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
            "../../../../rl/offline/position_potential_fit.json",
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

function pressure(
    attacker: PotentialPokemon,
    defender: PotentialPokemon,
): number {
    let best = 0;
    for (const attackType of attacker.offensiveTypes) {
        let multiplier = 1;
        for (const defenceType of defender.defensiveTypes) {
            if (!Dex.getImmunity(attackType, defenceType)) multiplier = 0;
            multiplier *= 2 ** Dex.getEffectiveness(attackType, defenceType);
        }
        best = Math.max(best, multiplier);
    }
    return Math.log2(Math.min(4, Math.max(0.25, best)));
}

export function evaluatePosition(
    own: PotentialPokemon[],
    opponent: PotentialPokemon[],
): number {
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
                                pressure(attacker, defender),
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
    return Math.tanh(
        (fit.hp_weight * (ownResources.hp - opponentResources.hp) +
            fit.alive_weight * (ownResources.alive - opponentResources.alive) +
            fit.matchup_weight * matchup) /
            2,
    );
}
