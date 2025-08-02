import { AnyObject } from "@pkmn/sim";
import { Pokemon } from "@pkmn/client";
import { EvalActionFnType } from "../eval";
import { GetMoveDamage } from "./max_dmg";
import { Action } from "../../../protos/service_pb";
import { ActionType } from "../../../protos/features_pb";

function scorePokemon(poke1: Pokemon, poke2: Pokemon) {
    if (!poke1 || poke1.fainted) {
        return -Infinity; // Fainted Pokémon can't fight, give a very low score
    }

    const baseStats = poke1.baseSpecies.baseStats;

    // Base stats contribution
    const hpFactor = poke1.hp / poke1.maxhp; // Remaining health as a fraction
    const offensivePower = baseStats.atk + baseStats.spa;
    const defensivePower = baseStats.def + baseStats.spd;
    const speedFactor = baseStats.spe / 100; // Speed contributes to turn order

    // Adjust the score based on health, base stats, and whether the Pokémon is statused
    let score =
        (offensivePower + defensivePower) * hpFactor * (1 + speedFactor);

    // Factor in Pokémon's status condition, if any
    if (poke1.status === "par") score *= 0.8; // Paralysis reduces utility
    else if (poke1.status === "slp" || poke1.status === "frz")
        score *= 0.5; // Sleep/freeze significantly reduce utility
    else if (poke1.status === "brn" || poke1.status === "psn")
        score *= 0.9; // Burn/poison reduce utility
    else if (poke1.status === "tox") {
        // Toxic (badly poisoned) penalty: the longer poisoned, the more severe the impact
        const toxicTurnCount = poke1.statusState.toxicTurns || 1; // Assuming toxicCounter keeps track of the number of turns poisoned
        const toxicMultiplier = Math.max(0.5, 1 - 0.1 * toxicTurnCount); // Reduces score more each turn
        score *= toxicMultiplier;
    }

    const calcTyping = (p1: Pokemon, p2: Pokemon) => {
        const out: number[] = [];
        for (const i in [0, 1]) {
            for (const j in [0, 1]) {
                let val = undefined;
                try {
                    val = poke1.species.dex.getEffectiveness(
                        p1.types[i],
                        p2.types[j],
                    );
                    // eslint-disable-next-line @typescript-eslint/no-unused-vars
                } catch (err) {
                    /* empty */
                } finally {
                    if (val !== undefined) out.push(val);
                }
            }
        }
        return out.reduce((a, b) => a + b) / out.length;
    };

    // Type advantages or disadvantages
    const typeAdvantageMultiplier = calcTyping(poke1, poke2); // Placeholder, as you might have access to type-checking functions
    // Example: `getTypeAdvantageMultiplier(poke, opponentType)` function that returns a multiplier based on effectiveness
    // typeAdvantageMultiplier = getTypeAdvantageMultiplier(poke, opponentType);

    // Final score adjustment based on type matchup
    score *= typeAdvantageMultiplier;

    return score;
}

export const GetHeuristicAction: EvalActionFnType = ({ player }) => {
    const action = new Action();
    action.setActionType(ActionType.ACTION_TYPE__DEFAULT);

    if (player.done) {
        return action;
    }

    const battle = player.privateBattle;
    const request = player.getRequest() as AnyObject;
    const active = request.active ?? [];
    const moves = active[0]?.moves;

    const playerIndex = player.getPlayerIndex();
    if (playerIndex === undefined) {
        throw new Error();
    }

    const mySide = battle.sides[playerIndex];
    const oppSide = battle.sides[1 - playerIndex];

    const attacker = mySide.active[0];
    const defender = oppSide.active[0];

    const argMax = (arr: number[]) => {
        return arr.reduce((maxIndex, currentElement, currentIndex, arr) => {
            return currentElement > arr[maxIndex] ? currentIndex : maxIndex;
        }, 0);
    };

    if (
        (!!(request.forceSwitch ?? [])[0] && defender !== null) ||
        (attacker !== null &&
            defender !== null &&
            scorePokemon(defender, attacker) >
                scorePokemon(attacker, defender) &&
            !attacker.trapped &&
            !attacker.maybeTrapped)
    ) {
        const scores = mySide.team.map((member) =>
            !!member.isActive() ||
            !!member.trapped ||
            !!member.maybeTrapped ||
            !!member.fainted
                ? -Infinity
                : scorePokemon(member, defender),
        );
        const maxScoreIdx = argMax(scores);
        const chosenMember = mySide.team[maxScoreIdx];
        if (scores.some((score) => score > -Infinity)) {
            if (chosenMember.fainted) {
                console.error("fainted");
            }
            if (attacker === null) {
                action.setActionType(ActionType.ACTION_TYPE__SWITCH);
                action.setSwitchSlot(maxScoreIdx);
                return action;
            }
            if (scorePokemon(attacker, defender) < scores[maxScoreIdx]) {
                action.setActionType(ActionType.ACTION_TYPE__SWITCH);
                action.setSwitchSlot(maxScoreIdx);
                return action;
            }
        }
    }
    if (moves !== undefined && attacker !== null && defender !== null) {
        const moveData: number[] = moves.map(
            ({ id, disabled }: { id: string; disabled: boolean }) => {
                let damage = 0;
                damage = GetMoveDamage({
                    battle,
                    attacker,
                    defender,
                    moveId: id,
                });

                return disabled ? -Infinity : damage;
            },
        );
        action.setActionType(ActionType.ACTION_TYPE__MOVE);
        action.setMoveSlot(argMax(moveData));
        return action;
    }
    return action;
};
