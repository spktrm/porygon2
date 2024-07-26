import { AnyObject } from "@pkmn/sim";
import { EvalActionFnType } from "../eval";
import { calculateADV } from "@smogon/calc/dist/mechanics/gen3";
import { Generation } from "@smogon/calc/dist/data/interface";
import {
    Generations,
    Pokemon as SmogonPoke,
    Move,
    Field,
    Side,
    Result,
} from "@smogon/calc";
import { Battle, Pokemon } from "@pkmn/client";

function fixMoveId(moveId: string) {
    if (moveId.startsWith("hiddenpower")) {
        return moveId.slice(0, -2);
    } else if (moveId.startsWith("return")) {
        return "return";
    }
    return moveId;
}

export const GetMoveDamange: (args: {
    battle: Battle;
    generation?: Generation;
    attacker: Pokemon;
    defender: Pokemon;
    moveId: string;
}) => number = ({ battle, attacker, defender, generation, moveId }) => {
    if (!generation) {
        generation = Generations.get(3);
    }
    if (moveId === "recharge") {
        return 0;
    }
    moveId = fixMoveId(moveId);
    const result = calculateADV(
        generation,
        new SmogonPoke(generation, attacker.baseSpecies.baseSpecies, {
            item: attacker.item,
            ability: attacker.ability,
            abilityOn: true,
            nature: attacker.nature ?? "Hardy",
            ivs: attacker.ivs,
            evs: attacker.evs ?? {
                hp: 84,
                atk: 84,
                def: 84,
                spa: 84,
                spd: 84,
                spe: 84,
            },
            boosts: attacker.boosts,
        }),
        new SmogonPoke(generation, defender.baseSpecies.baseSpecies, {
            item: defender.item,
            ability: defender.ability,
            abilityOn: true,
            nature: defender.nature ?? "Hardy",
            ivs: attacker.ivs,
            evs: defender.evs ?? {
                hp: 84,
                atk: 84,
                def: 84,
                spa: 84,
                spd: 84,
                spe: 84,
            },
            boosts: defender.boosts,
        }),
        new Move(generation, moveId, {
            species: attacker.baseSpecies.baseSpecies,
            item: attacker.item,
            ability: attacker.ability,
        }),
        new Field({
            weather: battle.field.weather,
            terrain: battle.field.terrain,
            attackerSide: new Side({
                isReflect:
                    attacker.side.sideConditions?.reflect?.level !== undefined,
            }),
            defenderSide: new Side({
                isReflect:
                    attacker.side.sideConditions?.reflect?.level !== undefined,
            }),
        }),
    ) as Result;
    const damage = result.damage as number[];
    if (typeof damage === "object") {
        return damage.reduce((a, b) => a + b) / damage.length;
    } else {
        return damage / result.defender.stats.hp;
    }
};

export const GetMaxDamageAction: EvalActionFnType = ({ handler }) => {
    const battle = handler.privatebattle;
    const generation = Generations.get(battle.gen.num);
    const request = battle.request as AnyObject;
    const active = request.active ?? [];
    const moves = active[0]?.moves;

    const playerIndex = handler.getPlayerIndex();
    if (playerIndex === undefined) {
        throw new Error();
    }

    const mySide = battle.sides[playerIndex];
    const oppSide = battle.sides[1 - playerIndex];

    const attacker = mySide.active[0];
    const defender = oppSide.active[0];

    if (moves !== undefined && attacker !== null && defender !== null) {
        const moveData: number[] = moves.map(
            ({ id, disabled }: { id: string; disabled: boolean }) => {
                let damage = 0;
                try {
                    damage = GetMoveDamange({
                        battle,
                        generation,
                        attacker,
                        defender,
                        moveId: id,
                    });
                } catch (err) {
                    throw err;
                }
                return disabled ? -100 : damage;
            },
        );
        const indexOfLargestNum = moveData.reduce(
            (maxIndex, currentElement, currentIndex, arr) => {
                return currentElement > arr[maxIndex] ? currentIndex : maxIndex;
            },
            0,
        );
        return indexOfLargestNum;
    }
    return -1;
};
