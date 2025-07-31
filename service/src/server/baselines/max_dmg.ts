import { AnyObject } from "@pkmn/sim";
import { calculateADV } from "@smogon/calc/dist/mechanics/gen3";
import {
    Generations,
    Pokemon as SmogonPoke,
    Move,
    Field,
    Side,
    Result,
} from "@smogon/calc";
import { Battle, Pokemon } from "@pkmn/client";
import { EvalActionFnType } from "../eval";

function fixMoveId(moveId: string) {
    if (moveId.startsWith("hiddenpower")) {
        const power = parseInt(moveId.slice(-2));
        if (isNaN(power)) {
            return "hiddenpower";
        } else {
            return moveId.slice(0, -2);
        }
    } else if (moveId.startsWith("return")) {
        return "return";
    } else if (moveId.startsWith("frustration")) {
        return "frustration";
    }
    return moveId;
}

export const GetMoveDamange: (args: {
    battle: Battle;
    attacker: Pokemon;
    defender: Pokemon;
    moveId: string;
}) => number = ({ battle, attacker, defender, moveId }) => {
    if (moveId === "recharge") {
        return 0;
    }
    moveId = fixMoveId(moveId);
    const moveData = battle.gens.dex.moves.get(moveId);
    const moveAccuracy =
        moveData.accuracy === true ? 1 : moveData.accuracy / 100;

    const potentialDefenderAbilities = Object.values(
        defender.baseSpecies.abilities,
    );
    const numDefenderAbilities = potentialDefenderAbilities.length;
    const defenderAbility =
        defender.ability === ""
            ? potentialDefenderAbilities[
                  Math.floor(Math.random() * numDefenderAbilities)
              ]
            : defender.ability;

    const Calc = (isCrit: boolean) => {
        const generation = Generations.get(battle.gen.num);
        let result: Result;
        try {
            result = calculateADV(
                generation,
                new SmogonPoke(generation, attacker.baseSpecies.baseSpecies, {
                    item: attacker.item,
                    ability: attacker.ability,
                    abilityOn: true,
                    nature: attacker.nature ?? "Hardy",
                    ivs: attacker.ivs,
                    level: attacker.level,
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
                    ability: defenderAbility,
                    abilityOn: true,
                    nature: defender.nature ?? "Hardy",
                    ivs: attacker.ivs,
                    level: defender.level,
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
                    isCrit,
                }),
                new Field({
                    weather: battle.field.weather,
                    terrain: battle.field.terrain,
                    attackerSide: new Side({
                        isReflect:
                            attacker.side.sideConditions?.reflect?.level !==
                            undefined,
                    }),
                    defenderSide: new Side({
                        isReflect:
                            attacker.side.sideConditions?.reflect?.level !==
                            undefined,
                    }),
                }),
            ) as Result;
        } catch (err) {
            console.log(
                `Error calculating damage for move ${moveId} in gen ${battle.gen.num}`,
                err,
            );
            return 0;
        }
        const damage = result.damage as number[];
        if (typeof damage === "object") {
            return (
                (damage.reduce((a, b) => a + b) / damage.length) * moveAccuracy
            );
        } else {
            return damage / result.defender.stats.hp;
        }
    };
    const calcCritChance = () => {
        let stage = 0;
        if (attacker.ability === "Super Luck") {
            stage += 1;
        }
        if ((moveData.critRatio ?? 1) === 2) {
            stage += 1;
        }
        if (["Razor Claw", "Scope Lens"].includes(attacker.item)) {
            stage += 1;
        }
        return (
            {
                0: 1 / 16,
                1: 1 / 8,
                2: 1 / 4,
                3: 1 / 3,
                4: 1 / 2,
            }[Math.min(4, stage)] ?? 1 / 16
        );
    };
    const critChance = calcCritChance();
    return critChance * Calc(true) + (1 - critChance) * Calc(false);
};

export const GetMaxDamageAction: EvalActionFnType = ({ player }) => {
    if (player.done) {
        return { actionIndex: -1 };
    }

    const battle = player.privateBattle;
    const request = battle.request as AnyObject;
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

    if (moves !== undefined && attacker !== null && defender !== null) {
        const moveData: number[] = moves.map(
            ({ id, disabled }: { id: string; disabled: boolean }) => {
                let damage = 0;
                damage = GetMoveDamange({
                    battle,
                    attacker,
                    defender,
                    moveId: id,
                });
                return disabled ? -100 : damage;
            },
        );
        const indexOfLargestNum = moveData.reduce(
            (maxIndex, currentElement, currentIndex, arr) => {
                return currentElement > arr[maxIndex] ? currentIndex : maxIndex;
            },
            0,
        );
        return { actionIndex: indexOfLargestNum };
    }

    return { actionIndex: -1 };
};
