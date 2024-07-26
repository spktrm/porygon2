import { AnyObject } from "@pkmn/sim";
import { evalFuncArgs } from "../eval";
import { Battle, Pokemon, Side } from "@pkmn/client";
import { BoostID, Type, TypeName, StatusName } from "@pkmn/dex";
import { Request } from "@pkmn/protocol";
import { Move as DexMove } from "@pkmn/dex-types";
import { StreamHandler } from "../handler";
import { GetMoveDamange } from "./max_dmg";

const TYPE_DAMAGE_MAPPING: { [key: string]: number } = {
    0: 1,
    1: 2,
    2: 0.5,
    3: 0,
};

const ENTRY_HAZARDS = ["spikes", "stealthrock", "stickyweb", "toxicspikes"];
const ANTI_HAZARD_MOVES = ["rapidspin", "defog"];

function ComputeTypeMatchup(types1: Type[], types2: Type[]) {
    return types1.map((t1) =>
        types2
            .map((t2) => TYPE_DAMAGE_MAPPING[t2.damageTaken[t1.name]])
            .reduce((a, b) => a * b, 1),
    );
}

const SPEED_TIER_COEFFECIENT = 0.1;
const STATUS_COEFFECIENT_MAPPING: { [k in StatusName]: number } = {
    slp: 0.3,
    psn: 0.05,
    brn: 0.1,
    frz: 0.3,
    par: 0.0025,
    tox: 0.2,
};
const HP_FRACTION_COEFICIENT = 0.8;

function MatchupPokemon(args: {
    battle: Battle;
    attacker: Pokemon;
    defender: Pokemon;
}) {
    const { battle, attacker, defender } = args;

    const attackerTypes = attacker.types.map(
        (t) => attacker.species.dex.types.get(t) as Type,
    );
    const defenderTypes = defender.types.map(
        (t) => defender.species.dex.types.get(t) as Type,
    );

    let score =
        Math.max(...ComputeTypeMatchup(attackerTypes, defenderTypes)) -
        Math.max(...ComputeTypeMatchup(defenderTypes, attackerTypes));

    const CalculateSpeedStat = (pokemon: Pokemon) => {
        return (
            pokemon.baseSpecies.baseStats.spe *
            (pokemon.status === "par" ? 0.25 : 1)
        );
    };

    const attackerSpeed = CalculateSpeedStat(attacker);
    const defenderSpeed = CalculateSpeedStat(defender);

    if (attackerSpeed > defenderSpeed) {
        score += SPEED_TIER_COEFFECIENT;
    } else if (attackerSpeed < defenderSpeed) {
        score -= SPEED_TIER_COEFFECIENT;
    }

    const CalculateHpRatio = (pokemon: Pokemon) => {
        return (
            (pokemon.baseSpecies.baseStats.hp / 255) *
            (pokemon.hp / pokemon.maxhp) *
            HP_FRACTION_COEFICIENT
        );
    };

    const CalculateStatRatio = (attacker: Pokemon, defender: Pokemon) => {
        return (
            attacker.baseSpecies.baseStats.atk /
                defender.baseSpecies.baseStats.def +
            attacker.baseSpecies.baseStats.spa /
                defender.baseSpecies.baseStats.spd
        );
    };

    score +=
        (CalculateHpRatio(attacker) + CalculateStatRatio(attacker, defender)) /
        3;
    score -=
        (CalculateHpRatio(defender) + CalculateStatRatio(defender, attacker)) /
        3;

    if (defender.status) {
        score += STATUS_COEFFECIENT_MAPPING[defender.status] ?? 0;
    }
    if (attacker.status) {
        score -= STATUS_COEFFECIENT_MAPPING[attacker.status] ?? 0;
    }

    return score;
}

const getNumRemainingMons: (side: Side) => number = (side) => {
    return side.team.map((x) => +!x.fainted).reduce((a, b) => a + b, 0);
};

const SLEEP_TALK_PRIORITY = 5;
const STATUS_PRIORITY = 3;
const ENTRY_HAZARD_PRIORITY = 1;
const KNOCKOFF_PRIORITY = 2;
const BOOST_PRIORITY = 0;

function calculateStatusPriority(args: {
    attacker: Pokemon;
    defender: Pokemon;
}): number {
    return STATUS_PRIORITY;
}

function calculateHealRatio(args: {
    battle: Battle;
    attacker: Pokemon;
    defender: Pokemon;
    moveData: DexMove;
}): number {
    const { battle, attacker, defender, moveData } = args;

    const [healRatioNum, healRatioDenom] = moveData.heal ?? [];
    const healRatio = healRatioNum / healRatioDenom;
    // const currentHpRatio = attacker.hp / attacker.maxhp;
    // const opponentHpRatio = defender.hp / defender.maxhp;
    const activeWeather = battle.field.weather;

    switch (moveData.id) {
        case "synthesis":
        case "moonlight":
        case "morningsun":
            if (
                activeWeather === undefined ||
                activeWeather === "Strong Winds"
            ) {
                return 0.5;
            }
            if (activeWeather === "Sun" || activeWeather === "Harsh Sunshine")
                return 2 / 3;
            return 0.25;
        case "gigadrain":
            const damage = GetMoveDamange({
                battle,
                attacker,
                defender,
                moveId: moveData.id,
            });
            return (0.5 * damage) / defender.maxhp;
        case "wish":
            return 0.5;
        case "rest":
            return 1;
        default:
            if (isNaN(healRatio)) {
                return 1;
            } else {
                return healRatio;
            }
    }
}

const calcMovePrior: (args: {
    handler: StreamHandler;
    move: Request.ActivePokemon["moves"][0];
    attacker: Pokemon;
    defender: Pokemon;
    score: number;
}) => number = ({ handler, move, attacker, defender, score }) => {
    const battle = handler.privatebattle;
    const { id } = move;
    const attackerRecentDamage = handler.eventHandler.getRecentDamage(
        attacker.ident,
    );

    const moveData = battle.gens.dex.moves.get(id);
    const moveAccuracy =
        moveData.accuracy === true ? 1 : moveData.accuracy / 100;
    const defenderAbilities = Object.values(defender.species.abilities);

    const boosts = moveData.boosts ?? {};
    const attackerHpRatio = attacker.hp / attacker.maxhp;

    const nRemainingMons = getNumRemainingMons(attacker.side);
    const nOppRemainingMons = getNumRemainingMons(defender.side);

    if (
        id === "sleeptalk" &&
        attacker.status === "slp" &&
        attacker.statusState.sleepTurns < 3
    ) {
        return SLEEP_TALK_PRIORITY;
    }
    if (
        id === "knockoff" &&
        defender.item === "" &&
        !(defender.itemEffect === "knocked off")
    ) {
        return KNOCKOFF_PRIORITY;
    }
    if (
        id === "substitute" &&
        !!!attacker.volatiles.substitute &&
        attackerRecentDamage < 0.25 &&
        attackerHpRatio >= 0.5
    ) {
        return KNOCKOFF_PRIORITY;
    }
    if (attackerHpRatio < 1) {
        if (id === "protect" && attacker.lastMove === "wish") {
            return KNOCKOFF_PRIORITY;
        }
        if (moveData.heal) {
            const healRatio = calculateHealRatio({
                battle,
                attacker,
                defender,
                moveData,
            });
            if (
                attackerHpRatio + attackerRecentDamage > 0 &&
                healRatio >= attackerRecentDamage &&
                attackerHpRatio + healRatio < 1.25 &&
                Math.random() <= moveAccuracy
            ) {
                return KNOCKOFF_PRIORITY;
            }
        }
    }

    if (
        moveData.status &&
        moveData.category === "Status" &&
        !defenderAbilities.includes("Guts") &&
        defender.status === undefined &&
        !Object.keys(defender.volatiles).includes("substitute")
    ) {
        const hasTypes = (defender: Pokemon, types: TypeName[]) => {
            return types.some((type) => defender.types.includes(type));
        };

        const hasAbility = (defender: Pokemon, abilities: string[]) => {
            return abilities.some((ability) =>
                defenderAbilities.includes(ability),
            );
        };

        if (
            ((["tox", "psn"].includes(moveData.status) &&
                !hasTypes(defender, ["Steel", "Poison"])) ||
                (moveData.status === "brn" &&
                    !hasAbility(defender, ["Water Bubble", "Water Veil"]) &&
                    !hasTypes(defender, ["Fire"])) ||
                (moveData.status === "slp" &&
                    !hasAbility(defender, ["Vital Spirit", "Insomnia"])) ||
                (moveData.status === "par" &&
                    !hasAbility(defender, ["Limber"]) &&
                    (moveData.type === "Grass"
                        ? true
                        : !hasTypes(defender, ["Ground", "Electric"]))) ||
                (moveData.status === "frz" && !hasTypes(defender, ["Ice"]))) &&
            defender.types.every(
                (type) =>
                    TYPE_DAMAGE_MAPPING[
                        battle.gen.dex.types.get(type).damageTaken[
                            moveData.type
                        ]
                    ] !== 0,
            )
        ) {
            return Math.random() < moveAccuracy
                ? calculateStatusPriority({ attacker, defender })
                : -100;
        }
    }

    const { target } = moveData;
    if (
        score > 0 &&
        Object.keys(boosts).length > 0 &&
        Object.values(boosts).reduce((a, b) => a + b) >= 1 &&
        target === "self" &&
        Math.min(
            ...Object.entries(boosts)
                .filter(([_, v]) => v > 0)
                .map(([k, _]) => attacker.boosts[k as BoostID] ?? 0),
        ) < 6
    ) {
        return BOOST_PRIORITY;
    }
    if (nOppRemainingMons >= 3 && ENTRY_HAZARDS.includes(id)) {
        if (attackerHpRatio >= 0.9) {
            if (
                id === "spikes" &&
                ((defender.side.sideConditions[id] ?? {}).level ?? 0) < 3
            ) {
                return ENTRY_HAZARD_PRIORITY;
            } else if (
                !Object.keys(defender.side.sideConditions).includes(id)
            ) {
                return ENTRY_HAZARD_PRIORITY;
            }
        }
    } else if (
        Object.keys(attacker.side.sideConditions).length > 0 &&
        ANTI_HAZARD_MOVES.includes(id) &&
        nRemainingMons >= 2
    ) {
        return ENTRY_HAZARD_PRIORITY + 1;
    }

    return -100;
};

type SwitcherEvalActionFnType = (
    args: evalFuncArgs & {
        switchThreshold?: number;
        boostThresold?: number;
    },
) => number;

export const GetBestSwitchAction: SwitcherEvalActionFnType = ({
    handler,
    switchThreshold = 0,
    boostThresold = 0,
}) => {
    const battle = handler.privatebattle;
    const request = battle.request as AnyObject;
    const active = request.active ?? [];
    const moves: Request.ActivePokemon["moves"] = active[0]?.moves;

    const playerIndex = handler.getPlayerIndex();
    if (playerIndex === undefined) {
        throw new Error();
    }

    const mySide = battle.sides[playerIndex];
    const oppSide = battle.sides[1 - playerIndex];

    const attacker = mySide.active[0];
    const defender = oppSide.active[0];

    const anySwitchesLeft = mySide.team.filter((t) => !t.fainted).length > 1;

    const MaximumBench = (defender: Pokemon) => {
        const benchScores = mySide.team.flatMap((bench, benchIndex) =>
            bench.fainted || bench.isActive()
                ? []
                : {
                      benchIndex,
                      score: MatchupPokemon({
                          battle,
                          attacker: bench,
                          defender,
                      }),
                  },
        );
        return benchScores.reduce((a, b) => (a.score > b.score ? a : b));
    };

    if (attacker === null && defender !== null && anySwitchesLeft) {
        const maxBench = MaximumBench(defender);
        return 4 + maxBench.benchIndex;
    }

    if (attacker !== null && defender !== null) {
        const score = MatchupPokemon({
            battle,
            attacker,
            defender,
        });
        if (score < switchThreshold || attacker.fainted) {
            if (
                !!!attacker.trapped &&
                !!!attacker.maybeTrapped &&
                anySwitchesLeft
            ) {
                const maxBench = MaximumBench(defender);
                if (maxBench.score > score || attacker.fainted) {
                    return 4 + maxBench.benchIndex;
                }
            }
        }
        if (moves !== undefined) {
            const movePriorities = moves
                .flatMap((move, moveIndex) =>
                    move.disabled
                        ? []
                        : {
                              move,
                              moveIndex,
                              prior: calcMovePrior({
                                  handler,
                                  move,
                                  attacker,
                                  defender,
                                  score,
                              }),
                          },
                )
                .filter(({ prior }) => prior >= 0);

            if (movePriorities.length > 0) {
                const maxMove = movePriorities.reduce((a, b) =>
                    a.prior > b.prior ? a : b,
                );
                return maxMove.moveIndex;
            }

            const moveData: number[] = moves.map(({ id, disabled }) => {
                let damage = 0;
                try {
                    damage = GetMoveDamange({
                        battle,
                        attacker,
                        defender,
                        moveId: id,
                    });
                } catch (err) {
                    throw err;
                }
                return disabled ? -100 : damage;
            });
            const indexOfLargestNum = moveData.reduce(
                (maxIndex, currentElement, currentIndex, arr) => {
                    return currentElement > arr[maxIndex]
                        ? currentIndex
                        : maxIndex;
                },
                0,
            );
            return indexOfLargestNum;
        }
    }

    return -1;
};
