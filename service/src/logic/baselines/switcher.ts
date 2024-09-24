import { AnyObject } from "@pkmn/sim";
import { evalFuncArgs } from "../eval";
import { Battle, Pokemon, Side } from "@pkmn/client";
import { BoostID, Type, TypeName, StatusName, StatID } from "@pkmn/dex";
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

const STATUS_COEFFICIENT_MAPPING: { [k in StatusName]: number } = {
    slp: 0.4, // Sleep is very impactful due to incapacitation
    psn: 0.1, // Regular Poison is less impactful than other conditions
    brn: 0.3, // Burn is significant due to damage and attack reduction
    frz: 0.4, // Freeze is highly impactful due to complete immobilization
    par: 0.2, // Paralysis affects speed and can prevent actions
    tox: 0.3, // Toxic is more impactful than regular poison due to scaling damage
};

const WEIGHTS = {
    typeMatchup: 0.35,
    speed: 0.05,
    hpRatio: 0.15,
    statRatio: 0.2,
    statusEffect: 0.1,
    abilityEffect: 0.05,
    itemEffect: 0.05,
    weatherEffect: 0.05,
    moveset: 0.2,
};

function calculateStat(pokemon: Pokemon, stat: Exclude<StatID, "hp">) {
    const baseStat = pokemon.baseSpecies.baseStats[stat];
    const iv = 0;
    const ev = 85;
    const boostStage = pokemon.boosts[stat] ?? 0;
    const boostMulti =
        boostStage <= 0 ? (boostStage + 8) / 8 : (boostStage + 2) / 2;
    const value = Math.floor(
        Math.floor(
            ((2 * baseStat + iv + Math.floor(ev / 4)) * pokemon.level) / 100 +
                5,
        ),
    );
    return boostMulti * value;
}

function MatchupPokemon(args: {
    privateBattle: Battle;
    publicBattle: Battle;
    attacker: Pokemon;
    defender: Pokemon;
}) {
    const { privateBattle, publicBattle, attacker, defender } = args;

    const attackerTypes = attacker.types.map(
        (t) => attacker.species.dex.types.get(t) as Type,
    );
    const defenderTypes = defender.types.map(
        (t) => defender.species.dex.types.get(t) as Type,
    );

    const calculateTypeMatchup = (
        attackerTypes: Type[],
        defenderTypes: Type[],
    ) => {
        return (
            Math.max(...ComputeTypeMatchup(attackerTypes, defenderTypes)) -
            Math.max(...ComputeTypeMatchup(defenderTypes, attackerTypes))
        );
    };

    const calculateSpeedStat = (pokemon: Pokemon) => {
        return (
            calculateStat(pokemon, "spe") *
            (pokemon.status === "par" ? 0.25 : 1)
        );
    };

    const calculateSpeedAdvantage = (attacker: Pokemon, defender: Pokemon) => {
        const attackerSpeed = calculateSpeedStat(attacker);
        const defenderSpeed = calculateSpeedStat(defender);
        return attackerSpeed > defenderSpeed
            ? 1
            : attackerSpeed < defenderSpeed
            ? -1
            : 0;
    };

    const calculateHpRatio = (pokemon: Pokemon) => {
        return (2 * pokemon.hp) / pokemon.maxhp - 1;
    };

    const calculateStatRatio = (attacker: Pokemon, defender: Pokemon) => {
        const attackerATK = calculateStat(attacker, "atk");
        const attackerSPA = calculateStat(attacker, "spa");
        const defenderDEF = calculateStat(defender, "def");
        const defenderSPD = calculateStat(defender, "spd");
        return (attackerATK / defenderDEF + attackerSPA / defenderSPD) / 2;
    };

    const calculateStatusEffect = (pokemon: Pokemon) => {
        if (pokemon.status) {
            return STATUS_COEFFICIENT_MAPPING[pokemon.status] ?? 0;
        }
        return 0;
    };

    const calculateAbilityEffect = (pokemon: Pokemon) => {
        // Add logic to calculate the effect of abilities if any
        return 0;
    };

    const calculateItemEffect = (pokemon: Pokemon) => {
        // Add logic to calculate the effect of items if any
        return 0;
    };

    const calculateWeatherEffect = (pokemon: Pokemon) => {
        // Add logic to calculate the effect of the current weather
        return 0;
    };

    const calculateMovesetEffectiveness = (
        attacker: Pokemon,
        defender: Pokemon,
    ) => {
        let numDamagingMoves = 0;
        const publicAttacker = publicBattle.getPokemon(attacker.ident);
        if (!publicAttacker) {
            return 0;
        }
        const defenderHp = calculateHpValue(defender);
        const defenderRatio = defender.hp / defender.maxhp;
        const damageRatios = attacker.moves
            .map((move) => {
                if (["selfdestruct", "explosion"].includes(move)) {
                    return 0;
                }
                let damage = 0;
                try {
                    damage = GetMoveDamange({
                        battle: privateBattle,
                        attacker,
                        defender,
                        moveId: move,
                    });
                } catch (err) {
                    // throw Error(move);
                }
                if (damage > 0) {
                    numDamagingMoves += 1;
                }
                return damage / defenderHp;
            })
            .filter((damageRatio) => damageRatio > 0);
        const bias = damageRatios.some(
            (damageRatio) => defenderRatio - damageRatio <= 0,
        )
            ? 10
            : 0;
        return (
            damageRatios.reduce((a, b) => a + b, 0) /
                Math.max(1, numDamagingMoves) +
            bias
        );
    };

    const typeMatchupScore = calculateTypeMatchup(attackerTypes, defenderTypes);
    const speedScore = calculateSpeedAdvantage(attacker, defender);
    const hpRatioScore =
        calculateHpRatio(attacker) - calculateHpRatio(defender);
    const statRatioScore =
        calculateStatRatio(attacker, defender) -
        calculateStatRatio(defender, attacker);
    const statusEffectScore =
        calculateStatusEffect(defender) - calculateStatusEffect(attacker);
    const abilityEffectScore =
        calculateAbilityEffect(attacker) - calculateAbilityEffect(defender);
    const itemEffectScore =
        calculateItemEffect(attacker) - calculateItemEffect(defender);
    const weatherEffectScore =
        calculateWeatherEffect(attacker) - calculateWeatherEffect(defender);

    const movesetScore =
        calculateMovesetEffectiveness(attacker, defender) -
        calculateMovesetEffectiveness(defender, attacker);

    const totalScore =
        WEIGHTS.typeMatchup * typeMatchupScore +
        WEIGHTS.speed * speedScore +
        WEIGHTS.hpRatio * hpRatioScore +
        WEIGHTS.statRatio * statRatioScore +
        WEIGHTS.statusEffect * statusEffectScore +
        WEIGHTS.abilityEffect * abilityEffectScore +
        WEIGHTS.itemEffect * itemEffectScore +
        WEIGHTS.weatherEffect * weatherEffectScore +
        WEIGHTS.moveset * movesetScore;

    return totalScore;
}

const getNumRemainingMons: (side: Side) => number = (side) => {
    return side.team.map((x) => +!x.fainted).reduce((a, b) => a + b, 0);
};

const SLEEP_TALK_PRIORITY = 5;
const STATUS_PRIORITY = 3;
const ENTRY_HAZARD_PRIORITY = 1;
const ENCORE_PRIORITY = 1;
const LEECHSEED_PRIORITY = 1;
const KNOCKOFF_PRIORITY = 2;
const BOOST_PRIORITY = 0;
const HEALBELL_PRIORITY = -1;
const HAZE_PRIORITY = -1;

function calculateHpValue(pokemon: Pokemon) {
    return (
        Math.floor(
            ((2 * pokemon.baseSpecies.baseStats.hp + 31 + Math.floor(85 / 4)) *
                pokemon.level) /
                100,
        ) +
        pokemon.level +
        10
    );
}

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

    const attackerHp = attacker.hp;
    const defenderHp = calculateHpValue(defender);

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
            return (0.5 * damage) / defenderHp;
        case "wish":
            return 0.5;
        case "rest":
            return 1;
        case "painsplit":
            const avgHp = (attackerHp + defenderHp) / 2;
            return (avgHp - attackerHp) / attacker.maxhp;
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
    attackerScore: number;
}) => number = ({ handler, move, attacker, defender, attackerScore }) => {
    const battle = handler.privateBattle;
    const { id } = move;
    const attackerRecentDamage = 0; // TODO: decouple this from general state creation

    const moveData = battle.gens.dex.moves.get(id);
    const moveAccuracy =
        moveData.accuracy === true ? 1 : moveData.accuracy / 100;
    const defenderAbilities = Object.values(defender.species.abilities);

    const boosts = moveData.boosts ?? {};
    const attackerHpRatio = attacker.hp / attacker.maxhp;
    const defenderHpRatio = defender.hp / defender.maxhp;

    const nRemainingMons = getNumRemainingMons(attacker.side);
    const nOppRemainingMons = getNumRemainingMons(defender.side);

    const { atk: defenderAtk, spa: defenderSpa } =
        defender.baseSpecies.baseStats;

    switch (id) {
        case "meanlook":
            if (!defender.volatiles.trapped) {
                return 1.5;
            }
            break;
        case "perishsong":
            if (
                !defender.volatiles["perishsong"] &&
                !Object.keys(defender.volatiles).some((volatile) =>
                    volatile.startsWith("perish"),
                )
            )
                return 1;
            break;
        case "spiderweb":
            if (!defender.volatiles["spiderweb"] && attackerScore > 0) {
                return 2;
            }
            break;
        case "trick":
            if (attacker.item.startsWith("choice")) {
                return HEALBELL_PRIORITY;
            }
            break;
        case "yawn":
            if (!defender.volatiles["yawn"]) {
                return 0;
            }
        case "endure":
            if (
                attackerHpRatio + attackerRecentDamage < 0 &&
                attackerScore < 0
            ) {
                return 0;
            }
            break;
        case "destinybond":
            if (attackerHpRatio + attackerRecentDamage < 0) {
                return 0;
            }
            break;
        case "raindance":
        case "sunnyday":
            if (moveData.weather) {
                const weather = battle.field.weather;
                const weatherState = battle.field.weatherState;
                if (
                    (id === "raindance" &&
                        (weather !== "Rain" ||
                            weatherState.maxDuration === 1)) ||
                    (id === "sunnyday" &&
                        (weather !== "Sun" || weatherState.maxDuration === 1))
                ) {
                    return 2;
                }
            }
            break;
        case "refresh":
            if (["brn", "tox", "par"].some((s) => s === attacker.status)) {
                return HEALBELL_PRIORITY;
            }
            break;
        case "roar":
        case "whirlwind":
            if (attackerHpRatio + attackerRecentDamage > 0) {
                return HEALBELL_PRIORITY;
            }
            if (
                Object.keys(defender.side.sideConditions).some(
                    (sideCondition) =>
                        sideCondition === "spikes" ||
                        sideCondition === "stealthrock" ||
                        sideCondition === "toxicspikes",
                )
            ) {
                return HEALBELL_PRIORITY + 1;
            }
            break;
        case "leechseed":
            if (!defender.volatiles["leechseed"]) {
                return LEECHSEED_PRIORITY;
            }
            break;
        case "healbell":
            if (
                attacker.side.team.some((member) => {
                    const { status } = member;
                    return (
                        (status === "slp" && !member.isActive()) ||
                        status !== undefined
                    );
                })
            ) {
                return HEALBELL_PRIORITY;
            }
            break;
        case "encore":
            if (!defender.volatiles["encore"]) {
                const defenderLastMoveData = battle.gens.dex.moves.get(
                    defender.lastMove,
                );
                if (
                    defenderLastMoveData.status &&
                    attacker.baseSpecies.baseStats.spe >
                        defender.baseSpecies.baseStats.spe
                ) {
                    return ENCORE_PRIORITY;
                }
            }
            break;
        case "haze":
            if (
                Object.values(defender.boosts ?? {}).reduce(
                    (a, b) => a + b,
                    0,
                ) > 0 ||
                Object.values(attacker.boosts ?? {}).reduce(
                    (a, b) => a + b,
                    0,
                ) < 0
            ) {
                return HAZE_PRIORITY;
            }
            break;
        case "mirrorcoat":
            return defenderSpa > defenderAtk ? 1 : -100;
        case "counter":
            return defenderAtk > defenderSpa ? 1 : -100;
        case "sleeptalk":
            if (
                attacker.status === "slp" &&
                attacker.statusState.sleepTurns < 3
            ) {
                return SLEEP_TALK_PRIORITY;
            }
            break;
        case "knockoff":
            if (defender.item === "" && defender.itemEffect !== "knocked off") {
                return KNOCKOFF_PRIORITY;
            }
            break;
        case "explosion":
        case "selfdestruct":
            if (attackerHpRatio + attackerRecentDamage <= 0) {
                return 1;
            }
            break;
        case "substitute":
            if (
                !attacker.volatiles.substitute &&
                -attackerRecentDamage < 0.25 &&
                attackerHpRatio >= 0.5
            ) {
                if (attacker.lastMove === "substitute") {
                    if (Math.random() < 0.5) {
                        return KNOCKOFF_PRIORITY;
                    }
                } else {
                    return KNOCKOFF_PRIORITY;
                }
            }
            break;
        case "protect":
            if (attackerHpRatio < 1) {
                if (attacker.lastMove === "wish") {
                    return KNOCKOFF_PRIORITY;
                }
                if (
                    moveData.flags.heal ||
                    (id === "painsplit" &&
                        attackerHpRatio < defenderHpRatio &&
                        !defender.volatiles.substitute)
                ) {
                    const healRatio = calculateHealRatio({
                        battle,
                        attacker,
                        defender,
                        moveData,
                    });
                    if (
                        attackerHpRatio +
                            (attacker.status === "tox"
                                ? 1.5 * attackerRecentDamage
                                : attackerRecentDamage) <=
                            0 &&
                        healRatio >= -attackerRecentDamage &&
                        attackerHpRatio + healRatio < 1.3 &&
                        Math.random() <= moveAccuracy
                    ) {
                        return KNOCKOFF_PRIORITY;
                    }
                }
            }
            break;
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
                    !hasAbility(defender, ["Vital Spirit", "Insomnia"]) &&
                    (defender.side.team.every(
                        (member) => member.status !== "slp",
                    ) ||
                        defender.statusState.sleepTurns === 2)) ||
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
        attackerHpRatio + attackerRecentDamage > 0 &&
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
                (defender.side.sideConditions[id]?.level ?? 0) < 3
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
) => Promise<number>;

const scores = [];

export const GetBestSwitchAction: SwitcherEvalActionFnType = async ({
    handler,
    switchThreshold = 0,
    boostThresold = 0,
}) => {
    const battle = handler.privateBattle;
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
                          privateBattle: handler.privateBattle,
                          publicBattle: handler.publicBattle,
                          attacker: bench,
                          defender,
                      }),
                  },
        );
        scores.push(...benchScores.map(({ score }) => score));
        return benchScores.reduce((a, b) => (a.score > b.score ? a : b));
    };

    if (attacker === null && defender !== null && anySwitchesLeft) {
        const maxBench = MaximumBench(defender);
        return 4 + maxBench.benchIndex;
    }

    if (attacker !== null && defender !== null) {
        const score = MatchupPokemon({
            privateBattle: handler.privateBattle,
            publicBattle: handler.publicBattle,
            attacker,
            defender,
        });
        let lastMoveDamage = 0;
        if (attacker.lastMove) {
            const lastMove =
                attacker.lastMove === "hiddenpower"
                    ? attacker.moveSlots.find((x) =>
                          x.id.startsWith("hiddenpower"),
                      )?.id ?? ""
                    : attacker.lastMove;
            try {
                lastMoveDamage = GetMoveDamange({
                    battle,
                    attacker,
                    defender,
                    moveId: lastMove,
                });
            } catch (err) {
                // throw err;
            }
        }
        if (
            score < switchThreshold ||
            (attacker.lastMove &&
                attacker.item.startsWith("choice") &&
                attacker.lastMove !== "switch-in" &&
                lastMoveDamage <= 0)
        ) {
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
                if (["selfdestruct", "explosion"].includes(id)) {
                    return -99;
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

            if (
                !["frz", "slp"].includes(defender.status ?? "") &&
                moveData.some(
                    (damage) =>
                        (defender.hp * calculateHpValue(defender)) /
                            defender.maxhp -
                            damage <=
                        0,
                )
            ) {
                return indexOfLargestNum;
            }

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
                                  attackerScore: score,
                              }),
                          },
                )
                .filter(({ prior }) => prior >= -99);

            if (movePriorities.length > 0) {
                const maxMove = movePriorities.reduce((a, b) =>
                    a.prior > b.prior ? a : b,
                );
                return maxMove.moveIndex;
            }

            return indexOfLargestNum;
        }
    }

    return -1;
};
