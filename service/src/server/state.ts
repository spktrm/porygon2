import { AnyObject } from "@pkmn/sim";
import {
    Args,
    BattleMajorArgName,
    BattleMinorArgName,
    BattleProgressArgName,
    KWArgs,
    PokemonIdent,
    Protocol,
} from "@pkmn/protocol";
import {
    AbilitiesEnum,
    BattlemajorargsEnum,
    BattleminorargsEnum,
    EffectEnum,
    EffecttypesEnum,
    GendernameEnum,
    ItemeffecttypesEnum,
    ItemsEnum,
    MovesEnum,
    SideconditionEnum,
    SpeciesEnum,
    StatusEnum,
    TypechartEnum,
    VolatilestatusEnum,
    WeatherEnum,
} from "../../protos/enums_pb";
import {
    EnumMappings,
    MoveIndex,
    NUM_HISTORY,
    jsonDatum,
    numEntityEdgeFeatures,
    numEntityNodeFeatures,
    numFieldFeatures,
    numInfoFeatures,
    numMoveFeatures,
    numMovesetFeatures,
} from "./data";
import { NA, Pokemon, Side } from "@pkmn/client";
import { Ability, Item, Move, BoostID } from "@pkmn/dex-types";
import { ID, MoveTarget, SideID } from "@pkmn/types";
import { Condition, Effect } from "@pkmn/data";
import { OneDBoolean, TypedArray } from "./utils";
import {
    EntityEdgeFeature,
    EntityEdgeFeatureMap,
    EntityNodeFeature,
    FieldFeature,
    FieldFeatureMap,
    InfoFeature,
    MovesetActionTypeEnum,
    MovesetFeature,
    MovesetHasPPEnum,
} from "../../protos/features_pb";
import { TrainablePlayerAI } from "./runner";
import { EnvironmentState } from "../../protos/service_pb";

type RemovePipes<T extends string> = T extends `|${infer U}|` ? U : T;
type MajorArgNames =
    | RemovePipes<BattleMajorArgName>
    | RemovePipes<BattleProgressArgName>;
type MinorArgNames = RemovePipes<BattleMinorArgName>;

const MAX_RATIO_TOKEN = 16384;

function int16ArrayToBitIndices(arr: Int16Array): number[] {
    const indices: number[] = [];

    for (let i = 0; i < arr.length; i++) {
        let num = arr[i];

        // Process each of the 16 bits in the int16 value
        for (let bitPosition = 0; bitPosition < 16; bitPosition++) {
            // Check if the least significant bit is 1
            if ((num & 1) !== 0) {
                indices.push(i * 16 + bitPosition); // Calculate the bit index
            }

            // Right shift the number to check the next bit
            num >>>= 1;
        }
    }

    return indices;
}

function bigIntToInt16Array(value: bigint): Int16Array {
    // Determine the number of 16-bit chunks needed to store the BigInt
    const bitSize = value.toString(2).length; // Number of bits in the BigInt
    const chunkCount = Math.ceil(bitSize / 16);

    // Create an Int16Array to store the chunks
    const result = new Int16Array(chunkCount);

    // Mask to extract 16 bits
    const mask = BigInt(0xffff);

    for (let i = 0; i < chunkCount; i++) {
        // Extract the lower 16 bits
        result[i] = Number(value & mask);
        // Shift the BigInt to the right by 16 bits
        value >>= BigInt(16);
    }

    return result;
}

const entityNodeArrayToObject = (array: Int16Array) => {
    const volatilesFlat = array.slice(
        EntityNodeFeature.ENTITY_NODE_FEATURE__VOLATILES0,
        EntityNodeFeature.ENTITY_NODE_FEATURE__VOLATILES8 + 1,
    );
    const volatilesIndices = int16ArrayToBitIndices(volatilesFlat);

    const typechangeFlat = array.slice(
        EntityNodeFeature.ENTITY_NODE_FEATURE__TYPECHANGE0,
        EntityNodeFeature.ENTITY_NODE_FEATURE__TYPECHANGE1 + 1,
    );
    const typechangeIndices = int16ArrayToBitIndices(typechangeFlat);

    const moveIndicies = Array.from(
        array.slice(
            EntityNodeFeature.ENTITY_NODE_FEATURE__MOVEID0,
            EntityNodeFeature.ENTITY_NODE_FEATURE__MOVEID3 + 1,
        ),
    );

    return {
        species:
            jsonDatum["species"][
                array[EntityNodeFeature.ENTITY_NODE_FEATURE__SPECIES]
            ],
        item: jsonDatum["items"][
            array[EntityNodeFeature.ENTITY_NODE_FEATURE__ITEM]
        ],
        hp:
            array[EntityNodeFeature.ENTITY_NODE_FEATURE__HP_RATIO] /
            MAX_RATIO_TOKEN,
        fainted: !!array[EntityNodeFeature.ENTITY_NODE_FEATURE__FAINTED],
        ability:
            jsonDatum["abilities"][
                array[EntityNodeFeature.ENTITY_NODE_FEATURE__ABILITY]
            ],
        moves: moveIndicies.map((index) => jsonDatum["moves"][index]),
        volatiles: volatilesIndices.map(
            (index) => jsonDatum["volatileStatus"][index],
        ),
        typechange: typechangeIndices.map(
            (index) => jsonDatum["typechart"][index],
        ),
        active: array[EntityNodeFeature.ENTITY_NODE_FEATURE__ACTIVE],
        side: array[EntityNodeFeature.ENTITY_NODE_FEATURE__SIDE],
        status: jsonDatum["status"][
            array[EntityNodeFeature.ENTITY_NODE_FEATURE__STATUS]
        ],
    };
};

const entityEdgeArrayToObject = (array: Int16Array) => {
    const minorArgsFlat = array.slice(
        EntityEdgeFeature.ENTITY_EDGE_FEATURE__MINOR_ARG0,
        EntityEdgeFeature.ENTITY_EDGE_FEATURE__MINOR_ARG3 + 1,
    );
    const minorArgIndices = int16ArrayToBitIndices(minorArgsFlat);

    return {
        majorArg:
            jsonDatum["battleMajorArgs"][
                array[EntityEdgeFeature.ENTITY_EDGE_FEATURE__MAJOR_ARG]
            ],
        minorArgs: minorArgIndices.map(
            (index) => jsonDatum["battleMinorArgs"][index],
        ),
        move: jsonDatum["moves"][
            array[EntityEdgeFeature.ENTITY_EDGE_FEATURE__MOVE_TOKEN]
        ],
        damage:
            array[EntityEdgeFeature.ENTITY_EDGE_FEATURE__DAMAGE_RATIO] /
            MAX_RATIO_TOKEN,
        heal:
            array[EntityEdgeFeature.ENTITY_EDGE_FEATURE__HEAL_RATIO] /
            MAX_RATIO_TOKEN,
        num_from_sources:
            array[EntityEdgeFeature.ENTITY_EDGE_FEATURE__NUM_FROM_SOURCES],
        from_source: [
            jsonDatum["Effect"][
                array[EntityEdgeFeature.ENTITY_EDGE_FEATURE__FROM_SOURCE_TOKEN0]
            ],
            jsonDatum["Effect"][
                array[EntityEdgeFeature.ENTITY_EDGE_FEATURE__FROM_SOURCE_TOKEN1]
            ],
            jsonDatum["Effect"][
                array[EntityEdgeFeature.ENTITY_EDGE_FEATURE__FROM_SOURCE_TOKEN2]
            ],
            jsonDatum["Effect"][
                array[EntityEdgeFeature.ENTITY_EDGE_FEATURE__FROM_SOURCE_TOKEN3]
            ],
            jsonDatum["Effect"][
                array[EntityEdgeFeature.ENTITY_EDGE_FEATURE__FROM_SOURCE_TOKEN4]
            ],
        ],
        boosts: {
            EDGE_BOOST_ATK_VALUE:
                array[EntityEdgeFeature.ENTITY_EDGE_FEATURE__BOOST_ATK_VALUE],
            EDGE_BOOST_DEF_VALUE:
                array[EntityEdgeFeature.ENTITY_EDGE_FEATURE__BOOST_DEF_VALUE],
            EDGE_BOOST_SPA_VALUE:
                array[EntityEdgeFeature.ENTITY_EDGE_FEATURE__BOOST_SPA_VALUE],
            EDGE_BOOST_SPD_VALUE:
                array[EntityEdgeFeature.ENTITY_EDGE_FEATURE__BOOST_SPD_VALUE],
            EDGE_BOOST_SPE_VALUE:
                array[EntityEdgeFeature.ENTITY_EDGE_FEATURE__BOOST_SPE_VALUE],
            EDGE_BOOST_ACCURACY_VALUE:
                array[
                    EntityEdgeFeature.ENTITY_EDGE_FEATURE__BOOST_ACCURACY_VALUE
                ],
            EDGE_BOOST_EVASION_VALUE:
                array[
                    EntityEdgeFeature.ENTITY_EDGE_FEATURE__BOOST_EVASION_VALUE
                ],
        },
        status: jsonDatum["status"][
            array[EntityEdgeFeature.ENTITY_EDGE_FEATURE__STATUS_TOKEN]
        ],
    };
};

const fieldArrayToObject = (array: Int16Array) => {
    const mySideConditionsFlat = array.slice(
        FieldFeature.FIELD_FEATURE__MY_SIDECONDITIONS0,
        FieldFeature.FIELD_FEATURE__MY_SIDECONDITIONS1 + 1,
    );
    const mySideConditionsIndices =
        int16ArrayToBitIndices(mySideConditionsFlat);
    const oppSideConditionsFlat = array.slice(
        FieldFeature.FIELD_FEATURE__OPP_SIDECONDITIONS0,
        FieldFeature.FIELD_FEATURE__OPP_SIDECONDITIONS1 + 1,
    );
    const oppSideConditionsIndices = int16ArrayToBitIndices(
        oppSideConditionsFlat,
    );

    return {
        mySideConditions: mySideConditionsIndices.map((index) => {
            return jsonDatum["sideCondition"][index];
        }),
        myNumSpikes: array[FieldFeature.FIELD_FEATURE__MY_SPIKES],
        oppSideConditions: oppSideConditionsIndices.map((index) => {
            return jsonDatum["sideCondition"][index];
        }),
        oppNumSpikes: array[FieldFeature.FIELD_FEATURE__OPP_SPIKES],
        turnOrder: array[FieldFeature.FIELD_FEATURE__TURN_ORDER_VALUE],
        requestCount: array[FieldFeature.FIELD_FEATURE__REQUEST_COUNT],
        weatherId:
            jsonDatum["weather"][array[FieldFeature.FIELD_FEATURE__WEATHER_ID]],
        weatherMinDuration:
            array[FieldFeature.FIELD_FEATURE__WEATHER_MIN_DURATION],
        weatherMaxDuration:
            array[FieldFeature.FIELD_FEATURE__WEATHER_MAX_DURATION],
    };
};

const WEATHERS = {
    sand: "sandstorm",
    sun: "sunnyday",
    rain: "raindance",
    hail: "hail",
    snowscape: "snowscape",
    harshsunshine: "desolateland",
    heavyrain: "primordialsea",
    strongwinds: "deltastream",
};

function isMySide(n: number, playerIndex: number) {
    return +(n === playerIndex);
}

const enumDatumPrefixCache = new WeakMap<object, string>();
const sanitizeKeyCache = new Map<string, string>();

function getPrefix<T extends EnumMappings>(enumDatum: T): string | null {
    if (enumDatumPrefixCache.has(enumDatum)) {
        return enumDatumPrefixCache.get(enumDatum)!;
    }

    for (const key in enumDatum) {
        const prefix = key.split("__")[0];
        enumDatumPrefixCache.set(enumDatum, prefix);
        return prefix;
    }

    return null; // Handle cases where enumDatum has no keys
}

function SanitizeKey<T extends EnumMappings>(
    enumDatum: T,
    key: string,
): string {
    const prefix = getPrefix(enumDatum);
    if (!prefix) {
        throw new Error(
            "Prefix could not be determined for the given enumDatum",
        );
    }

    // Construct the raw key
    const rawKey = `${prefix}__${key}`;

    // Check if the sanitized key is cached
    if (sanitizeKeyCache.has(rawKey)) {
        return sanitizeKeyCache.get(rawKey)!;
    }

    // Sanitize the key (remove non-alphanumeric characters and make uppercase)
    const sanitizedKey = rawKey.replace(/\W/g, "").toUpperCase();

    // Cache the sanitized key
    sanitizeKeyCache.set(rawKey, sanitizedKey);
    return sanitizedKey;
}

export function IndexValueFromEnum<T extends EnumMappings>(
    enumDatum: T,
    key: string,
): T[keyof T] {
    const sanitizedKey = SanitizeKey(enumDatum, key) as keyof T;

    // Retrieve the value from the enumDatum using the sanitized key
    const value = enumDatum[sanitizedKey];
    if (value === undefined) {
        throw new Error(`${sanitizedKey.toString()} not in mapping`);
    }
    return value;
}

export function concatenateArrays<T extends TypedArray>(arrays: T[]): T {
    // Step 1: Calculate the total length
    let totalLength = 0;
    for (const arr of arrays) {
        totalLength += arr.length;
    }

    // Step 2: Create a new array using the constructor of the first array in the list
    const result = new (arrays[0].constructor as { new (length: number): T })(
        totalLength,
    );

    // Step 3: Copy each array into the result
    let offset = 0;
    for (const arr of arrays) {
        result.set(arr, offset);
        offset += arr.length;
    }

    return result;
}

const POKEMON_ARRAY_CONSTRUCTOR = Int16Array;

function getBlankPokemonArr() {
    return new POKEMON_ARRAY_CONSTRUCTOR(numEntityNodeFeatures);
}

function getUnkPokemon(n: number) {
    const data = getBlankPokemonArr();
    data[EntityNodeFeature.ENTITY_NODE_FEATURE__SPECIES] =
        SpeciesEnum.SPECIES_ENUM___UNK;
    data[EntityNodeFeature.ENTITY_NODE_FEATURE__ITEM] =
        ItemsEnum.ITEMS_ENUM___UNK;
    data[EntityNodeFeature.ENTITY_NODE_FEATURE__GENDER] =
        GendernameEnum.GENDERNAME_ENUM___UNK;
    data[EntityNodeFeature.ENTITY_NODE_FEATURE__ITEM_EFFECT] =
        ItemeffecttypesEnum.ITEMEFFECTTYPES_ENUM___NULL;
    data[EntityNodeFeature.ENTITY_NODE_FEATURE__ABILITY] =
        AbilitiesEnum.ABILITIES_ENUM___UNK;
    data[EntityNodeFeature.ENTITY_NODE_FEATURE__FAINTED] = 0;
    data[EntityNodeFeature.ENTITY_NODE_FEATURE__LEVEL] = 100;
    data[EntityNodeFeature.ENTITY_NODE_FEATURE__HP] = 100;
    data[EntityNodeFeature.ENTITY_NODE_FEATURE__MAXHP] = 100;
    data[EntityNodeFeature.ENTITY_NODE_FEATURE__HP_RATIO] = MAX_RATIO_TOKEN; // Full Health
    data[EntityNodeFeature.ENTITY_NODE_FEATURE__STATUS] =
        StatusEnum.STATUS_ENUM___NULL;
    data[EntityNodeFeature.ENTITY_NODE_FEATURE__MOVEID0] =
        MovesEnum.MOVES_ENUM___UNK;
    data[EntityNodeFeature.ENTITY_NODE_FEATURE__MOVEID1] =
        MovesEnum.MOVES_ENUM___UNK;
    data[EntityNodeFeature.ENTITY_NODE_FEATURE__MOVEID2] =
        MovesEnum.MOVES_ENUM___UNK;
    data[EntityNodeFeature.ENTITY_NODE_FEATURE__MOVEID3] =
        MovesEnum.MOVES_ENUM___UNK;
    data[EntityNodeFeature.ENTITY_NODE_FEATURE__MOVEPP0] = 0;
    data[EntityNodeFeature.ENTITY_NODE_FEATURE__MOVEPP1] = 0;
    data[EntityNodeFeature.ENTITY_NODE_FEATURE__MOVEPP2] = 0;
    data[EntityNodeFeature.ENTITY_NODE_FEATURE__MOVEPP3] = 0;
    data[EntityNodeFeature.ENTITY_NODE_FEATURE__LAST_MOVE] =
        MovesEnum.MOVES_ENUM___UNK;
    data[EntityNodeFeature.ENTITY_NODE_FEATURE__NUM_MOVES] = 4;
    data[EntityNodeFeature.ENTITY_NODE_FEATURE__HAS_STATUS] = 0;
    data[EntityNodeFeature.ENTITY_NODE_FEATURE__SIDE] = n;
    return data;
}

const unkPokemon0 = getUnkPokemon(0);
const unkPokemon1 = getUnkPokemon(1);

function getNullPokemon() {
    const data = getBlankPokemonArr();
    data[EntityNodeFeature.ENTITY_NODE_FEATURE__SPECIES] =
        SpeciesEnum.SPECIES_ENUM___NULL;
    return data;
}

const nullPokemon = getNullPokemon();

function getArrayFromPokemon(
    candidate: Pokemon | null,
    playerIndex: number,
    isPublic: boolean = true,
) {
    if (candidate === null) {
        return nullPokemon;
    }

    let pokemon: Pokemon;
    let isTransformed = false;
    if (
        candidate.volatiles.transform !== undefined &&
        candidate.volatiles.transform.pokemon !== undefined
    ) {
        pokemon = candidate.volatiles.transform.pokemon as Pokemon;
        isTransformed = true;
    } else {
        pokemon = candidate;
    }

    const baseSpecies = pokemon.species.baseSpecies.toLowerCase();
    const ability = pokemon.ability;

    // We take candidate item here instead of pokemons since
    // transformed does not copy item
    const item = candidate.item ?? candidate.lastItem;
    const itemEffect = candidate.itemEffect ?? candidate.lastItemEffect;

    // Moves are stored on candidate
    const moveSlots = candidate.moveSlots.slice(0, 4);
    const moveIds = [];
    const movePps = [];

    if (moveSlots) {
        for (const move of moveSlots) {
            const { id, ppUsed } = move;
            const maxPP = isTransformed
                ? 5
                : pokemon.side.battle.gens.dex.moves.get(id).pp;

            // Remove pp up assumption (5/8)
            const correctUsed =
                (isNaN(ppUsed) ? +!!ppUsed : ppUsed) *
                (isTransformed ? 1 : 5 / 8);

            moveIds.push(IndexValueFromEnum(MovesEnum, id));
            movePps.push(Math.floor((31 * correctUsed) / maxPP));
        }
    }
    let remainingIndex: MoveIndex = moveSlots.length as MoveIndex;

    const dataArr = getBlankPokemonArr();

    if (isPublic) {
        for (remainingIndex; remainingIndex < 4; remainingIndex++) {
            moveIds.push(MovesEnum.MOVES_ENUM___UNK);
            movePps.push(0);
        }
        dataArr[EntityNodeFeature.ENTITY_NODE_FEATURE__NUM_MOVES] = 4;
    } else {
        for (remainingIndex; remainingIndex < 4; remainingIndex++) {
            moveIds.push(MovesEnum.MOVES_ENUM___NULL);
            movePps.push(0);
        }
        dataArr[EntityNodeFeature.ENTITY_NODE_FEATURE__NUM_MOVES] =
            moveSlots.length;
    }

    dataArr[EntityNodeFeature.ENTITY_NODE_FEATURE__IS_PUBLIC] = +isPublic;
    dataArr[EntityNodeFeature.ENTITY_NODE_FEATURE__SPECIES] =
        IndexValueFromEnum<typeof SpeciesEnum>(SpeciesEnum, baseSpecies);
    dataArr[EntityNodeFeature.ENTITY_NODE_FEATURE__ITEM] = item
        ? IndexValueFromEnum<typeof ItemsEnum>(ItemsEnum, item)
        : ItemsEnum.ITEMS_ENUM___UNK;
    dataArr[EntityNodeFeature.ENTITY_NODE_FEATURE__ITEM_EFFECT] = itemEffect
        ? IndexValueFromEnum(ItemeffecttypesEnum, itemEffect)
        : ItemeffecttypesEnum.ITEMEFFECTTYPES_ENUM___NULL;

    const possibleAbilities = Object.values(pokemon.baseSpecies.abilities);
    if (ability) {
        const actualAbility = IndexValueFromEnum(AbilitiesEnum, ability);
        dataArr[EntityNodeFeature.ENTITY_NODE_FEATURE__ABILITY] = actualAbility;
    } else if (possibleAbilities.length === 1) {
        const onlyAbility = possibleAbilities[0]
            ? IndexValueFromEnum(AbilitiesEnum, possibleAbilities[0])
            : AbilitiesEnum.ABILITIES_ENUM___UNK;
        dataArr[EntityNodeFeature.ENTITY_NODE_FEATURE__ABILITY] = onlyAbility;
    } else {
        dataArr[EntityNodeFeature.ENTITY_NODE_FEATURE__ABILITY] =
            AbilitiesEnum.ABILITIES_ENUM___UNK;
    }

    // We take candidate lastMove here instead of pokemons since
    // transformed does not lastMove
    if (candidate.lastMove === "") {
        dataArr[EntityNodeFeature.ENTITY_NODE_FEATURE__LAST_MOVE] =
            MovesEnum.MOVES_ENUM___NULL;
    } else if (candidate.lastMove === "switch-in") {
        dataArr[EntityNodeFeature.ENTITY_NODE_FEATURE__LAST_MOVE] =
            MovesEnum.MOVES_ENUM___SWITCH;
    } else {
        dataArr[EntityNodeFeature.ENTITY_NODE_FEATURE__LAST_MOVE] =
            IndexValueFromEnum(MovesEnum, candidate.lastMove);
    }

    dataArr[EntityNodeFeature.ENTITY_NODE_FEATURE__GENDER] = IndexValueFromEnum(
        GendernameEnum,
        pokemon.gender,
    );
    dataArr[EntityNodeFeature.ENTITY_NODE_FEATURE__ACTIVE] =
        candidate.isActive() ? 1 : 0;
    dataArr[EntityNodeFeature.ENTITY_NODE_FEATURE__FAINTED] = candidate.fainted
        ? 1
        : 0;

    // We take candidate HP here instead of pokemons since
    // transformed does not copy HP
    const isHpBug = !candidate.fainted && candidate.hp === 0;
    const hp = isHpBug ? 100 : candidate.hp;
    const maxHp = isHpBug ? 100 : candidate.maxhp;
    const hpRatio = hp / maxHp;
    dataArr[EntityNodeFeature.ENTITY_NODE_FEATURE__HP] = hp;
    dataArr[EntityNodeFeature.ENTITY_NODE_FEATURE__MAXHP] = maxHp;
    dataArr[EntityNodeFeature.ENTITY_NODE_FEATURE__HP_RATIO] = Math.floor(
        MAX_RATIO_TOKEN * hpRatio,
    );

    // We take candidate status here instead of pokemons since
    // transformed does not copy status
    dataArr[EntityNodeFeature.ENTITY_NODE_FEATURE__STATUS] = candidate.status
        ? IndexValueFromEnum(StatusEnum, candidate.status)
        : StatusEnum.STATUS_ENUM___NULL;
    dataArr[EntityNodeFeature.ENTITY_NODE_FEATURE__HAS_STATUS] =
        candidate.status ? 1 : 0;
    dataArr[EntityNodeFeature.ENTITY_NODE_FEATURE__TOXIC_TURNS] =
        candidate.statusState.toxicTurns;
    dataArr[EntityNodeFeature.ENTITY_NODE_FEATURE__SLEEP_TURNS] =
        candidate.statusState.sleepTurns;
    dataArr[EntityNodeFeature.ENTITY_NODE_FEATURE__BEING_CALLED_BACK] =
        candidate.beingCalledBack ? 1 : 0;
    dataArr[EntityNodeFeature.ENTITY_NODE_FEATURE__TRAPPED] = candidate.trapped
        ? 1
        : 0;
    dataArr[EntityNodeFeature.ENTITY_NODE_FEATURE__NEWLY_SWITCHED] =
        candidate.newlySwitched ? 1 : 0;

    // We take pokemon level here
    dataArr[EntityNodeFeature.ENTITY_NODE_FEATURE__LEVEL] = pokemon.level;

    for (let moveIndex = 0; moveIndex < 4; moveIndex++) {
        dataArr[
            EntityNodeFeature[
                `ENTITY_NODE_FEATURE__MOVEID${moveIndex as MoveIndex}`
            ]
        ] = moveIds[moveIndex];
        dataArr[
            EntityNodeFeature[
                `ENTITY_NODE_FEATURE__MOVEPP${moveIndex as MoveIndex}`
            ]
        ] = movePps[moveIndex];
    }

    // Only copy candidate volatiles
    let volatiles = BigInt(0b0);
    for (const [key] of Object.entries(candidate.volatiles)) {
        const index = IndexValueFromEnum(VolatilestatusEnum, key);
        volatiles |= BigInt(1) << BigInt(index);
    }
    dataArr.set(
        bigIntToInt16Array(volatiles),
        EntityNodeFeature.ENTITY_NODE_FEATURE__VOLATILES0,
    );

    dataArr[EntityNodeFeature.ENTITY_NODE_FEATURE__SIDE] = isMySide(
        pokemon.side.n,
        playerIndex,
    );

    // Only copy pokemon boosts
    dataArr[EntityNodeFeature.ENTITY_NODE_FEATURE__BOOST_ATK_VALUE] =
        pokemon.boosts.atk ?? 0;
    dataArr[EntityNodeFeature.ENTITY_NODE_FEATURE__BOOST_DEF_VALUE] =
        pokemon.boosts.def ?? 0;
    dataArr[EntityNodeFeature.ENTITY_NODE_FEATURE__BOOST_SPA_VALUE] =
        pokemon.boosts.spa ?? 0;
    dataArr[EntityNodeFeature.ENTITY_NODE_FEATURE__BOOST_SPD_VALUE] =
        pokemon.boosts.spd ?? 0;
    dataArr[EntityNodeFeature.ENTITY_NODE_FEATURE__BOOST_SPE_VALUE] =
        pokemon.boosts.spe ?? 0;
    dataArr[EntityNodeFeature.ENTITY_NODE_FEATURE__BOOST_EVASION_VALUE] =
        pokemon.boosts.evasion ?? 0;
    dataArr[EntityNodeFeature.ENTITY_NODE_FEATURE__BOOST_ACCURACY_VALUE] =
        pokemon.boosts.accuracy ?? 0;

    // Copy candidate type change
    let typeChanged = BigInt(0b0);
    const typechangeVolatile = candidate.volatiles.typechange;
    if (typechangeVolatile) {
        if (typechangeVolatile.apparentType) {
            for (const type of typechangeVolatile.apparentType.split("/")) {
                const index = IndexValueFromEnum(TypechartEnum, type);
                typeChanged |= BigInt(1) << BigInt(index);
            }
        }
    }
    dataArr.set(
        bigIntToInt16Array(typeChanged),
        EntityNodeFeature.ENTITY_NODE_FEATURE__TYPECHANGE0,
    );

    return dataArr;
}

class Edge {
    player: TrainablePlayerAI;

    entityNodeData: Int16Array;
    entityEdgeData: Int16Array;
    fieldData: Int16Array;

    constructor(player: TrainablePlayerAI) {
        this.player = player;

        this.entityNodeData = new Int16Array(12 * numEntityNodeFeatures);
        this.entityEdgeData = new Int16Array(12 * numEntityEdgeFeatures);
        this.fieldData = new Int16Array(numFieldFeatures);

        this.updateSideData();
        this.updateFieldData();
    }

    clone() {
        const edge = new Edge(this.player);
        edge.entityNodeData.set(this.entityNodeData);
        edge.entityEdgeData.set(this.entityEdgeData);
        edge.fieldData.set(this.fieldData);
        return edge;
    }

    updateSideData() {
        const playerIndex = this.player.getPlayerIndex()!;

        let offset = 0;
        for (const side of this.player.publicBattle.sides) {
            this.updateEntityData(side, playerIndex);
            this.updateSideConditionData(side, playerIndex);
            offset += side.team.slice(0, 6).length * numEntityNodeFeatures;
        }
        for (const side of this.player.publicBattle.sides) {
            const team = side.team.slice(0, 6);
            const unkPokemon = isMySide(side.n, playerIndex)
                ? unkPokemon1
                : unkPokemon0;
            for (let i = team.length; i < side.totalPokemon; i++) {
                this.entityNodeData.set(unkPokemon, offset);
                offset += numEntityNodeFeatures;
            }
            for (let i = side.totalPokemon; i < 6; i++) {
                this.entityNodeData.set(nullPokemon, offset);
                offset += numEntityNodeFeatures;
            }
        }
    }

    updateEntityData(side: Side, playerIndex: number) {
        const team = side.team.slice(0, side.totalPokemon);
        for (const pokemon of team) {
            const pokemonBuffer = getArrayFromPokemon(pokemon, playerIndex);
            const index = this.player.eventHandler.identToIndex.get(
                pokemon.originalIdent,
            );
            if (index === undefined) {
                throw new Error(
                    `Pokemon ${pokemon.originalIdent} not found in eventHandler`,
                );
            }
            this.entityNodeData.set(
                pokemonBuffer,
                index * numEntityNodeFeatures,
            );
        }
    }

    updateSideConditionData(side: Side, playerIndex: number) {
        const isMe = isMySide(side.n, playerIndex);
        let sideConditionBuffer = BigInt(0b0);
        for (const [id] of Object.entries(side.sideConditions)) {
            const featureIndex = IndexValueFromEnum(SideconditionEnum, id);
            sideConditionBuffer |= BigInt(1) << BigInt(featureIndex);
        }
        this.fieldData.set(
            bigIntToInt16Array(sideConditionBuffer),
            isMe
                ? FieldFeature.FIELD_FEATURE__MY_SIDECONDITIONS0
                : FieldFeature.FIELD_FEATURE__OPP_SIDECONDITIONS0,
        );
        if (side.sideConditions.spikes) {
            this.setFieldFeature({
                featureIndex: isMe
                    ? FieldFeature.FIELD_FEATURE__MY_SPIKES
                    : FieldFeature.FIELD_FEATURE__OPP_SPIKES,
                value: side.sideConditions.spikes.level,
            });
        }
        if (side.sideConditions.toxicspikes) {
            this.setFieldFeature({
                featureIndex: isMe
                    ? FieldFeature.FIELD_FEATURE__MY_TOXIC_SPIKES
                    : FieldFeature.FIELD_FEATURE__OPP_TOXIC_SPIKES,
                value: side.sideConditions.toxicspikes.level,
            });
        }
    }

    updateFieldData() {
        const field = this.player.publicBattle.field;
        const weatherIndex = field.weatherState.id
            ? IndexValueFromEnum(
                  WeatherEnum,
                  WEATHERS[field.weatherState.id as never],
              )
            : WeatherEnum.WEATHER_ENUM___NULL;

        this.setFieldFeature({
            featureIndex: FieldFeature.FIELD_FEATURE__WEATHER_ID,
            value: weatherIndex,
        });
        this.setFieldFeature({
            featureIndex: FieldFeature.FIELD_FEATURE__WEATHER_MAX_DURATION,
            value: field.weatherState.maxDuration,
        });
        this.setFieldFeature({
            featureIndex: FieldFeature.FIELD_FEATURE__WEATHER_MIN_DURATION,
            value: field.weatherState.minDuration,
        });
    }

    setEntityEdgeFeature(args: {
        edgeIndex: number;
        featureIndex: EntityEdgeFeatureMap[keyof EntityEdgeFeatureMap];
        value: number;
    }) {
        const { edgeIndex, featureIndex, value } = args;
        if (edgeIndex > 11 || edgeIndex < 0) {
            throw new Error("edgeIndex out of bounds");
        }
        if (featureIndex === undefined) {
            throw new Error("featureIndex cannot be undefined");
        }
        if (value === undefined) {
            throw new Error("Value cannot be undefined");
        }
        this.entityEdgeData[edgeIndex * numEntityEdgeFeatures + featureIndex] =
            value;
    }

    setFieldFeature(args: {
        featureIndex: FieldFeatureMap[keyof FieldFeatureMap];
        value: number;
    }) {
        const { featureIndex, value } = args;
        if (featureIndex === undefined) {
            throw new Error("Index cannot be undefined");
        }
        if (value === undefined) {
            throw new Error("Value cannot be undefined");
        }
        this.fieldData[featureIndex] = value;
    }

    getEntityEdgeFeature(args: {
        edgeIndex: number;
        featureIndex: EntityEdgeFeatureMap[keyof EntityEdgeFeatureMap];
    }) {
        const { edgeIndex, featureIndex } = args;
        const index = edgeIndex * numEntityEdgeFeatures + featureIndex;
        return this.entityEdgeData.at(index);
    }

    getFieldFeature(args: {
        featureIndex: FieldFeatureMap[keyof FieldFeatureMap];
    }) {
        const { featureIndex } = args;
        return this.fieldData[featureIndex];
    }

    addMajorArg(args: { argName: MajorArgNames; edgeIndex: number }) {
        const { argName, edgeIndex } = args;
        const index = IndexValueFromEnum(BattlemajorargsEnum, argName);
        this.setEntityEdgeFeature({
            featureIndex: EntityEdgeFeature.ENTITY_EDGE_FEATURE__MAJOR_ARG,
            edgeIndex,
            value: index,
        });
    }

    updateEdgeFromOf(args: { effect: Partial<Effect>; edgeIndex: number }) {
        const { effect, edgeIndex } = args;
        const { effectType } = effect;
        if (effectType) {
            const fromTypeToken = IndexValueFromEnum(
                EffecttypesEnum,
                effectType,
            );
            const fromSourceToken = getEffectToken(effect);

            const numFromTypes =
                this.getEntityEdgeFeature({
                    featureIndex:
                        EntityEdgeFeature.ENTITY_EDGE_FEATURE__NUM_FROM_TYPES,
                    edgeIndex,
                }) ?? 0;
            const numFromSources =
                this.getEntityEdgeFeature({
                    featureIndex:
                        EntityEdgeFeature.ENTITY_EDGE_FEATURE__NUM_FROM_SOURCES,
                    edgeIndex,
                }) ?? 0;

            if (numFromTypes < 5) {
                this.setEntityEdgeFeature({
                    featureIndex:
                        (EntityEdgeFeature.ENTITY_EDGE_FEATURE__FROM_TYPE_TOKEN0 +
                            numFromTypes) as EntityEdgeFeatureMap[keyof EntityEdgeFeatureMap],
                    edgeIndex,
                    value: fromTypeToken,
                });
                this.setEntityEdgeFeature({
                    featureIndex:
                        EntityEdgeFeature.ENTITY_EDGE_FEATURE__NUM_FROM_TYPES,
                    edgeIndex,
                    value: numFromTypes + 1,
                });
            }
            if (numFromSources < 5) {
                this.setEntityEdgeFeature({
                    featureIndex:
                        (EntityEdgeFeature.ENTITY_EDGE_FEATURE__FROM_SOURCE_TOKEN0 +
                            numFromSources) as EntityEdgeFeatureMap[keyof EntityEdgeFeatureMap],
                    edgeIndex,
                    value: fromSourceToken,
                });
                this.setEntityEdgeFeature({
                    featureIndex:
                        EntityEdgeFeature.ENTITY_EDGE_FEATURE__NUM_FROM_SOURCES,
                    edgeIndex,
                    value: numFromSources + 1,
                });
            }
        }
    }
}

function getEffectToken(effect: Partial<Effect>): number {
    const { id } = effect;
    if (id) {
        return IndexValueFromEnum(EffectEnum, id);
    }
    return EffectEnum.EFFECT_ENUM___NULL;
}

export class EdgeBuffer {
    player: TrainablePlayerAI;

    entityNodeData: Int16Array;
    entityEdgeData: Int16Array;
    fieldData: Int16Array;

    entityNodeCursor: number;
    entityEdgeCursor: number;
    fieldCursor: number;

    prevEntityNodeCursor: number;
    prevEntityEdgeCursor: number;
    prevFieldCursor: number;

    numEdges: number;
    maxEdges: number;

    constructor(player: TrainablePlayerAI) {
        this.player = player;

        const maxEdges = 4000;
        this.maxEdges = maxEdges;

        this.entityNodeData = new Int16Array(
            maxEdges * 12 * numEntityNodeFeatures,
        );
        this.entityEdgeData = new Int16Array(
            maxEdges * 12 * numEntityEdgeFeatures,
        );
        this.fieldData = new Int16Array(maxEdges * numFieldFeatures);

        this.entityNodeCursor = 0;
        this.entityEdgeCursor = 0;
        this.fieldCursor = 0;

        this.prevEntityNodeCursor = 0;
        this.prevEntityEdgeCursor = 0;
        this.prevFieldCursor = 0;

        this.numEdges = 0;
    }

    setLatestEntityEdgeFeature(args: {
        edgeIndex: number;
        featureIndex: EntityEdgeFeatureMap[keyof EntityEdgeFeatureMap];
        value: number;
    }) {
        const { featureIndex, edgeIndex, value } = args;
        if (featureIndex === undefined) {
            throw new Error("Index cannot be undefined");
        }
        if (value === undefined) {
            throw new Error("Value cannot be undefined");
        }
        const index =
            this.prevEntityEdgeCursor +
            edgeIndex * numEntityEdgeFeatures +
            featureIndex;
        this.entityEdgeData[index] = value;
    }

    getLatestEntityEdgeFeature(args: {
        edgeIndex: number;
        featureIndex: EntityEdgeFeatureMap[keyof EntityEdgeFeatureMap];
    }) {
        const { featureIndex, edgeIndex } = args;
        const index =
            this.prevEntityEdgeCursor +
            edgeIndex * numEntityEdgeFeatures +
            featureIndex;
        const value = this.entityEdgeData.at(index);
        if (value === undefined) {
            throw new Error(
                `Feature index ${featureIndex} not found for edge index ${edgeIndex}`,
            );
        }
        return value;
    }

    updateLatestMinorArgs(args: {
        argName: MinorArgNames;
        edgeIndex: number;
        precision?: number;
    }) {
        const { argName, edgeIndex } = args;
        const precision = args.precision ?? 16;

        const index = IndexValueFromEnum(BattleminorargsEnum, argName);
        const featureIndex = {
            0: EntityEdgeFeature.ENTITY_EDGE_FEATURE__MINOR_ARG0,
            1: EntityEdgeFeature.ENTITY_EDGE_FEATURE__MINOR_ARG1,
            2: EntityEdgeFeature.ENTITY_EDGE_FEATURE__MINOR_ARG2,
            3: EntityEdgeFeature.ENTITY_EDGE_FEATURE__MINOR_ARG3,
        }[Math.floor(index / precision)];
        if (featureIndex === undefined) {
            throw new Error();
        }
        const currentValue = this.getLatestEntityEdgeFeature({
            featureIndex,
            edgeIndex,
        })!;
        const newValue = currentValue | (1 << index % precision);
        this.setLatestEntityEdgeFeature({
            featureIndex,
            edgeIndex,
            value: newValue,
        });
    }

    updateLatestMajorArg(args: { argName: MajorArgNames; edgeIndex: number }) {
        const { argName, edgeIndex } = args;
        const index = IndexValueFromEnum(BattlemajorargsEnum, argName);
        this.setLatestEntityEdgeFeature({
            featureIndex: EntityEdgeFeature.ENTITY_EDGE_FEATURE__MAJOR_ARG,
            edgeIndex,
            value: index,
        });
    }

    setLatestFieldFeature(args: {
        featureIndex: FieldFeatureMap[keyof FieldFeatureMap];
        value: number;
    }) {
        const { featureIndex, value } = args;
        if (featureIndex === undefined) {
            throw new Error("Index cannot be undefined");
        }
        if (value === undefined) {
            throw new Error("Value cannot be undefined");
        }
        const index = this.prevFieldCursor + featureIndex;
        this.fieldData[index] = value;
    }

    getLatestFieldFeature(args: {
        featureIndex: FieldFeatureMap[keyof FieldFeatureMap];
    }) {
        const { featureIndex } = args;
        const index = this.prevFieldCursor + featureIndex;
        return this.fieldData[index];
    }

    updateLatestEdgeFromOf(args: {
        effect: Partial<Effect>;
        edgeIndex: number;
    }) {
        const { effect, edgeIndex } = args;
        const { id, effectType, kind } = effect;
        const trueEffectType = effectType === undefined ? kind : effectType;
        if (trueEffectType !== undefined && id !== undefined) {
            const fromTypeToken = IndexValueFromEnum(
                EffecttypesEnum,
                trueEffectType,
            );
            const fromSourceToken = getEffectToken(effect);
            const numFromTypes =
                this.getLatestEntityEdgeFeature({
                    featureIndex:
                        EntityEdgeFeature.ENTITY_EDGE_FEATURE__NUM_FROM_TYPES,
                    edgeIndex,
                }) ?? 0;
            const numFromSources =
                this.getLatestEntityEdgeFeature({
                    featureIndex:
                        EntityEdgeFeature.ENTITY_EDGE_FEATURE__NUM_FROM_SOURCES,
                    edgeIndex,
                }) ?? 0;
            if (numFromTypes < 5) {
                this.setLatestEntityEdgeFeature({
                    featureIndex:
                        (EntityEdgeFeature.ENTITY_EDGE_FEATURE__FROM_TYPE_TOKEN0 +
                            numFromTypes) as EntityEdgeFeatureMap[keyof EntityEdgeFeatureMap],
                    edgeIndex,
                    value: fromTypeToken,
                });
                this.setLatestEntityEdgeFeature({
                    featureIndex:
                        EntityEdgeFeature.ENTITY_EDGE_FEATURE__NUM_FROM_TYPES,
                    edgeIndex,
                    value: numFromTypes + 1,
                });
            }
            if (numFromSources < 5) {
                this.setLatestEntityEdgeFeature({
                    featureIndex:
                        (EntityEdgeFeature.ENTITY_EDGE_FEATURE__FROM_SOURCE_TOKEN0 +
                            numFromSources) as EntityEdgeFeatureMap[keyof EntityEdgeFeatureMap],
                    edgeIndex,
                    value: fromSourceToken,
                });
                this.setLatestEntityEdgeFeature({
                    featureIndex:
                        EntityEdgeFeature.ENTITY_EDGE_FEATURE__NUM_FROM_SOURCES,
                    edgeIndex,
                    value: numFromSources + 1,
                });
            }
        }
    }

    addEdge(edge: Edge) {
        this.entityNodeData.set(edge.entityNodeData, this.entityNodeCursor);
        this.entityEdgeData.set(edge.entityEdgeData, this.entityEdgeCursor);
        this.fieldData.set(edge.fieldData, this.fieldCursor);

        this.prevEntityNodeCursor = this.entityNodeCursor;
        this.prevEntityEdgeCursor = this.entityEdgeCursor;
        this.prevFieldCursor = this.fieldCursor;

        this.entityNodeCursor += 12 * numEntityNodeFeatures;
        this.entityEdgeCursor += 12 * numEntityEdgeFeatures;
        this.fieldCursor += numFieldFeatures;

        this.numEdges += 1;
    }

    getHistory(numHistory: number = NUM_HISTORY) {
        const historyLength = Math.max(1, Math.min(this.numEdges, numHistory));
        const historyEntityNodes = new Uint8Array(
            this.entityNodeData.slice(
                this.entityNodeCursor -
                    historyLength * 12 * numEntityNodeFeatures,
                this.entityNodeCursor,
            ).buffer,
        );
        const historyEntityEdges = new Uint8Array(
            this.entityEdgeData.slice(
                this.entityEdgeCursor -
                    historyLength * 12 * numEntityEdgeFeatures,
                this.entityEdgeCursor,
            ).buffer,
        );
        const historyField = new Uint8Array(
            this.fieldData.slice(
                this.fieldCursor - historyLength * numFieldFeatures,
                this.fieldCursor,
            ).buffer,
        );
        return {
            historyEntityNodes,
            historyEntityEdges,
            historyField,
            historyLength,
        };
    }

    static toReadableHistory(args: {
        historyEntityNodesBuffer: Uint8Array;
        historyEntityEdgesBuffer: Uint8Array;
        historyFieldBuffer: Uint8Array;
        historyLength: number;
    }) {
        const {
            historyEntityNodesBuffer,
            historyEntityEdgesBuffer,
            historyFieldBuffer,
            historyLength,
        } = args;
        const historyItems = [];
        const historyEntityNodes = new Int16Array(
            historyEntityNodesBuffer.buffer,
        );
        const historyEntityEdges = new Int16Array(
            historyEntityEdgesBuffer.buffer,
        );
        const historyField = new Int16Array(historyFieldBuffer.buffer);

        for (
            let historyIndex = 0;
            historyIndex < historyLength;
            historyIndex++
        ) {
            const stepEntityNodes = historyEntityNodes.slice(
                historyIndex * 12 * numEntityNodeFeatures,
                (historyIndex + 1) * 12 * numEntityNodeFeatures,
            );
            const stepEntityEdges = historyEntityEdges.slice(
                historyIndex * 12 * numEntityEdgeFeatures,
                (historyIndex + 1) * 12 * numEntityEdgeFeatures,
            );
            const stepField = historyField.slice(
                historyIndex * numFieldFeatures,
                (historyIndex + 1) * numFieldFeatures,
            );
            const oneToEleven = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
            historyItems.push({
                entities: oneToEleven.map((memberIndex) => {
                    const start = memberIndex * numEntityNodeFeatures;
                    const end = (memberIndex + 1) * numEntityNodeFeatures;
                    return entityNodeArrayToObject(
                        stepEntityNodes.slice(start, end),
                    );
                }),
                edges: oneToEleven.map((memberIndex) => {
                    const start = memberIndex * numEntityEdgeFeatures;
                    const end = (memberIndex + 1) * numEntityEdgeFeatures;
                    return entityEdgeArrayToObject(
                        stepEntityEdges.slice(start, end),
                    );
                }),
                field: fieldArrayToObject(stepField),
            });
        }
        return historyItems;
    }
}

export class EventHandler implements Protocol.Handler {
    readonly player: TrainablePlayerAI;

    turnOrder: number;
    turnNum: number;
    timestamp: number;
    edgeBuffer: EdgeBuffer;
    log: string[];
    identToIndex: Map<PokemonIdent | SideID, number>;

    constructor(player: TrainablePlayerAI) {
        this.player = player;

        this.edgeBuffer = new EdgeBuffer(player);
        this.turnOrder = 0;
        this.turnNum = 0;
        this.timestamp = 0;
        this.log = [];

        this.identToIndex = new Map<PokemonIdent, number>();
    }

    addLine(line: string) {
        this.log.push(line);
    }

    getPokemon(
        pokemonid: PokemonIdent,
        isPublic: boolean = true,
    ): {
        pokemon: Pokemon | null;
        index: number;
    } {
        if (
            !pokemonid ||
            pokemonid === "??" ||
            pokemonid === "null" ||
            pokemonid === "false"
        ) {
            return { pokemon: null, index: -1 };
        }

        if (isPublic) {
            const { pokemonid: parsedPokemonid } =
                this.player.publicBattle.parsePokemonId(pokemonid);
            if (!this.identToIndex.has(parsedPokemonid)) {
                this.identToIndex.set(parsedPokemonid, this.identToIndex.size);
            }

            for (const side of this.player.publicBattle.sides) {
                for (const pokemon of side.team.slice(0, side.totalPokemon)) {
                    if (pokemon.originalIdent === parsedPokemonid) {
                        return {
                            pokemon,
                            index: this.identToIndex.get(parsedPokemonid) ?? -1,
                        };
                    }
                }
            }
            return {
                pokemon: null,
                index: this.identToIndex.get(parsedPokemonid) ?? -1,
            };
        } else {
            const { siden, pokemonid: parsedPokemonid } =
                this.player.privateBattle.parsePokemonId(pokemonid);
            const side = this.player.privateBattle.sides[siden];
            for (const [index, pokemon] of side.team.entries()) {
                if (pokemon.originalIdent === parsedPokemonid) {
                    return { pokemon, index };
                }
            }
            return { pokemon: null, index: -1 };
        }
    }

    getMove(ident?: string) {
        return this.player.publicBattle.get("moves", ident) as Partial<Move> &
            NA;
    }

    getAbility(ident?: string) {
        return this.player.publicBattle.get(
            "abilities",
            ident,
        ) as Partial<Ability> & NA;
    }

    getItem(ident: string) {
        return this.player.publicBattle.get("items", ident) as Partial<Item> &
            NA;
    }

    getCondition(ident?: string) {
        return this.player.publicBattle.get(
            "conditions",
            ident,
        ) as Partial<Condition>;
    }

    getSide(ident: Protocol.Side) {
        return this.player.publicBattle.getSide(ident);
    }

    _preprocessEdge(edge: Edge) {
        edge.setFieldFeature({
            featureIndex: FieldFeature.FIELD_FEATURE__REQUEST_COUNT,
            value: this.player.requestCount,
        });
        edge.setFieldFeature({
            featureIndex: FieldFeature.FIELD_FEATURE__VALID,
            value: 1,
        });
        edge.setFieldFeature({
            featureIndex: FieldFeature.FIELD_FEATURE__INDEX,
            value: this.edgeBuffer.numEdges,
        });
        edge.setFieldFeature({
            featureIndex: FieldFeature.FIELD_FEATURE__TURN_ORDER_VALUE,
            value: this.turnOrder,
        });
        this.turnOrder += 1;
        edge.setFieldFeature({
            featureIndex: FieldFeature.FIELD_FEATURE__TURN_VALUE,
            value: this.turnNum,
        });
        return edge;
    }

    addEdge(edge: Edge) {
        const preprocessedEdge = this._preprocessEdge(edge);
        this.edgeBuffer.addEdge(preprocessedEdge);
    }

    "|move|"(args: Args["|move|"], kwArgs: KWArgs["|move|"]) {
        const [argName, poke1Ident, moveId] = args;

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const { index: edgeIndex } = this.getPokemon(
            poke1Ident as PokemonIdent,
        );

        const move = this.getMove(moveId);

        const edge = new Edge(this.player);
        edge.addMajorArg({ argName, edgeIndex });
        edge.setEntityEdgeFeature({
            edgeIndex,
            featureIndex: EntityEdgeFeature.ENTITY_EDGE_FEATURE__MOVE_TOKEN,
            value: IndexValueFromEnum(MovesEnum, move.id),
        });
        edge.setEntityEdgeFeature({
            edgeIndex,
            featureIndex: EntityEdgeFeature.ENTITY_EDGE_FEATURE__MOVE_TOKEN,
            value: IndexValueFromEnum(MovesEnum, move.id),
        });
        if (kwArgs.from) {
            const fromEffect = this.getCondition(kwArgs.from);
            edge.updateEdgeFromOf({ effect: fromEffect, edgeIndex });
        }
        edge.setEntityEdgeFeature({
            edgeIndex,
            featureIndex: EntityEdgeFeature.ENTITY_EDGE_FEATURE__HIT_COUNT,
            value: 1,
        });
        this.addEdge(edge);
    }

    "|player|"(args: Args["|player|"]) {
        // eslint-disable-next-line @typescript-eslint/no-unused-vars
        const [argName, playerId, userName] = args;
        const playerIndex = playerId.at(1);
        if (playerIndex !== undefined && this.player.userName === userName) {
            this.player.playerIndex = parseInt(playerIndex) - 1;
        }
    }

    "|drag|"(args: Args["|drag|"]) {
        this.handleSwitch(args, {});
    }

    "|switch|"(args: Args["|switch|"], kwArgs: KWArgs["|switch|"]) {
        this.handleSwitch(args, kwArgs);
    }

    handleSwitch(
        args: Args["|switch|" | "|drag|"],
        kwArgs: KWArgs["|switch|" | "|drag|"],
    ) {
        const [argName, poke1Ident] = args;

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const { index: edgeIndex } = this.getPokemon(poke1Ident)!;
        const edge = new Edge(this.player);

        edge.addMajorArg({ argName, edgeIndex });
        edge.setEntityEdgeFeature({
            edgeIndex,
            featureIndex: EntityEdgeFeature.ENTITY_EDGE_FEATURE__MOVE_TOKEN,
            value: MovesEnum.MOVES_ENUM___SWITCH,
        });
        if (argName === "switch") {
            const from = (kwArgs as KWArgs["|switch|"]).from;
            if (from) {
                const effect = this.getCondition(from);
                edge.updateEdgeFromOf({ effect, edgeIndex });
            }
        }
        this.addEdge(edge);
    }

    "|cant|"(args: Args["|cant|"]) {
        const [argName, poke1Ident, conditionId, moveId] = args;

        const edge = new Edge(this.player);

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const { index: edgeIndex } = this.getPokemon(poke1Ident)!;

        if (moveId) {
            const move = this.getMove(moveId);
            edge.setEntityEdgeFeature({
                edgeIndex,
                featureIndex: EntityEdgeFeature.ENTITY_EDGE_FEATURE__MOVE_TOKEN,
                value: IndexValueFromEnum(MovesEnum, move.id),
            });
        }

        const condition = this.getCondition(conditionId);

        edge.addMajorArg({ argName, edgeIndex });
        edge.updateEdgeFromOf({ effect: condition, edgeIndex });

        this.addEdge(edge);
    }

    "|faint|"(args: Args["|faint|"]) {
        const [argName, poke1Ident] = args;

        const edge = new Edge(this.player);

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const { index: edgeIndex } = this.getPokemon(poke1Ident)!;

        edge.addMajorArg({ argName, edgeIndex });
        this.addEdge(edge);
    }

    "|-fail|"(args: Args["|-fail|"], kwArgs: KWArgs["|-fail|"]) {
        const [argName, poke1Ident] = args;

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const { index: edgeIndex } = this.getPokemon(poke1Ident)!;

        const effect = this.getCondition(kwArgs.from);

        this.edgeBuffer.updateLatestMinorArgs({ argName, edgeIndex });
        this.edgeBuffer.updateLatestEdgeFromOf({
            effect,
            edgeIndex,
        });
    }

    "|-block|"(args: Args["|-block|"], kwArgs: KWArgs["|-block|"]) {
        const [argName, poke1Ident] = args;

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const { index: edgeIndex } = this.getPokemon(poke1Ident)!;

        const effect = this.getCondition(kwArgs.from);

        this.edgeBuffer.updateLatestMinorArgs({ argName, edgeIndex });
        this.edgeBuffer.updateLatestEdgeFromOf({ effect, edgeIndex });
    }

    "|-notarget|"(args: Args["|-notarget|"]) {
        const [argName, poke1Ident] = args;

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        if (poke1Ident !== undefined) {
            const { index: edgeIndex } = this.getPokemon(poke1Ident)!;
            this.edgeBuffer.updateLatestMinorArgs({ argName, edgeIndex });
        } else {
            throw new Error(
                `Pokemon identifier is required for |-notarget| event: ${args}`,
            );
        }
    }

    "|-miss|"(args: Args["|-miss|"], kwArgs: KWArgs["|-miss|"]) {
        const [argName, poke1Ident, poke2Ident] = args;

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const idents = [poke1Ident];
        if (poke2Ident !== undefined) {
            idents.push(poke2Ident);
        }
        for (const ident of idents) {
            const { index: edgeIndex } = this.getPokemon(ident)!;
            const effect = this.getCondition(kwArgs.from);
            this.edgeBuffer.updateLatestMinorArgs({ argName, edgeIndex });
            this.edgeBuffer.updateLatestEdgeFromOf({
                effect,
                edgeIndex,
            });
        }
    }

    "|-damage|"(
        args: Args["|-damage|"] | Args["|-sethp|"],
        kwArgs: KWArgs["|-damage|"] | KWArgs["|-sethp|"],
    ) {
        const [argName, poke1Ident, hpStatus] = args;

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const { pokemon, index: edgeIndex } = this.getPokemon(poke1Ident);
        if (pokemon === null) {
            throw new Error(`Pokemon ${poke1Ident} not found`);
        }

        if (kwArgs.from) {
            const effect = this.getCondition(kwArgs.from);
            this.edgeBuffer.updateLatestEdgeFromOf({ effect, edgeIndex });
        }

        const damage = pokemon.healthParse(hpStatus);
        if (damage === null) {
            throw new Error(`Invalid damage value: ${damage}`);
        }

        const addedDamageToken = Math.abs(
            Math.floor((MAX_RATIO_TOKEN * damage[0]) / damage[1]),
        );

        this.edgeBuffer.updateLatestMinorArgs({ argName, edgeIndex });
        const currentDamageToken =
            this.edgeBuffer.getLatestEntityEdgeFeature({
                featureIndex:
                    EntityEdgeFeature.ENTITY_EDGE_FEATURE__DAMAGE_RATIO,
                edgeIndex,
            }) ?? 0;
        this.edgeBuffer.setLatestEntityEdgeFeature({
            featureIndex: EntityEdgeFeature.ENTITY_EDGE_FEATURE__DAMAGE_RATIO,
            edgeIndex,
            value: Math.min(
                MAX_RATIO_TOKEN,
                currentDamageToken + addedDamageToken,
            ),
        });
    }

    "|-heal|"(
        args: Args["|-heal|"] | Args["|-sethp|"],
        kwArgs: KWArgs["|-heal|"] | KWArgs["|-sethp|"],
    ) {
        const [argName, poke1Ident, hpStatus] = args;

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const { pokemon, index: edgeIndex } = this.getPokemon(poke1Ident);
        if (pokemon === null) {
            throw new Error(`Pokemon ${poke1Ident} not found`);
        }

        if (kwArgs.from) {
            const effect = this.getCondition(kwArgs.from);
            this.edgeBuffer.updateLatestEdgeFromOf({
                effect,
                edgeIndex,
            });
        }

        const damage = pokemon.healthParse(hpStatus);
        if (damage === null) {
            throw new Error(`Invalid damage value: ${damage}`);
        }

        const addedHealToken = Math.abs(
            Math.floor((MAX_RATIO_TOKEN * damage[0]) / damage[1]),
        );

        this.edgeBuffer.updateLatestMinorArgs({ argName, edgeIndex });
        const currentHealToken = this.edgeBuffer.getLatestEntityEdgeFeature({
            featureIndex: EntityEdgeFeature.ENTITY_EDGE_FEATURE__HEAL_RATIO,
            edgeIndex,
        });
        this.edgeBuffer.setLatestEntityEdgeFeature({
            featureIndex: EntityEdgeFeature.ENTITY_EDGE_FEATURE__HEAL_RATIO,
            edgeIndex,
            value: Math.min(MAX_RATIO_TOKEN, currentHealToken + addedHealToken),
        });
    }

    "|-sethp|"(args: Args["|-sethp|"], kwArgs: KWArgs["|-sethp|"]) {
        const [argName, poke1Ident, hpStatus] = args;
        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }
        const { pokemon, index: edgeIndex } = this.getPokemon(poke1Ident)!;
        if (pokemon === null) {
            throw new Error(`Pokemon ${poke1Ident} not found`);
        }
        const damage = pokemon.healthParse(hpStatus);
        if (damage === null) {
            throw new Error(`Invalid damage value: ${damage}`);
        }
        if (damage[0] < 0) {
            this["|-damage|"](
                ["-damage", args[1], args[2]] as Args["|-damage|"],
                kwArgs,
            );
        } else if (damage[0] > 0) {
            this["|-heal|"](
                ["-heal", args[1], args[2]] as Args["|-heal|"],
                kwArgs,
            );
        }
        this.edgeBuffer.updateLatestMinorArgs({ argName, edgeIndex });
    }

    "|-status|"(args: Args["|-status|"], kwArgs: KWArgs["|-status|"]) {
        const [argName, poke1Ident, statusId] = args;

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const { index: edgeIndex } = this.getPokemon(poke1Ident)!;

        const fromEffect = this.getCondition(kwArgs.from);
        const statusToken = IndexValueFromEnum(StatusEnum, statusId);

        this.edgeBuffer.updateLatestMinorArgs({ argName, edgeIndex });
        this.edgeBuffer.updateLatestEdgeFromOf({
            effect: fromEffect,
            edgeIndex,
        });
        this.edgeBuffer.setLatestEntityEdgeFeature({
            featureIndex: EntityEdgeFeature.ENTITY_EDGE_FEATURE__STATUS_TOKEN,
            edgeIndex,
            value: statusToken,
        });
    }

    "|-curestatus|"(args: Args["|-curestatus|"]) {
        const [argName, poke1Ident, statusId] = args;

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const { index: edgeIndex } = this.getPokemon(poke1Ident)!;

        const statusToken = IndexValueFromEnum(StatusEnum, statusId);

        this.edgeBuffer.updateLatestMinorArgs({ argName, edgeIndex });
        this.edgeBuffer.setLatestEntityEdgeFeature({
            featureIndex: EntityEdgeFeature.ENTITY_EDGE_FEATURE__STATUS_TOKEN,
            edgeIndex,
            value: statusToken,
        });
    }

    "|-cureteam|"(args: Args["|-cureteam|"]) {
        const [argName, poke1Ident] = args;

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const { index: edgeIndex } = this.getPokemon(poke1Ident)!;

        this.edgeBuffer.updateLatestMinorArgs({ argName, edgeIndex });
    }

    static getStatBoostEdgeFeatureIndex(stat: BoostID) {
        return EntityEdgeFeature[
            `ENTITY_EDGE_FEATURE__BOOST_${stat.toLocaleUpperCase()}_VALUE` as `ENTITY_EDGE_FEATURE__BOOST_${Uppercase<BoostID>}_VALUE`
        ];
    }

    "|-boost|"(args: Args["|-boost|"]) {
        const [argName, poke1Ident, stat, value] = args;

        const featureIndex = EventHandler.getStatBoostEdgeFeatureIndex(stat);

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const { index: edgeIndex } = this.getPokemon(poke1Ident)!;

        this.edgeBuffer.updateLatestMinorArgs({ argName, edgeIndex });
        this.edgeBuffer.setLatestEntityEdgeFeature({
            featureIndex,
            edgeIndex,
            value: parseInt(value),
        });
    }

    "|-unboost|"(args: Args["|-unboost|"]) {
        const [argName, poke1Ident, stat, value] = args;

        const featureIndex = EventHandler.getStatBoostEdgeFeatureIndex(stat);

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const { index: edgeIndex } = this.getPokemon(poke1Ident)!;

        this.edgeBuffer.updateLatestMinorArgs({ argName, edgeIndex });
        this.edgeBuffer.setLatestEntityEdgeFeature({
            featureIndex,
            edgeIndex,
            value: -parseInt(value),
        });
    }

    "|-setboost|"(args: Args["|-setboost|"], kwArgs: KWArgs["|-setboost|"]) {
        const [argName, poke1Ident, stat, value] = args;

        const featureIndex = EventHandler.getStatBoostEdgeFeatureIndex(stat);
        const effect = this.getCondition(kwArgs.from);

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const { index: edgeIndex } = this.getPokemon(poke1Ident)!;

        this.edgeBuffer.updateLatestMinorArgs({ argName, edgeIndex });
        this.edgeBuffer.setLatestEntityEdgeFeature({
            featureIndex,
            edgeIndex,
            value: parseInt(value),
        });
        this.edgeBuffer.updateLatestEdgeFromOf({ effect, edgeIndex });
    }

    "|-swapboost|"(args: Args["|-swapboost|"], kwArgs: KWArgs["|-swapboost|"]) {
        const [argName, poke1Ident] = args;

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const { index: edgeIndex } = this.getPokemon(poke1Ident)!;

        const effect = this.getCondition(kwArgs.from);

        this.edgeBuffer.updateLatestMinorArgs({ argName, edgeIndex });
        this.edgeBuffer.updateLatestEdgeFromOf({ effect, edgeIndex });
    }

    "|-invertboost|"(
        args: Args["|-invertboost|"],
        kwArgs: KWArgs["|-invertboost|"],
    ) {
        const [argName, poke1Ident] = args;

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const { index: edgeIndex } = this.getPokemon(poke1Ident)!;

        const effect = this.getCondition(kwArgs.from);

        this.edgeBuffer.updateLatestMinorArgs({ argName, edgeIndex });
        this.edgeBuffer.updateLatestEdgeFromOf({ effect, edgeIndex });
    }

    "|-clearboost|"(
        args: Args["|-clearboost|"],
        kwArgs: KWArgs["|-clearboost|"],
    ) {
        const [argName, poke1Ident] = args;

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const { index: edgeIndex } = this.getPokemon(poke1Ident)!;

        const effect = this.getCondition(kwArgs.from);

        this.edgeBuffer.updateLatestMinorArgs({ argName, edgeIndex });
        this.edgeBuffer.updateLatestEdgeFromOf({ effect, edgeIndex });
    }

    "|-clearnegativeboost|"(
        args: Args["|-clearnegativeboost|"],
        kwArgs: KWArgs["|-clearnegativeboost|"],
    ) {
        const [argName, poke1Ident] = args;

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const { index: edgeIndex } = this.getPokemon(poke1Ident);

        const effect = this.getCondition(kwArgs.from);

        this.edgeBuffer.updateLatestMinorArgs({ argName, edgeIndex });
        this.edgeBuffer.updateLatestEdgeFromOf({ effect, edgeIndex });
    }

    "|-copyboost|"() {}

    "|-weather|"(args: Args["|-weather|"]) {
        // eslint-disable-next-line @typescript-eslint/no-unused-vars
        const [argName, weatherId] = args;

        const weatherIndex =
            weatherId === "none"
                ? WeatherEnum.WEATHER_ENUM___NULL
                : IndexValueFromEnum(WeatherEnum, weatherId);

        this.edgeBuffer.setLatestFieldFeature({
            featureIndex: FieldFeature.FIELD_FEATURE__WEATHER_ID,
            value: weatherIndex,
        });
    }

    "|-fieldstart|"() {
        // kwArgs: KWArgs["|-fieldstart|"], // args: Args["|-fieldstart|"],
        // const [argName] = args;
    }

    "|-fieldend|"() {
        // args: Args["|-fieldend|"], kwArgs: KWArgs["|-fieldend|"]
        // const [argName] = args;
    }

    "|-sidestart|"(args: Args["|-sidestart|"]) {
        const [argName, sideId, conditionId] = args;

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const side = this.getSide(sideId);
        const effect = this.getCondition(conditionId);

        for (const pokemon of side.team) {
            const ident = pokemon.originalIdent;
            const { index: edgeIndex } = this.getPokemon(ident);
            this.edgeBuffer.updateLatestMinorArgs({ argName, edgeIndex });
            this.edgeBuffer.updateLatestEdgeFromOf({ effect, edgeIndex });
        }
    }

    "|-sideend|"(args: Args["|-sideend|"]) {
        const [argName, sideId, conditionId] = args;

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const side = this.getSide(sideId);
        const effect = this.getCondition(conditionId);

        for (const pokemon of side.team) {
            const ident = pokemon.originalIdent;
            const { index: edgeIndex } = this.getPokemon(ident);
            this.edgeBuffer.updateLatestMinorArgs({ argName, edgeIndex });
            this.edgeBuffer.updateLatestEdgeFromOf({ effect, edgeIndex });
        }
    }

    "|-swapsideconditions|"() {}

    "|-start|"(args: Args["|-start|"], kwArgs: KWArgs["|-start|"]) {
        const [argName, poke1Ident, conditionId] = args;

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const { index: edgeIndex } = this.getPokemon(poke1Ident)!;

        this.edgeBuffer.updateLatestMinorArgs({ argName, edgeIndex });
        this.edgeBuffer.updateLatestEdgeFromOf({
            effect: this.getCondition(kwArgs.from),
            edgeIndex,
        });
        this.edgeBuffer.updateLatestEdgeFromOf({
            effect: this.getCondition(
                conditionId.startsWith("perish") ? "perishsong" : conditionId,
            ),
            edgeIndex,
        });
    }

    "|-end|"(args: Args["|-end|"], kwArgs: KWArgs["|-end|"]) {
        const [argName, poke1Ident, conditionId] = args;

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const { index: edgeIndex } = this.getPokemon(poke1Ident)!;

        this.edgeBuffer.updateLatestMinorArgs({ argName, edgeIndex });
        this.edgeBuffer.updateLatestEdgeFromOf({
            effect: this.getCondition(kwArgs.from),
            edgeIndex,
        });
        this.edgeBuffer.updateLatestEdgeFromOf({
            effect: this.getCondition(conditionId),
            edgeIndex,
        });
    }

    "|-crit|"(args: Args["|-crit|"]) {
        const [argName, poke1Ident] = args;

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const { index: edgeIndex } = this.getPokemon(poke1Ident);
        this.edgeBuffer.updateLatestMinorArgs({ argName, edgeIndex });
    }

    "|-supereffective|"(args: Args["|-supereffective|"]) {
        const [argName, poke1Ident] = args;

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const { index: edgeIndex } = this.getPokemon(poke1Ident);
        this.edgeBuffer.updateLatestMinorArgs({ argName, edgeIndex });
    }

    "|-resisted|"(args: Args["|-resisted|"]) {
        const [argName, poke1Ident] = args;

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const { index: edgeIndex } = this.getPokemon(poke1Ident);
        this.edgeBuffer.updateLatestMinorArgs({ argName, edgeIndex });
    }

    "|-immune|"(args: Args["|-immune|"], kwArgs: KWArgs["|-immune|"]) {
        const [argName, poke1Ident, conditionId] = args;

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const { index: edgeIndex } = this.getPokemon(poke1Ident);
        this.edgeBuffer.updateLatestMinorArgs({ argName, edgeIndex });
        this.edgeBuffer.updateLatestEdgeFromOf({
            effect: this.getCondition(kwArgs.from),
            edgeIndex,
        });
        this.edgeBuffer.updateLatestEdgeFromOf({
            effect: this.getCondition(conditionId),
            edgeIndex,
        });
    }

    "|-item|"(args: Args["|-item|"], kwArgs: KWArgs["|-item|"]) {
        const [argName, poke1Ident, itemId] = args;

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const { index: edgeIndex } = this.getPokemon(
            poke1Ident as PokemonIdent,
        );
        this.edgeBuffer.updateLatestMinorArgs({ argName, edgeIndex });

        const itemIndex = IndexValueFromEnum(ItemsEnum, itemId);
        const effect = this.getCondition(kwArgs.from);

        this.edgeBuffer.updateLatestMinorArgs({ argName, edgeIndex });
        this.edgeBuffer.updateLatestEdgeFromOf({ effect, edgeIndex });
        this.edgeBuffer.setLatestEntityEdgeFeature({
            featureIndex: EntityEdgeFeature.ENTITY_EDGE_FEATURE__ITEM_TOKEN,
            edgeIndex,
            value: itemIndex,
        });
    }

    "|-enditem|"(args: Args["|-enditem|"], kwArgs: KWArgs["|-enditem|"]) {
        const [argName, poke1Ident, itemId] = args;

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const { index: edgeIndex } = this.getPokemon(poke1Ident);
        this.edgeBuffer.updateLatestMinorArgs({ argName, edgeIndex });

        const itemIndex = IndexValueFromEnum(ItemsEnum, itemId);
        const effect = this.getCondition(kwArgs.from);

        this.edgeBuffer.updateLatestMinorArgs({ argName, edgeIndex });
        this.edgeBuffer.updateLatestEdgeFromOf({ effect, edgeIndex });
        this.edgeBuffer.setLatestEntityEdgeFeature({
            featureIndex: EntityEdgeFeature.ENTITY_EDGE_FEATURE__ITEM_TOKEN,
            edgeIndex,
            value: itemIndex,
        });
    }

    "|-ability|"(args: Args["|-ability|"], kwArgs: KWArgs["|-ability|"]) {
        const [argName, poke1Ident, abilityId] = args;

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const { index: edgeIndex } = this.getPokemon(poke1Ident);

        const abilityIndex = IndexValueFromEnum(AbilitiesEnum, abilityId);

        if (kwArgs.from) {
            const effect = this.getCondition(kwArgs.from);
            this.edgeBuffer.updateLatestEdgeFromOf({ effect, edgeIndex });
        }

        this.edgeBuffer.updateLatestMinorArgs({ argName, edgeIndex });
        this.edgeBuffer.setLatestEntityEdgeFeature({
            featureIndex: EntityEdgeFeature.ENTITY_EDGE_FEATURE__ABILITY_TOKEN,
            edgeIndex,
            value: abilityIndex,
        });
    }

    "|-endability|"(
        args: Args["|-endability|"],
        kwArgs: KWArgs["|-endability|"],
    ) {
        const [argName, poke1Ident] = args;

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const { index: edgeIndex } = this.getPokemon(poke1Ident)!;

        const effect = this.getCondition(kwArgs.from);

        this.edgeBuffer.updateLatestMinorArgs({ argName, edgeIndex });
        this.edgeBuffer.updateLatestEdgeFromOf({ effect, edgeIndex });
    }

    rememberTransformed(
        poke1Ident: Protocol.PokemonIdent,
        poke2Ident: Protocol.PokemonIdent,
    ) {
        const { pokemon: srcPokemon } = this.getPokemon(poke1Ident, false)!;
        const { pokemon: tgtPokemon } = this.getPokemon(poke2Ident)!;

        if (srcPokemon !== null && tgtPokemon !== null) {
            const transformedPokemon =
                srcPokemon?.volatiles?.transform?.pokemon;
            if (transformedPokemon === undefined) {
                return;
            }
            const currentRememberedMoves = new Set(
                tgtPokemon.moveSlots.map((x) => x.id),
            );
            for (const { id } of srcPokemon.moveSlots.slice(0, 4)) {
                if (!currentRememberedMoves.has(id)) {
                    tgtPokemon.rememberMove(id);
                }
            }
            tgtPokemon.rememberAbility(transformedPokemon.ability);
        }
    }

    "|-transform|"(args: Args["|-transform|"]) {
        const [argName, poke1Ident, poke2Ident] = args;

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const { index: edgeIndex } = this.getPokemon(poke1Ident);

        this.edgeBuffer.updateLatestMinorArgs({ argName, edgeIndex });
        this.rememberTransformed(poke1Ident, poke2Ident);
    }

    "|-mega|"() {}

    "|-primal|"() {}

    "|-burst|"() {}

    "|-zpower|"() {}

    "|-zbroken|"() {}

    "|-activate|"(args: Args["|-activate|"], kwArgs: KWArgs["|-activate|"]) {
        const [argName, poke1Ident, conditionId1] = args;

        if (poke1Ident) {
            const playerIndex = this.player.getPlayerIndex();
            if (playerIndex === undefined) {
                throw new Error();
            }

            const { index: edgeIndex } = this.getPokemon(poke1Ident)!;

            this.edgeBuffer.updateLatestMinorArgs({ argName, edgeIndex });
            for (const effect of [
                this.getCondition(kwArgs.from),
                this.getCondition(conditionId1),
                // this.getCondition(conditionId2),
            ]) {
                this.edgeBuffer.updateLatestEdgeFromOf({ effect, edgeIndex });
            }
        }
    }

    "|-mustrecharge|"(args: Args["|-mustrecharge|"]) {
        const [argName, poke1Ident] = args;

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const { index: edgeIndex } = this.getPokemon(poke1Ident)!;

        this.edgeBuffer.updateLatestMinorArgs({ argName, edgeIndex });
    }

    "|-prepare|"(args: Args["|-prepare|"]) {
        const [argName, poke1Ident] = args;

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const { index: edgeIndex } = this.getPokemon(poke1Ident)!;

        this.edgeBuffer.updateLatestMinorArgs({ argName, edgeIndex });
    }

    "|-hitcount|"(args: Args["|-hitcount|"]) {
        const [argName, poke1Ident, numHits] = args;

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const { index: edgeIndex } = this.getPokemon(poke1Ident)!;

        this.edgeBuffer.updateLatestMinorArgs({ argName, edgeIndex });
        this.edgeBuffer.setLatestEntityEdgeFeature({
            featureIndex: EntityEdgeFeature.ENTITY_EDGE_FEATURE__HIT_COUNT,
            edgeIndex,
            value: parseInt(numHits),
        });
    }

    "|done|"(args: Args["|done|"]) {
        const [argName] = args;

        const edge = new Edge(this.player);
        for (const side of this.player.privateBattle.sides) {
            for (const active of side.active) {
                if (active !== null) {
                    const { index: edgeIndex } = this.getPokemon(
                        active.originalIdent,
                    );
                    if (edgeIndex >= 0) {
                        edge.addMajorArg({ argName, edgeIndex });
                    }
                }
            }
        }
        if (this.turnOrder > 0) {
            this.addEdge(edge);
        }
    }

    "|start|"() {
        this.turnOrder = 0;

        const edge = new Edge(this.player);
        this.addEdge(edge);
    }

    "|t:|"(args: Args["|t:|"]) {
        // eslint-disable-next-line @typescript-eslint/no-unused-vars
        const [_, timestamp] = args;
        this.timestamp = parseInt(timestamp);
    }

    "|turn|"(args: Args["|turn|"]) {
        const turnNum = (args.at(1) ?? "").toString();

        this.turnOrder = 0;
        this.turnNum = parseInt(turnNum);
    }

    "|win|"() {
        this.player.done = true;
    }

    "|tie|"() {
        this.player.done = true;
    }
}

export class StateHandler {
    player: TrainablePlayerAI;

    constructor(player: TrainablePlayerAI) {
        this.player = player;
    }

    static getLegalActions(
        request?: AnyObject | null,
        maskMoves: boolean = false,
    ): {
        legalActions: OneDBoolean;
        isStruggling: boolean;
    } {
        const legalActions = new OneDBoolean(10, Uint8Array);
        let isStruggling = false;

        if (request === undefined || request === null) {
            for (const index of [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
                legalActions.set(index, true);
        } else {
            if (request.wait) {
                for (const index of [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
                    legalActions.set(index, true);
            } else if (request.forceSwitch) {
                const pokemon = request.side.pokemon;
                const forceSwitchLength = request.forceSwitch.length;
                const isReviving = !!pokemon[0].reviving;

                for (let j = 1; j <= 6; j++) {
                    const currentPokemon = pokemon[j - 1];
                    if (
                        currentPokemon &&
                        j > forceSwitchLength &&
                        (isReviving ? 1 : 0) ^
                            (currentPokemon.condition.endsWith(" fnt") ? 0 : 1)
                    ) {
                        const switchIndex = j as 1 | 2 | 3 | 4 | 5 | 6;
                        legalActions.set(3 + switchIndex, true);
                    }
                }
            } else if (request.active) {
                const pokemon = request.side.pokemon;
                const active = request.active[0];
                const possibleMoves = active.moves ?? [];
                const canSwitch = [];

                for (let j = 0; j < 6; j++) {
                    const currentPokemon = pokemon[j];
                    if (
                        currentPokemon &&
                        !currentPokemon.active &&
                        !currentPokemon.condition.endsWith(" fnt")
                    ) {
                        canSwitch.push(j);
                    }
                }

                const switches =
                    active.trapped || active.maybeTrapped ? [] : canSwitch;
                const canAddMove = !maskMoves || switches.length === 0;

                for (let j = 1; j <= possibleMoves.length; j++) {
                    const currentMove = possibleMoves[j - 1];
                    if (currentMove.id === "struggle") {
                        isStruggling = true;
                    }
                    if ((!currentMove.disabled && canAddMove) || isStruggling) {
                        const moveIndex = j as 1 | 2 | 3 | 4;
                        legalActions.set(-1 + moveIndex, true);
                    }
                }

                for (const j of switches) {
                    const switchIndex = (j + 1) as 1 | 2 | 3 | 4 | 5 | 6;
                    legalActions.set(3 + switchIndex, true);
                }
            } else if (request.teamPreview) {
                const pokemon = request.side.pokemon;
                const canSwitch = [];

                for (let j = 0; j < 6; j++) {
                    const currentPokemon = pokemon[j];
                    if (currentPokemon) {
                        canSwitch.push(j);
                    }
                }

                for (const j of canSwitch) {
                    const switchIndex = (j + 1) as 1 | 2 | 3 | 4 | 5 | 6;
                    legalActions.set(3 + switchIndex, true);
                }
            }
        }
        return { legalActions, isStruggling };
    }

    getMoveset(): Uint8Array {
        const request = this.player.getRequest() as AnyObject;

        const active = (request?.active ??
            [])[0] as Protocol.MoveRequest["active"][0];
        const activeMoves = (active ?? {})?.moves ?? [];
        const switches = (request?.side?.pokemon ??
            []) as Protocol.Request.SideInfo["pokemon"];

        const actionBuffer = new Int16Array(numMovesetFeatures);
        let bufferOffset = 0;

        const assignActionBuffer = (index: number, value: number) => {
            actionBuffer[bufferOffset + index] = value;
        };

        const switchIndices = switches.map((poke) => {
            const { index } = this.player.eventHandler.getPokemon(
                poke.ident,
                false,
            );
            return index;
        });
        let activeIndex = 0;
        for (const [index, poke] of switches.entries()) {
            if (poke.active) {
                activeIndex = switchIndices[index];
                break;
            }
        }

        const pushMoveAction = (
            action:
                | { name: "Recharge"; id: "recharge" }
                | { name: Protocol.MoveName; id: ID }
                | {
                      name: Protocol.MoveName;
                      id: ID;
                      pp: number;
                      maxpp: number;
                      target: MoveTarget;
                      disabled?: boolean;
                  },
        ) => {
            // assignActionBuffer(
            //     MovesetFeature.MOVESET_FEATURE__ACTION_ID,
            //     IndexValueFromEnum(ActionsEnum, `move_${action.id}`),
            // );
            if ("pp" in action) {
                assignActionBuffer(
                    MovesetFeature.MOVESET_FEATURE__HAS_PP,
                    MovesetHasPPEnum.MOVESET_HAS_PP_ENUM__YES,
                );
                assignActionBuffer(
                    MovesetFeature.MOVESET_FEATURE__PP,
                    action.pp,
                );
                assignActionBuffer(
                    MovesetFeature.MOVESET_FEATURE__MAXPP,
                    action.maxpp,
                );
                assignActionBuffer(
                    MovesetFeature.MOVESET_FEATURE__PP_RATIO,
                    MAX_RATIO_TOKEN * (action.pp / action.maxpp),
                );
            } else {
                assignActionBuffer(
                    MovesetFeature.MOVESET_FEATURE__HAS_PP,
                    MovesetHasPPEnum.MOVESET_HAS_PP_ENUM__NO,
                );
            }
            assignActionBuffer(
                MovesetFeature.MOVESET_FEATURE__MOVE_ID,
                IndexValueFromEnum(MovesEnum, action.id),
            );
            // assignActionBuffer(
            //     MovesetFeature.MOVESET_FEATURE__SPECIES_ID,
            //     SpeciesEnum.SPECIES_ENUM___UNSPECIFIED,
            // );
            assignActionBuffer(
                MovesetFeature.MOVESET_FEATURE__ACTION_TYPE,
                MovesetActionTypeEnum.MOVESET_ACTION_TYPE_ENUM__MOVE,
            );
            assignActionBuffer(
                MovesetFeature.MOVESET_FEATURE__ENTITY_IDX,
                activeIndex,
            );
        };

        const pushSwitchAction = (action: Protocol.Request.Pokemon) => {
            let member = this.player.privateBattle.getPokemon(action.ident);
            if (member === null) {
                const activeIdent = (action.ident.slice(0, 2) +
                    "a" +
                    action.ident.slice(2)) as PokemonIdent;
                member = this.player.privateBattle.getPokemon(activeIdent);
            }
            if (member === null) {
                throw new Error();
            }
            // const species = member.species.baseSpecies.toLowerCase();
            // assignActionBuffer(
            //     MovesetFeature.MOVESET_FEATURE__ACTION_ID,
            //     IndexValueFromEnum(ActionsEnum, `switch_${species}`),
            // );
            assignActionBuffer(
                MovesetFeature.MOVESET_FEATURE__MOVE_ID,
                MovesEnum.MOVES_ENUM___SWITCH,
            );
            // assignActionBuffer(
            //     MovesetFeature.MOVESET_FEATURE__SPECIES_ID,
            //     IndexValueFromEnum(SpeciesEnum, species),
            // );
            assignActionBuffer(
                MovesetFeature.MOVESET_FEATURE__ACTION_TYPE,
                MovesetActionTypeEnum.MOVESET_ACTION_TYPE_ENUM__SWITCH,
            );
            assignActionBuffer(
                MovesetFeature.MOVESET_FEATURE__HAS_PP,
                MovesetHasPPEnum.MOVESET_HAS_PP_ENUM__NO,
            );
        };

        for (const action of activeMoves) {
            pushMoveAction(action);
            bufferOffset += numMoveFeatures;
        }
        bufferOffset += (4 - activeMoves.length) * numMoveFeatures;
        // eslint-disable-next-line @typescript-eslint/no-unused-vars
        for (const [_, action] of switches.entries()) {
            const { index: entityIndex } = this.player.eventHandler.getPokemon(
                action.ident,
                false,
            );
            // This was to check a test
            // if (actionIndex !== entityIndex) {
            //     throw new Error(
            //         `Switch action index ${actionIndex} does not match entity index ${entityIndex} for ${action.ident}`,
            //     );
            // }
            pushSwitchAction(action);
            assignActionBuffer(
                MovesetFeature.MOVESET_FEATURE__ENTITY_IDX,
                entityIndex,
            );
            bufferOffset += numMoveFeatures;
        }
        bufferOffset += (6 - switches.length) * numMoveFeatures;

        return new Uint8Array(actionBuffer.buffer);
    }

    getTeamFromSide(
        side: Side,
        playerIndex: number,
        isPublic: boolean = true,
    ): Int16Array {
        const buffer = new Int16Array(6 * numEntityNodeFeatures);

        let offset = 0;
        const team = side.team.slice(0, 6);

        for (const member of team) {
            buffer.set(
                getArrayFromPokemon(member, playerIndex, isPublic),
                offset,
            );
            offset += numEntityNodeFeatures;
        }
        const unkPoke = isMySide(side.n, playerIndex)
            ? unkPokemon1
            : unkPokemon0;
        for (let i = team.length; i < side.totalPokemon; i++) {
            buffer.set(unkPoke, offset);
            offset += numEntityNodeFeatures;
        }
        for (let i = side.totalPokemon; i < 6; i++) {
            buffer.set(nullPokemon, offset);
            offset += numEntityNodeFeatures;
        }
        return buffer;
    }

    // getAllMyMoves(): Int16Array {
    //     const request = this.player.getRequest() as AnyObject;

    //     const active = (request?.active ??
    //         [])[0] as Protocol.MoveRequest["active"][0];
    //     const activeMoves = (active ?? {})?.moves ?? [];
    //     const switches = (request?.side?.pokemon ??
    //         []) as Protocol.Request.SideInfo["pokemon"];

    //     const buffer = new Int16Array(6 * 4 * numActionFields);
    //     let bufferOffset = 0;

    //     for (const [actionIndex, action] of activeMoves.entries()) {
    //         buffer[bufferOffset + ActionsFeature.ACTIONS_FEATURE__MOVE_ID] =
    //             IndexValueFromEnum(MovesEnum, action.id);
    //         buffer[
    //             bufferOffset + ActionsFeature.ACTIONS_FEATURE__ACTION_INDEX
    //         ] = actionIndex;
    //         bufferOffset += numActionFields;
    //     }
    //     for (let i = 0; i < 4 - Math.min(4, activeMoves.length); i++) {
    //         buffer[bufferOffset] = MovesEnum.MOVES_ENUM___NULL;
    //         buffer[
    //             bufferOffset + ActionsFeature.ACTIONS_FEATURE__ACTION_INDEX
    //         ] = i + activeMoves.length;
    //         bufferOffset += numActionFields;
    //     }
    //     for (const [potentialSwitchIndex, potentialSwitch] of switches
    //         .slice(0, 6)
    //         .entries()) {
    //         const numMoves = potentialSwitch.moves.length;
    //         if (!potentialSwitch.active) {
    //             for (const move of potentialSwitch.moves) {
    //                 buffer[
    //                     bufferOffset + ActionsFeature.ACTIONS_FEATURE__MOVE_ID
    //                 ] = IndexValueFromEnum(MovesEnum, move);
    //                 buffer[
    //                     bufferOffset +
    //                         ActionsFeature.ACTIONS_FEATURE__ACTION_INDEX
    //                 ] = potentialSwitchIndex + 4;
    //                 bufferOffset += numActionFields;
    //             }
    //             for (let i = 0; i < 4 - numMoves; i++) {
    //                 buffer[
    //                     bufferOffset + ActionsFeature.ACTIONS_FEATURE__MOVE_ID
    //                 ] = MovesEnum.MOVES_ENUM___NULL;
    //                 bufferOffset += numActionFields;
    //             }
    //         }
    //     }
    //     for (let i = 0; i < 6 - switches.length; i++) {
    //         for (let i = 0; i < 4; i++) {
    //             buffer[bufferOffset + ActionsFeature.ACTIONS_FEATURE__MOVE_ID] =
    //                 MovesEnum.MOVES_ENUM___NULL;
    //             buffer[
    //                 bufferOffset + ActionsFeature.ACTIONS_FEATURE__ACTION_INDEX
    //             ] = i + 4 + Math.min(6, switches.length);
    //             bufferOffset += numActionFields;
    //         }
    //     }

    //     return buffer;
    // }

    // getAllOppMoves(): Int16Array {
    //     const playerIndex = this.player.getPlayerIndex()!;
    //     const oppSide = this.player.publicBattle.sides[1 - playerIndex];

    //     const buffer = new Int16Array(6 * 4 * numActionFields);
    //     let bufferOffset = 0;

    //     let activeIndex = 0;
    //     let switchIndex = 4;

    //     for (const member of oppSide.team.slice(0, 6)) {
    //         for (const move of member.moves.slice(0, 4)) {
    //             buffer[bufferOffset + ActionsFeature.ACTIONS_FEATURE__MOVE_ID] =
    //                 member.fainted || member.hp === 0
    //                     ? MovesEnum.MOVES_ENUM___NULL
    //                     : IndexValueFromEnum(MovesEnum, move);
    //             if (member.isActive()) {
    //                 buffer[
    //                     bufferOffset +
    //                         ActionsFeature.ACTIONS_FEATURE__ACTION_INDEX
    //                 ] = activeIndex;
    //                 activeIndex += 1;
    //             } else {
    //                 buffer[
    //                     bufferOffset +
    //                         ActionsFeature.ACTIONS_FEATURE__ACTION_INDEX
    //                 ] = switchIndex;
    //             }
    //             bufferOffset += numActionFields;
    //         }
    //         const numMoves = member.moves.length;
    //         for (let i = 0; i < 4 - numMoves; i++) {
    //             buffer[bufferOffset + ActionsFeature.ACTIONS_FEATURE__MOVE_ID] =
    //                 member.fainted
    //                     ? MovesEnum.MOVES_ENUM___NULL
    //                     : MovesEnum.MOVES_ENUM___UNK;
    //             if (member.isActive()) {
    //                 buffer[
    //                     bufferOffset +
    //                         ActionsFeature.ACTIONS_FEATURE__ACTION_INDEX
    //                 ] = activeIndex;
    //                 activeIndex += 1;
    //             } else {
    //                 buffer[
    //                     bufferOffset +
    //                         ActionsFeature.ACTIONS_FEATURE__ACTION_INDEX
    //                 ] = switchIndex;
    //             }
    //             bufferOffset += numActionFields;
    //         }
    //         switchIndex += 1;
    //     }
    //     const teamLength = Math.min(oppSide.team.length, 6);
    //     for (
    //         let i = 0;
    //         i < Math.min(6, oppSide.totalPokemon - teamLength);
    //         i++
    //     ) {
    //         for (let i = 0; i < 4; i++) {
    //             buffer[bufferOffset + ActionsFeature.ACTIONS_FEATURE__MOVE_ID] =
    //                 MovesEnum.MOVES_ENUM___UNK;
    //             buffer[
    //                 bufferOffset + ActionsFeature.ACTIONS_FEATURE__ACTION_INDEX
    //             ] = switchIndex;
    //             bufferOffset += numActionFields;
    //         }
    //         switchIndex += 1;
    //     }
    //     for (let i = 0; i < 6 - Math.min(6, oppSide.totalPokemon); i++) {
    //         for (let i = 0; i < 4; i++) {
    //             buffer[bufferOffset + ActionsFeature.ACTIONS_FEATURE__MOVE_ID] =
    //                 MovesEnum.MOVES_ENUM___NULL;
    //             bufferOffset += numActionFields;
    //         }
    //     }

    //     return buffer;
    // }

    getPrivateTeam(playerIndex: number): Int16Array {
        const team = [
            this.getTeamFromSide(
                this.player.privateBattle.sides[playerIndex],
                playerIndex,
                false,
            ),
            // this.getTeamFromSide(
            //     this.player.privateBattle.sides[1 - playerIndex],
            //     playerIndex,
            // ),
        ];
        return concatenateArrays(team);
    }

    getPublicTeam(playerIndex: number): Int16Array {
        const team = [
            this.getTeamFromSide(
                this.player.publicBattle.sides[playerIndex],
                playerIndex,
            ),
            this.getTeamFromSide(
                this.player.publicBattle.sides[1 - playerIndex],
                playerIndex,
            ),
        ];
        return concatenateArrays(team);
    }

    getHistory(numHistory: number = NUM_HISTORY) {
        return this.player.eventHandler.edgeBuffer.getHistory(numHistory);
    }

    getReward() {
        if (this.player.done) {
            if (this.player.finishedEarly) {
                const playerIndex = this.player.getPlayerIndex()!;
                const sideHpSquares = this.player.privateBattle.sides.map(
                    (side) => {
                        const knownHpTotal = side.team.reduce(
                            (acc, pokemon) =>
                                acc + (pokemon.hp / pokemon.maxhp) ** 2,
                            0,
                        );
                        const unknownHpTotal =
                            side.totalPokemon - side.team.length;
                        return (
                            (knownHpTotal + unknownHpTotal) / side.totalPokemon
                        );
                    },
                );
                const ratioDiff =
                    sideHpSquares[playerIndex] - sideHpSquares[1 - playerIndex];
                return Math.floor((MAX_RATIO_TOKEN * ratioDiff) / 2);
            }
            for (let i = this.player.log.length - 1; i >= 0; i--) {
                const line = this.player.log.at(i) ?? "";
                // eslint-disable-next-line @typescript-eslint/no-unused-vars
                const [_, cmd, winner] = line.split("|");
                if (cmd === "win") {
                    return this.player.userName === winner
                        ? MAX_RATIO_TOKEN
                        : -MAX_RATIO_TOKEN;
                } else if (cmd === "tie") {
                    return 0;
                }
            }
        }
        return 0;
    }

    getInfo() {
        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error("Player index is undefined");
        }

        const infoBuffer = new Int16Array(numInfoFeatures);

        infoBuffer[InfoFeature.INFO_FEATURE__PLAYER_INDEX] = playerIndex;
        infoBuffer[InfoFeature.INFO_FEATURE__TURN] =
            this.player.privateBattle.turn;
        infoBuffer[InfoFeature.INFO_FEATURE__DONE] = +this.player.done;
        infoBuffer[InfoFeature.INFO_FEATURE__REQUEST_COUNT] =
            this.player.requestCount;

        infoBuffer[InfoFeature.INFO_FEATURE__WIN_REWARD] = this.getReward();

        return new Uint8Array(infoBuffer.buffer);
    }

    static toReadableTeam(buffer: Int16Array) {
        const entityDatums = [];
        const numEntites = buffer.length / numEntityNodeFeatures;
        for (let entityIndex = 0; entityIndex < numEntites; entityIndex++) {
            const start = entityIndex * numEntityNodeFeatures;
            const end = (entityIndex + 1) * numEntityNodeFeatures;
            entityDatums.push(
                entityNodeArrayToObject(buffer.slice(start, end)),
            );
        }
        return entityDatums;
    }

    getCurrentContext() {
        const currentContextBuffer = new Int16Array(numFieldFeatures);
        const playerIndex = this.player.getPlayerIndex()!;
        for (const side of this.player.privateBattle.sides) {
            const mySide = isMySide(side.n, playerIndex) === 1;
            const sideOffset = mySide
                ? FieldFeature.FIELD_FEATURE__MY_SIDECONDITIONS0
                : FieldFeature.FIELD_FEATURE__OPP_SIDECONDITIONS0;
            const spikesOffset = mySide
                ? FieldFeature.FIELD_FEATURE__MY_SPIKES
                : FieldFeature.FIELD_FEATURE__OPP_SPIKES;
            const toxisSpikesOffset = mySide
                ? FieldFeature.FIELD_FEATURE__MY_TOXIC_SPIKES
                : FieldFeature.FIELD_FEATURE__OPP_TOXIC_SPIKES;

            let sideConditionBuffer = BigInt(0b0);
            for (const [id] of Object.entries(side.sideConditions)) {
                const featureIndex = IndexValueFromEnum(SideconditionEnum, id);
                sideConditionBuffer |= BigInt(1) << BigInt(featureIndex);
            }
            currentContextBuffer.set(
                bigIntToInt16Array(sideConditionBuffer),
                sideOffset,
            );
            if (side.sideConditions.spikes) {
                currentContextBuffer[spikesOffset] =
                    side.sideConditions.spikes.level;
            }
            if (side.sideConditions.toxicspikes) {
                currentContextBuffer[toxisSpikesOffset] =
                    side.sideConditions.toxicspikes.level;
            }
        }
        const field = this.player.publicBattle.field;
        const weatherIndex = field.weatherState.id
            ? IndexValueFromEnum(
                  WeatherEnum,
                  WEATHERS[field.weatherState.id as never],
              )
            : WeatherEnum.WEATHER_ENUM___NULL;

        currentContextBuffer[FieldFeature.FIELD_FEATURE__WEATHER_ID] =
            weatherIndex;
        currentContextBuffer[FieldFeature.FIELD_FEATURE__WEATHER_MAX_DURATION] =
            field.weatherState.maxDuration;
        currentContextBuffer[FieldFeature.FIELD_FEATURE__WEATHER_MIN_DURATION] =
            field.weatherState.minDuration;
        return new Uint8Array(currentContextBuffer.buffer);
    }

    build(): EnvironmentState {
        const request = this.player.getRequest();
        if (!this.player.done && request === undefined) {
            throw new Error("Need Request");
        }

        const state = new EnvironmentState();
        const info = this.getInfo();
        state.setInfo(info);

        const { legalActions } = StateHandler.getLegalActions(request);
        state.setLegalActions(legalActions.buffer);

        const {
            historyEntityNodes,
            historyEntityEdges,
            historyField,
            historyLength,
        } = this.getHistory(NUM_HISTORY);
        state.setHistoryEntityNodes(historyEntityNodes);
        state.setHistoryEntityEdges(historyEntityEdges);
        state.setHistoryField(historyField);
        state.setHistoryLength(historyLength);

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error("Player index is undefined");
        }

        const privateTeam = this.getPrivateTeam(playerIndex);
        state.setPrivateTeam(new Uint8Array(privateTeam.buffer));

        const publicTeam = this.getPublicTeam(playerIndex);
        state.setPublicTeam(new Uint8Array(publicTeam.buffer));

        state.setMoveset(this.getMoveset());

        state.setCurrentContext(this.getCurrentContext());

        return state;
    }
}
