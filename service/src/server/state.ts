import { AnyObject, PokemonSet, Teams, TeamValidator, toID } from "@pkmn/sim";
import {
    Args,
    BattleInitArgName,
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
    NaturesEnum,
    SideconditionEnum,
    SpeciesEnum,
    StatusEnum,
    TypechartEnum,
    VolatilestatusEnum,
    WeatherEnum,
} from "../../protos/enums_pb";
import {
    EnumMappings,
    ITOS,
    MoveIndex,
    NUM_HISTORY,
    WILDCARDS,
    jsonDatum,
    numActionFeatures,
    numEntityEdgeFeatures,
    numFieldFeatures,
    numInfoFeatures,
    numMoveFeatures,
    numPackedSetFeatures,
    numPrivateEntityNodeFeatures,
    numPublicEntityNodeFeatures,
    numRevealedEntityNodeFeatures,
    sampleTeams,
} from "./data";
import { Battle, NA, Pokemon, Side } from "@pkmn/client";
import { Ability, Item, BoostID } from "@pkmn/dex-types";
import { ID, MoveTarget, SideID } from "@pkmn/types";
import { Condition, Effect } from "@pkmn/data";
import { OneDBoolean, TypedArray } from "./utils";
import {
    ActionType,
    EntityEdgeFeature,
    EntityEdgeFeatureMap,
    EntityPrivateNodeFeature,
    EntityPublicNodeFeature,
    EntityRevealedNodeFeature,
    FieldFeature,
    FieldFeatureMap,
    InfoFeature,
    MovesetFeature,
    MovesetHasPP,
    PackedSetFeature,
    RequestType,
    RewardFeature,
} from "../../protos/features_pb";
import { TrainablePlayerAI } from "./runner";
import { EnvironmentState, ActionEnum } from "../../protos/service_pb";
import { Move } from "@pkmn/dex";

type RemovePipes<T extends string> = T extends `|${infer U}|` ? U : T;
type MajorArgNames =
    | RemovePipes<BattleMajorArgName>
    | RemovePipes<BattleProgressArgName>
    | RemovePipes<BattleInitArgName>;
type MinorArgNames = RemovePipes<BattleMinorArgName>;

const MAX_RATIO_TOKEN = 16384;

export function getSampleTeam(format: string, include?: string): string {
    const internalFormat = format
        .replace("_ou_all_formats", "ou")
        .replace("_ou_only_format", "ou");
    const teams = sampleTeams[internalFormat];
    if (!teams || teams.length === 0) {
        throw new Error(`No sample teams found for format: ${format}`);
    }
    if (include) {
        const filteredTeams = teams.filter((team) => team.includes(include));
        if (filteredTeams.length > 0) {
            return filteredTeams[
                Math.floor(Math.random() * filteredTeams.length)
            ];
        }
    }
    return teams[Math.floor(Math.random() * teams.length)];
}

export function randomSampleTeam(format: string): string {
    const formatSampleTeams = sampleTeams[format];
    if (!formatSampleTeams || formatSampleTeams.length === 0) {
        throw new Error(`No sample teams found for format: ${format}`);
    }
    return formatSampleTeams[
        Math.floor(Math.random() * formatSampleTeams.length)
    ];
}

function capitalize(str: string): string {
    if (str.length === 0) {
        return str;
    }
    return str[0].toUpperCase() + str.slice(1);
}

export function packedSetFromArraySlice(packedSetSlice: number[]): string {
    const species =
        ITOS.species[
            packedSetSlice[PackedSetFeature.PACKED_SET_FEATURE__SPECIES]
        ];
    const nickname = species;
    const item =
        ITOS.items[packedSetSlice[PackedSetFeature.PACKED_SET_FEATURE__ITEM]];
    const ability =
        ITOS.abilities[
            packedSetSlice[PackedSetFeature.PACKED_SET_FEATURE__ABILITY]
        ];
    const moves = [];
    for (const moveIndex of [
        PackedSetFeature.PACKED_SET_FEATURE__MOVE1,
        PackedSetFeature.PACKED_SET_FEATURE__MOVE2,
        PackedSetFeature.PACKED_SET_FEATURE__MOVE3,
        PackedSetFeature.PACKED_SET_FEATURE__MOVE4,
    ]) {
        const moveId = packedSetSlice[moveIndex];
        const moveName =
            moveId < MovesEnum.MOVES_ENUM__10000000VOLTTHUNDERBOLT
                ? ""
                : ITOS.moves[moveId];
        moves.push(moveName);
    }
    const nature =
        ITOS.natures[
            packedSetSlice[PackedSetFeature.PACKED_SET_FEATURE__NATURE]
        ];
    const evs = [];
    for (const evIndex of [
        PackedSetFeature.PACKED_SET_FEATURE__HP_EV,
        PackedSetFeature.PACKED_SET_FEATURE__ATK_EV,
        PackedSetFeature.PACKED_SET_FEATURE__DEF_EV,
        PackedSetFeature.PACKED_SET_FEATURE__SPA_EV,
        PackedSetFeature.PACKED_SET_FEATURE__SPD_EV,
        PackedSetFeature.PACKED_SET_FEATURE__SPE_EV,
    ]) {
        evs.push(packedSetSlice[evIndex] * 4);
    }
    const gender =
        ITOS.genders[
            packedSetSlice[PackedSetFeature.PACKED_SET_FEATURE__GENDER]
        ].toUpperCase();
    const ivs = "";
    const shiny = "";
    const level = "";
    const happiness = "";
    const pokeball = "";

    const hiddenPowerToken =
        packedSetSlice[PackedSetFeature.PACKED_SET_FEATURE__HIDDENPOWERTYPE];
    const hiddenpowertype =
        hiddenPowerToken == 0
            ? ""
            : capitalize(
                  ITOS.typechart[
                      packedSetSlice[
                          PackedSetFeature.PACKED_SET_FEATURE__HIDDENPOWERTYPE
                      ]
                  ],
              );
    const gigantamax = "";
    const dynamaxlevel = "";
    const teratype = capitalize(
        ITOS.typechart[
            packedSetSlice[PackedSetFeature.PACKED_SET_FEATURE__TERATYPE]
        ],
    );
    return `${nickname}|${species}|${item}|${ability}|${moves.join(",")}|${nature}|${evs.join(",")}|${gender}|${ivs}|${shiny}|${level}|${happiness},${hiddenpowertype},${pokeball},${gigantamax},${dynamaxlevel},${teratype}`;
}

export function generateTeamFromArray(packedTeam: number[]): string {
    const generatedSets: string[] = [];
    for (let i = 0; i < 6; i += 1) {
        const packedSetSlice = packedTeam.slice(
            i * numPackedSetFeatures,
            (i + 1) * numPackedSetFeatures,
        );
        const packedSet = packedSetFromArraySlice(packedSetSlice);
        generatedSets.push(packedSet);
    }
    return generatedSets.join("]");
}

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

const entityPrivateArrayToObject = (array: Int16Array) => {
    const moveIndicies = Array.from(
        array.slice(
            EntityPrivateNodeFeature.ENTITY_PRIVATE_NODE_FEATURE__MOVEID0,
            EntityPrivateNodeFeature.ENTITY_PRIVATE_NODE_FEATURE__MOVEID3 + 1,
        ),
    );

    return {
        species:
            jsonDatum["species"][
                array[
                    EntityPrivateNodeFeature
                        .ENTITY_PRIVATE_NODE_FEATURE__SPECIES
                ]
            ],
        item: jsonDatum["items"][
            array[EntityPrivateNodeFeature.ENTITY_PRIVATE_NODE_FEATURE__ITEM]
        ],
        ability:
            jsonDatum["abilities"][
                array[
                    EntityPrivateNodeFeature
                        .ENTITY_PRIVATE_NODE_FEATURE__ABILITY
                ]
            ],
        moves: moveIndicies.map((index) => jsonDatum["moves"][index]),
        teraType:
            jsonDatum["typechart"][
                array[
                    EntityPrivateNodeFeature
                        .ENTITY_PRIVATE_NODE_FEATURE__TERA_TYPE
                ]
            ],
    };
};

const entityPublicArrayToObject = (array: Int16Array) => {
    const volatilesFlat = array.slice(
        EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__VOLATILES0,
        EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__VOLATILES8 + 1,
    );
    const volatilesIndices = int16ArrayToBitIndices(volatilesFlat);

    const typechangeFlat = array.slice(
        EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__TYPECHANGE0,
        EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__TYPECHANGE1 + 1,
    );
    const typechangeIndices = int16ArrayToBitIndices(typechangeFlat);

    return {
        hp:
            array[
                EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__HP_RATIO
            ] / MAX_RATIO_TOKEN,
        fainted:
            !!array[
                EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__FAINTED
            ],
        volatiles: volatilesIndices.map(
            (index) => jsonDatum["volatileStatus"][index],
        ),
        typechange: typechangeIndices.map(
            (index) => jsonDatum["typechart"][index],
        ),
        active: array[
            EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__ACTIVE
        ],
        side: array[EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__SIDE],
        status: jsonDatum["status"][
            array[EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__STATUS]
        ],
    };
};

const entityRevealedArrayToObject = (array: Int16Array) => {
    const moveIndicies = Array.from(
        array.slice(
            EntityRevealedNodeFeature.ENTITY_REVEALED_NODE_FEATURE__MOVEID0,
            EntityRevealedNodeFeature.ENTITY_REVEALED_NODE_FEATURE__MOVEID3 + 1,
        ),
    );

    return {
        species:
            jsonDatum["species"][
                array[
                    EntityRevealedNodeFeature
                        .ENTITY_REVEALED_NODE_FEATURE__SPECIES
                ]
            ],
        item: jsonDatum["items"][
            array[EntityRevealedNodeFeature.ENTITY_REVEALED_NODE_FEATURE__ITEM]
        ],
        ability:
            jsonDatum["abilities"][
                array[
                    EntityRevealedNodeFeature
                        .ENTITY_REVEALED_NODE_FEATURE__ABILITY
                ]
            ],
        moves: moveIndicies.map((index) => jsonDatum["moves"][index]),
        teraType:
            jsonDatum["typechart"][
                array[
                    EntityRevealedNodeFeature
                        .ENTITY_REVEALED_NODE_FEATURE__TERA_TYPE
                ]
            ],
    };
};

const moveArrayToObject = (array: Int16Array) => {
    return {
        pp_ratio:
            array[MovesetFeature.MOVESET_FEATURE__PP_RATIO] / MAX_RATIO_TOKEN,
        move_id:
            jsonDatum["moves"][array[MovesetFeature.MOVESET_FEATURE__MOVE_ID]],
        pp: array[MovesetFeature.MOVESET_FEATURE__PP],
        maxpp: array[MovesetFeature.MOVESET_FEATURE__MAXPP],
        has_pp: !!array[MovesetFeature.MOVESET_FEATURE__HAS_PP],
        action_type: array[MovesetFeature.MOVESET_FEATURE__ACTION_TYPE],
        entity_idx: array[MovesetFeature.MOVESET_FEATURE__ENTITY_IDX],
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
        numRelevant: array[FieldFeature.FIELD_FEATURE__NUM_RELEVANT],
    };
};

const WEATHERS = {
    sand: "sandstorm",
    sun: "sunnyday",
    rain: "raindance",
    hail: "hail",
    snow: "snowscape",
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

export function EnumFromIndexValue<T extends EnumMappings>(
    enumDatum: T,
    value: T[keyof T],
): string {
    for (const key in enumDatum) {
        if (enumDatum[key] === value) {
            const sanitizedKey = SanitizeKey(enumDatum, key);
            return sanitizedKey.toString().split("__")[1];
        }
    }
    throw new Error(`${value} not in mapping`);
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

function getBlankPublicPokemonArr() {
    return new POKEMON_ARRAY_CONSTRUCTOR(numPublicEntityNodeFeatures);
}

function getBlankRevealedPokemonArr() {
    return new POKEMON_ARRAY_CONSTRUCTOR(numRevealedEntityNodeFeatures);
}

function getBlankPrivatePokemonArr() {
    return new POKEMON_ARRAY_CONSTRUCTOR(numPrivateEntityNodeFeatures);
}

function getUnkRevealedPokemon() {
    const data = getBlankRevealedPokemonArr();
    data[EntityRevealedNodeFeature.ENTITY_REVEALED_NODE_FEATURE__SPECIES] =
        SpeciesEnum.SPECIES_ENUM___UNK;

    // Item
    data[EntityRevealedNodeFeature.ENTITY_REVEALED_NODE_FEATURE__ITEM] =
        ItemsEnum.ITEMS_ENUM___UNK;

    // Ability
    data[EntityRevealedNodeFeature.ENTITY_REVEALED_NODE_FEATURE__ABILITY] =
        AbilitiesEnum.ABILITIES_ENUM___UNK;

    // Moves
    data[EntityRevealedNodeFeature.ENTITY_REVEALED_NODE_FEATURE__MOVEID0] =
        MovesEnum.MOVES_ENUM___UNK;
    data[EntityRevealedNodeFeature.ENTITY_REVEALED_NODE_FEATURE__MOVEID1] =
        MovesEnum.MOVES_ENUM___UNK;
    data[EntityRevealedNodeFeature.ENTITY_REVEALED_NODE_FEATURE__MOVEID2] =
        MovesEnum.MOVES_ENUM___UNK;
    data[EntityRevealedNodeFeature.ENTITY_REVEALED_NODE_FEATURE__MOVEID3] =
        MovesEnum.MOVES_ENUM___UNK;

    // Teratype
    data[EntityRevealedNodeFeature.ENTITY_REVEALED_NODE_FEATURE__TERA_TYPE] =
        TypechartEnum.TYPECHART_ENUM___UNK;
    return data;
}

function getUnkPublicPokemon() {
    const data = getBlankPublicPokemonArr();

    data[EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__ITEM_EFFECT] =
        ItemeffecttypesEnum.ITEMEFFECTTYPES_ENUM___NULL;
    data[EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__GENDER] =
        GendernameEnum.GENDERNAME_ENUM___UNK;
    data[EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__ACTIVE] = 0;
    data[EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__FAINTED] = 0;
    data[EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__HP] = 100;
    data[EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__MAXHP] = 100;
    data[EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__HP_RATIO] =
        MAX_RATIO_TOKEN; // Full Health;
    data[EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__STATUS] =
        StatusEnum.STATUS_ENUM___NULL;
    data[EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__TOXIC_TURNS] = 0;
    data[EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__SLEEP_TURNS] = 0;
    data[
        EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__BEING_CALLED_BACK
    ] = 0;
    data[EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__TRAPPED] = 0;
    data[EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__NEWLY_SWITCHED] =
        0;
    data[EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__LEVEL] = 100;
    data[EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__HAS_STATUS] = 0;
    data[EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__BOOST_ATK_VALUE] =
        0;
    data[EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__BOOST_DEF_VALUE] =
        0;
    data[EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__BOOST_SPA_VALUE] =
        0;
    data[EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__BOOST_SPD_VALUE] =
        0;
    data[EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__BOOST_SPE_VALUE] =
        0;
    data[
        EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__BOOST_ACCURACY_VALUE
    ] = 0;
    data[
        EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__BOOST_EVASION_VALUE
    ] = 0;
    data[EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__VOLATILES0] = 0;
    data[EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__VOLATILES1] = 0;
    data[EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__VOLATILES2] = 0;
    data[EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__VOLATILES3] = 0;
    data[EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__VOLATILES4] = 0;
    data[EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__VOLATILES5] = 0;
    data[EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__VOLATILES6] = 0;
    data[EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__VOLATILES7] = 0;
    data[EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__VOLATILES8] = 0;
    data[EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__TYPECHANGE0] = 0;
    data[EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__TYPECHANGE1] = 0;
    data[EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__NUM_MOVES] = 0;
    data[EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__LAST_MOVE] =
        MovesEnum.MOVES_ENUM___UNK;
    data[EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__MEGA] = 0;
    data[EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__PRIMAL] = 0;
    data[EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__MOVEPP0] = 0;
    data[EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__MOVEPP1] = 0;
    data[EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__MOVEPP2] = 0;
    data[EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__MOVEPP3] = 0;

    return data;
}

function getUnkPokemon(n: number) {
    const publicData = getUnkPublicPokemon();
    const revealedData = getUnkRevealedPokemon();

    // Side
    publicData[EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__SIDE] = n;
    return { publicData, revealedData };
}

const unkPokemon0 = getUnkPokemon(0);
const unkPokemon1 = getUnkPokemon(1);

function getNullPokemon() {
    const publicData = getBlankPublicPokemonArr();
    const revealedData = getUnkRevealedPokemon();
    revealedData[
        EntityRevealedNodeFeature.ENTITY_REVEALED_NODE_FEATURE__SPECIES
    ] = SpeciesEnum.SPECIES_ENUM___NULL;
    return { publicData, revealedData };
}

const nullPokemon = getNullPokemon();

function tryFindIndex(enumDatum: EnumMappings, keys: string[]) {
    for (const key of keys) {
        try {
            return IndexValueFromEnum(enumDatum, key);
        } catch (err) {
            console.log(err);
            continue;
        }
    }
    throw new Error(`None of the keys ${keys} found in enum mapping`);
}

function getArrayFromPrivatePokemon(
    candidate: Pokemon | null | undefined,
    // pokemonSet: PokemonSet,
    pokemonSet: Protocol.Request.Pokemon,
) {
    const dataArr = getBlankPrivatePokemonArr();

    if (candidate === null || candidate === undefined) {
        return dataArr;
    }

    let pokemon: Pokemon;
    if (
        candidate.volatiles.transform !== undefined &&
        candidate.volatiles.transform.pokemon !== undefined
    ) {
        pokemon = candidate.volatiles.transform.pokemon as Pokemon;
    } else {
        pokemon = candidate;
    }

    if (pokemonSet === null || pokemonSet === undefined) {
        throw new Error("Private data requested for null or undefined set");
    }

    dataArr[EntityPrivateNodeFeature.ENTITY_PRIVATE_NODE_FEATURE__SPECIES] =
        tryFindIndex(SpeciesEnum, [
            pokemon.baseSpecies.id,
            pokemon.baseSpecies.baseSpecies.toLowerCase(),
        ]);

    dataArr[EntityPrivateNodeFeature.ENTITY_PRIVATE_NODE_FEATURE__ITEM] =
        !!pokemonSet.item
            ? IndexValueFromEnum(ItemsEnum, pokemonSet.item)
            : ItemsEnum.ITEMS_ENUM___NULL;
    dataArr[EntityPrivateNodeFeature.ENTITY_PRIVATE_NODE_FEATURE__ABILITY] =
        IndexValueFromEnum(AbilitiesEnum, pokemonSet.ability);

    const moveset = pokemonSet.moves;
    dataArr[EntityPrivateNodeFeature.ENTITY_PRIVATE_NODE_FEATURE__MOVEID0] =
        moveset[0]
            ? IndexValueFromEnum(MovesEnum, moveset[0])
            : MovesEnum.MOVES_ENUM___NULL;
    dataArr[EntityPrivateNodeFeature.ENTITY_PRIVATE_NODE_FEATURE__MOVEID1] =
        moveset[1]
            ? IndexValueFromEnum(MovesEnum, moveset[1])
            : MovesEnum.MOVES_ENUM___NULL;
    dataArr[EntityPrivateNodeFeature.ENTITY_PRIVATE_NODE_FEATURE__MOVEID2] =
        moveset[2]
            ? IndexValueFromEnum(MovesEnum, moveset[2])
            : MovesEnum.MOVES_ENUM___NULL;
    dataArr[EntityPrivateNodeFeature.ENTITY_PRIVATE_NODE_FEATURE__MOVEID3] =
        moveset[3]
            ? IndexValueFromEnum(MovesEnum, moveset[3])
            : MovesEnum.MOVES_ENUM___NULL;

    const stats = pokemonSet.stats ?? {};
    dataArr[EntityPrivateNodeFeature.ENTITY_PRIVATE_NODE_FEATURE__HP_STAT] =
        pokemonSet.maxhp;
    dataArr[EntityPrivateNodeFeature.ENTITY_PRIVATE_NODE_FEATURE__ATK_STAT] =
        stats.atk ?? 0;
    dataArr[EntityPrivateNodeFeature.ENTITY_PRIVATE_NODE_FEATURE__DEF_STAT] =
        stats.def ?? 0;
    dataArr[EntityPrivateNodeFeature.ENTITY_PRIVATE_NODE_FEATURE__SPA_STAT] =
        stats.spa ?? 0;
    dataArr[EntityPrivateNodeFeature.ENTITY_PRIVATE_NODE_FEATURE__SPD_STAT] =
        stats.spd ?? 0;
    dataArr[EntityPrivateNodeFeature.ENTITY_PRIVATE_NODE_FEATURE__SPE_STAT] =
        stats.spe ?? 0;

    const teraType = pokemonSet.teraType
        ? IndexValueFromEnum(TypechartEnum, pokemonSet.teraType)
        : TypechartEnum.TYPECHART_ENUM___NULL;
    dataArr[EntityPrivateNodeFeature.ENTITY_PRIVATE_NODE_FEATURE__TERA_TYPE] =
        teraType;

    return dataArr;
}

function getArrayFromPublicPokemon(
    candidate: Pokemon | null,
    relativeSide: number,
): {
    publicData: Int16Array;
    revealedData: Int16Array;
} {
    const { publicData, revealedData } =
        relativeSide === 0 ? getUnkPokemon(0) : getUnkPokemon(1);

    if (candidate === null || candidate === undefined) {
        return { publicData, revealedData: nullPokemon.revealedData };
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

    // Terastallized
    const teraType = pokemon.terastallized
        ? IndexValueFromEnum(TypechartEnum, pokemon.terastallized)
        : TypechartEnum.TYPECHART_ENUM___UNK;
    revealedData[
        EntityRevealedNodeFeature.ENTITY_REVEALED_NODE_FEATURE__TERA_TYPE
    ] = teraType;

    const ability = pokemon.ability;

    // We take candidate item here instead of pokemons since
    // transformed does not copy item
    const item = candidate.item;
    const itemEffect = candidate.itemEffect;

    // Moves are stored on candidate
    const moveSlots = candidate.moveSlots.slice(0, 4);
    const moveIds = [];
    const movePps = [];

    if (moveSlots) {
        for (const move of moveSlots) {
            let { id } = move;
            if (id.startsWith("return")) {
                id = "return" as ID;
            } else if (id.startsWith("frustration")) {
                id = "frustration" as ID;
            } else if (id.startsWith("hiddenpower")) {
                const power = parseInt(id.slice(-2));
                if (isNaN(power)) {
                    id = "hiddenpower" as ID;
                } else {
                    id = id.slice(0, -2) as ID;
                }
            } else if (id.startsWith("move: ")) {
                id = id.slice("move: ".length) as ID;
            }
            const ppUsed = move.ppUsed;
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

    for (remainingIndex; remainingIndex < 4; remainingIndex++) {
        moveIds.push(MovesEnum.MOVES_ENUM___UNK);
        movePps.push(0);
    }

    revealedData[
        EntityRevealedNodeFeature.ENTITY_REVEALED_NODE_FEATURE__SPECIES
    ] = tryFindIndex(SpeciesEnum, [
        pokemon.baseSpecies.id,
        pokemon.baseSpecies.baseSpecies.toLowerCase(),
    ]);

    if (item) {
        revealedData[
            EntityRevealedNodeFeature.ENTITY_REVEALED_NODE_FEATURE__ITEM
        ] = IndexValueFromEnum<typeof ItemsEnum>(ItemsEnum, item);
    } else if (candidate.volatiles.itemremoved) {
        revealedData[
            EntityRevealedNodeFeature.ENTITY_REVEALED_NODE_FEATURE__ITEM
        ] = ItemsEnum.ITEMS_ENUM___NULL;
    } else {
        revealedData[
            EntityRevealedNodeFeature.ENTITY_REVEALED_NODE_FEATURE__ITEM
        ] = ItemsEnum.ITEMS_ENUM___UNK;
    }
    publicData[
        EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__ITEM_EFFECT
    ] = itemEffect
        ? IndexValueFromEnum(ItemeffecttypesEnum, itemEffect)
        : ItemeffecttypesEnum.ITEMEFFECTTYPES_ENUM___NULL;

    const possibleAbilities = Object.values(pokemon.baseSpecies.abilities);
    if (ability) {
        if (ability === "noability" || ability === "none") {
            revealedData[
                EntityRevealedNodeFeature.ENTITY_REVEALED_NODE_FEATURE__ABILITY
            ] = AbilitiesEnum.ABILITIES_ENUM__NOABILITY;
        } else {
            const actualAbility = IndexValueFromEnum(AbilitiesEnum, ability);
            revealedData[
                EntityRevealedNodeFeature.ENTITY_REVEALED_NODE_FEATURE__ABILITY
            ] = actualAbility;
        }
    } else if (possibleAbilities.length === 1) {
        const onlyAbility = possibleAbilities[0]
            ? IndexValueFromEnum(AbilitiesEnum, possibleAbilities[0])
            : AbilitiesEnum.ABILITIES_ENUM___UNK;
        revealedData[
            EntityRevealedNodeFeature.ENTITY_REVEALED_NODE_FEATURE__ABILITY
        ] = onlyAbility;
    } else {
        revealedData[
            EntityRevealedNodeFeature.ENTITY_REVEALED_NODE_FEATURE__ABILITY
        ] = AbilitiesEnum.ABILITIES_ENUM___UNK;
    }

    // We take candidate lastMove here instead of pokemons since
    // transformed does not lastMove
    if (candidate.lastMove === "") {
        publicData[
            EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__LAST_MOVE
        ] = MovesEnum.MOVES_ENUM___NULL;
    } else if (candidate.lastMove === "switch-in") {
        publicData[
            EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__LAST_MOVE
        ] = MovesEnum.MOVES_ENUM___SWITCH_IN;
    } else {
        publicData[
            EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__LAST_MOVE
        ] = IndexValueFromEnum(MovesEnum, candidate.lastMove);
    }

    publicData[EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__GENDER] =
        IndexValueFromEnum(GendernameEnum, pokemon.gender);

    const position = candidate.ident.at(2);
    if (candidate.isActive()) {
        if (position === "a") {
            publicData[
                EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__ACTIVE
            ] = 1;
        } else if (position === "b") {
            publicData[
                EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__ACTIVE
            ] = 2;
        }
    } else {
        publicData[EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__ACTIVE] =
            0;
    }

    publicData[EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__FAINTED] =
        +candidate.fainted;

    // We take candidate HP here instead of pokemons since
    // transformed does not copy HP
    const isHpBug = !candidate.fainted && candidate.hp === 0;
    const hp = isHpBug ? 100 : candidate.hp;
    const maxHp = isHpBug ? 100 : candidate.maxhp;
    const hpRatio = hp / maxHp;
    publicData[EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__HP] = hp;
    publicData[EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__MAXHP] =
        maxHp;
    publicData[EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__HP_RATIO] =
        Math.floor(MAX_RATIO_TOKEN * hpRatio);

    // We take candidate status here instead of pokemons since
    // transformed does not copy status
    publicData[EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__STATUS] =
        candidate.status
            ? IndexValueFromEnum(StatusEnum, candidate.status)
            : StatusEnum.STATUS_ENUM___NULL;
    publicData[EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__HAS_STATUS] =
        candidate.status ? 1 : 0;
    publicData[
        EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__TOXIC_TURNS
    ] = candidate.statusState.toxicTurns;
    publicData[
        EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__SLEEP_TURNS
    ] = candidate.statusState.sleepTurns;
    publicData[
        EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__BEING_CALLED_BACK
    ] = candidate.beingCalledBack ? 1 : 0;
    publicData[EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__TRAPPED] =
        candidate.trapped ? 1 : 0;
    publicData[
        EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__NEWLY_SWITCHED
    ] = candidate.newlySwitched ? 1 : 0;

    // We take pokemon level here
    publicData[EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__LEVEL] =
        pokemon.level;

    for (let moveIndex = 0; moveIndex < 4; moveIndex++) {
        revealedData[
            EntityRevealedNodeFeature[
                `ENTITY_REVEALED_NODE_FEATURE__MOVEID${moveIndex as MoveIndex}`
            ]
        ] = moveIds[moveIndex];
        publicData[
            EntityPublicNodeFeature[
                `ENTITY_PUBLIC_NODE_FEATURE__MOVEPP${moveIndex as MoveIndex}`
            ]
        ] = movePps[moveIndex];
    }

    // Only copy candidate volatiles
    let volatiles = BigInt(0b0);
    for (const [key] of Object.entries(candidate.volatiles)) {
        const index = getVolatileStatusToken(
            key.startsWith("fallen") ? "fallen" : key,
        );
        volatiles |= BigInt(1) << BigInt(index);
    }
    publicData.set(
        bigIntToInt16Array(volatiles),
        EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__VOLATILES0,
    );

    publicData[EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__SIDE] =
        relativeSide;

    // Only copy pokemon boosts
    publicData[
        EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__BOOST_ATK_VALUE
    ] = pokemon.boosts.atk ?? 0;
    publicData[
        EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__BOOST_DEF_VALUE
    ] = pokemon.boosts.def ?? 0;
    publicData[
        EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__BOOST_SPA_VALUE
    ] = pokemon.boosts.spa ?? 0;
    publicData[
        EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__BOOST_SPD_VALUE
    ] = pokemon.boosts.spd ?? 0;
    publicData[
        EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__BOOST_SPE_VALUE
    ] = pokemon.boosts.spe ?? 0;
    publicData[
        EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__BOOST_EVASION_VALUE
    ] = pokemon.boosts.evasion ?? 0;
    publicData[
        EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__BOOST_ACCURACY_VALUE
    ] = pokemon.boosts.accuracy ?? 0;

    // Copy candidate type change
    let typeChanged = BigInt(0b0);
    const typechangeVolatile = candidate.volatiles.typechange;
    if (typechangeVolatile) {
        if (typechangeVolatile.apparentType) {
            for (const type of typechangeVolatile.apparentType.split("/")) {
                const index =
                    type === "???"
                        ? TypechartEnum.TYPECHART_ENUM__TYPELESS
                        : IndexValueFromEnum(TypechartEnum, type);
                typeChanged |= BigInt(1) << BigInt(index);
            }
        }
    }
    publicData.set(
        bigIntToInt16Array(typeChanged),
        EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__TYPECHANGE0,
    );

    return { publicData, revealedData };
}

function getEffectToken(effect: Partial<Effect>): number {
    const { id } = effect;
    if (id) {
        let value = undefined;
        const testIds: string[] = [id];
        let haveAdded = false;
        while (testIds.length > 0) {
            const testId = testIds.shift()!;
            try {
                value = IndexValueFromEnum(EffectEnum, testId);
                return value;
            } catch (err) {
                if (testIds.length === 0) {
                    if (!haveAdded) {
                        testIds.push(
                            ...[
                                id.slice("move".length),
                                id.slice("item".length),
                                id.slice("pokemon".length),
                                id.slice("ability".length),
                                id.slice("condition".length),
                            ],
                        );
                        haveAdded = true;
                    } else {
                        return EffectEnum.EFFECT_ENUM___UNK;
                    }
                }
            }
        }
        return EffectEnum.EFFECT_ENUM___UNK;
    }
    return EffectEnum.EFFECT_ENUM___NULL;
}

function getVolatileStatusToken(id: string): number {
    let value = undefined;
    const testIds: string[] = [id];
    let haveAdded = false;
    while (testIds.length > 0) {
        const testId = testIds.shift()!;
        try {
            value = IndexValueFromEnum(VolatilestatusEnum, testId);
            return value;
        } catch (err) {
            if (testIds.length === 0) {
                if (!haveAdded) {
                    testIds.push(
                        ...[
                            id.slice("move".length),
                            id.slice("item".length),
                            id.slice("ability".length),
                            id.slice("condition".length),
                        ],
                    );
                    haveAdded = true;
                } else {
                    console.log(id, err);
                    return VolatilestatusEnum.VOLATILESTATUS_ENUM___UNK;
                }
            }
        }
    }
    throw new Error("Volatile status token not found");
}

class DynamicArray {
    buffer: Int16Array;
    length: number = 0;
    increment: number = 0;
    currentCursor: number = 0;
    prevCursor: number = 0;

    constructor(length: number, increment: number) {
        this.buffer = new Int16Array(length * increment);
        this.length = length;
        this.increment = increment;
        this.currentCursor = 0;
        this.prevCursor = 0;
    }

    setNextSlice(data: Int16Array, sizeOverride?: number) {
        this.buffer.set(data, this.currentCursor);
        this.prevCursor = this.currentCursor;
        this.currentCursor += (sizeOverride ?? 1) * this.increment;
    }

    setValue(index: number, value: number) {
        this.buffer[this.prevCursor + index] = value;
    }

    set(data: DynamicArray, offset: number = 0) {
        this.buffer.set(data.buffer, offset);
    }

    getLatestSlice(length: number): Int16Array {
        if (this.currentCursor === 0) {
            return this.buffer.slice(0, length * this.increment);
        }
        const start = this.currentCursor - length * this.increment;
        const end = this.currentCursor;
        return this.buffer.slice(start, end);
    }

    size() {
        return this.currentCursor / this.increment;
    }
}

class Edge {
    player: TrainablePlayerAI;

    entity2Idx: Map<string, number>;

    entityPublicData: DynamicArray;
    entityRevealedData: DynamicArray;
    entityEdgeData: DynamicArray;
    fieldData: DynamicArray;

    unkEntityIndex: number;

    constructor(player: TrainablePlayerAI) {
        this.player = player;

        this.entity2Idx = new Map<string, number>();

        const numLocalEdges = 8;
        this.entityPublicData = new DynamicArray(
            numLocalEdges,
            numPublicEntityNodeFeatures,
        );
        this.entityRevealedData = new DynamicArray(
            numLocalEdges,
            numRevealedEntityNodeFeatures,
        );
        this.entityEdgeData = new DynamicArray(
            numLocalEdges,
            numEntityEdgeFeatures,
        );
        this.fieldData = new DynamicArray(1, numFieldFeatures);

        this.unkEntityIndex = 0;
        try {
            this.updateSideData();
        } catch (err) {}
        this.updateFieldData();
    }

    isEmpty() {
        return false;
    }

    clone() {
        const edge = new Edge(this.player);
        edge.entityPublicData.set(this.entityPublicData);
        edge.entityRevealedData.set(this.entityRevealedData);
        edge.entityEdgeData.set(this.entityEdgeData);
        edge.fieldData.set(this.fieldData);
        return edge;
    }

    _getIdent(pokemon: Pokemon) {
        return toID(pokemon.side.id + pokemon.details);
    }

    updatePokemon(pokemon: Pokemon) {
        const ident = this._getIdent(pokemon);

        const index = this.entity2Idx.get(ident) ?? this.entity2Idx.size;
        this.entity2Idx.set(ident, index);

        const relativeSide = isMySide(
            pokemon.side.n,
            this.player.getPlayerIndex(),
        );
        const { revealedData, publicData } = getArrayFromPublicPokemon(
            pokemon,
            relativeSide,
        );
        this.entityRevealedData.buffer.set(
            revealedData,
            index * numRevealedEntityNodeFeatures,
        );
        this.entityPublicData.buffer.set(
            publicData,
            index * numPublicEntityNodeFeatures,
        );
    }

    updateSideData() {
        const playerIndex = this.player.getPlayerIndex();

        let publicOffset = 0;
        let revealedOffset = 0;

        for (const side of this.player.publicBattle.sides) {
            const relativeSide = isMySide(side.n, playerIndex);

            this.updateSideConditionData(side, relativeSide);

            const teamLength = side.team.slice(0, 6).length;
            publicOffset += teamLength * numPublicEntityNodeFeatures;
            revealedOffset += teamLength * numRevealedEntityNodeFeatures;
        }
    }

    updateSideConditionData(side: Side, relativeSide: number) {
        let sideConditionBuffer = BigInt(0b0);
        for (const [id] of Object.entries(side.sideConditions)) {
            const featureIndex = IndexValueFromEnum(SideconditionEnum, id);
            sideConditionBuffer |= BigInt(1) << BigInt(featureIndex);
        }
        this.fieldData.buffer.set(
            bigIntToInt16Array(sideConditionBuffer),
            relativeSide
                ? FieldFeature.FIELD_FEATURE__MY_SIDECONDITIONS0
                : FieldFeature.FIELD_FEATURE__OPP_SIDECONDITIONS0,
        );
        if (side.sideConditions.spikes) {
            this.setFieldFeature({
                featureIndex: relativeSide
                    ? FieldFeature.FIELD_FEATURE__MY_SPIKES
                    : FieldFeature.FIELD_FEATURE__OPP_SPIKES,
                value: side.sideConditions.spikes.level,
            });
        }
        if (side.sideConditions.toxicspikes) {
            this.setFieldFeature({
                featureIndex: relativeSide
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
        pokemon: Pokemon;
        featureIndex: EntityEdgeFeatureMap[keyof EntityEdgeFeatureMap];
        value: number;
    }) {
        const { pokemon, featureIndex, value } = args;
        const edgeIndex = this.getEntityIdx(pokemon);
        if (edgeIndex > 11 || edgeIndex < 0) {
            throw new Error("edgeIndex out of bounds");
        }
        if (featureIndex === undefined) {
            throw new Error("featureIndex cannot be undefined");
        }
        if (value === undefined) {
            throw new Error("Value cannot be undefined");
        }
        this.entityEdgeData.buffer[
            edgeIndex * numEntityEdgeFeatures + featureIndex
        ] = value;
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
        this.fieldData.buffer[featureIndex] = value;
    }

    getEntityEdgeFeature(args: {
        pokemon: Pokemon;
        featureIndex: EntityEdgeFeatureMap[keyof EntityEdgeFeatureMap];
    }) {
        const { pokemon, featureIndex } = args;
        const edgeIndex = this.getEntityIdx(pokemon);
        const index = edgeIndex * numEntityEdgeFeatures + featureIndex;
        return this.entityEdgeData.buffer[index];
    }

    getFieldFeature(args: {
        featureIndex: FieldFeatureMap[keyof FieldFeatureMap];
    }) {
        const { featureIndex } = args;
        return this.fieldData.buffer[featureIndex];
    }

    getEntityIdx(pokemon: Pokemon) {
        const ident = this._getIdent(pokemon);
        const index = this.entity2Idx.get(ident);
        if (index === undefined) {
            this.updatePokemon(pokemon);
            const index = this.entity2Idx.get(ident);
            if (index === undefined) {
                throw new Error("Entity index not found after update");
            }
            return index;
        }
        return index;
    }

    addMajorArg(args: { argName: MajorArgNames; pokemon: Pokemon }) {
        const { argName, pokemon } = args;
        const index = IndexValueFromEnum(BattlemajorargsEnum, argName);
        this.setEntityEdgeFeature({
            featureIndex: EntityEdgeFeature.ENTITY_EDGE_FEATURE__MAJOR_ARG,
            pokemon,
            value: index,
        });
    }

    updateEdgeFromOf(args: { effect: Partial<Effect>; pokemon: Pokemon }) {
        const { effect, pokemon } = args;
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
                    pokemon,
                }) ?? 0;
            const numFromSources =
                this.getEntityEdgeFeature({
                    featureIndex:
                        EntityEdgeFeature.ENTITY_EDGE_FEATURE__NUM_FROM_SOURCES,
                    pokemon,
                }) ?? 0;

            if (numFromTypes < 5) {
                this.setEntityEdgeFeature({
                    featureIndex:
                        (EntityEdgeFeature.ENTITY_EDGE_FEATURE__FROM_TYPE_TOKEN0 +
                            numFromTypes) as EntityEdgeFeatureMap[keyof EntityEdgeFeatureMap],
                    pokemon,
                    value: fromTypeToken,
                });
                this.setEntityEdgeFeature({
                    featureIndex:
                        EntityEdgeFeature.ENTITY_EDGE_FEATURE__NUM_FROM_TYPES,
                    pokemon,
                    value: numFromTypes + 1,
                });
            }
            if (numFromSources < 5) {
                this.setEntityEdgeFeature({
                    featureIndex:
                        (EntityEdgeFeature.ENTITY_EDGE_FEATURE__FROM_SOURCE_TOKEN0 +
                            numFromSources) as EntityEdgeFeatureMap[keyof EntityEdgeFeatureMap],
                    pokemon,
                    value: fromSourceToken,
                });
                this.setEntityEdgeFeature({
                    featureIndex:
                        EntityEdgeFeature.ENTITY_EDGE_FEATURE__NUM_FROM_SOURCES,
                    pokemon,
                    value: numFromSources + 1,
                });
            }
        }
    }

    updateMinorArgs(args: {
        argName: MinorArgNames;
        pokemon: Pokemon;
        precision?: number;
    }) {
        const { argName, pokemon } = args;
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
        const currentValue = this.getEntityEdgeFeature({
            featureIndex,
            pokemon,
        })!;
        const newValue = currentValue | (1 << (index % precision));
        this.setEntityEdgeFeature({
            featureIndex,
            pokemon,
            value: newValue,
        });
    }
}

export class EdgeBuffer {
    player: TrainablePlayerAI;

    entity2Idx: Map<string, number>;

    entityPublicData: DynamicArray;
    entityRevealedData: DynamicArray;
    entityEdgeData: DynamicArray;
    fieldData: DynamicArray;

    numEdges: number;
    maxEdges: number;

    constructor(player: TrainablePlayerAI) {
        this.player = player;

        this.entity2Idx = new Map<string, number>();

        const maxEdges = 4000;
        this.maxEdges = maxEdges;

        this.entityPublicData = new DynamicArray(
            maxEdges,
            numPublicEntityNodeFeatures,
        );
        this.entityRevealedData = new DynamicArray(
            maxEdges,
            numRevealedEntityNodeFeatures,
        );
        this.entityEdgeData = new DynamicArray(maxEdges, numEntityEdgeFeatures);
        this.fieldData = new DynamicArray(maxEdges, numFieldFeatures);

        this.numEdges = 0;
    }

    addEdge(edge: Edge) {
        if (this.numEdges >= this.maxEdges) {
            throw new Error("EdgeBuffer max edges exceeded");
        }
        const numEntities = edge.entity2Idx.size;
        const prevSize = this.entityPublicData.size();
        for (const [ident, index] of edge.entity2Idx.entries()) {
            this.entity2Idx.set(ident, prevSize + index);
        }
        if (numEntities === 0) {
            return;
        }
        this.entityPublicData.setNextSlice(
            edge.entityPublicData.buffer,
            numEntities,
        );
        const postSize = this.entityPublicData.size();
        this.entityRevealedData.setNextSlice(
            edge.entityRevealedData.buffer,
            numEntities,
        );
        this.entityEdgeData.setNextSlice(
            edge.entityEdgeData.buffer,
            numEntities,
        );
        let offset = 0;
        for (let i = prevSize; i < postSize; i++) {
            edge.setFieldFeature({
                featureIndex:
                    (FieldFeature.FIELD_FEATURE__RELEVANT_ENTITY_IDX0 +
                        offset) as FieldFeatureMap[keyof FieldFeatureMap],
                value: i,
            });
            offset += 1;
        }
        edge.setFieldFeature({
            featureIndex: FieldFeature.FIELD_FEATURE__NUM_RELEVANT,
            value: postSize - prevSize,
        });
        this.fieldData.setNextSlice(edge.fieldData.buffer);
        this.numEdges += 1;
    }

    getHistory(numHistory: number = NUM_HISTORY) {
        const historyLength = Math.max(1, Math.min(this.numEdges, numHistory));
        const numEdges = Math.max(1, this.entityPublicData.size());
        const historyEntityPublic = new Uint8Array(
            this.entityPublicData.getLatestSlice(numEdges).buffer,
        );
        const historyEntityRevealed = new Uint8Array(
            this.entityRevealedData.getLatestSlice(numEdges).buffer,
        );
        const historyEntityEdges = new Uint8Array(
            this.entityEdgeData.getLatestSlice(numEdges).buffer,
        );
        const historyField = new Uint8Array(
            this.fieldData.getLatestSlice(historyLength).buffer,
        );
        return {
            historyEntityPublic,
            historyEntityRevealed,
            historyEntityEdges,
            historyField,
            historyLength,
            historyPackedLength: numEdges,
        };
    }

    static toReadableHistory(args: {
        historyEntityPublicBuffer: Uint8Array;
        historyEntityRevealedBuffer: Uint8Array;
        historyEntityEdgesBuffer: Uint8Array;
        historyFieldBuffer: Uint8Array;
        historyLength: number;
    }) {
        const {
            historyEntityPublicBuffer,
            historyEntityRevealedBuffer,
            historyEntityEdgesBuffer,
            historyFieldBuffer,
            historyLength,
        } = args;
        const historyItems = [];
        const historyEntityPublic = new Int16Array(
            historyEntityPublicBuffer.buffer,
        );
        const historyEntityRevealed = new Int16Array(
            historyEntityRevealedBuffer.buffer,
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
            const stepField = historyField.slice(
                historyIndex * numFieldFeatures,
                (historyIndex + 1) * numFieldFeatures,
            );
            const fieldObject = fieldArrayToObject(stepField);
            const numRelevant = fieldObject.numRelevant;

            const offset =
                stepField[FieldFeature.FIELD_FEATURE__RELEVANT_ENTITY_IDX0];
            const oneToEleven = [...Array(numRelevant).keys()];

            const stepEntityPublic = historyEntityPublic.slice(
                offset * numPublicEntityNodeFeatures,
                (offset + numRelevant) * numPublicEntityNodeFeatures,
            );
            const stepEntityRevealed = historyEntityRevealed.slice(
                offset * numRevealedEntityNodeFeatures,
                (offset + numRelevant) * numRevealedEntityNodeFeatures,
            );
            const stepEntityEdges = historyEntityEdges.slice(
                offset * numEntityEdgeFeatures,
                (offset + numRelevant) * numEntityEdgeFeatures,
            );

            historyItems.push({
                public: oneToEleven.map((memberIndex) => {
                    const start = memberIndex * numPublicEntityNodeFeatures;
                    const end = (memberIndex + 1) * numPublicEntityNodeFeatures;
                    return entityPublicArrayToObject(
                        stepEntityPublic.slice(start, end),
                    );
                }),
                revealed: oneToEleven.map((memberIndex) => {
                    const start = memberIndex * numRevealedEntityNodeFeatures;
                    const end =
                        (memberIndex + 1) * numRevealedEntityNodeFeatures;
                    return entityRevealedArrayToObject(
                        stepEntityRevealed.slice(start, end),
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

    edgeBuffer: EdgeBuffer;
    latestEdge: Edge;
    prevEdge: Edge;

    turnOrder: number;
    turnNum: number;
    timestamp: number;

    identToIndex: Map<PokemonIdent | SideID, number>;

    constructor(player: TrainablePlayerAI) {
        this.player = player;

        this.edgeBuffer = new EdgeBuffer(player);
        this.latestEdge = new Edge(player);
        this.prevEdge = this.latestEdge.clone();

        this.turnOrder = 0;
        this.turnNum = 0;
        this.timestamp = 0;

        this.identToIndex = new Map<PokemonIdent, number>();
    }

    getPokemon(
        pokemonIdent: PokemonIdent,
        isPublic: boolean = true,
    ): {
        pokemon: Pokemon | null;
        index: number;
    } {
        if (
            !pokemonIdent ||
            pokemonIdent === "??" ||
            pokemonIdent === "null" ||
            pokemonIdent === "false"
        ) {
            return { pokemon: null, index: -1 };
        }

        if (isPublic) {
            const { siden, pokemonid: parsedPokemonid } =
                this.player.publicBattle.parsePokemonId(pokemonIdent);

            if (!this.identToIndex.has(parsedPokemonid)) {
                this.identToIndex.set(parsedPokemonid, this.identToIndex.size);
            }

            let pokemonid = parsedPokemonid;

            let found: Pokemon | null = null;
            const side = this.player.publicBattle.sides[siden];

            for (const pokemon of side.team) {
                if (pokemon.originalIdent === pokemonid) {
                    found = pokemon;
                }
            }

            return {
                pokemon: found,
                index: this.identToIndex.get(parsedPokemonid) ?? -1,
            };
        } else {
            const { siden, pokemonid: parsedPokemonid } =
                this.player.privateBattle.parsePokemonId(pokemonIdent);
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
        if (ident) {
            if (ident.startsWith("fallen")) {
                ident = "fallen";
            }
        }
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
        edge.setFieldFeature({
            featureIndex: FieldFeature.FIELD_FEATURE__TURN_VALUE,
            value: this.turnNum,
        });
        return edge;
    }

    addEdge() {
        const currentEdge = this.latestEdge;
        if (currentEdge.isEmpty()) {
            return;
        }

        this.prevEdge = currentEdge.clone();
        this.latestEdge = new Edge(this.player);

        const preprocessedEdge = this._preprocessEdge(currentEdge);
        this.turnOrder += 1;
        this.edgeBuffer.addEdge(preprocessedEdge);
    }

    "|move|"(args: Args["|move|"], kwArgs: KWArgs["|move|"]) {
        this.addEdge();

        const [argName, pokeIdent, moveId] = args;

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const { pokemon } = this.getPokemon(pokeIdent as PokemonIdent);
        if (pokemon === null) {
            throw new Error(`Pokemon not found for ident ${pokeIdent}`);
        }
        const move = this.getMove(moveId);

        this.latestEdge.addMajorArg({ argName, pokemon });
        this.latestEdge.setEntityEdgeFeature({
            pokemon,
            featureIndex: EntityEdgeFeature.ENTITY_EDGE_FEATURE__MOVE_TOKEN,
            value: IndexValueFromEnum(MovesEnum, move.id),
        });
        this.latestEdge.setEntityEdgeFeature({
            pokemon,
            featureIndex: EntityEdgeFeature.ENTITY_EDGE_FEATURE__MOVE_TOKEN,
            value: IndexValueFromEnum(MovesEnum, move.id),
        });
        if (kwArgs.from) {
            const fromEffect = this.getCondition(kwArgs.from);
            this.latestEdge.updateEdgeFromOf({ effect: fromEffect, pokemon });
        }
        this.latestEdge.setEntityEdgeFeature({
            pokemon,
            featureIndex: EntityEdgeFeature.ENTITY_EDGE_FEATURE__HIT_COUNT,
            value: 1,
        });
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

    "|request|"() {
        this.latestEdge.updateSideData();
    }

    "|poke|"(args: Args["|poke|"]) {
        const [argName, sideId, pokeIdent] = args;

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }
    }

    handleSwitch(
        args: Args["|switch|" | "|drag|"],
        kwArgs: KWArgs["|switch|" | "|drag|"],
    ) {
        this.addEdge();

        const [argName, pokeIdent, pokeDetails] = args;

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const switchedOut = this.player.publicBattle.getSwitchedOutPokemon(
            pokeIdent,
            pokeDetails,
        );
        const { pokemon: switchedIn } = this.getPokemon(
            pokeIdent as PokemonIdent,
        );

        for (const affected of [switchedIn, switchedOut]) {
            if (!affected) {
                continue;
            }
            this.latestEdge.updatePokemon(affected);
            this.latestEdge.addMajorArg({ argName, pokemon: affected });
        }

        if (switchedOut) {
            this.latestEdge.setEntityEdgeFeature({
                pokemon: switchedOut,
                featureIndex: EntityEdgeFeature.ENTITY_EDGE_FEATURE__MOVE_TOKEN,
                value: MovesEnum.MOVES_ENUM___SWITCH_OUT,
            });
            if (argName !== "switch") {
                const from = (kwArgs as KWArgs["|switch|"]).from;
                if (from) {
                    const effect = this.getCondition(from);
                    this.latestEdge.updateEdgeFromOf({
                        effect,
                        pokemon: switchedOut,
                    });
                }
            }
        }
        if (switchedIn) {
            this.latestEdge.setEntityEdgeFeature({
                pokemon: switchedIn,
                featureIndex: EntityEdgeFeature.ENTITY_EDGE_FEATURE__MOVE_TOKEN,
                value: MovesEnum.MOVES_ENUM___SWITCH_IN,
            });
        }
    }

    "|cant|"(args: Args["|cant|"]) {
        this.addEdge();

        const [argName, pokeIdent, conditionId, moveId] = args;

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const { pokemon } = this.getPokemon(pokeIdent);
        if (pokemon === null) {
            throw new Error(`Pokemon not found for ident ${pokeIdent}`);
        }

        if (moveId) {
            const move = this.getMove(moveId);
            this.latestEdge.setEntityEdgeFeature({
                pokemon,
                featureIndex: EntityEdgeFeature.ENTITY_EDGE_FEATURE__MOVE_TOKEN,
                value: IndexValueFromEnum(MovesEnum, move.id),
            });
        }

        const condition = this.getCondition(conditionId);

        this.latestEdge.addMajorArg({ argName, pokemon });
        this.latestEdge.updateEdgeFromOf({ effect: condition, pokemon });
    }

    "|faint|"(args: Args["|faint|"]) {
        this.addEdge();

        const [argName, pokeIdent] = args;

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const { pokemon } = this.getPokemon(pokeIdent);
        if (pokemon === null) {
            throw new Error(`Pokemon not found for ident ${pokeIdent}`);
        }

        this.latestEdge.addMajorArg({ argName, pokemon });
    }

    "|-fail|"(args: Args["|-fail|"], kwArgs: KWArgs["|-fail|"]) {
        const [argName, pokeIdent] = args;

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const { pokemon } = this.getPokemon(pokeIdent)!;
        if (pokemon === null) {
            throw new Error(`Pokemon not found for ident ${pokeIdent}`);
        }

        const effect = this.getCondition(kwArgs.from);

        this.latestEdge.updateMinorArgs({ argName, pokemon });
        this.latestEdge.updateEdgeFromOf({
            effect,
            pokemon,
        });
    }

    "|-block|"(args: Args["|-block|"], kwArgs: KWArgs["|-block|"]) {
        const [argName, pokeIdent] = args;

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const { pokemon } = this.getPokemon(pokeIdent)!;
        if (pokemon === null) {
            throw new Error(`Pokemon not found for ident ${pokeIdent}`);
        }

        const effect = this.getCondition(kwArgs.from);

        this.latestEdge.updateMinorArgs({ argName, pokemon });
        this.latestEdge.updateEdgeFromOf({ effect, pokemon });
    }

    "|-notarget|"(args: Args["|-notarget|"]) {
        const [argName, pokeIdent] = args;

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        if (pokeIdent !== undefined) {
            const { pokemon } = this.getPokemon(pokeIdent)!;
            if (pokemon === null) {
                throw new Error(`Pokemon not found for ident ${pokeIdent}`);
            }
            this.latestEdge.updateMinorArgs({ argName, pokemon });
        } else {
            throw new Error(
                `Pokemon identifier is required for |-notarget| event: ${args}`,
            );
        }
    }

    "|-miss|"(args: Args["|-miss|"], kwArgs: KWArgs["|-miss|"]) {
        const [argName, pokeIdent, poke2Ident] = args;

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const idents = [pokeIdent];
        if (poke2Ident !== undefined) {
            idents.push(poke2Ident);
        }
        for (const ident of idents) {
            const { pokemon } = this.getPokemon(ident)!;
            if (pokemon === null) {
                throw new Error(`Pokemon not found for ident ${ident}`);
            }
            const effect = this.getCondition(kwArgs.from);
            this.latestEdge.updateMinorArgs({ argName, pokemon });
            this.latestEdge.updateEdgeFromOf({
                effect,
                pokemon,
            });
        }
    }

    "|-damage|"(
        args: Args["|-damage|"] | Args["|-sethp|"],
        kwArgs: KWArgs["|-damage|"] | KWArgs["|-sethp|"],
    ) {
        const [argName, pokeIdent, hpStatus] = args;

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const { pokemon } = this.getPokemon(pokeIdent);
        if (pokemon === null) {
            throw new Error(`Pokemon ${pokeIdent} not found`);
        }

        if (kwArgs.from) {
            const effect = this.getCondition(kwArgs.from);
            this.latestEdge.updateEdgeFromOf({ effect, pokemon });
        }

        const damage = pokemon.healthParse(hpStatus);
        if (damage === null) {
            throw new Error(`Invalid damage value: ${damage}`);
        }

        const addedDamageToken = Math.abs(
            Math.floor((MAX_RATIO_TOKEN * damage[0]) / damage[1]),
        );

        this.latestEdge.updateMinorArgs({ argName, pokemon });
        const currentDamageToken =
            this.latestEdge.getEntityEdgeFeature({
                featureIndex:
                    EntityEdgeFeature.ENTITY_EDGE_FEATURE__DAMAGE_RATIO,
                pokemon,
            }) ?? 0;
        this.latestEdge.setEntityEdgeFeature({
            featureIndex: EntityEdgeFeature.ENTITY_EDGE_FEATURE__DAMAGE_RATIO,
            pokemon,
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
        const [argName, pokeIdent, hpStatus] = args;

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const { pokemon } = this.getPokemon(pokeIdent);
        if (pokemon === null) {
            throw new Error(`Pokemon ${pokeIdent} not found`);
        }

        if (kwArgs.from) {
            const effect = this.getCondition(kwArgs.from);
            this.latestEdge.updateEdgeFromOf({
                effect,
                pokemon,
            });
        }

        const damage = pokemon.healthParse(hpStatus);
        if (damage === null) {
            throw new Error(`Invalid damage value: ${damage}`);
        }

        const addedHealToken = Math.abs(
            Math.floor((MAX_RATIO_TOKEN * damage[0]) / damage[1]),
        );

        this.latestEdge.updateMinorArgs({ argName, pokemon });
        const currentHealToken = this.latestEdge.getEntityEdgeFeature({
            featureIndex: EntityEdgeFeature.ENTITY_EDGE_FEATURE__HEAL_RATIO,
            pokemon,
        });
        this.latestEdge.setEntityEdgeFeature({
            featureIndex: EntityEdgeFeature.ENTITY_EDGE_FEATURE__HEAL_RATIO,
            pokemon,
            value: Math.min(MAX_RATIO_TOKEN, currentHealToken + addedHealToken),
        });
    }

    "|-sethp|"(args: Args["|-sethp|"], kwArgs: KWArgs["|-sethp|"]) {
        const [argName, pokeIdent, hpStatus] = args;
        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }
        const { pokemon } = this.getPokemon(pokeIdent)!;
        if (pokemon === null) {
            throw new Error(`Pokemon ${pokeIdent} not found`);
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
        this.latestEdge.updateMinorArgs({ argName, pokemon });
    }

    "|-status|"(args: Args["|-status|"], kwArgs: KWArgs["|-status|"]) {
        const [argName, pokeIdent, statusId] = args;

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const { pokemon } = this.getPokemon(pokeIdent)!;
        if (pokemon === null) {
            throw new Error(`Pokemon ${pokeIdent} not found`);
        }

        const fromEffect = this.getCondition(kwArgs.from);
        const statusToken = IndexValueFromEnum(StatusEnum, statusId);

        this.latestEdge.updateMinorArgs({ argName, pokemon });
        this.latestEdge.updateEdgeFromOf({
            effect: fromEffect,
            pokemon,
        });
        this.latestEdge.setEntityEdgeFeature({
            featureIndex: EntityEdgeFeature.ENTITY_EDGE_FEATURE__STATUS_TOKEN,
            pokemon,
            value: statusToken,
        });
    }

    "|-curestatus|"(args: Args["|-curestatus|"]) {
        const [argName, pokeIdent, statusId] = args;

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const { pokemon } = this.getPokemon(pokeIdent)!;
        if (pokemon === null) {
            console.warn(`${args} Pokemon ${pokeIdent} not found`);
            return;
            throw new Error(`Pokemon ${pokeIdent} not found`);
        }

        const statusToken = IndexValueFromEnum(StatusEnum, statusId);

        this.latestEdge.updateMinorArgs({ argName, pokemon });
        this.latestEdge.setEntityEdgeFeature({
            featureIndex: EntityEdgeFeature.ENTITY_EDGE_FEATURE__STATUS_TOKEN,
            pokemon,
            value: statusToken,
        });
    }

    "|-cureteam|"(args: Args["|-cureteam|"]) {
        const [argName, pokeIdent] = args;

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const { pokemon } = this.getPokemon(pokeIdent)!;
        if (pokemon === null) {
            throw new Error(`Pokemon ${pokeIdent} not found`);
        }

        this.latestEdge.updateMinorArgs({ argName, pokemon });
    }

    static getStatBoostEdgeFeatureIndex(stat: BoostID) {
        return EntityEdgeFeature[
            `ENTITY_EDGE_FEATURE__BOOST_${stat.toLocaleUpperCase()}_VALUE` as `ENTITY_EDGE_FEATURE__BOOST_${Uppercase<BoostID>}_VALUE`
        ];
    }

    "|-boost|"(args: Args["|-boost|"]) {
        const [argName, pokeIdent, stat, value] = args;

        const featureIndex = EventHandler.getStatBoostEdgeFeatureIndex(stat);

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const { pokemon } = this.getPokemon(pokeIdent)!;
        if (pokemon === null) {
            throw new Error(`Pokemon ${pokeIdent} not found`);
        }

        this.latestEdge.updateMinorArgs({ argName, pokemon });
        this.latestEdge.setEntityEdgeFeature({
            featureIndex,
            pokemon,
            value: parseInt(value),
        });
    }

    "|-unboost|"(args: Args["|-unboost|"]) {
        const [argName, pokeIdent, stat, value] = args;

        const featureIndex = EventHandler.getStatBoostEdgeFeatureIndex(stat);

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const { pokemon } = this.getPokemon(pokeIdent)!;
        if (pokemon === null) {
            throw new Error(`Pokemon ${pokeIdent} not found`);
        }

        this.latestEdge.updateMinorArgs({ argName, pokemon });
        this.latestEdge.setEntityEdgeFeature({
            featureIndex,
            pokemon,
            value: -parseInt(value),
        });
    }

    "|-setboost|"(args: Args["|-setboost|"], kwArgs: KWArgs["|-setboost|"]) {
        const [argName, pokeIdent, stat, value] = args;

        const featureIndex = EventHandler.getStatBoostEdgeFeatureIndex(stat);
        const effect = this.getCondition(kwArgs.from);

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const { pokemon } = this.getPokemon(pokeIdent)!;
        if (pokemon === null) {
            throw new Error(`Pokemon ${pokeIdent} not found`);
        }

        this.latestEdge.updateMinorArgs({ argName, pokemon });
        this.latestEdge.setEntityEdgeFeature({
            featureIndex,
            pokemon,
            value: parseInt(value),
        });
        this.latestEdge.updateEdgeFromOf({ effect, pokemon });
    }

    "|-swapboost|"(args: Args["|-swapboost|"], kwArgs: KWArgs["|-swapboost|"]) {
        const [argName, pokeIdent] = args;

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const { pokemon } = this.getPokemon(pokeIdent)!;
        if (pokemon === null) {
            throw new Error(`Pokemon ${pokeIdent} not found`);
        }

        const effect = this.getCondition(kwArgs.from);

        this.latestEdge.updateMinorArgs({ argName, pokemon });
        this.latestEdge.updateEdgeFromOf({ effect, pokemon });
    }

    "|-invertboost|"(
        args: Args["|-invertboost|"],
        kwArgs: KWArgs["|-invertboost|"],
    ) {
        const [argName, pokeIdent] = args;

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const { pokemon } = this.getPokemon(pokeIdent)!;
        if (pokemon === null) {
            throw new Error(`Pokemon ${pokeIdent} not found`);
        }

        const effect = this.getCondition(kwArgs.from);

        this.latestEdge.updateMinorArgs({ argName, pokemon });
        this.latestEdge.updateEdgeFromOf({ effect, pokemon });
    }

    "|-clearboost|"(
        args: Args["|-clearboost|"],
        kwArgs: KWArgs["|-clearboost|"],
    ) {
        const [argName, pokeIdent] = args;

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const { pokemon } = this.getPokemon(pokeIdent)!;
        if (pokemon === null) {
            throw new Error(`Pokemon ${pokeIdent} not found`);
        }

        const effect = this.getCondition(kwArgs.from);

        this.latestEdge.updateMinorArgs({ argName, pokemon });
        this.latestEdge.updateEdgeFromOf({ effect, pokemon });
    }

    "|-clearnegativeboost|"(
        args: Args["|-clearnegativeboost|"],
        kwArgs: KWArgs["|-clearnegativeboost|"],
    ) {
        const [argName, pokeIdent] = args;

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const { pokemon } = this.getPokemon(pokeIdent)!;
        if (pokemon === null) {
            throw new Error(`Pokemon ${pokeIdent} not found`);
        }

        const effect = this.getCondition(kwArgs.from);

        this.latestEdge.updateMinorArgs({ argName, pokemon });
        this.latestEdge.updateEdgeFromOf({ effect, pokemon });
    }

    "|-copyboost|"() {}

    "|-weather|"(args: Args["|-weather|"]) {
        // eslint-disable-next-line @typescript-eslint/no-unused-vars
        const [argName, weatherId] = args;

        const weatherIndex =
            weatherId === "none"
                ? WeatherEnum.WEATHER_ENUM___NULL
                : IndexValueFromEnum(WeatherEnum, weatherId);

        this.latestEdge.setFieldFeature({
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

        for (const pokemon of side.active) {
            if (!pokemon) {
                continue;
            }
            this.latestEdge.updateMinorArgs({ argName, pokemon });
            this.latestEdge.updateEdgeFromOf({ effect, pokemon });
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

        for (const pokemon of side.active) {
            if (!pokemon) {
                continue;
            }
            this.latestEdge.updateMinorArgs({ argName, pokemon });
            this.latestEdge.updateEdgeFromOf({ effect, pokemon });
        }
    }

    "|-swapsideconditions|"() {}

    "|-start|"(args: Args["|-start|"], kwArgs: KWArgs["|-start|"]) {
        const [argName, pokeIdent, conditionId] = args;

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const { pokemon } = this.getPokemon(pokeIdent)!;
        if (pokemon === null) {
            throw new Error(`Pokemon ${pokeIdent} not found`);
        }

        this.latestEdge.updateMinorArgs({ argName, pokemon });
        this.latestEdge.updateEdgeFromOf({
            effect: this.getCondition(kwArgs.from),
            pokemon,
        });
        this.latestEdge.updateEdgeFromOf({
            effect: this.getCondition(
                conditionId.startsWith("perish") ? "perishsong" : conditionId,
            ),
            pokemon,
        });
    }

    "|-end|"(args: Args["|-end|"], kwArgs: KWArgs["|-end|"]) {
        const [argName, pokeIdent, conditionId] = args;

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const { pokemon } = this.getPokemon(pokeIdent)!;
        if (pokemon === null) {
            throw new Error(`Pokemon ${pokeIdent} not found`);
        }

        this.latestEdge.updateMinorArgs({ argName, pokemon });
        this.latestEdge.updateEdgeFromOf({
            effect: this.getCondition(kwArgs.from),
            pokemon,
        });
        this.latestEdge.updateEdgeFromOf({
            effect: this.getCondition(conditionId),
            pokemon,
        });
    }

    "|-crit|"(args: Args["|-crit|"]) {
        const [argName, pokeIdent] = args;

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const { pokemon } = this.getPokemon(pokeIdent);
        if (pokemon === null) {
            throw new Error(`Pokemon ${pokeIdent} not found`);
        }
        this.latestEdge.updateMinorArgs({ argName, pokemon });
    }

    "|-supereffective|"(args: Args["|-supereffective|"]) {
        const [argName, pokeIdent] = args;

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const { pokemon } = this.getPokemon(pokeIdent);
        if (pokemon === null) {
            throw new Error(`Pokemon ${pokeIdent} not found`);
        }
        this.latestEdge.updateMinorArgs({ argName, pokemon });
    }

    "|-resisted|"(args: Args["|-resisted|"]) {
        const [argName, pokeIdent] = args;

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const { pokemon } = this.getPokemon(pokeIdent);
        if (pokemon === null) {
            throw new Error(`Pokemon ${pokeIdent} not found`);
        }
        this.latestEdge.updateMinorArgs({ argName, pokemon });
    }

    "|-immune|"(args: Args["|-immune|"], kwArgs: KWArgs["|-immune|"]) {
        const [argName, pokeIdent, conditionId] = args;

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const { pokemon } = this.getPokemon(pokeIdent);
        if (pokemon === null) {
            throw new Error(`Pokemon ${pokeIdent} not found`);
        }
        this.latestEdge.updateMinorArgs({ argName, pokemon });
        this.latestEdge.updateEdgeFromOf({
            effect: this.getCondition(kwArgs.from),
            pokemon,
        });
        this.latestEdge.updateEdgeFromOf({
            effect: this.getCondition(conditionId),
            pokemon,
        });
    }

    "|-item|"(args: Args["|-item|"], kwArgs: KWArgs["|-item|"]) {
        const [argName, pokeIdent, itemId] = args;

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const { pokemon } = this.getPokemon(pokeIdent as PokemonIdent);
        if (pokemon === null) {
            throw new Error(`Pokemon ${pokeIdent} not found`);
        }
        this.latestEdge.updateMinorArgs({ argName, pokemon });
        const itemIndex = IndexValueFromEnum(ItemsEnum, itemId);
        const effect = this.getCondition(kwArgs.from);

        this.latestEdge.updateMinorArgs({ argName, pokemon });
        this.latestEdge.updateEdgeFromOf({ effect, pokemon });
        this.latestEdge.setEntityEdgeFeature({
            featureIndex: EntityEdgeFeature.ENTITY_EDGE_FEATURE__ITEM_TOKEN,
            pokemon,
            value: itemIndex,
        });
    }

    "|-enditem|"(args: Args["|-enditem|"], kwArgs: KWArgs["|-enditem|"]) {
        const [argName, pokeIdent, itemId] = args;
        if (!itemId) {
            return;
        }

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const { pokemon } = this.getPokemon(pokeIdent);
        if (pokemon === null) {
            throw new Error(`Pokemon ${pokeIdent} not found`);
        }
        this.latestEdge.updateMinorArgs({ argName, pokemon });
        const itemIndex = IndexValueFromEnum(ItemsEnum, itemId);
        const effect = this.getCondition(kwArgs.from);

        this.latestEdge.updateMinorArgs({ argName, pokemon });
        this.latestEdge.updateEdgeFromOf({ effect, pokemon });
        this.latestEdge.setEntityEdgeFeature({
            featureIndex: EntityEdgeFeature.ENTITY_EDGE_FEATURE__ITEM_TOKEN,
            pokemon,
            value: itemIndex,
        });
    }

    "|-ability|"(args: Args["|-ability|"], kwArgs: KWArgs["|-ability|"]) {
        const [argName, pokeIdent, abilityId] = args;

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const { pokemon } = this.getPokemon(pokeIdent);
        if (pokemon === null) {
            throw new Error(`Pokemon ${pokeIdent} not found`);
        }

        const abilityIndex = IndexValueFromEnum(AbilitiesEnum, abilityId);

        if (kwArgs.from) {
            const effect = this.getCondition(kwArgs.from);
            this.latestEdge.updateEdgeFromOf({ effect, pokemon });
        }

        this.latestEdge.updateMinorArgs({ argName, pokemon });
        this.latestEdge.setEntityEdgeFeature({
            featureIndex: EntityEdgeFeature.ENTITY_EDGE_FEATURE__ABILITY_TOKEN,
            pokemon,
            value: abilityIndex,
        });
    }

    "|-endability|"(
        args: Args["|-endability|"],
        kwArgs: KWArgs["|-endability|"],
    ) {
        const [argName, pokeIdent] = args;

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const { pokemon } = this.getPokemon(pokeIdent)!;
        if (pokemon === null) {
            throw new Error(`Pokemon ${pokeIdent} not found`);
        }

        const effect = this.getCondition(kwArgs.from);

        this.latestEdge.updateMinorArgs({ argName, pokemon });
        this.latestEdge.updateEdgeFromOf({ effect, pokemon });
    }

    rememberTransformed(
        pokeIdent: Protocol.PokemonIdent,
        poke2Ident: Protocol.PokemonIdent,
    ) {
        const { pokemon: srcPokemon } = this.getPokemon(pokeIdent, false)!;
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
        const [argName, pokeIdent, poke2Ident] = args;

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const { pokemon } = this.getPokemon(pokeIdent);
        if (pokemon === null) {
            throw new Error(`Pokemon ${pokeIdent} not found`);
        }

        this.latestEdge.updateMinorArgs({ argName, pokemon });
        this.rememberTransformed(pokeIdent, poke2Ident);
    }

    "|-mega|"() {}

    "|-primal|"() {}

    "|-burst|"() {}

    "|-zpower|"() {}

    "|-zbroken|"() {}

    // Suprisingly not needed?

    "|-terastallize|"(args: Args["|-terastallize|"]) {
        const [argName, pokeIdent] = args;
        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }
        const { pokemon } = this.getPokemon(pokeIdent)!;
        if (pokemon === null) {
            throw new Error(`Pokemon ${pokeIdent} not found`);
        }
        this.latestEdge.updateMinorArgs({ argName, pokemon });
    }

    "|-activate|"(args: Args["|-activate|"], kwArgs: KWArgs["|-activate|"]) {
        const [argName, pokeIdent, conditionId1] = args;

        if (pokeIdent) {
            const playerIndex = this.player.getPlayerIndex();
            if (playerIndex === undefined) {
                throw new Error();
            }

            const { pokemon } = this.getPokemon(pokeIdent)!;
            if (pokemon === null) {
                throw new Error(`Pokemon ${pokeIdent} not found`);
            }

            this.latestEdge.updateMinorArgs({ argName, pokemon });
            for (const effect of [
                this.getCondition(kwArgs.from),
                this.getCondition(conditionId1),
                // this.getCondition(conditionId2),
            ]) {
                this.latestEdge.updateEdgeFromOf({ effect, pokemon });
            }
        }
    }

    "|-mustrecharge|"(args: Args["|-mustrecharge|"]) {
        const [argName, pokeIdent] = args;

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const { pokemon } = this.getPokemon(pokeIdent)!;
        if (pokemon === null) {
            throw new Error(`Pokemon ${pokeIdent} not found`);
        }

        this.latestEdge.updateMinorArgs({ argName, pokemon });
    }

    "|-prepare|"(args: Args["|-prepare|"]) {
        const [argName, pokeIdent] = args;

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const { pokemon } = this.getPokemon(pokeIdent)!;
        if (pokemon === null) {
            throw new Error(`Pokemon ${pokeIdent} not found`);
        }

        this.latestEdge.updateMinorArgs({ argName, pokemon });
    }

    "|-hitcount|"(args: Args["|-hitcount|"]) {
        const [argName, pokeIdent, numHits] = args;

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const { pokemon } = this.getPokemon(pokeIdent);
        if (pokemon === null) {
            throw new Error(`Pokemon ${pokeIdent} not found`);
        }

        this.latestEdge.updateMinorArgs({ argName, pokemon });
        this.latestEdge.setEntityEdgeFeature({
            featureIndex: EntityEdgeFeature.ENTITY_EDGE_FEATURE__HIT_COUNT,
            pokemon,
            value: parseInt(numHits),
        });
    }

    "|done|"(args: Args["|done|"]) {
        const [argName] = args;

        // let edge = undefined;
        // for (const side of this.player.publicBattle.sides) {
        //     for (const active of side.active) {
        //         if (active !== null) {
        //             const { pokemon } = this.getPokemon(active.originalIdent);
        //             if (pokemon === null) {
        //                 throw new Error(`Pokemon ${pokeIdent} not found`);
        //             }
        //             if (edge === undefined) {
        //                 edge = new Edge(this.player);
        //             }
        //             if (edgeIndex >= 0) {
        //                 edge.addMajorArg({ argName, edgeIndex });
        //             }
        //         }
        //     }
        // }
        // if (edge !== undefined && this.turnOrder > 0) {
        //     this.addEdge();
        // }
        this.addEdge();
    }

    "|start|"() {
        this.turnOrder = 0;
    }

    "|teampreview|"() {
        this.turnOrder = 0;
    }

    "|t:|"(args: Args["|t:|"]) {
        // eslint-disable-next-line @typescript-eslint/no-unused-vars
        const [_, timestamp] = args;
        this.timestamp = parseInt(timestamp);
    }

    "|turn|"(args: Args["|turn|"]) {
        this.addEdge();

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

class PrivateActionHandler {
    player: TrainablePlayerAI;
    request: AnyObject;
    actionBuffer: Int16Array;

    constructor(player: TrainablePlayerAI) {
        this.player = player;
        this.request = player.getRequest();
        this.actionBuffer = new Int16Array(4 * 4 * numMoveFeatures);
    }

    assignActionBuffer(args: { offset: number; index: number; value: number }) {
        const { offset, index, value } = args;
        this.actionBuffer[offset + index] = value;
    }

    pushMoveAction(
        actionOffset: number,
        move:
            | { name: "Recharge"; id: "recharge" }
            | { name: Protocol.MoveName; id: ID }
            | {
                  name: Protocol.MoveName;
                  id: ID;
                  pp: number;
                  maxpp: number;
                  target: MoveTarget;
                  disabled?: boolean;
              }
            | {
                  id: ID;
                  target: MoveTarget;
                  disabled?: boolean;
              },
    ) {
        if ("pp" in move) {
            this.assignActionBuffer({
                offset: actionOffset,
                index: MovesetFeature.MOVESET_FEATURE__HAS_PP,
                value: MovesetHasPP.MOVESET_HAS_PP__YES,
            });
            this.assignActionBuffer({
                offset: actionOffset,
                index: MovesetFeature.MOVESET_FEATURE__PP,
                value: move.pp,
            });
            this.assignActionBuffer({
                offset: actionOffset,
                index: MovesetFeature.MOVESET_FEATURE__MAXPP,
                value: move.maxpp,
            });
            this.assignActionBuffer({
                offset: actionOffset,
                index: MovesetFeature.MOVESET_FEATURE__PP_RATIO,
                value: MAX_RATIO_TOKEN * (move.pp / move.maxpp),
            });
        } else {
            this.assignActionBuffer({
                offset: actionOffset,
                index: MovesetFeature.MOVESET_FEATURE__HAS_PP,
                value: MovesetHasPP.MOVESET_HAS_PP__NO,
            });
        }
        let moveId = move.id;
        if (moveId.startsWith("return")) {
            moveId = "return" as ID;
        } else if (moveId.startsWith("frustration")) {
            moveId = "frustration" as ID;
        } else if (moveId.startsWith("hiddenpower")) {
            const power = parseInt(moveId.slice(-2));
            if (isNaN(power)) {
                moveId = "hiddenpower" as ID;
            } else {
                moveId = moveId.slice(0, -2) as ID;
            }
        }
        this.assignActionBuffer({
            offset: actionOffset,
            index: MovesetFeature.MOVESET_FEATURE__MOVE_ID,
            value: IndexValueFromEnum(MovesEnum, moveId),
        });
        this.assignActionBuffer({
            offset: actionOffset,
            index: MovesetFeature.MOVESET_FEATURE__ACTION_TYPE,
            value: ActionType.ACTION_TYPE__MOVE,
        });
        if ("disabled" in move) {
            this.assignActionBuffer({
                offset: actionOffset,
                index: MovesetFeature.MOVESET_FEATURE__DISABLED,
                value: move.disabled ? 1 : 0,
            });
        }
    }

    build() {
        const actives = [
            ...(this.request?.active ?? [null]),
        ] as Protocol.MoveRequest["active"];
        if (actives.length < 2) {
            actives.push(null);
        }

        const switches = (this.request?.side?.pokemon ??
            []) as Protocol.Request.SideInfo["pokemon"];

        let actionOffset = 0;

        const addPadRows = (numRows: number) => {
            for (let i = 0; i < numRows; i++) {
                this.assignActionBuffer({
                    offset: actionOffset,
                    index: MovesetFeature.MOVESET_FEATURE__MOVE_ID,
                    value: MovesEnum.MOVES_ENUM___PAD,
                });
                actionOffset += numMoveFeatures;
            }
        };

        for (const [activeIndex, activePokemon] of actives.entries()) {
            const moves = activePokemon?.moves ?? [];
            const wildcardMoves = activePokemon?.maxMoves ?? moves;
            if (activePokemon !== null) {
                const { pokemon, index: entityIndex } =
                    this.player.eventHandler.getPokemon(
                        switches[activeIndex].ident,
                        false,
                    );
                if (pokemon === null) {
                    throw new Error(
                        `Pokemon ${switches[activeIndex].ident} not found`,
                    );
                }
                for (const action of moves) {
                    this.pushMoveAction(actionOffset, action);
                    this.assignActionBuffer({
                        offset: actionOffset,
                        index: MovesetFeature.MOVESET_FEATURE__ENTITY_IDX,
                        value: entityIndex,
                    });
                    this.assignActionBuffer({
                        offset: actionOffset,
                        index: MovesetFeature.MOVESET_FEATURE__IS_WILDCARD,
                        value: 0,
                    });
                    actionOffset += numMoveFeatures;
                }
                addPadRows(4 - moves.length);

                for (const action of wildcardMoves) {
                    this.pushMoveAction(actionOffset, action);
                    this.assignActionBuffer({
                        offset: actionOffset,
                        index: MovesetFeature.MOVESET_FEATURE__IS_WILDCARD,
                        value: 1,
                    });
                    actionOffset += numMoveFeatures;
                }
                addPadRows(4 - wildcardMoves.length);
            } else {
                addPadRows(8);
            }
        }

        return new Uint8Array(this.actionBuffer.buffer);
    }
}

class PublicActionHandler {
    player: TrainablePlayerAI;
    actionBuffer: Int16Array;

    constructor(player: TrainablePlayerAI) {
        this.player = player;
        this.actionBuffer = new Int16Array(4 * 4 * numMoveFeatures);
    }

    assignActionBuffer(args: { offset: number; index: number; value: number }) {
        const { offset, index, value } = args;
        this.actionBuffer[offset + index] = value;
    }

    pushMoveAction(
        actionOffset: number,
        move:
            | {
                  name: Protocol.MoveName;
                  id: ID;
                  ppUsed: number;
                  virtual?: boolean;
              }
            | {
                  name: Protocol.MoveName;
                  id: ID;
                  ppUsed: number;
                  pp: number;
                  maxpp: number;
                  target: MoveTarget;
                  disabled?: boolean;
                  virtual?: boolean;
              },
    ) {
        const moveDexData = this.player.publicBattle.gens.dex.moves.get(
            move.id,
        );
        const maxpp = Math.floor((moveDexData.pp * 8) / 5);

        if ("ppUsed" in move) {
            const pp = maxpp - move.ppUsed;

            this.assignActionBuffer({
                offset: actionOffset,
                index: MovesetFeature.MOVESET_FEATURE__HAS_PP,
                value: MovesetHasPP.MOVESET_HAS_PP__YES,
            });
            this.assignActionBuffer({
                offset: actionOffset,
                index: MovesetFeature.MOVESET_FEATURE__PP,
                value: pp,
            });
            this.assignActionBuffer({
                offset: actionOffset,
                index: MovesetFeature.MOVESET_FEATURE__MAXPP,
                value: maxpp,
            });
            this.assignActionBuffer({
                offset: actionOffset,
                index: MovesetFeature.MOVESET_FEATURE__PP_RATIO,
                value: MAX_RATIO_TOKEN * (pp / maxpp),
            });
        } else {
            this.assignActionBuffer({
                offset: actionOffset,
                index: MovesetFeature.MOVESET_FEATURE__HAS_PP,
                value: MovesetHasPP.MOVESET_HAS_PP__NO,
            });
        }
        let moveId = move.id;
        if (moveId.startsWith("return")) {
            moveId = "return" as ID;
        } else if (moveId.startsWith("frustration")) {
            moveId = "frustration" as ID;
        } else if (moveId.startsWith("hiddenpower")) {
            const power = parseInt(moveId.slice(-2));
            if (isNaN(power)) {
                moveId = "hiddenpower" as ID;
            } else {
                moveId = moveId.slice(0, -2) as ID;
            }
        }
        this.assignActionBuffer({
            offset: actionOffset,
            index: MovesetFeature.MOVESET_FEATURE__MOVE_ID,
            value: IndexValueFromEnum(MovesEnum, moveId),
        });
        this.assignActionBuffer({
            offset: actionOffset,
            index: MovesetFeature.MOVESET_FEATURE__ACTION_TYPE,
            value: ActionType.ACTION_TYPE__MOVE,
        });
        if ("disabled" in move) {
            this.assignActionBuffer({
                offset: actionOffset,
                index: MovesetFeature.MOVESET_FEATURE__DISABLED,
                value: move.disabled ? 1 : 0,
            });
        }
    }

    build(playerIndex: number) {
        const side = this.player.publicBattle.sides[playerIndex];

        const actives = [...side.active];
        if (actives.length < 2) {
            actives.push(null);
        }

        const switches = side.team;
        const hasTera = switches.some((poke) => poke?.isTerastallized);

        let actionOffset = 0;

        const addTokenRows = (numRows: number, value: number) => {
            for (let i = 0; i < numRows; i++) {
                this.assignActionBuffer({
                    offset: actionOffset,
                    index: MovesetFeature.MOVESET_FEATURE__MOVE_ID,
                    value,
                });
                actionOffset += numMoveFeatures;
            }
        };

        for (const [activeIndex, activePokemon] of actives.entries()) {
            const moves = (activePokemon?.moveSlots ?? []).slice(0, 4);
            const wildcardMoves = hasTera ? [] : moves;

            if (activePokemon !== null) {
                const { pokemon, index: entityIndex } =
                    this.player.eventHandler.getPokemon(
                        activePokemon.ident,
                        true,
                    );
                if (pokemon === null) {
                    throw new Error(`Pokemon ${activePokemon.ident} not found`);
                }
                for (const action of moves) {
                    this.pushMoveAction(actionOffset, action);
                    this.assignActionBuffer({
                        offset: actionOffset,
                        index: MovesetFeature.MOVESET_FEATURE__ENTITY_IDX,
                        value: entityIndex,
                    });
                    this.assignActionBuffer({
                        offset: actionOffset,
                        index: MovesetFeature.MOVESET_FEATURE__IS_WILDCARD,
                        value: 0,
                    });
                    actionOffset += numMoveFeatures;
                }
                addTokenRows(4 - moves.length, MovesEnum.MOVES_ENUM___UNK);

                for (const action of wildcardMoves) {
                    this.pushMoveAction(actionOffset, action);
                    this.assignActionBuffer({
                        offset: actionOffset,
                        index: MovesetFeature.MOVESET_FEATURE__IS_WILDCARD,
                        value: 1,
                    });
                    actionOffset += numMoveFeatures;
                }
                addTokenRows(
                    4 - wildcardMoves.length,
                    MovesEnum.MOVES_ENUM___UNK,
                );
            } else {
                addTokenRows(8, MovesEnum.MOVES_ENUM___PAD);
            }
        }

        return new Uint8Array(this.actionBuffer.buffer);
    }
}

export class RewardTracker {
    prevFaintedCount: [number, number];
    currFaintedCount: [number, number];

    constructor() {
        this.prevFaintedCount = [0, 0];
        this.currFaintedCount = [0, 0];
    }

    updateFaintedCount(battle: Battle) {
        this.prevFaintedCount = this.currFaintedCount;
        this.currFaintedCount = battle.sides.map(
            (side) => side.team.filter((poke) => poke.fainted).length,
        ) as [number, number];
    }
}

export class StateHandler {
    player: TrainablePlayerAI;

    constructor(player: TrainablePlayerAI) {
        this.player = player;
    }

    getActionMask(args: { request?: AnyObject | null; format?: string }): {
        actionMask: OneDBoolean;
        isStruggling: boolean;
    } {
        const { request, format } = args;

        const arrLength = numActionFeatures ** 2;
        const actionMask = new OneDBoolean(
            arrLength,
            Uint8Array,
            numActionFeatures,
        );
        let isStruggling = false;

        const setAll = (val: boolean) => {
            for (let i = 0; i < arrLength; i++) {
                actionMask.set(i, val);
            }
        };
        setAll(false);

        if (request === undefined || request === null) {
            setAll(true);
        } else {
            if (request.wait) {
                setAll(true);
            } else if (request.forceSwitch) {
                const pokemon = request.side
                    .pokemon as Protocol.Request.SideInfo["pokemon"];

                for (const [i, mustSwitch] of request.forceSwitch.entries()) {
                    if (i != this.player.choices.length) {
                        continue;
                    }

                    const rowColValPassValue =
                        ActionEnum[
                            `ACTION_ENUM__ALLY_${
                                i + 1
                            }_PASS` as keyof typeof ActionEnum
                        ];
                    if (!mustSwitch) {
                        actionMask.setRowCol(
                            rowColValPassValue,
                            rowColValPassValue,
                            true,
                        );
                        continue;
                    }

                    const canSwitch = Array.from(pokemon.entries()).filter(
                        ([j, p]) => {
                            return (
                                p &&
                                j >= request.forceSwitch.length &&
                                !this.player.choices.includes(
                                    `switch ${j + 1}`,
                                ) &&
                                !p.condition.endsWith(" fnt") ===
                                    !pokemon[i].reviving
                            );
                        },
                    );
                    if (canSwitch.length === 0) {
                        actionMask.setRowCol(
                            rowColValPassValue,
                            rowColValPassValue,
                            true,
                        );
                        continue;
                    }

                    for (const [j, _] of canSwitch) {
                        actionMask.setRowCol(
                            ActionEnum[
                                `ACTION_ENUM__RESERVE_${
                                    j + 1
                                }` as keyof typeof ActionEnum
                            ],
                            ActionEnum.ACTION_ENUM__ALLY_1 + i,
                            true,
                        );
                    }
                }
            } else if (request.active) {
                let [
                    canMegaEvo,
                    canUltraBurst,
                    canZMove,
                    canDynamax,
                    canTerastallize,
                ] = [true, true, true, true, true];

                const pokemon = request.side
                    .pokemon as Protocol.Request.SideInfo["pokemon"];

                for (const [i, active] of request.active.entries()) {
                    if (i != this.player.choices.length) {
                        continue;
                    }

                    const rowColValPassValue =
                        ActionEnum[
                            `ACTION_ENUM__ALLY_${
                                i + 1
                            }_PASS` as keyof typeof ActionEnum
                        ];
                    if (
                        pokemon[i].condition.endsWith(` fnt`) ||
                        pokemon[i].commanding
                    ) {
                        actionMask.setRowCol(
                            rowColValPassValue,
                            rowColValPassValue,
                            true,
                        );
                        continue;
                    }

                    canMegaEvo = canMegaEvo && active.canMegaEvo;
                    canUltraBurst = canUltraBurst && active.canUltraBurst;
                    canZMove = canZMove && !!active.canZMove;
                    canDynamax = canDynamax && !!active.canDynamax;
                    canTerastallize =
                        canTerastallize && !!active.canTerastallize;

                    const useMaxMoves =
                        (!active.canDynamax && active.maxMoves) || canDynamax;
                    const possibleMoves = (
                        useMaxMoves ? active.maxMoves.maxMoves : active.moves
                    ) as Protocol.Request.ActivePokemon["moves"];

                    let canMove = Array.from(possibleMoves.entries()).filter(
                        ([_, move]) => !("disabled" in move) || !move.disabled,
                    );
                    const hasAlly =
                        pokemon.length > 1 &&
                        !pokemon[i ^ 1].condition.endsWith(` fnt`);
                    const filtered = canMove.filter(([_, move]) => {
                        if (!("target" in move)) {
                            // move has no target property (e.g. Recharge), keep it
                            return true;
                        }
                        return move.target !== `adjacentAlly` || hasAlly;
                    });
                    canMove = filtered.length ? filtered : canMove;

                    for (const [j, move] of canMove) {
                        const actionIndices: number[] = [];
                        if (
                            this.player.privateBattle.gameType === "doubles" &&
                            "target" in move
                        ) {
                            switch (move.target) {
                                case "any":
                                case "normal":
                                    actionIndices.push(
                                        ActionEnum[
                                            `ACTION_ENUM__ALLY_${
                                                2 - i
                                            }` as keyof typeof ActionEnum
                                        ],
                                        ActionEnum.ACTION_ENUM__ENEMY_1,
                                        ActionEnum.ACTION_ENUM__ENEMY_2,
                                    );
                                    break;

                                case "adjacentAlly":
                                    actionIndices.push(
                                        ActionEnum[
                                            `ACTION_ENUM__ALLY_${
                                                2 - i
                                            }` as keyof typeof ActionEnum
                                        ],
                                    );
                                    break;

                                case "adjacentFoe":
                                    actionIndices.push(
                                        ActionEnum.ACTION_ENUM__ENEMY_1,
                                        ActionEnum.ACTION_ENUM__ENEMY_2,
                                    );
                                    break;

                                case "adjacentAllyOrSelf":
                                    actionIndices.push(
                                        ActionEnum.ACTION_ENUM__ALLY_1,
                                        ActionEnum.ACTION_ENUM__ALLY_2,
                                    );
                                    break;

                                case "self":
                                case "all":
                                case "allySide":
                                case "foeSide":
                                case "allyTeam":
                                case "randomNormal":
                                case "allAdjacent":
                                case "allAdjacentFoes":
                                case "allies":
                                case "scripted":
                                    actionIndices.push(
                                        ActionEnum.ACTION_ENUM__TARGET_AUTO,
                                    );
                                    break;

                                default:
                                    actionIndices.push(
                                        ActionEnum.ACTION_ENUM__TARGET_AUTO,
                                    );
                                    break;
                            }
                        } else {
                            actionIndices.push(
                                ActionEnum.ACTION_ENUM__TARGET_AUTO,
                            );
                        }
                        for (const actionIndex of actionIndices) {
                            if (actionIndex === undefined) {
                                throw new Error(
                                    `Undefined action index for move ${
                                        move.name
                                    } with target ${
                                        "target" in move ? move.target : "N/A"
                                    }`,
                                );
                            }
                            const rowIndex =
                                ActionEnum[
                                    `ACTION_ENUM__ALLY_${i + 1}_MOVE_${
                                        j + 1
                                    }` as keyof typeof ActionEnum
                                ];
                            actionMask.setRowCol(rowIndex, actionIndex, true);
                            const wildCardRowIndex =
                                ActionEnum[
                                    `ACTION_ENUM__ALLY_${i + 1}_MOVE_${
                                        j + 1
                                    }_WILDCARD` as keyof typeof ActionEnum
                                ];

                            const wildCardChosenAlready = this.player.choices
                                .map((x) =>
                                    WILDCARDS.map((wildcard) =>
                                        x.endsWith(wildcard),
                                    ).some((x) => x),
                                )
                                .some((x) => x);
                            const canWildCard =
                                (!!canMegaEvo ||
                                    !!canUltraBurst ||
                                    !!canZMove ||
                                    !!canTerastallize) &&
                                !wildCardChosenAlready;

                            actionMask.setRowCol(
                                wildCardRowIndex,
                                actionIndex,
                                canWildCard,
                            );
                        }
                    }

                    const canSwitch = Array.from(pokemon.entries()).filter(
                        ([j, p]) => {
                            return (
                                p &&
                                !p.active &&
                                !this.player.choices.includes(
                                    `switch ${j + 1}`,
                                ) &&
                                !p.condition.endsWith(" fnt")
                            );
                        },
                    );
                    const switches = active.trapped ? [] : canSwitch;

                    const switchToIndex = ActionEnum.ACTION_ENUM__ALLY_1 + i;

                    if (switches.length > 0) {
                        for (const [j, _] of switches) {
                            const switchFromIndex =
                                ActionEnum.ACTION_ENUM__RESERVE_1 + j;

                            actionMask.setRowCol(
                                switchFromIndex,
                                switchToIndex,
                                true,
                            );
                        }
                    }
                }
            } else if (request.teamPreview) {
                const pokemon = request.side.pokemon;

                for (let i = this.player.choices.length; i < 6; i++) {
                    for (let j = this.player.choices.length; j < 6; j++) {
                        const poke_i = pokemon[i];
                        const poke_j = pokemon[j];

                        if (poke_i && poke_j) {
                            const switchFromIndex =
                                ActionEnum.ACTION_ENUM__RESERVE_1 + i;
                            const switchToIndex =
                                ActionEnum.ACTION_ENUM__RESERVE_1 + j;

                            actionMask.setRowCol(
                                switchFromIndex,
                                switchToIndex,
                                true,
                            );
                        }
                    }
                }
            }
        }

        return { actionMask, isStruggling };
    }

    getMyMoveset(): Uint8Array {
        const actionHandler = new PrivateActionHandler(this.player);
        return actionHandler.build();
    }

    getOppMoveset(): Uint8Array {
        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }
        const actionHandler = new PublicActionHandler(this.player);
        return actionHandler.build(1 - playerIndex);
    }

    getPublicTeamFromSide(playerIndex: number): {
        publicBuffer: Int16Array;
        revealedBuffer: Int16Array;
    } {
        const side = this.player.publicBattle.sides[playerIndex];
        const publicBuffer = new Int16Array(6 * numPublicEntityNodeFeatures);
        const revealedBuffer = new Int16Array(
            6 * numRevealedEntityNodeFeatures,
        );

        let publicOffset = 0;
        let revealedOffset = 0;
        const team = side.team.slice(0, 6);
        if (team.length > 6) {
            console.log(
                `Warning: team length is greater than 6: ${team.length}`,
            );
        }

        const relativeSide = isMySide(side.n, this.player.getPlayerIndex());

        const scoreOrder = (pokemon: Pokemon) => {
            let score = 0;
            if (pokemon.isActive()) {
                score += 1;
            }
            score += pokemon.slot;
            return score;
        };

        try {
            for (const member of [...team].sort(
                (a, b) => scoreOrder(b) - scoreOrder(a),
            )) {
                const { publicData, revealedData } = getArrayFromPublicPokemon(
                    member,
                    relativeSide,
                );
                publicBuffer.set(publicData, publicOffset);
                revealedBuffer.set(revealedData, revealedOffset);
                publicOffset += numPublicEntityNodeFeatures;
                revealedOffset += numRevealedEntityNodeFeatures;
            }

            const { publicData: publicUnkData, revealedData: revealedUnkData } =
                relativeSide ? unkPokemon1 : unkPokemon0;
            for (let i = team.length; i < 6; i++) {
                publicBuffer.set(publicUnkData, publicOffset);
                revealedBuffer.set(revealedUnkData, revealedOffset);
                publicOffset += numPublicEntityNodeFeatures;
                revealedOffset += numRevealedEntityNodeFeatures;
            }

            // const {
            //     publicData: publicNullData,
            //     revealedData: revealedNullData,
            // } = nullPokemon;
            // for (let i = Math.max(team.length, 6); i < 11; i++) {
            //     publicBuffer.set(publicNullData, publicOffset);
            //     revealedBuffer.set(revealedNullData, revealedOffset);
            //     publicOffset += numPublicEntityNodeFeatures;
            //     revealedOffset += numRevealedEntityNodeFeatures;
            // }

            // for (let i = side.totalPokemon; i < 6; i++) {
            //     revealedBuffer.set(nullPokemon, revealedOffset);
            //     revealedOffset += numRevealedEntityNodeFeatures;
            // }
        } catch (error) {
            console.log(error);
            console.log(team);
            return { publicBuffer, revealedBuffer };
        }

        if (publicOffset !== publicBuffer.length) {
            throw new Error(
                `Buffer length mismatch: expected ${publicBuffer.length}, got ${publicOffset}`,
            );
        }
        return { publicBuffer, revealedBuffer };
    }

    getPrivateTeam(playerIndex: number): Int16Array {
        const request = this.player.getRequest();
        if (request === undefined) {
            throw new Error("Request is undefined");
        }
        const requestPokemon = request.side?.pokemon as
            | Protocol.Request.SideInfo["pokemon"]
            | undefined;

        let offset = 0;
        const buffer = new Int16Array(6 * numPrivateEntityNodeFeatures);

        if (requestPokemon === undefined) {
            throw new Error("Request pokemon is undefined");
        } else {
            let privateOrder;
            privateOrder = [...requestPokemon];

            // TODO: Fix the sorting here
            // if (request.teamPreview) {
            //     privateOrder = [...requestPokemon];
            //     for (const [toIdx, choice] of this.player.choices.entries()) {
            //         const fromIdx = parseInt(choice.split(" ")[1]) - 1;
            //         [privateOrder[toIdx], privateOrder[fromIdx]] = [
            //             privateOrder[fromIdx],
            //             privateOrder[toIdx],
            //         ];
            //     }
            // } else {
            //     privateOrder = [...requestPokemon].sort((a, b) => {
            //         return a.ident.localeCompare(b.ident);
            //     });
            // }

            for (const member of privateOrder) {
                const name = toID(member.speciesForme);

                const matchedTeamMate = this.player.privateBattle.sides[
                    playerIndex
                ].team.find((teamMate) => {
                    const setSpecies = toID(
                        teamMate.baseSpecies.baseSpecies.toLowerCase(),
                    );
                    return (
                        setSpecies === name ||
                        setSpecies.includes(name) ||
                        name.includes(setSpecies)
                    );
                });
                buffer.set(
                    getArrayFromPrivatePokemon(matchedTeamMate, member),
                    offset,
                );
                offset += numPrivateEntityNodeFeatures;
            }
        }

        return buffer;
    }

    getPublicTeam(playerIndex: number): {
        publicData: Int16Array;
        revealedData: Int16Array;
    } {
        const publicDataArr = [];
        const revealedDataArr = [];
        for (const idx of [playerIndex, 1 - playerIndex]) {
            const { publicBuffer, revealedBuffer } =
                this.getPublicTeamFromSide(idx);
            publicDataArr.push(publicBuffer);
            revealedDataArr.push(revealedBuffer);
        }

        return {
            publicData: concatenateArrays(publicDataArr),
            revealedData: concatenateArrays(revealedDataArr),
        };
    }

    getHistory(numHistory: number = NUM_HISTORY) {
        return this.player.eventHandler.edgeBuffer.getHistory(numHistory);
    }

    getWinReward(): {
        winReward?: number;
        lossReward?: number;
        tieReward?: number;
    } {
        if (this.player.done) {
            if (this.player.finishedEarly) {
                return { tieReward: 1 };
            }
            for (let i = this.player.log.length - 1; i >= 0; i--) {
                const line = this.player.log.at(i) ?? "";
                // eslint-disable-next-line @typescript-eslint/no-unused-vars
                const [_, cmd, winner] = line.split("|");
                if (cmd === "win") {
                    return this.player.userName === winner
                        ? { winReward: 1 }
                        : { lossReward: 1 };
                } else if (cmd === "tie") {
                    return { tieReward: 1 };
                }
            }
        }
        return {};
    }

    getHpRatio(member: Pokemon) {
        const isHpBug = !member.fainted && member.hp === 0;
        const hp = isHpBug ? 100 : member.hp;
        const maxHp = isHpBug ? 100 : member.maxhp;
        return hp / maxHp;
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

        const { winReward, lossReward, tieReward } = this.getWinReward();
        infoBuffer[InfoFeature.INFO_FEATURE__WIN_REWARD] =
            MAX_RATIO_TOKEN * (winReward ?? 0);
        infoBuffer[InfoFeature.INFO_FEATURE__LOSS_REWARD] =
            MAX_RATIO_TOKEN * (lossReward ?? 0);
        infoBuffer[InfoFeature.INFO_FEATURE__TIE_REWARD] =
            MAX_RATIO_TOKEN * (tieReward ?? 0);

        const mySide = this.player.privateBattle.sides[playerIndex];
        infoBuffer[InfoFeature.INFO_FEATURE__NUM_ACTIVE] = mySide.active.length;

        const request = this.player.getRequest();
        if (request === undefined) {
            throw new Error("Request is undefined");
        }

        if (request.teamPreview) {
            infoBuffer[InfoFeature.INFO_FEATURE__REQUEST_TYPE] =
                RequestType.REQUEST_TYPE__TEAM;
        } else if (request.forceSwitch) {
            infoBuffer[InfoFeature.INFO_FEATURE__REQUEST_TYPE] =
                RequestType.REQUEST_TYPE__SWITCH;
        } else {
            infoBuffer[InfoFeature.INFO_FEATURE__REQUEST_TYPE] =
                RequestType.REQUEST_TYPE__MOVE;
        }

        const requestPokemon = request.side?.pokemon as
            | Protocol.Request.SideInfo["pokemon"]
            | undefined;

        if (requestPokemon) {
            let privateOrder;

            if (request.teamPreview) {
                privateOrder = [...requestPokemon];
                for (const [toIdx, choice] of this.player.choices.entries()) {
                    const fromIdx = parseInt(choice.split(" ")[1]) - 1;
                    [privateOrder[toIdx], privateOrder[fromIdx]] = [
                        privateOrder[fromIdx],
                        privateOrder[toIdx],
                    ];
                }
            } else {
                privateOrder = [...requestPokemon].sort((a, b) => {
                    return a.ident.localeCompare(b.ident);
                });
            }
            for (const [sortedIdx, sortedMember] of privateOrder.entries()) {
                for (const [
                    unsortedIdx,
                    unsortedMember,
                ] of requestPokemon.entries()) {
                    if (sortedMember.ident === unsortedMember.ident) {
                        infoBuffer[
                            InfoFeature[
                                `INFO_FEATURE__SWITCH_ORDER_VALUE${unsortedIdx}` as keyof typeof InfoFeature
                            ]
                        ] = sortedIdx;
                        break;
                    }
                }
            }
        }

        infoBuffer[InfoFeature.INFO_FEATURE__HAS_PREV_ACTION] = 0;
        if (!request.teamPreview && this.player.actions.length > 0) {
            infoBuffer[InfoFeature.INFO_FEATURE__HAS_PREV_ACTION] = 1;
            const lastAction = this.player.actions.at(-1)!;
            infoBuffer[InfoFeature.INFO_FEATURE__PREV_ACTION_SRC] =
                lastAction.getSrc();
            infoBuffer[InfoFeature.INFO_FEATURE__PREV_ACTION_TGT] =
                lastAction.getTgt();
        }

        return new Uint8Array(infoBuffer.buffer);
    }

    static toReadablePrivate(buffer: Uint8Array) {
        const teamBuffer = new Int16Array(buffer.buffer);
        const entityDatums = [];
        const numEntites = teamBuffer.length / numPrivateEntityNodeFeatures;
        for (let entityIndex = 0; entityIndex < numEntites; entityIndex++) {
            const start = entityIndex * numPrivateEntityNodeFeatures;
            const end = (entityIndex + 1) * numPrivateEntityNodeFeatures;
            entityDatums.push(
                entityPrivateArrayToObject(teamBuffer.slice(start, end)),
            );
        }
        return entityDatums;
    }

    static toReadablePublic(buffer: Uint8Array) {
        const teamBuffer = new Int16Array(buffer.buffer);
        const entityDatums = [];
        const numEntites = teamBuffer.length / numPublicEntityNodeFeatures;
        for (let entityIndex = 0; entityIndex < numEntites; entityIndex++) {
            const start = entityIndex * numPublicEntityNodeFeatures;
            const end = (entityIndex + 1) * numPublicEntityNodeFeatures;
            entityDatums.push(
                entityPublicArrayToObject(teamBuffer.slice(start, end)),
            );
        }
        return entityDatums;
    }

    static toReadableRevealed(buffer: Uint8Array) {
        const teamBuffer = new Int16Array(buffer.buffer);
        const entityDatums = [];
        const numEntites = teamBuffer.length / numRevealedEntityNodeFeatures;
        for (let entityIndex = 0; entityIndex < numEntites; entityIndex++) {
            const start = entityIndex * numRevealedEntityNodeFeatures;
            const end = (entityIndex + 1) * numRevealedEntityNodeFeatures;
            entityDatums.push(
                entityRevealedArrayToObject(teamBuffer.slice(start, end)),
            );
        }
        return entityDatums;
    }

    static toReadableMoveset(buffer: Uint8Array) {
        const movesetBuffer = new Int16Array(buffer.buffer);
        const entityDatums = [];
        const numMoves = movesetBuffer.length / numMoveFeatures;
        for (let moveIndex = 0; moveIndex < numMoves; moveIndex++) {
            const start = moveIndex * numMoveFeatures;
            const end = (moveIndex + 1) * numMoveFeatures;
            entityDatums.push(
                moveArrayToObject(movesetBuffer.slice(start, end)),
            );
        }
        return entityDatums;
    }

    getField() {
        const fieldBuffer = new Int16Array(numFieldFeatures);
        const playerIndex = this.player.getPlayerIndex()!;
        for (const side of this.player.privateBattle.sides) {
            const relativeSide = isMySide(side.n, playerIndex);
            const sideOffset = relativeSide
                ? FieldFeature.FIELD_FEATURE__MY_SIDECONDITIONS0
                : FieldFeature.FIELD_FEATURE__OPP_SIDECONDITIONS0;
            const spikesOffset = relativeSide
                ? FieldFeature.FIELD_FEATURE__MY_SPIKES
                : FieldFeature.FIELD_FEATURE__OPP_SPIKES;
            const toxisSpikesOffset = relativeSide
                ? FieldFeature.FIELD_FEATURE__MY_TOXIC_SPIKES
                : FieldFeature.FIELD_FEATURE__OPP_TOXIC_SPIKES;

            let sideConditionBuffer = BigInt(0b0);
            for (const [id] of Object.entries(side.sideConditions)) {
                const featureIndex = IndexValueFromEnum(SideconditionEnum, id);
                sideConditionBuffer |= BigInt(1) << BigInt(featureIndex);
            }
            fieldBuffer.set(
                bigIntToInt16Array(sideConditionBuffer),
                sideOffset,
            );
            if (side.sideConditions.spikes) {
                fieldBuffer[spikesOffset] = side.sideConditions.spikes.level;
            }
            if (side.sideConditions.toxicspikes) {
                fieldBuffer[toxisSpikesOffset] =
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

        fieldBuffer[FieldFeature.FIELD_FEATURE__WEATHER_ID] = weatherIndex;
        fieldBuffer[FieldFeature.FIELD_FEATURE__WEATHER_MAX_DURATION] =
            field.weatherState.maxDuration;
        fieldBuffer[FieldFeature.FIELD_FEATURE__WEATHER_MIN_DURATION] =
            field.weatherState.minDuration;
        return new Uint8Array(fieldBuffer.buffer);
    }

    build(): EnvironmentState {
        this.player.rewardTracker.updateFaintedCount(this.player.privateBattle);

        const request = this.player.getRequest();
        if (!this.player.done && request === undefined) {
            throw new Error("Need Request");
        }

        const state = new EnvironmentState();
        const info = this.getInfo();
        state.setInfo(info);

        const { actionMask } = this.getActionMask({
            request,
            format: this.player.privateBattle.gameType,
        });
        state.setActionMask(actionMask.buffer);

        const {
            historyEntityPublic,
            historyEntityRevealed,
            historyEntityEdges,
            historyField,
            historyLength,
            historyPackedLength,
        } = this.getHistory(NUM_HISTORY);
        state.setHistoryEntityPublic(historyEntityPublic);
        state.setHistoryEntityRevealed(historyEntityRevealed);
        state.setHistoryEntityEdges(historyEntityEdges);
        state.setHistoryField(historyField);
        state.setHistoryLength(historyLength);
        state.setHistoryPackedLength(historyPackedLength);

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error("Player index is undefined");
        }

        const privateTeam = this.getPrivateTeam(playerIndex);
        state.setPrivateTeam(new Uint8Array(privateTeam.buffer));

        const { publicData, revealedData } = this.getPublicTeam(playerIndex);
        state.setPublicTeam(new Uint8Array(publicData.buffer));
        state.setRevealedTeam(new Uint8Array(revealedData.buffer));

        state.setMyMoveset(this.getMyMoveset());
        state.setOppMoveset(this.getOppMoveset());

        state.setField(this.getField());

        return state;
    }
}
