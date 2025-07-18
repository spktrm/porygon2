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
    numPokemonFields as numPokemonFeatures,
    numMovesetFields,
    numMoveFields,
    NUM_HISTORY,
    jsonDatum,
    numContextFields,
    numInfoFields,
} from "./data";
import { NA, Pokemon, Side } from "@pkmn/client";
import { Ability, Item, Move, BoostID } from "@pkmn/dex-types";
import { ID, MoveTarget } from "@pkmn/types";
import { Condition, Effect } from "@pkmn/data";
import { History } from "../../protos/history_pb";
import { OneDBoolean, TypedArray } from "./utils";
import {
    AbsoluteEdgeFeature,
    AbsoluteEdgeFeatureMap,
    ContextFeature,
    EntityFeature,
    InfoFeature,
    MovesetActionTypeEnum,
    MovesetFeature,
    MovesetHasPPEnum,
    RelativeEdgeFeature,
    RelativeEdgeFeatureMap,
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

const entityArrayToObject = (array: Int16Array) => {
    const volatilesFlat = array.slice(
        EntityFeature.ENTITY_FEATURE__VOLATILES0,
        EntityFeature.ENTITY_FEATURE__VOLATILES8 + 1,
    );
    const volatilesIndices = int16ArrayToBitIndices(volatilesFlat);

    const typechangeFlat = array.slice(
        EntityFeature.ENTITY_FEATURE__TYPECHANGE0,
        EntityFeature.ENTITY_FEATURE__TYPECHANGE1 + 1,
    );
    const typechangeIndices = int16ArrayToBitIndices(typechangeFlat);

    const moveIndicies = Array.from(
        array.slice(
            EntityFeature.ENTITY_FEATURE__MOVEID0,
            EntityFeature.ENTITY_FEATURE__MOVEID3 + 1,
        ),
    );

    return {
        species:
            jsonDatum["species"][array[EntityFeature.ENTITY_FEATURE__SPECIES]],
        item: jsonDatum["items"][array[EntityFeature.ENTITY_FEATURE__ITEM]],
        hp: array[EntityFeature.ENTITY_FEATURE__HP_RATIO] / MAX_RATIO_TOKEN,
        fainted: !!array[EntityFeature.ENTITY_FEATURE__FAINTED],
        ability:
            jsonDatum["abilities"][
                array[EntityFeature.ENTITY_FEATURE__ABILITY]
            ],
        moves: moveIndicies.map((index) => jsonDatum["moves"][index]),
        volatiles: volatilesIndices.map(
            (index) => jsonDatum["volatileStatus"][index],
        ),
        typechange: typechangeIndices.map(
            (index) => jsonDatum["typechart"][index],
        ),
        active: array[EntityFeature.ENTITY_FEATURE__ACTIVE],
        side: array[EntityFeature.ENTITY_FEATURE__SIDE],
        status: jsonDatum["status"][
            array[EntityFeature.ENTITY_FEATURE__STATUS]
        ],
    };
};

const relativeArrayToObject = (array: Int16Array) => {
    const minorArgsFlat = array.slice(
        RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__MINOR_ARG0,
        RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__MINOR_ARG3 + 1,
    );
    const minorArgIndices = int16ArrayToBitIndices(minorArgsFlat);

    const sideConditionsFlat = array.slice(
        RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__SIDECONDITIONS0,
        RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__SIDECONDITIONS1 + 1,
    );
    const sideConditionIndices = int16ArrayToBitIndices(sideConditionsFlat);

    return {
        majorArg:
            jsonDatum["battleMajorArgs"][
                array[RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__MAJOR_ARG]
            ],
        minorArgs: minorArgIndices.map(
            (index) => jsonDatum["battleMinorArgs"][index],
        ),
        move: jsonDatum["moves"][
            array[RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__MOVE_TOKEN]
        ],
        damage:
            array[RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__DAMAGE_RATIO] /
            MAX_RATIO_TOKEN,
        heal:
            array[RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__HEAL_RATIO] /
            MAX_RATIO_TOKEN,
        sideConditions: sideConditionIndices.map(
            (index) => jsonDatum["sideCondition"][index],
        ),
        num_from_sources:
            array[RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__NUM_FROM_SOURCES],
        from_source: [
            jsonDatum["Effect"][
                array[
                    RelativeEdgeFeature
                        .RELATIVE_EDGE_FEATURE__FROM_SOURCE_TOKEN0
                ]
            ],
            jsonDatum["Effect"][
                array[
                    RelativeEdgeFeature
                        .RELATIVE_EDGE_FEATURE__FROM_SOURCE_TOKEN1
                ]
            ],
            jsonDatum["Effect"][
                array[
                    RelativeEdgeFeature
                        .RELATIVE_EDGE_FEATURE__FROM_SOURCE_TOKEN2
                ]
            ],
            jsonDatum["Effect"][
                array[
                    RelativeEdgeFeature
                        .RELATIVE_EDGE_FEATURE__FROM_SOURCE_TOKEN3
                ]
            ],
            jsonDatum["Effect"][
                array[
                    RelativeEdgeFeature
                        .RELATIVE_EDGE_FEATURE__FROM_SOURCE_TOKEN4
                ]
            ],
        ],
        boosts: {
            EDGE_BOOST_ATK_VALUE:
                array[
                    RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__BOOST_ATK_VALUE
                ],
            EDGE_BOOST_DEF_VALUE:
                array[
                    RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__BOOST_DEF_VALUE
                ],
            EDGE_BOOST_SPA_VALUE:
                array[
                    RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__BOOST_SPA_VALUE
                ],
            EDGE_BOOST_SPD_VALUE:
                array[
                    RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__BOOST_SPD_VALUE
                ],
            EDGE_BOOST_SPE_VALUE:
                array[
                    RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__BOOST_SPE_VALUE
                ],
            EDGE_BOOST_ACCURACY_VALUE:
                array[
                    RelativeEdgeFeature
                        .RELATIVE_EDGE_FEATURE__BOOST_ACCURACY_VALUE
                ],
            EDGE_BOOST_EVASION_VALUE:
                array[
                    RelativeEdgeFeature
                        .RELATIVE_EDGE_FEATURE__BOOST_EVASION_VALUE
                ],
        },
        status: jsonDatum["status"][
            array[RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__STATUS_TOKEN]
        ],
    };
};

const absoluteArrayToObject = (array: Int16Array) => {
    return {
        turnOrder:
            array[AbsoluteEdgeFeature.ABSOLUTE_EDGE_FEATURE__TURN_ORDER_VALUE],
        requestCount:
            array[AbsoluteEdgeFeature.ABSOLUTE_EDGE_FEATURE__REQUEST_COUNT],
        weatherId:
            jsonDatum["weather"][
                array[AbsoluteEdgeFeature.ABSOLUTE_EDGE_FEATURE__WEATHER_ID]
            ],
    };
};

const WEATHERS = {
    sand: "sandstorm",
    sun: "sunnyday",
    rain: "raindance",
    hail: "hail",
    snow: "snow",
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
    return new POKEMON_ARRAY_CONSTRUCTOR(numPokemonFeatures);
}

function getUnkPokemon(n: number) {
    const data = getBlankPokemonArr();
    data[EntityFeature.ENTITY_FEATURE__SPECIES] =
        SpeciesEnum.SPECIES_ENUM___UNK;
    data[EntityFeature.ENTITY_FEATURE__ITEM] = ItemsEnum.ITEMS_ENUM___UNK;
    data[EntityFeature.ENTITY_FEATURE__GENDER] =
        GendernameEnum.GENDERNAME_ENUM___UNK;
    data[EntityFeature.ENTITY_FEATURE__ITEM_EFFECT] =
        ItemeffecttypesEnum.ITEMEFFECTTYPES_ENUM___NULL;
    data[EntityFeature.ENTITY_FEATURE__ABILITY] =
        AbilitiesEnum.ABILITIES_ENUM___UNK;
    data[EntityFeature.ENTITY_FEATURE__FAINTED] = 0;
    data[EntityFeature.ENTITY_FEATURE__LEVEL] = 100;
    data[EntityFeature.ENTITY_FEATURE__HP] = 100;
    data[EntityFeature.ENTITY_FEATURE__MAXHP] = 100;
    data[EntityFeature.ENTITY_FEATURE__HP_RATIO] = MAX_RATIO_TOKEN; // Full Health
    data[EntityFeature.ENTITY_FEATURE__STATUS] = StatusEnum.STATUS_ENUM___NULL;
    data[EntityFeature.ENTITY_FEATURE__MOVEID0] = MovesEnum.MOVES_ENUM___UNK;
    data[EntityFeature.ENTITY_FEATURE__MOVEID1] = MovesEnum.MOVES_ENUM___UNK;
    data[EntityFeature.ENTITY_FEATURE__MOVEID2] = MovesEnum.MOVES_ENUM___UNK;
    data[EntityFeature.ENTITY_FEATURE__MOVEID3] = MovesEnum.MOVES_ENUM___UNK;
    data[EntityFeature.ENTITY_FEATURE__MOVEPP0] = 0;
    data[EntityFeature.ENTITY_FEATURE__MOVEPP1] = 0;
    data[EntityFeature.ENTITY_FEATURE__MOVEPP2] = 0;
    data[EntityFeature.ENTITY_FEATURE__MOVEPP3] = 0;
    data[EntityFeature.ENTITY_FEATURE__LAST_MOVE] = MovesEnum.MOVES_ENUM___UNK;
    data[EntityFeature.ENTITY_FEATURE__NUM_MOVES] = 4;
    data[EntityFeature.ENTITY_FEATURE__HAS_STATUS] = 0;
    data[EntityFeature.ENTITY_FEATURE__SIDE] = n;
    return data;
}

const unkPokemon0 = getUnkPokemon(0);
const unkPokemon1 = getUnkPokemon(1);

function getNullPokemon() {
    const data = getBlankPokemonArr();
    data[EntityFeature.ENTITY_FEATURE__SPECIES] =
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
        dataArr[EntityFeature.ENTITY_FEATURE__NUM_MOVES] = 4;
    } else {
        for (remainingIndex; remainingIndex < 4; remainingIndex++) {
            moveIds.push(MovesEnum.MOVES_ENUM___NULL);
            movePps.push(0);
        }
        dataArr[EntityFeature.ENTITY_FEATURE__NUM_MOVES] = moveSlots.length;
    }

    dataArr[EntityFeature.ENTITY_FEATURE__IS_PUBLIC] = +isPublic;
    dataArr[EntityFeature.ENTITY_FEATURE__SPECIES] = IndexValueFromEnum<
        typeof SpeciesEnum
    >(SpeciesEnum, baseSpecies);
    dataArr[EntityFeature.ENTITY_FEATURE__ITEM] = item
        ? IndexValueFromEnum<typeof ItemsEnum>(ItemsEnum, item)
        : ItemsEnum.ITEMS_ENUM___UNK;
    dataArr[EntityFeature.ENTITY_FEATURE__ITEM_EFFECT] = itemEffect
        ? IndexValueFromEnum(ItemeffecttypesEnum, itemEffect)
        : ItemeffecttypesEnum.ITEMEFFECTTYPES_ENUM___NULL;

    const possibleAbilities = Object.values(pokemon.baseSpecies.abilities);
    if (ability) {
        const actualAbility = IndexValueFromEnum(AbilitiesEnum, ability);
        dataArr[EntityFeature.ENTITY_FEATURE__ABILITY] = actualAbility;
    } else if (possibleAbilities.length === 1) {
        const onlyAbility = possibleAbilities[0]
            ? IndexValueFromEnum(AbilitiesEnum, possibleAbilities[0])
            : AbilitiesEnum.ABILITIES_ENUM___UNK;
        dataArr[EntityFeature.ENTITY_FEATURE__ABILITY] = onlyAbility;
    } else {
        dataArr[EntityFeature.ENTITY_FEATURE__ABILITY] =
            AbilitiesEnum.ABILITIES_ENUM___UNK;
    }

    // We take candidate lastMove here instead of pokemons since
    // transformed does not lastMove
    if (candidate.lastMove === "") {
        dataArr[EntityFeature.ENTITY_FEATURE__LAST_MOVE] =
            MovesEnum.MOVES_ENUM___NULL;
    } else if (candidate.lastMove === "switch-in") {
        dataArr[EntityFeature.ENTITY_FEATURE__LAST_MOVE] =
            MovesEnum.MOVES_ENUM___SWITCH;
    } else {
        dataArr[EntityFeature.ENTITY_FEATURE__LAST_MOVE] = IndexValueFromEnum(
            MovesEnum,
            candidate.lastMove,
        );
    }

    dataArr[EntityFeature.ENTITY_FEATURE__GENDER] = IndexValueFromEnum(
        GendernameEnum,
        pokemon.gender,
    );
    dataArr[EntityFeature.ENTITY_FEATURE__ACTIVE] = candidate.isActive()
        ? 1
        : 0;
    dataArr[EntityFeature.ENTITY_FEATURE__FAINTED] = candidate.fainted ? 1 : 0;

    // We take candidate HP here instead of pokemons since
    // transformed does not copy HP
    const isHpBug = !candidate.fainted && candidate.hp === 0;
    const hp = isHpBug ? 100 : candidate.hp;
    const maxHp = isHpBug ? 100 : candidate.maxhp;
    const hpRatio = hp / maxHp;
    dataArr[EntityFeature.ENTITY_FEATURE__HP] = hp;
    dataArr[EntityFeature.ENTITY_FEATURE__MAXHP] = maxHp;
    dataArr[EntityFeature.ENTITY_FEATURE__HP_RATIO] = Math.floor(
        MAX_RATIO_TOKEN * hpRatio,
    );

    // We take candidate status here instead of pokemons since
    // transformed does not copy status
    dataArr[EntityFeature.ENTITY_FEATURE__STATUS] = candidate.status
        ? IndexValueFromEnum(StatusEnum, candidate.status)
        : StatusEnum.STATUS_ENUM___NULL;
    dataArr[EntityFeature.ENTITY_FEATURE__HAS_STATUS] = candidate.status
        ? 1
        : 0;
    dataArr[EntityFeature.ENTITY_FEATURE__TOXIC_TURNS] =
        candidate.statusState.toxicTurns;
    dataArr[EntityFeature.ENTITY_FEATURE__SLEEP_TURNS] =
        candidate.statusState.sleepTurns;
    dataArr[EntityFeature.ENTITY_FEATURE__BEING_CALLED_BACK] =
        candidate.beingCalledBack ? 1 : 0;
    dataArr[EntityFeature.ENTITY_FEATURE__TRAPPED] = candidate.trapped ? 1 : 0;
    dataArr[EntityFeature.ENTITY_FEATURE__NEWLY_SWITCHED] =
        candidate.newlySwitched ? 1 : 0;

    // We take pokemon level here
    dataArr[EntityFeature.ENTITY_FEATURE__LEVEL] = pokemon.level;

    for (let moveIndex = 0; moveIndex < 4; moveIndex++) {
        dataArr[
            EntityFeature[`ENTITY_FEATURE__MOVEID${moveIndex as MoveIndex}`]
        ] = moveIds[moveIndex];
        dataArr[
            EntityFeature[`ENTITY_FEATURE__MOVEPP${moveIndex as MoveIndex}`]
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
        EntityFeature.ENTITY_FEATURE__VOLATILES0,
    );

    dataArr[EntityFeature.ENTITY_FEATURE__SIDE] = isMySide(
        pokemon.side.n,
        playerIndex,
    );

    // Only copy pokemon boosts
    dataArr[EntityFeature.ENTITY_FEATURE__BOOST_ATK_VALUE] =
        pokemon.boosts.atk ?? 0;
    dataArr[EntityFeature.ENTITY_FEATURE__BOOST_DEF_VALUE] =
        pokemon.boosts.def ?? 0;
    dataArr[EntityFeature.ENTITY_FEATURE__BOOST_SPA_VALUE] =
        pokemon.boosts.spa ?? 0;
    dataArr[EntityFeature.ENTITY_FEATURE__BOOST_SPD_VALUE] =
        pokemon.boosts.spd ?? 0;
    dataArr[EntityFeature.ENTITY_FEATURE__BOOST_SPE_VALUE] =
        pokemon.boosts.spe ?? 0;
    dataArr[EntityFeature.ENTITY_FEATURE__BOOST_EVASION_VALUE] =
        pokemon.boosts.evasion ?? 0;
    dataArr[EntityFeature.ENTITY_FEATURE__BOOST_ACCURACY_VALUE] =
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
        EntityFeature.ENTITY_FEATURE__TYPECHANGE0,
    );

    return dataArr;
}

const numRelativeEdgeFeatures = Object.keys(RelativeEdgeFeature).length;
const numAbsoluteEdgeFeatures = Object.keys(AbsoluteEdgeFeature).length;

class Edge {
    player: TrainablePlayerAI;

    entityData: Int16Array;
    relativeEdgeData: Int16Array;
    absoluteEdgeData: Int16Array;

    constructor(player: TrainablePlayerAI) {
        this.player = player;

        this.entityData = new Int16Array(2 * numPokemonFeatures);
        this.relativeEdgeData = new Int16Array(2 * numRelativeEdgeFeatures);
        this.absoluteEdgeData = new Int16Array(numAbsoluteEdgeFeatures);

        this.updateSideData();
        this.updateFieldData();
    }

    clone() {
        const edge = new Edge(this.player);
        edge.relativeEdgeData.set(this.relativeEdgeData);
        edge.entityData.set(this.entityData);
        return edge;
    }

    setPokeFromData(data: Int16Array, offset: number) {
        this.entityData.set(data, offset);
    }

    setPoke(pokemon: Pokemon | null) {
        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }
        if (pokemon !== null) {
            const data = getArrayFromPokemon(pokemon, playerIndex);
            const offset = isMySide(pokemon!.side.n, playerIndex)
                ? 0
                : numPokemonFeatures;
            this.setPokeFromData(data, offset);
        }
    }

    updateSideData() {
        const playerIndex = this.player.getPlayerIndex()!;

        for (const side of this.player.publicBattle.sides) {
            const isMe = isMySide(side.n, playerIndex);
            const entityOffset = isMe ? 0 : numPokemonFeatures;
            const sideConditionOffset =
                (isMe ? 0 : numRelativeEdgeFeatures) +
                RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__SIDECONDITIONS0;
            this.updateEntityData(side, entityOffset, playerIndex);
            this.updateSideConditionData(
                side,
                sideConditionOffset,
                playerIndex,
            );
        }
    }

    updateEntityData(side: Side, entityOffset: number, playerIndex: number) {
        const onlyActive = side.active[0];
        const activePokemonBuffer = getArrayFromPokemon(
            onlyActive,
            playerIndex,
        );
        this.entityData.set(activePokemonBuffer, entityOffset);
    }

    updateSideConditionData(
        side: Side,
        relativeEdgeOffset: number,
        playerIndex: number,
    ) {
        let sideConditionBuffer = BigInt(0b0);
        for (const [id] of Object.entries(side.sideConditions)) {
            const featureIndex = IndexValueFromEnum(SideconditionEnum, id);
            sideConditionBuffer |= BigInt(1) << BigInt(featureIndex);
        }
        this.relativeEdgeData.set(
            bigIntToInt16Array(sideConditionBuffer),
            relativeEdgeOffset,
        );
        if (side.sideConditions.spikes) {
            this.setRelativeEdgeFeature({
                featureIndex: RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__SPIKES,
                isMe: isMySide(side.n, playerIndex),
                value: side.sideConditions.spikes.level,
            });
        }
        if (side.sideConditions.toxicspikes) {
            this.setRelativeEdgeFeature({
                featureIndex:
                    RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__TOXIC_SPIKES,
                isMe: isMySide(side.n, playerIndex),
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

        this.absoluteEdgeData[
            AbsoluteEdgeFeature.ABSOLUTE_EDGE_FEATURE__WEATHER_ID
        ] = weatherIndex;
        this.absoluteEdgeData[
            AbsoluteEdgeFeature.ABSOLUTE_EDGE_FEATURE__WEATHER_MAX_DURATION
        ] = field.weatherState.maxDuration;
        this.absoluteEdgeData[
            AbsoluteEdgeFeature.ABSOLUTE_EDGE_FEATURE__WEATHER_MIN_DURATION
        ] = field.weatherState.minDuration;
    }

    setRelativeEdgeFeature(args: {
        featureIndex: RelativeEdgeFeatureMap[keyof RelativeEdgeFeatureMap];
        isMe: number;
        value: number;
    }) {
        const { featureIndex, isMe, value } = args;
        if (featureIndex === undefined) {
            throw new Error("Index cannot be undefined");
        }
        if (value === undefined) {
            throw new Error("Value cannot be undefined");
        }
        const index = featureIndex + (isMe ? 0 : numRelativeEdgeFeatures);
        this.relativeEdgeData[index] = value;
    }

    setAbsoluteEdgeFeature(
        featureIndex: AbsoluteEdgeFeatureMap[keyof AbsoluteEdgeFeatureMap],

        value: number,
    ) {
        if (featureIndex === undefined) {
            throw new Error("Index cannot be undefined");
        }
        if (value === undefined) {
            throw new Error("Value cannot be undefined");
        }

        this.absoluteEdgeData[featureIndex] = value;
    }

    getRelativeEdgeFeature(
        featureIndex: RelativeEdgeFeatureMap[keyof RelativeEdgeFeatureMap],
        isMe: number,
    ) {
        const index = featureIndex + (isMe ? 0 : numRelativeEdgeFeatures);
        return this.relativeEdgeData.at(index);
    }

    addMajorArg(argName: MajorArgNames, isMe: number) {
        const index = IndexValueFromEnum(BattlemajorargsEnum, argName);
        this.setRelativeEdgeFeature({
            featureIndex: RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__MAJOR_ARG,
            isMe,
            value: index,
        });
    }

    updateEdgeFromOf(effect: Partial<Effect>, isMe: number) {
        const { effectType } = effect;
        if (effectType) {
            const fromTypeToken = IndexValueFromEnum(
                EffecttypesEnum,
                effectType,
            );
            const fromSourceToken = getEffectToken(effect);

            const numFromTypes =
                this.getRelativeEdgeFeature(
                    RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__NUM_FROM_TYPES,
                    isMe,
                ) ?? 0;
            const numFromSources =
                this.getRelativeEdgeFeature(
                    RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__NUM_FROM_SOURCES,
                    isMe,
                ) ?? 0;

            if (numFromTypes < 5) {
                this.setRelativeEdgeFeature({
                    featureIndex:
                        (RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__FROM_TYPE_TOKEN0 +
                            numFromTypes) as RelativeEdgeFeatureMap[keyof RelativeEdgeFeatureMap],
                    isMe,
                    value: fromTypeToken,
                });
                this.setRelativeEdgeFeature({
                    featureIndex:
                        RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__NUM_FROM_TYPES,
                    isMe,
                    value: numFromTypes + 1,
                });
            }
            if (numFromSources < 5) {
                this.setRelativeEdgeFeature({
                    featureIndex:
                        (RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__FROM_SOURCE_TOKEN0 +
                            numFromSources) as RelativeEdgeFeatureMap[keyof RelativeEdgeFeatureMap],
                    isMe,
                    value: fromSourceToken,
                });
                this.setRelativeEdgeFeature({
                    featureIndex:
                        RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__NUM_FROM_SOURCES,
                    isMe,
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

class EdgeBuffer {
    player: TrainablePlayerAI;

    entityData: Int16Array;
    relativeEdgeData: Int16Array;
    absoluteEdgeData: Int16Array;

    entityCursor: number;
    relativeEdgeCursor: number;
    absoluteEdgeCursor: number;

    prevEntityCursor: number;
    prevRelativeEdgeCursor: number;
    prevAbsoluteEdgeCursor: number;

    numEdges: number;
    maxEdges: number;

    constructor(player: TrainablePlayerAI) {
        this.player = player;

        const maxEdges = 4000;
        this.maxEdges = maxEdges;

        this.entityData = new Int16Array(maxEdges * 2 * numPokemonFeatures);
        this.relativeEdgeData = new Int16Array(
            maxEdges * 2 * numRelativeEdgeFeatures,
        );
        this.absoluteEdgeData = new Int16Array(
            maxEdges * numAbsoluteEdgeFeatures,
        );

        this.entityCursor = 0;
        this.relativeEdgeCursor = 0;
        this.absoluteEdgeCursor = 0;

        this.prevEntityCursor = 0;
        this.prevRelativeEdgeCursor = 0;
        this.prevAbsoluteEdgeCursor = 0;

        this.numEdges = 0;
    }

    updateLatestEntityData(
        activePokemon: (Pokemon | null)[],
        entityOffset: number,
        playerIndex: number,
    ) {
        const onlyActive = activePokemon[0];
        const activePokemonBuffer =
            onlyActive === null
                ? nullPokemon
                : getArrayFromPokemon(onlyActive, playerIndex);
        this.entityData.set(activePokemonBuffer, entityOffset);
    }

    updateLatestSideConditionData(
        side: Side,
        relativeEdgeOffset: number,
        playerIndex: number,
    ) {
        let sideConditionBuffer = BigInt(0b0);
        for (const [id] of Object.entries(side.sideConditions)) {
            const featureIndex = IndexValueFromEnum(SideconditionEnum, id);
            sideConditionBuffer |= BigInt(1) << BigInt(featureIndex);
        }
        this.relativeEdgeData.set(
            bigIntToInt16Array(sideConditionBuffer),
            relativeEdgeOffset,
        );
        if (side.sideConditions.spikes) {
            this.setLatestRelativeEdgeFeature({
                featureIndex: RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__SPIKES,
                isMe: isMySide(side.n, playerIndex),
                value: side.sideConditions.spikes.level,
            });
        }
        if (side.sideConditions.toxicspikes) {
            this.setLatestRelativeEdgeFeature({
                featureIndex:
                    RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__TOXIC_SPIKES,
                isMe: isMySide(side.n, playerIndex),
                value: side.sideConditions.toxicspikes.level,
            });
        }
    }

    updateLatestFieldData() {
        const field = this.player.publicBattle.field;
        const weatherIndex = field.weatherState.id
            ? IndexValueFromEnum(
                  WeatherEnum,
                  WEATHERS[field.weatherState.id as never],
              )
            : WeatherEnum.WEATHER_ENUM___NULL;

        this.absoluteEdgeData[
            this.prevAbsoluteEdgeCursor +
                AbsoluteEdgeFeature.ABSOLUTE_EDGE_FEATURE__WEATHER_ID
        ] = weatherIndex;
        this.absoluteEdgeData[
            this.prevAbsoluteEdgeCursor +
                AbsoluteEdgeFeature.ABSOLUTE_EDGE_FEATURE__WEATHER_MAX_DURATION
        ] = field.weatherState.maxDuration;
        this.absoluteEdgeData[
            this.prevAbsoluteEdgeCursor +
                AbsoluteEdgeFeature.ABSOLUTE_EDGE_FEATURE__WEATHER_MIN_DURATION
        ] = field.weatherState.minDuration;
    }

    updateLatestSideData() {
        const playerIndex = this.player.getPlayerIndex()!;

        for (const side of this.player.publicBattle.sides) {
            const isMe = isMySide(side.n, playerIndex);
            const entityOffset =
                this.prevEntityCursor + (isMe ? 0 : numPokemonFeatures);
            const sideConditionOffset =
                this.prevRelativeEdgeCursor +
                (isMe ? 0 : numRelativeEdgeFeatures) +
                RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__SIDECONDITIONS0;
            this.updateLatestEntityData(side.active, entityOffset, playerIndex);
            this.updateLatestSideConditionData(
                side,
                sideConditionOffset,
                playerIndex,
            );
        }
    }

    setLatestRelativeEdgeFeature(args: {
        featureIndex: RelativeEdgeFeatureMap[keyof RelativeEdgeFeatureMap];
        isMe: number;
        value: number;
    }) {
        const { featureIndex, isMe, value } = args;
        if (featureIndex === undefined) {
            throw new Error("Index cannot be undefined");
        }
        if (value === undefined) {
            throw new Error("Value cannot be undefined");
        }
        const index =
            this.prevRelativeEdgeCursor +
            featureIndex +
            (isMe ? 0 : numRelativeEdgeFeatures);
        this.relativeEdgeData[index] = value;
    }

    getLatestRelativeEdgeFeature(
        featureIndex: RelativeEdgeFeatureMap[keyof RelativeEdgeFeatureMap],
        isMe: number,
    ) {
        const index =
            this.prevRelativeEdgeCursor +
            featureIndex +
            (isMe ? 0 : numRelativeEdgeFeatures);
        return this.relativeEdgeData.at(index);
    }

    updateLatestMinorArgs(
        argName: MinorArgNames,
        isMe: number,
        precision: number = 16,
    ) {
        this.updateLatestSideData();
        this.updateLatestFieldData();

        const index = IndexValueFromEnum(BattleminorargsEnum, argName);
        const featureIndex = {
            0: RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__MINOR_ARG0,
            1: RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__MINOR_ARG1,
            2: RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__MINOR_ARG2,
            3: RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__MINOR_ARG3,
        }[Math.floor(index / precision)];
        if (featureIndex === undefined) {
            throw new Error();
        }
        const currentValue = this.getLatestRelativeEdgeFeature(
            featureIndex,
            isMe,
        )!;
        const newValue = currentValue | (1 << index % precision);
        this.setLatestRelativeEdgeFeature({
            featureIndex,
            isMe,
            value: newValue,
        });
    }

    updateLatestMajorArg(argName: MajorArgNames, isMe: number) {
        const index = IndexValueFromEnum(BattlemajorargsEnum, argName);
        this.setLatestRelativeEdgeFeature({
            featureIndex: RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__MAJOR_ARG,
            isMe,
            value: index,
        });
    }

    setLatestAbsoluteEdgeFeature(
        featureIndex: AbsoluteEdgeFeatureMap[keyof AbsoluteEdgeFeatureMap],
        value: number,
    ) {
        if (featureIndex === undefined) {
            throw new Error("Index cannot be undefined");
        }
        if (value === undefined) {
            throw new Error("Value cannot be undefined");
        }
        const index = this.prevAbsoluteEdgeCursor + featureIndex;
        this.absoluteEdgeData[index] = value;
    }

    updateLatestEdgeFromOf(effect: Partial<Effect>, isMe: number) {
        const { id, effectType, kind } = effect;
        const trueEffectType = effectType === undefined ? kind : effectType;
        if (trueEffectType !== undefined && id !== undefined) {
            const fromTypeToken = IndexValueFromEnum(
                EffecttypesEnum,
                trueEffectType,
            );
            const fromSourceToken = getEffectToken(effect);
            const numFromTypes =
                this.getLatestRelativeEdgeFeature(
                    RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__NUM_FROM_TYPES,
                    isMe,
                ) ?? 0;
            const numFromSources =
                this.getLatestRelativeEdgeFeature(
                    RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__NUM_FROM_SOURCES,
                    isMe,
                ) ?? 0;
            if (numFromTypes < 5) {
                this.setLatestRelativeEdgeFeature({
                    featureIndex:
                        (RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__FROM_TYPE_TOKEN0 +
                            numFromTypes) as RelativeEdgeFeatureMap[keyof RelativeEdgeFeatureMap],
                    isMe,
                    value: fromTypeToken,
                });
                this.setLatestRelativeEdgeFeature({
                    featureIndex:
                        RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__NUM_FROM_TYPES,
                    isMe,
                    value: numFromTypes + 1,
                });
            }
            if (numFromSources < 5) {
                this.setLatestRelativeEdgeFeature({
                    featureIndex:
                        (RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__FROM_SOURCE_TOKEN0 +
                            numFromSources) as RelativeEdgeFeatureMap[keyof RelativeEdgeFeatureMap],
                    isMe,
                    value: fromSourceToken,
                });
                this.setLatestRelativeEdgeFeature({
                    featureIndex:
                        RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__NUM_FROM_SOURCES,
                    isMe,
                    value: numFromSources + 1,
                });
            }
        }
    }

    addEdge(edge: Edge) {
        this.entityData.set(edge.entityData, this.entityCursor);
        this.relativeEdgeData.set(
            edge.relativeEdgeData,
            this.relativeEdgeCursor,
        );
        this.absoluteEdgeData.set(
            edge.absoluteEdgeData,
            this.absoluteEdgeCursor,
        );

        this.prevEntityCursor = this.entityCursor;
        this.prevRelativeEdgeCursor = this.relativeEdgeCursor;
        this.prevAbsoluteEdgeCursor = this.absoluteEdgeCursor;

        this.entityCursor += 2 * numPokemonFeatures;
        this.relativeEdgeCursor += 2 * numRelativeEdgeFeatures;
        this.absoluteEdgeCursor += numAbsoluteEdgeFeatures;

        this.numEdges += 1;
    }

    getHistory(numHistory: number = NUM_HISTORY) {
        const historyLength = Math.max(1, Math.min(this.numEdges, numHistory));
        const historyEntities = new Uint8Array(
            this.entityData.slice(
                this.entityCursor - historyLength * 2 * numPokemonFeatures,
                this.entityCursor,
            ).buffer,
        );
        const historyRelativeEdges = new Uint8Array(
            this.relativeEdgeData.slice(
                this.relativeEdgeCursor -
                    historyLength * 2 * numRelativeEdgeFeatures,
                this.relativeEdgeCursor,
            ).buffer,
        );
        const historyAbsoluteEdge = new Uint8Array(
            this.absoluteEdgeData.slice(
                this.absoluteEdgeCursor -
                    historyLength * numAbsoluteEdgeFeatures,
                this.absoluteEdgeCursor,
            ).buffer,
        );
        return {
            historyEntities,
            historyRelativeEdges,
            historyAbsoluteEdge,
            historyLength,
        };
    }

    static toReadableHistory(history: History) {
        const historyItems = [];
        const historyLength = history.getLength() ?? 0;
        const historyEntities = new Int16Array(
            history.getEntities_asU8().buffer,
        );
        const historyRelativeEdges = new Int16Array(
            history.getRelativeEdges_asU8().buffer,
        );
        const historyAbsoluteEdges = new Int16Array(
            history.getAbsoluteEdge_asU8().buffer,
        );

        for (
            let historyIndex = 0;
            historyIndex < historyLength;
            historyIndex++
        ) {
            const stepEntities = historyEntities.slice(
                historyIndex * 2 * numPokemonFeatures,
                (historyIndex + 1) * 2 * numPokemonFeatures,
            );
            const stepEntity0 = stepEntities.slice(0, numPokemonFeatures);
            const stepEntity1 = stepEntities.slice(
                numPokemonFeatures,
                2 * numPokemonFeatures,
            );
            const stepRelativeEdges = historyRelativeEdges.slice(
                historyIndex * 2 * numRelativeEdgeFeatures,
                (historyIndex + 1) * 2 * numRelativeEdgeFeatures,
            );
            const stepRelativeEdge0 = stepRelativeEdges.slice(
                0,
                numRelativeEdgeFeatures,
            );
            const stepRelativeEdge1 = stepRelativeEdges.slice(
                numRelativeEdgeFeatures,
                2 * numRelativeEdgeFeatures,
            );
            const stepAbsoluteEdge = historyAbsoluteEdges.slice(
                historyIndex * numAbsoluteEdgeFeatures,
                (historyIndex + 1) * numAbsoluteEdgeFeatures,
            );
            historyItems.push({
                myEntites: [0].map((memberIndex) => {
                    const start = memberIndex * numPokemonFeatures;
                    const end = (memberIndex + 1) * numPokemonFeatures;
                    return entityArrayToObject(stepEntity0.slice(start, end));
                }),
                oppEntites: [0].map((memberIndex) => {
                    const start = memberIndex * numPokemonFeatures;
                    const end = (memberIndex + 1) * numPokemonFeatures;
                    return entityArrayToObject(stepEntity1.slice(start, end));
                }),
                relativeEdges: [
                    relativeArrayToObject(stepRelativeEdge0),
                    relativeArrayToObject(stepRelativeEdge1),
                ],
                absoluteEdge: absoluteArrayToObject(stepAbsoluteEdge),
            });
        }
        return historyItems;
    }
}

export class EventHandler implements Protocol.Handler {
    readonly player: TrainablePlayerAI;

    prevHp: Map<string, number>;
    actives: Map<ID, PokemonIdent>;
    turnOrder: number;
    turnNum: number;
    timestamp: number;
    edgeBuffer: EdgeBuffer;
    log: string[];

    constructor(player: TrainablePlayerAI) {
        this.player = player;
        this.prevHp = new Map();
        this.actives = new Map();

        this.edgeBuffer = new EdgeBuffer(player);
        this.turnOrder = 0;
        this.turnNum = 0;
        this.timestamp = 0;
        this.log = [];
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
            const { siden, pokemonid: parsedPokemonid } =
                this.player.publicBattle.parsePokemonId(pokemonid);
            const side = this.player.publicBattle.sides[siden];
            for (const [index, pokemon] of side.team.entries()) {
                if (pokemon.originalIdent === parsedPokemonid) {
                    return { pokemon, index };
                }
            }
        } else {
            const { siden, pokemonid: parsedPokemonid } =
                this.player.privateBattle.parsePokemonId(pokemonid);
            const side = this.player.privateBattle.sides[siden];
            for (const [index, pokemon] of side.team.entries()) {
                if (pokemon.originalIdent === parsedPokemonid) {
                    return { pokemon, index };
                }
            }
        }

        return { pokemon: null, index: -1 };
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
        edge.setAbsoluteEdgeFeature(
            AbsoluteEdgeFeature.ABSOLUTE_EDGE_FEATURE__REQUEST_COUNT,
            this.player.requestCount,
        );
        edge.setAbsoluteEdgeFeature(
            AbsoluteEdgeFeature.ABSOLUTE_EDGE_FEATURE__VALID,
            1,
        );
        edge.setAbsoluteEdgeFeature(
            AbsoluteEdgeFeature.ABSOLUTE_EDGE_FEATURE__INDEX,
            this.edgeBuffer.numEdges,
        );
        edge.setAbsoluteEdgeFeature(
            AbsoluteEdgeFeature.ABSOLUTE_EDGE_FEATURE__TURN_ORDER_VALUE,
            this.turnOrder,
        );
        this.turnOrder += 1;
        edge.setAbsoluteEdgeFeature(
            AbsoluteEdgeFeature.ABSOLUTE_EDGE_FEATURE__TURN_VALUE,
            this.turnNum,
        );
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

        const { pokemon } = this.getPokemon(poke1Ident as PokemonIdent);
        const isMe = isMySide(pokemon!.side.n, playerIndex);

        const move = this.getMove(moveId);

        const edge = new Edge(this.player);
        edge.addMajorArg(argName, isMe);
        // edge.setRelativeEdgeFeature({
        //     featureIndex:
        //         RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__ACTION_TOKEN,
        //     isMe,
        //     value: IndexValueFromEnum(ActionsEnum, `move_${move.id}`),
        // });
        edge.setRelativeEdgeFeature({
            featureIndex: RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__MOVE_TOKEN,
            isMe,
            value: IndexValueFromEnum(MovesEnum, move.id),
        });
        this.addEdge(edge);

        if (kwArgs.from) {
            const fromEffect = this.getCondition(kwArgs.from);
            this.edgeBuffer.updateLatestEdgeFromOf(fromEffect, isMe);
        }
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

        const { pokemon } = this.getPokemon(poke1Ident)!;
        const isMe = isMySide(pokemon!.side.n, playerIndex);

        const edge = new Edge(this.player);
        edge.addMajorArg(argName, isMe);
        // edge.setRelativeEdgeFeature({
        //     featureIndex:
        //         RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__ACTION_TOKEN,
        //     isMe,
        //     value: IndexValueFromEnum(
        //         ActionsEnum,
        //         `switch_${poke.species.baseSpecies.toLowerCase()}`,
        //     ),
        // });
        edge.setRelativeEdgeFeature({
            featureIndex: RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__MOVE_TOKEN,
            isMe,
            value: MovesEnum.MOVES_ENUM___SWITCH,
        });
        this.addEdge(edge);

        if (argName === "switch") {
            const from = (kwArgs as KWArgs["|switch|"]).from;
            if (from) {
                const fromEffect = this.getCondition(from);
                this.edgeBuffer.updateLatestEdgeFromOf(fromEffect, isMe);
            }
        }
    }

    "|cant|"(args: Args["|cant|"]) {
        const [argName, poke1Ident, conditionId, moveId] = args;

        const edge = new Edge(this.player);

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const { pokemon } = this.getPokemon(poke1Ident)!;
        const isMe = isMySide(pokemon!.side.n, playerIndex);

        if (moveId) {
            const move = this.getMove(moveId);
            // edge.setRelativeEdgeFeature({
            //     featureIndex:
            //         RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__ACTION_TOKEN,
            //     isMe,
            //     value: IndexValueFromEnum(ActionsEnum, `move_${move.id}`),
            // });
            edge.setRelativeEdgeFeature({
                featureIndex:
                    RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__MOVE_TOKEN,
                isMe,
                value: IndexValueFromEnum(MovesEnum, move.id),
            });
        }

        const condition = this.getCondition(conditionId);

        edge.addMajorArg(argName, isMe);
        edge.updateEdgeFromOf(condition, isMe);

        this.addEdge(edge);
    }

    "|faint|"(args: Args["|faint|"]) {
        const [argName, poke1Ident] = args;

        const edge = new Edge(this.player);

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const { pokemon } = this.getPokemon(poke1Ident)!;
        const isMe = isMySide(pokemon!.side.n, playerIndex);

        edge.addMajorArg(argName, isMe);
        this.addEdge(edge);
    }

    "|-fail|"(args: Args["|-fail|"], kwArgs: KWArgs["|-fail|"]) {
        const [argName, poke1Ident] = args;

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const { pokemon } = this.getPokemon(poke1Ident)!;
        const isMe = isMySide(pokemon!.side.n, playerIndex);

        const fromEffect = this.getCondition(kwArgs.from);

        this.edgeBuffer.updateLatestMinorArgs(argName, isMe);
        this.edgeBuffer.updateLatestEdgeFromOf(fromEffect, isMe);
    }

    "|-block|"(args: Args["|-block|"], kwArgs: KWArgs["|-block|"]) {
        const [argName, poke1Ident] = args;

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const { pokemon } = this.getPokemon(poke1Ident)!;
        const isMe = isMySide(pokemon!.side.n, playerIndex);

        const fromEffect = this.getCondition(kwArgs.from);

        this.edgeBuffer.updateLatestMinorArgs(argName, isMe);
        this.edgeBuffer.updateLatestEdgeFromOf(fromEffect, isMe);
    }

    "|-notarget|"(args: Args["|-notarget|"]) {
        const [argName, poke1Ident] = args;

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        if (poke1Ident !== undefined) {
            const { pokemon } = this.getPokemon(poke1Ident)!;
            const isMe = isMySide(pokemon!.side.n, playerIndex);
            this.edgeBuffer.updateLatestMinorArgs(argName, isMe);
        } else {
            for (const isMe of [0, 1]) {
                this.edgeBuffer.updateLatestMinorArgs(argName, isMe);
            }
        }
    }

    "|-miss|"(args: Args["|-miss|"], kwArgs: KWArgs["|-miss|"]) {
        const [argName, poke1Ident, poke2Ident] = args;

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const { pokemon } = this.getPokemon(poke1Ident ?? poke2Ident)!;
        const isMe = isMySide(pokemon!.side.n, playerIndex);

        const fromEffect = this.getCondition(kwArgs.from);

        this.edgeBuffer.updateLatestMinorArgs(argName, isMe);
        this.edgeBuffer.updateLatestEdgeFromOf(fromEffect, isMe);
    }

    "|-damage|"(
        args: Args["|-damage|"] | Args["|-sethp|"],
        kwArgs: KWArgs["|-damage|"] | KWArgs["|-sethp|"],
    ) {
        const [argName, poke1Ident] = args;

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const { pokemon } = this.getPokemon(poke1Ident)!;
        const isMe = isMySide(pokemon!.side.n, playerIndex);
        const trueIdent = pokemon!.originalIdent;

        if (kwArgs.from) {
            const fromEffect = this.getCondition(kwArgs.from);
            this.edgeBuffer.updateLatestEdgeFromOf(fromEffect, isMe);
        }

        if (!this.prevHp.has(trueIdent)) {
            this.prevHp.set(trueIdent, 1);
        }
        const prevHp = this.prevHp.get(trueIdent) ?? 1;
        const currHp = pokemon!.hp / pokemon!.maxhp;
        const diffRatio = currHp - prevHp;
        this.prevHp.set(trueIdent, currHp);
        const addedDamageToken = Math.abs(
            Math.floor(MAX_RATIO_TOKEN * diffRatio),
        );

        this.edgeBuffer.updateLatestMinorArgs(argName, isMe);
        const currentDamageToken =
            this.edgeBuffer.getLatestRelativeEdgeFeature(
                RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__DAMAGE_RATIO,
                isMe,
            ) ?? 0;
        this.edgeBuffer.setLatestRelativeEdgeFeature({
            featureIndex:
                RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__DAMAGE_RATIO,
            isMe,
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
        const [argName, poke1Ident] = args;

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const { pokemon } = this.getPokemon(poke1Ident)!;
        const isMe = isMySide(pokemon!.side.n, playerIndex);
        const trueIdent = pokemon!.originalIdent;

        if (kwArgs.from) {
            const fromEffect = this.getCondition(kwArgs.from);
            this.edgeBuffer.updateLatestEdgeFromOf(fromEffect, isMe);
        }

        if (!this.prevHp.has(trueIdent)) {
            this.prevHp.set(trueIdent, 1);
        }
        const prevHp = this.prevHp.get(trueIdent) ?? 1;
        const currHp = pokemon!.hp / pokemon!.maxhp;
        const diffRatio = currHp - prevHp;
        this.prevHp.set(trueIdent, currHp);
        const addedHealToken = Math.abs(
            Math.floor(MAX_RATIO_TOKEN * diffRatio),
        );

        this.edgeBuffer.updateLatestMinorArgs(argName, isMe);
        const currentHealToken =
            this.edgeBuffer.getLatestRelativeEdgeFeature(
                RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__HEAL_RATIO,
                isMe,
            ) ?? 0;
        this.edgeBuffer.setLatestRelativeEdgeFeature({
            featureIndex: RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__HEAL_RATIO,
            isMe,
            value: Math.min(MAX_RATIO_TOKEN, currentHealToken + addedHealToken),
        });
    }

    "|-sethp|"(args: Args["|-sethp|"], kwArgs: KWArgs["|-sethp|"]) {
        const [argName, poke1Ident] = args;
        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }
        const { pokemon } = this.getPokemon(poke1Ident)!;
        const isMe = isMySide(pokemon!.side.n, playerIndex);
        const trueIdent = pokemon!.originalIdent;
        if (!this.prevHp.has(trueIdent)) {
            this.prevHp.set(trueIdent, 1);
        }
        const prevHp = this.prevHp.get(trueIdent) ?? 1;
        const currHp = pokemon!.hp / pokemon!.maxhp;
        const diffRatio = currHp - prevHp;
        if (diffRatio < 0) {
            this["|-damage|"](
                ["-damage", args[1], args[2]] as Args["|-damage|"],
                kwArgs,
            );
        } else if (diffRatio > 0) {
            this["|-heal|"](
                ["-heal", args[1], args[2]] as Args["|-heal|"],
                kwArgs,
            );
        }
        this.edgeBuffer.updateLatestMinorArgs(argName, isMe);
    }

    "|-status|"(args: Args["|-status|"], kwArgs: KWArgs["|-status|"]) {
        const [argName, poke1Ident, statusId] = args;

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const { pokemon } = this.getPokemon(poke1Ident)!;
        const isMe = isMySide(pokemon!.side.n, playerIndex);

        const fromEffect = this.getCondition(kwArgs.from);
        const statusToken = IndexValueFromEnum(StatusEnum, statusId);

        this.edgeBuffer.updateLatestMinorArgs(argName, isMe);
        this.edgeBuffer.updateLatestEdgeFromOf(fromEffect, isMe);
        this.edgeBuffer.setLatestRelativeEdgeFeature({
            featureIndex:
                RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__STATUS_TOKEN,
            isMe,
            value: statusToken,
        });
    }

    "|-curestatus|"(args: Args["|-curestatus|"]) {
        const [argName, poke1Ident, statusId] = args;

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const { pokemon } = this.getPokemon(poke1Ident)!;
        const isMe = isMySide(pokemon!.side.n, playerIndex);

        const statusToken = IndexValueFromEnum(StatusEnum, statusId);

        this.edgeBuffer.updateLatestMinorArgs(argName, isMe);
        this.edgeBuffer.setLatestRelativeEdgeFeature({
            featureIndex:
                RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__STATUS_TOKEN,
            isMe,
            value: statusToken,
        });
    }

    "|-cureteam|"(args: Args["|-cureteam|"]) {
        const [argName, poke1Ident] = args;

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const { pokemon } = this.getPokemon(poke1Ident)!;
        const isMe = isMySide(pokemon!.side.n, playerIndex);

        this.edgeBuffer.updateLatestMinorArgs(argName, isMe);
    }

    static getStatBoostEdgeFeatureIndex(stat: BoostID) {
        return RelativeEdgeFeature[
            `RELATIVE_EDGE_FEATURE__BOOST_${stat.toLocaleUpperCase()}_VALUE` as `RELATIVE_EDGE_FEATURE__BOOST_${Uppercase<BoostID>}_VALUE`
        ];
    }

    "|-boost|"(args: Args["|-boost|"]) {
        const [argName, poke1Ident, stat, value] = args;

        const featureIndex = EventHandler.getStatBoostEdgeFeatureIndex(stat);

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const { pokemon } = this.getPokemon(poke1Ident)!;
        const isMe = isMySide(pokemon!.side.n, playerIndex);

        this.edgeBuffer.updateLatestMinorArgs(argName, isMe);
        this.edgeBuffer.setLatestRelativeEdgeFeature({
            featureIndex,
            isMe,
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

        const { pokemon } = this.getPokemon(poke1Ident)!;
        const isMe = isMySide(pokemon!.side.n, playerIndex);

        this.edgeBuffer.updateLatestMinorArgs(argName, isMe);
        this.edgeBuffer.setLatestRelativeEdgeFeature({
            featureIndex,
            isMe,
            value: -parseInt(value),
        });
    }

    "|-setboost|"(args: Args["|-setboost|"], kwArgs: KWArgs["|-setboost|"]) {
        const [argName, poke1Ident, stat, value] = args;

        const featureIndex = EventHandler.getStatBoostEdgeFeatureIndex(stat);
        const fromEffect = this.getCondition(kwArgs.from);

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const { pokemon } = this.getPokemon(poke1Ident)!;
        const isMe = isMySide(pokemon!.side.n, playerIndex);

        this.edgeBuffer.updateLatestMinorArgs(argName, isMe);
        this.edgeBuffer.setLatestRelativeEdgeFeature({
            featureIndex,
            isMe,
            value: parseInt(value),
        });
        this.edgeBuffer.updateLatestEdgeFromOf(fromEffect, isMe);
    }

    "|-swapboost|"(args: Args["|-swapboost|"], kwArgs: KWArgs["|-swapboost|"]) {
        const [argName, poke1Ident] = args;

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const { pokemon } = this.getPokemon(poke1Ident)!;
        const isMe = isMySide(pokemon!.side.n, playerIndex);

        const fromEffect = this.getCondition(kwArgs.from);

        this.edgeBuffer.updateLatestMinorArgs(argName, isMe);
        this.edgeBuffer.updateLatestEdgeFromOf(fromEffect, isMe);
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

        const { pokemon } = this.getPokemon(poke1Ident)!;
        const isMe = isMySide(pokemon!.side.n, playerIndex);

        const fromEffect = this.getCondition(kwArgs.from);

        this.edgeBuffer.updateLatestMinorArgs(argName, isMe);
        this.edgeBuffer.updateLatestEdgeFromOf(fromEffect, isMe);
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

        const { pokemon } = this.getPokemon(poke1Ident)!;
        const isMe = isMySide(pokemon!.side.n, playerIndex);

        const fromEffect = this.getCondition(kwArgs.from);

        this.edgeBuffer.updateLatestMinorArgs(argName, isMe);
        this.edgeBuffer.updateLatestEdgeFromOf(fromEffect, isMe);
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

        const { pokemon } = this.getPokemon(poke1Ident)!;
        const isMe = isMySide(pokemon!.side.n, playerIndex);

        const fromEffect = this.getCondition(kwArgs.from);

        this.edgeBuffer.updateLatestMinorArgs(argName, isMe);
        this.edgeBuffer.updateLatestEdgeFromOf(fromEffect, isMe);
    }

    "|-copyboost|"() {}

    "|-weather|"(args: Args["|-weather|"]) {
        const [argName, weatherId] = args;

        const fromEffect = this.getCondition(weatherId);

        const weatherIndex =
            weatherId === "none"
                ? WeatherEnum.WEATHER_ENUM___NULL
                : IndexValueFromEnum(WeatherEnum, weatherId);
        for (const isMe of [0, 1]) {
            this.edgeBuffer.updateLatestMinorArgs(argName, isMe);
            if (weatherId !== "none")
                this.edgeBuffer.updateLatestEdgeFromOf(fromEffect, isMe);
        }
        this.edgeBuffer.setLatestAbsoluteEdgeFeature(
            AbsoluteEdgeFeature.ABSOLUTE_EDGE_FEATURE__WEATHER_ID,
            weatherIndex,
        );
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
        const isMe = isMySide(side.n, playerIndex);

        const fromEffect = this.getCondition(conditionId);

        this.edgeBuffer.updateLatestMinorArgs(argName, isMe);
        this.edgeBuffer.updateLatestEdgeFromOf(fromEffect, isMe);
    }

    "|-sideend|"(args: Args["|-sideend|"]) {
        const [argName, sideId, conditionId] = args;

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const side = this.getSide(sideId);
        const isMe = isMySide(side.n, playerIndex);

        const fromEffect = this.getCondition(conditionId);

        this.edgeBuffer.updateLatestMinorArgs(argName, isMe);
        this.edgeBuffer.updateLatestEdgeFromOf(fromEffect, isMe);
    }

    "|-swapsideconditions|"() {}

    "|-start|"(args: Args["|-start|"], kwArgs: KWArgs["|-start|"]) {
        const [argName, poke1Ident, conditionId] = args;

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const { pokemon } = this.getPokemon(poke1Ident)!;
        const isMe = isMySide(pokemon!.side.n, playerIndex);

        this.edgeBuffer.updateLatestMinorArgs(argName, isMe);
        this.edgeBuffer.updateLatestEdgeFromOf(
            this.getCondition(kwArgs.from),
            isMe,
        );
        this.edgeBuffer.updateLatestEdgeFromOf(
            this.getCondition(
                conditionId.startsWith("perish") ? "perishsong" : conditionId,
            ),
            isMe,
        );
    }

    "|-end|"(args: Args["|-end|"], kwArgs: KWArgs["|-end|"]) {
        const [argName, poke1Ident, conditionId] = args;

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const { pokemon } = this.getPokemon(poke1Ident)!;
        const isMe = isMySide(pokemon!.side.n, playerIndex);

        this.edgeBuffer.updateLatestMinorArgs(argName, isMe);
        this.edgeBuffer.updateLatestEdgeFromOf(
            this.getCondition(kwArgs.from),
            isMe,
        );
        this.edgeBuffer.updateLatestEdgeFromOf(
            this.getCondition(conditionId),
            isMe,
        );
    }

    "|-crit|"(args: Args["|-crit|"]) {
        const [argName, poke1Ident] = args;

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const { pokemon } = this.getPokemon(poke1Ident)!;
        const isMe = isMySide(pokemon!.side.n, playerIndex);

        this.edgeBuffer.updateLatestMinorArgs(argName, isMe);
    }

    "|-supereffective|"(args: Args["|-supereffective|"]) {
        const [argName, poke1Ident] = args;

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const { pokemon } = this.getPokemon(poke1Ident)!;
        const isMe = isMySide(pokemon!.side.n, playerIndex);

        this.edgeBuffer.updateLatestMinorArgs(argName, isMe);
    }

    "|-resisted|"(args: Args["|-resisted|"]) {
        const [argName, poke1Ident] = args;

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const { pokemon } = this.getPokemon(poke1Ident)!;
        const isMe = isMySide(pokemon!.side.n, playerIndex);

        this.edgeBuffer.updateLatestMinorArgs(argName, isMe);
    }

    "|-immune|"(args: Args["|-immune|"], kwArgs: KWArgs["|-immune|"]) {
        const [argName, poke1Ident, conditionId] = args;

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const { pokemon } = this.getPokemon(poke1Ident)!;
        const isMe = isMySide(pokemon!.side.n, playerIndex);

        this.edgeBuffer.updateLatestMinorArgs(argName, isMe);
        this.edgeBuffer.updateLatestEdgeFromOf(
            this.getCondition(kwArgs.from),
            isMe,
        );
        this.edgeBuffer.updateLatestEdgeFromOf(
            this.getCondition(conditionId),
            isMe,
        );
    }

    "|-item|"(args: Args["|-item|"], kwArgs: KWArgs["|-item|"]) {
        const [argName, poke1Ident, itemId] = args;

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const { pokemon } = this.getPokemon(poke1Ident as PokemonIdent)!;
        const isMe = isMySide(pokemon!.side.n, playerIndex);

        const itemIndex = IndexValueFromEnum(ItemsEnum, itemId);
        const fromEffect = this.getCondition(kwArgs.from);

        this.edgeBuffer.updateLatestMinorArgs(argName, isMe);
        this.edgeBuffer.updateLatestEdgeFromOf(fromEffect, isMe);
        this.edgeBuffer.setLatestRelativeEdgeFeature({
            featureIndex: RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__ITEM_TOKEN,
            isMe,
            value: itemIndex,
        });
    }

    "|-enditem|"(args: Args["|-enditem|"], kwArgs: KWArgs["|-enditem|"]) {
        const [argName, poke1Ident, itemId] = args;

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const { pokemon } = this.getPokemon(poke1Ident)!;
        const isMe = isMySide(pokemon!.side.n, playerIndex);

        const itemIndex = IndexValueFromEnum(ItemsEnum, itemId);
        const fromEffect = this.getCondition(kwArgs.from);

        this.edgeBuffer.updateLatestMinorArgs(argName, isMe);
        this.edgeBuffer.updateLatestEdgeFromOf(fromEffect, isMe);
        this.edgeBuffer.setLatestRelativeEdgeFeature({
            featureIndex: RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__ITEM_TOKEN,
            isMe,
            value: itemIndex,
        });
    }

    "|-ability|"(args: Args["|-ability|"], kwArgs: KWArgs["|-ability|"]) {
        const [argName, poke1Ident, abilityId] = args;

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const { pokemon } = this.getPokemon(poke1Ident)!;
        const isMe = isMySide(pokemon!.side.n, playerIndex);

        const abilityIndex = IndexValueFromEnum(AbilitiesEnum, abilityId);

        if (kwArgs.from) {
            const fromEffect = this.getCondition(kwArgs.from);
            this.edgeBuffer.updateLatestEdgeFromOf(fromEffect, isMe);
        }

        this.edgeBuffer.updateLatestMinorArgs(argName, isMe);
        this.edgeBuffer.setLatestRelativeEdgeFeature({
            featureIndex:
                RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__ABILITY_TOKEN,
            isMe,
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

        const { pokemon } = this.getPokemon(poke1Ident)!;
        const isMe = isMySide(pokemon!.side.n, playerIndex);

        const fromEffect = this.getCondition(kwArgs.from);

        this.edgeBuffer.updateLatestMinorArgs(argName, isMe);
        this.edgeBuffer.updateLatestEdgeFromOf(fromEffect, isMe);
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

    "|-transform|"(args: Args["|-transform|"], kwArgs: KWArgs["|-transform|"]) {
        const [argName, poke1Ident, poke2Ident] = args;
        const { pokemon } = this.getPokemon(poke1Ident)!;

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }
        const isMe = isMySide(pokemon!.side.n, playerIndex);

        this.edgeBuffer.updateLatestMinorArgs(argName, isMe);
        this.rememberTransformed(poke1Ident, poke2Ident);
    }

    "|-mega|"() {}

    "|-primal|"() {}

    "|-burst|"() {}

    "|-zpower|"() {}

    "|-zbroken|"() {}

    "|-activate|"(args: Args["|-activate|"], kwArgs: KWArgs["|-activate|"]) {
        const [argName, poke1Ident, conditionId1, conditionId2] = args;

        if (poke1Ident) {
            const playerIndex = this.player.getPlayerIndex();
            if (playerIndex === undefined) {
                throw new Error();
            }

            const { pokemon } = this.getPokemon(poke1Ident)!;
            const isMe = isMySide(pokemon!.side.n, playerIndex);

            this.edgeBuffer.updateLatestMinorArgs(argName, isMe);
            for (const fromEffect of [
                this.getCondition(kwArgs.from),
                this.getCondition(conditionId1),
                // this.getCondition(conditionId2),
            ]) {
                this.edgeBuffer.updateLatestEdgeFromOf(fromEffect, isMe);
            }
        }
    }

    "|-mustrecharge|"(args: Args["|-mustrecharge|"]) {
        const [argName, poke1Ident] = args;

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const { pokemon } = this.getPokemon(poke1Ident)!;
        const isMe = isMySide(pokemon!.side.n, playerIndex);

        this.edgeBuffer.updateLatestMinorArgs(argName, isMe);
    }

    "|-prepare|"(args: Args["|-prepare|"]) {
        const [argName, poke1Ident] = args;

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const { pokemon } = this.getPokemon(poke1Ident)!;
        const isMe = isMySide(pokemon!.side.n, playerIndex);

        this.edgeBuffer.updateLatestMinorArgs(argName, isMe);
    }

    "|-hitcount|"(args: Args["|-hitcount|"]) {
        const [argName, poke1Ident] = args;

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const { pokemon } = this.getPokemon(poke1Ident)!;
        const isMe = isMySide(pokemon!.side.n, playerIndex);

        this.edgeBuffer.updateLatestMinorArgs(argName, isMe);
    }

    "|done|"(args: Args["|done|"]) {
        const [argName] = args;

        const edge = new Edge(this.player);
        for (const isMe of [0, 1]) edge.addMajorArg(argName, isMe);
        if (this.turnOrder > 0) {
            this.addEdge(edge);
        }
    }

    "|start|"() {
        this.turnOrder = 0;
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

    reset() {
        this.prevHp = new Map();
        this.actives = new Map();
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

        const actionBuffer = new Int16Array(numMovesetFields);
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
            bufferOffset += numMoveFields;
        }
        bufferOffset += (4 - activeMoves.length) * numMoveFields;
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
            bufferOffset += numMoveFields;
        }
        bufferOffset += (6 - switches.length) * numMoveFields;

        return new Uint8Array(actionBuffer.buffer);
    }

    getTeamFromSide(
        side: Side,
        playerIndex: number,
        isPublic: boolean = true,
    ): Int16Array {
        const buffer = new Int16Array(6 * numPokemonFeatures);

        let offset = 0;
        const team = side.team.slice(0, 6);

        for (const member of team) {
            buffer.set(
                getArrayFromPokemon(member, playerIndex, isPublic),
                offset,
            );
            offset += numPokemonFeatures;
        }

        const unkPoke = isMySide(side.n, playerIndex)
            ? unkPokemon1
            : unkPokemon0;
        for (
            let memberIndex = team.length;
            memberIndex < side.totalPokemon;
            memberIndex++
        ) {
            buffer.set(unkPoke, offset);
            offset += numPokemonFeatures;
        }

        for (
            let memberIndex = side.totalPokemon;
            memberIndex < 6;
            memberIndex++
        ) {
            buffer.set(nullPokemon, offset);
            offset += numPokemonFeatures;
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
            for (let i = this.player.log.length - 1; i >= 0; i--) {
                const line = this.player.log.at(i) ?? "";
                // eslint-disable-next-line @typescript-eslint/no-unused-vars
                const [_, cmd, winner] = line.split("|");
                if (cmd === "win") {
                    return this.player.userName === winner ? 1 : -1;
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

        const infoBuffer = new Int16Array(numInfoFields);

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
        const numEntites = buffer.length / numPokemonFeatures;
        for (let entityIndex = 0; entityIndex < numEntites; entityIndex++) {
            const start = entityIndex * numPokemonFeatures;
            const end = (entityIndex + 1) * numPokemonFeatures;
            entityDatums.push(entityArrayToObject(buffer.slice(start, end)));
        }
        return entityDatums;
    }

    getCurrentContext() {
        const currentContextBuffer = new Int16Array(numContextFields);
        const playerIndex = this.player.getPlayerIndex()!;
        for (const side of this.player.privateBattle.sides) {
            const mySide = isMySide(side.n, playerIndex) === 1;
            const sideOffset = mySide
                ? ContextFeature.CONTEXT_FEATURE__MY_SIDECONDITIONS0
                : ContextFeature.CONTEXT_FEATURE__OPP_SIDECONDITIONS0;
            const spikesOffset = mySide
                ? ContextFeature.CONTEXT_FEATURE__MY_SPIKES
                : ContextFeature.CONTEXT_FEATURE__OPP_SPIKES;
            const toxisSpikesOffset = mySide
                ? ContextFeature.CONTEXT_FEATURE__MY_TOXIC_SPIKES
                : ContextFeature.CONTEXT_FEATURE__OPP_TOXIC_SPIKES;

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

        currentContextBuffer[
            AbsoluteEdgeFeature.ABSOLUTE_EDGE_FEATURE__WEATHER_ID
        ] = weatherIndex;
        currentContextBuffer[
            AbsoluteEdgeFeature.ABSOLUTE_EDGE_FEATURE__WEATHER_MAX_DURATION
        ] = field.weatherState.maxDuration;
        currentContextBuffer[
            AbsoluteEdgeFeature.ABSOLUTE_EDGE_FEATURE__WEATHER_MIN_DURATION
        ] = field.weatherState.minDuration;
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
            historyEntities,
            historyRelativeEdges,
            historyAbsoluteEdge,
            historyLength,
        } = this.getHistory(NUM_HISTORY);
        state.setHistoryEntities(historyEntities);
        state.setHistoryRelativeEdges(historyRelativeEdges);
        state.setHistoryAbsoluteEdge(historyAbsoluteEdge);
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
