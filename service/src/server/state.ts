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
import { Heuristics, Info, Rewards, State } from "../../protos/state_pb";
import {
    AbilitiesEnum,
    ActionsEnum,
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
} from "./data";
import { NA, Pokemon, Side } from "@pkmn/client";
import { Ability, Item, Move, BoostID } from "@pkmn/dex-types";
import { ID, MoveTarget } from "@pkmn/types";
import { Condition, Effect } from "@pkmn/data";
import { History } from "../../protos/history_pb";
import { OneDBoolean, TypedArray } from "./utils";
import { Player } from "./player";
import { DRAW_TURNS } from "./game";
import { GetHeuristicAction } from "./baselines/heuristic";
import {
    AbsoluteEdgeFeature,
    AbsoluteEdgeFeatureMap,
    EntityFeature,
    MovesetActionTypeEnum,
    MovesetFeature,
    RelativeEdgeFeature,
    RelativeEdgeFeatureMap,
} from "../../protos/features_pb";

type RemovePipes<T extends string> = T extends `|${infer U}|` ? U : T;
type MajorArgNames =
    | RemovePipes<BattleMajorArgName>
    | RemovePipes<BattleProgressArgName>;
type MinorArgNames = RemovePipes<BattleMinorArgName>;

const MAX_16BIT_SIGNED = 32767;

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

function hashArrayToInt32(numbers: number[]): number {
    if (numbers.length !== 4)
        throw new Error("Array must contain exactly 4 numbers.");

    let hash = 0;

    // Process each number
    for (const num of numbers) {
        hash = (hash * 31 + num) | 0; // Multiply by a prime number and add the value
    }

    return hash; // Resulting 32-bit integer
}
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

function getUnkPokemon(n?: number) {
    const data = getBlankPokemonArr();
    data[EntityFeature.ENTITY_FEATURE__SPECIES] =
        SpeciesEnum.SPECIES_ENUM___UNK;
    data[EntityFeature.ENTITY_FEATURE__ITEM] = ItemsEnum.ITEMS_ENUM___UNK;
    data[EntityFeature.ENTITY_FEATURE__ITEM_EFFECT] =
        ItemeffecttypesEnum.ITEMEFFECTTYPES_ENUM___NULL;
    data[EntityFeature.ENTITY_FEATURE__ABILITY] =
        AbilitiesEnum.ABILITIES_ENUM___UNK;
    data[EntityFeature.ENTITY_FEATURE__FAINTED] = 0;
    data[EntityFeature.ENTITY_FEATURE__HP] = 100;
    data[EntityFeature.ENTITY_FEATURE__MAXHP] = 100;
    data[EntityFeature.ENTITY_FEATURE__STATUS] = StatusEnum.STATUS_ENUM___NULL;
    data[EntityFeature.ENTITY_FEATURE__MOVEID0] = MovesEnum.MOVES_ENUM___UNK;
    data[EntityFeature.ENTITY_FEATURE__MOVEID1] = MovesEnum.MOVES_ENUM___UNK;
    data[EntityFeature.ENTITY_FEATURE__MOVEID2] = MovesEnum.MOVES_ENUM___UNK;
    data[EntityFeature.ENTITY_FEATURE__MOVEID3] = MovesEnum.MOVES_ENUM___UNK;
    data[EntityFeature.ENTITY_FEATURE__HAS_STATUS] = 0;
    data[EntityFeature.ENTITY_FEATURE__SIDE] = n ?? 0;
    data[EntityFeature.ENTITY_FEATURE__HP] = 31;
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

function getArrayFromPokemon(candidate: Pokemon | null, playerIndex: number) {
    if (candidate === null) {
        return getNullPokemon();
    }

    let pokemon: Pokemon;
    if (candidate.volatiles.transform !== undefined) {
        pokemon = candidate.volatiles.transform.pokemon as Pokemon;
    } else {
        pokemon = candidate;
    }

    const baseSpecies = pokemon.species.baseSpecies.toLowerCase();
    const item = pokemon.item ?? pokemon.lastItem;
    const itemEffect = pokemon.itemEffect ?? pokemon.lastItemEffect;
    const ability = pokemon.ability;

    const moveSlots = pokemon.moveSlots.slice(0, 4);
    const moveIds = [];
    const movePps = [];
    if (moveSlots) {
        for (const move of moveSlots) {
            const { id, ppUsed } = move;
            const maxPP = pokemon.side.battle.gens.dex.moves.get(id).pp;
            const idValue = IndexValueFromEnum(ActionsEnum, `move_${id}`);
            const correctUsed = ((isNaN(ppUsed) ? +!!ppUsed : ppUsed) * 5) / 8;

            moveIds.push(idValue);
            movePps.push(Math.floor((31 * correctUsed) / maxPP));
        }
    }
    let remainingIndex: MoveIndex = moveSlots.length as MoveIndex;
    for (remainingIndex; remainingIndex < 4; remainingIndex++) {
        moveIds.push(ActionsEnum.ACTIONS_ENUM___UNK);
        movePps.push(0);
    }

    const dataArr = getBlankPokemonArr();
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

    dataArr[EntityFeature.ENTITY_FEATURE__GENDER] = IndexValueFromEnum(
        GendernameEnum,
        pokemon.gender,
    );
    dataArr[EntityFeature.ENTITY_FEATURE__ACTIVE] = pokemon.isActive() ? 1 : 0;
    dataArr[EntityFeature.ENTITY_FEATURE__FAINTED] = pokemon.fainted ? 1 : 0;
    dataArr[EntityFeature.ENTITY_FEATURE__HP] = pokemon.hp;
    dataArr[EntityFeature.ENTITY_FEATURE__MAXHP] = pokemon.maxhp;
    dataArr[EntityFeature.ENTITY_FEATURE__HP_RATIO] = Math.floor(
        (31 * pokemon.hp) / pokemon.maxhp,
    );
    dataArr[EntityFeature.ENTITY_FEATURE__STATUS] = pokemon.status
        ? IndexValueFromEnum(StatusEnum, pokemon.status)
        : StatusEnum.STATUS_ENUM___NULL;
    dataArr[EntityFeature.ENTITY_FEATURE__HAS_STATUS] = pokemon.status ? 1 : 0;
    dataArr[EntityFeature.ENTITY_FEATURE__TOXIC_TURNS] =
        pokemon.statusState.toxicTurns;
    dataArr[EntityFeature.ENTITY_FEATURE__SLEEP_TURNS] =
        pokemon.statusState.sleepTurns;
    dataArr[EntityFeature.ENTITY_FEATURE__BEING_CALLED_BACK] =
        pokemon.beingCalledBack ? 1 : 0;
    dataArr[EntityFeature.ENTITY_FEATURE__TRAPPED] = pokemon.trapped ? 1 : 0;
    dataArr[EntityFeature.ENTITY_FEATURE__NEWLY_SWITCHED] =
        pokemon.newlySwitched ? 1 : 0;
    dataArr[EntityFeature.ENTITY_FEATURE__LEVEL] = pokemon.level;
    for (let moveIndex = 0; moveIndex < 4; moveIndex++) {
        dataArr[
            EntityFeature[`ENTITY_FEATURE__MOVEID${moveIndex as MoveIndex}`]
        ] = moveIds[moveIndex];
        dataArr[
            EntityFeature[`ENTITY_FEATURE__MOVEPP${moveIndex as MoveIndex}`]
        ] = movePps[moveIndex];
    }

    let volatiles = BigInt(0b0);
    for (const [key] of Object.entries({
        ...pokemon.volatiles,
        ...candidate.volatiles,
    })) {
        const index = IndexValueFromEnum(VolatilestatusEnum, key);
        volatiles |= BigInt(1) << BigInt(index);
    }
    dataArr.set(
        bigIntToInt16Array(volatiles),
        EntityFeature.ENTITY_FEATURE__VOLATILES0,
    );

    dataArr[EntityFeature.ENTITY_FEATURE__SIDE] = pokemon.side.n ^ playerIndex;

    dataArr[EntityFeature.ENTITY_FEATURE__BOOST_ATK_VALUE] =
        (pokemon.boosts.atk ?? 0) + 6;
    dataArr[EntityFeature.ENTITY_FEATURE__BOOST_DEF_VALUE] =
        (pokemon.boosts.def ?? 0) + 6;
    dataArr[EntityFeature.ENTITY_FEATURE__BOOST_SPA_VALUE] =
        (pokemon.boosts.spa ?? 0) + 6;
    dataArr[EntityFeature.ENTITY_FEATURE__BOOST_SPD_VALUE] =
        (pokemon.boosts.spd ?? 0) + 6;
    dataArr[EntityFeature.ENTITY_FEATURE__BOOST_SPE_VALUE] =
        (pokemon.boosts.spe ?? 0) + 6;
    dataArr[EntityFeature.ENTITY_FEATURE__BOOST_EVASION_VALUE] =
        (pokemon.boosts.evasion ?? 0) + 6;
    dataArr[EntityFeature.ENTITY_FEATURE__BOOST_ACCURACY_VALUE] =
        (pokemon.boosts.accuracy ?? 0) + 6;

    let typeChanged = BigInt(0b0);
    const typechangeVolatile =
        pokemon.volatiles.typechange ?? candidate.volatiles.typechange;
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
    player: Player;

    entityData: Int16Array;
    relativeEdgeData: Int16Array;
    absoluteEdgeData: Int16Array;

    constructor(player: Player) {
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

    setPoke(poke: Pokemon | null) {
        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }
        if (poke !== null) {
            const data = getArrayFromPokemon(poke, playerIndex);
            const isMySide = (poke.side.n ^ playerIndex) === 0;
            const offset = isMySide ? 0 : numPokemonFeatures;
            this.setPokeFromData(data, offset);
        }
    }

    updateSideData() {
        const playerIndex = this.player.getPlayerIndex()!;

        for (const side of this.player.publicBattle.sides) {
            const isMySide = (side.n ^ playerIndex) === 0;
            const entityOffset = isMySide ? 0 : numPokemonFeatures;
            const sideConditionOffset =
                (isMySide ? 0 : numRelativeEdgeFeatures) +
                RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__SIDECONDITIONS0;
            this.updateEntityData(side.active, entityOffset, playerIndex);
            this.updateSideConditionData(
                side,
                sideConditionOffset,
                playerIndex,
            );
        }
    }

    updateEntityData(
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
            this.setRelativeEdgeFeature(
                RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__SPIKES,
                side.n ^ playerIndex,
                side.sideConditions.spikes.level,
            );
        }
        if (side.sideConditions.toxicspikes) {
            this.setRelativeEdgeFeature(
                RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__TOXIC_SPIKES,
                side.n ^ playerIndex,
                side.sideConditions.toxicspikes.level,
            );
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

    setRelativeEdgeFeature(
        featureIndex: RelativeEdgeFeatureMap[keyof RelativeEdgeFeatureMap],
        sideIndex: number,
        value: number,
    ) {
        if (featureIndex === undefined) {
            throw new Error("Index cannot be undefined");
        }
        if (value === undefined) {
            throw new Error("Value cannot be undefined");
        }
        const index =
            featureIndex + (sideIndex === 0 ? 0 : numRelativeEdgeFeatures);
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
        sideIndex: number,
    ) {
        const index =
            featureIndex + (sideIndex === 0 ? 0 : numRelativeEdgeFeatures);
        return this.relativeEdgeData.at(index);
    }

    addMajorArg(argName: MajorArgNames, sideIndex: number) {
        const index = IndexValueFromEnum(BattlemajorargsEnum, argName);
        this.setRelativeEdgeFeature(
            RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__MAJOR_ARG,
            sideIndex,
            index,
        );
    }

    updateEdgeFromOf(effect: Partial<Effect>, sideIndex: number) {
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
                    sideIndex,
                ) ?? 0;
            const numFromSources =
                this.getRelativeEdgeFeature(
                    RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__NUM_FROM_SOURCES,
                    sideIndex,
                ) ?? 0;
            const prevFromType = this.getRelativeEdgeFeature(
                (RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__FROM_TYPE_TOKEN0 +
                    numFromTypes) as RelativeEdgeFeatureMap[keyof RelativeEdgeFeatureMap],
                sideIndex,
            );
            const prevFromSource = this.getRelativeEdgeFeature(
                (RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__FROM_SOURCE_TOKEN0 +
                    numFromSources) as RelativeEdgeFeatureMap[keyof RelativeEdgeFeatureMap],
                sideIndex,
            );
            if (prevFromType !== fromTypeToken && numFromTypes < 5) {
                this.setRelativeEdgeFeature(
                    (RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__FROM_TYPE_TOKEN0 +
                        numFromTypes) as RelativeEdgeFeatureMap[keyof RelativeEdgeFeatureMap],
                    sideIndex,
                    fromTypeToken,
                );
                this.setRelativeEdgeFeature(
                    RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__NUM_FROM_TYPES,
                    sideIndex,
                    numFromTypes + 1,
                );
            }
            if (prevFromSource !== fromSourceToken && numFromSources < 5) {
                this.setRelativeEdgeFeature(
                    (RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__FROM_SOURCE_TOKEN0 +
                        numFromSources) as RelativeEdgeFeatureMap[keyof RelativeEdgeFeatureMap],
                    sideIndex,
                    fromSourceToken,
                );
                this.setRelativeEdgeFeature(
                    RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__NUM_FROM_SOURCES,
                    sideIndex,
                    numFromSources + 1,
                );
            }
        }
    }
}

function getEffectToken(effect: Partial<Effect>): number {
    const { id } = effect;
    if (id) {
        return IndexValueFromEnum(EffectEnum, id);
    }
    return 0;
}

class EdgeBuffer {
    player: Player;

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

    constructor(player: Player) {
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
            this.setLatestRelativeEdgeFeature(
                RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__SPIKES,
                side.n ^ playerIndex,
                side.sideConditions.spikes.level,
            );
        }
        if (side.sideConditions.toxicspikes) {
            this.setLatestRelativeEdgeFeature(
                RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__TOXIC_SPIKES,
                side.n ^ playerIndex,
                side.sideConditions.toxicspikes.level,
            );
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
            const entityOffset =
                this.prevEntityCursor +
                ((side.n ^ playerIndex) === 0 ? 0 : numPokemonFeatures);
            const sideConditionOffset =
                this.prevRelativeEdgeCursor +
                ((side.n ^ playerIndex) === 0 ? 0 : numRelativeEdgeFeatures) +
                RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__SIDECONDITIONS0;
            this.updateLatestEntityData(side.active, entityOffset, playerIndex);
            this.updateLatestSideConditionData(
                side,
                sideConditionOffset,
                playerIndex,
            );
        }
    }

    setLatestRelativeEdgeFeature(
        featureIndex: RelativeEdgeFeatureMap[keyof RelativeEdgeFeatureMap],
        sideIndex: number,
        value: number,
    ) {
        if (featureIndex === undefined) {
            throw new Error("Index cannot be undefined");
        }
        if (value === undefined) {
            throw new Error("Value cannot be undefined");
        }
        const index =
            this.prevRelativeEdgeCursor +
            featureIndex +
            (sideIndex === 0 ? 0 : numRelativeEdgeFeatures);
        this.relativeEdgeData[index] = value;
    }

    getLatestRelativeEdgeFeature(
        featureIndex: RelativeEdgeFeatureMap[keyof RelativeEdgeFeatureMap],
        sideIndex: number,
    ) {
        const index =
            this.prevRelativeEdgeCursor +
            featureIndex +
            (sideIndex === 0 ? 0 : numRelativeEdgeFeatures);
        return this.relativeEdgeData.at(index);
    }

    updateLatestMinorArgs(
        argName: MinorArgNames,
        sideIndex: number,
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
            sideIndex,
        )!;
        const newValue = currentValue | (1 << index % precision);
        this.setLatestRelativeEdgeFeature(featureIndex, sideIndex, newValue);
    }

    updateLatestMajorArg(argName: MajorArgNames, sideIndex: number) {
        const index = IndexValueFromEnum(BattlemajorargsEnum, argName);
        this.setLatestRelativeEdgeFeature(
            RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__MAJOR_ARG,
            sideIndex,
            index,
        );
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

    updateLatestEdgeFromOf(effect: Partial<Effect>, sideIndex: number) {
        const { effectType } = effect;
        if (effectType) {
            const fromTypeToken = IndexValueFromEnum(
                EffecttypesEnum,
                effectType,
            );
            const fromSourceToken = getEffectToken(effect);

            const numFromTypes =
                this.getLatestRelativeEdgeFeature(
                    RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__NUM_FROM_TYPES,
                    sideIndex,
                ) ?? 0;
            const numFromSources =
                this.getLatestRelativeEdgeFeature(
                    RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__NUM_FROM_SOURCES,
                    sideIndex,
                ) ?? 0;
            const prevFromType = this.getLatestRelativeEdgeFeature(
                (RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__FROM_TYPE_TOKEN0 +
                    Math.max(
                        0,
                        numFromTypes - 1,
                    )) as RelativeEdgeFeatureMap[keyof RelativeEdgeFeatureMap],
                sideIndex,
            );
            const prevFromSource = this.getLatestRelativeEdgeFeature(
                (RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__FROM_SOURCE_TOKEN0 +
                    Math.max(
                        0,
                        numFromSources - 1,
                    )) as RelativeEdgeFeatureMap[keyof RelativeEdgeFeatureMap],
                sideIndex,
            );
            if (prevFromType !== fromTypeToken && numFromTypes < 5) {
                this.setLatestRelativeEdgeFeature(
                    (RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__FROM_TYPE_TOKEN0 +
                        numFromTypes) as RelativeEdgeFeatureMap[keyof RelativeEdgeFeatureMap],
                    sideIndex,
                    fromTypeToken,
                );
                this.setLatestRelativeEdgeFeature(
                    RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__NUM_FROM_TYPES,
                    sideIndex,
                    numFromTypes + 1,
                );
            }
            if (prevFromSource !== fromSourceToken && numFromSources < 5) {
                this.setLatestRelativeEdgeFeature(
                    (RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__FROM_SOURCE_TOKEN0 +
                        numFromSources) as RelativeEdgeFeatureMap[keyof RelativeEdgeFeatureMap],
                    sideIndex,
                    fromSourceToken,
                );
                this.setLatestRelativeEdgeFeature(
                    RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__NUM_FROM_SOURCES,
                    sideIndex,
                    numFromSources + 1,
                );
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
        const history = new History();
        const width = Math.max(1, Math.min(this.numEdges, numHistory));
        history.setEntities(
            new Uint8Array(
                this.entityData.slice(
                    this.entityCursor - width * 2 * numPokemonFeatures,
                    this.entityCursor,
                ).buffer,
            ),
        );
        history.setRelativeEdges(
            new Uint8Array(
                this.relativeEdgeData.slice(
                    this.relativeEdgeCursor -
                        width * 2 * numRelativeEdgeFeatures,
                    this.relativeEdgeCursor,
                ).buffer,
            ),
        );
        history.setAbsoluteEdge(
            new Uint8Array(
                this.absoluteEdgeData.slice(
                    this.absoluteEdgeCursor - width * numAbsoluteEdgeFeatures,
                    this.absoluteEdgeCursor,
                ).buffer,
            ),
        );
        history.setLength(width);
        return history;
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
                    jsonDatum["species"][
                        array[EntityFeature.ENTITY_FEATURE__SPECIES]
                    ],
                item: jsonDatum["items"][
                    array[EntityFeature.ENTITY_FEATURE__ITEM]
                ],
                ability:
                    jsonDatum["abilities"][
                        array[EntityFeature.ENTITY_FEATURE__ABILITY]
                    ],
                moves: moveIndicies.map((index) => jsonDatum["Actions"][index]),
                volatiles: volatilesIndices.map(
                    (index) => jsonDatum["volatileStatus"][index],
                ),
                typechange: typechangeIndices.map(
                    (index) => jsonDatum["typechart"][index],
                ),
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
            const sideConditionIndices =
                int16ArrayToBitIndices(sideConditionsFlat);

            return {
                majorArg:
                    jsonDatum["battleMajorArgs"][
                        array[
                            RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__MAJOR_ARG
                        ]
                    ],
                minorArgs: minorArgIndices.map(
                    (index) => jsonDatum["battleMinorArgs"][index],
                ),
                action: jsonDatum["Actions"][
                    array[
                        RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__ACTION_TOKEN
                    ]
                ],
                damage:
                    array[
                        RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__DAMAGE_RATIO
                    ] / MAX_16BIT_SIGNED,
                heal:
                    array[
                        RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__HEAL_RATIO
                    ] / MAX_16BIT_SIGNED,
                sideConditions: sideConditionIndices.map(
                    (index) => jsonDatum["sideCondition"][index],
                ),
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
                            RelativeEdgeFeature
                                .RELATIVE_EDGE_FEATURE__BOOST_ATK_VALUE
                        ],
                    EDGE_BOOST_DEF_VALUE:
                        array[
                            RelativeEdgeFeature
                                .RELATIVE_EDGE_FEATURE__BOOST_DEF_VALUE
                        ],
                    EDGE_BOOST_SPA_VALUE:
                        array[
                            RelativeEdgeFeature
                                .RELATIVE_EDGE_FEATURE__BOOST_SPA_VALUE
                        ],
                    EDGE_BOOST_SPD_VALUE:
                        array[
                            RelativeEdgeFeature
                                .RELATIVE_EDGE_FEATURE__BOOST_SPD_VALUE
                        ],
                    EDGE_BOOST_SPE_VALUE:
                        array[
                            RelativeEdgeFeature
                                .RELATIVE_EDGE_FEATURE__BOOST_SPE_VALUE
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
            };
        };

        const absoluteArrayToObject = (array: Int16Array) => {
            return {
                turnOrder:
                    array[
                        AbsoluteEdgeFeature
                            .ABSOLUTE_EDGE_FEATURE__TURN_ORDER_VALUE
                    ],
                requestCount:
                    array[
                        AbsoluteEdgeFeature.ABSOLUTE_EDGE_FEATURE__REQUEST_COUNT
                    ],
                weatherId:
                    jsonDatum["weather"][
                        array[
                            AbsoluteEdgeFeature
                                .ABSOLUTE_EDGE_FEATURE__WEATHER_ID
                        ]
                    ],
            };
        };

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
                entites: [
                    entityArrayToObject(stepEntity0),
                    entityArrayToObject(stepEntity1),
                ],
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
    readonly player: Player;

    prevHp: Map<string, number>;
    actives: Map<ID, PokemonIdent>;
    turnOrder: number;
    turnNum: number;

    edgeBuffer: EdgeBuffer;

    constructor(player: Player) {
        this.player = player;
        this.prevHp = new Map();
        this.actives = new Map();

        this.edgeBuffer = new EdgeBuffer(player);
        this.turnOrder = 0;
        this.turnNum = 0;
    }

    getPokemon(pokemonid: PokemonIdent) {
        if (
            !pokemonid ||
            pokemonid === "??" ||
            pokemonid === "null" ||
            pokemonid === "false"
        ) {
            return null;
        }
        const { siden, pokemonid: parsedPokemonid } =
            this.player.publicBattle.parsePokemonId(pokemonid);

        const side = this.player.publicBattle.sides[siden];

        for (const pokemon of side.team) {
            if (pokemon.originalIdent === parsedPokemonid) {
                return pokemon;
            }
        }

        return null;
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

    "|move|"(args: Args["|move|"]) {
        const [argName, poke1Ident, moveId] = args;

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const poke = this.getPokemon(poke1Ident)!;
        const relativePlayerIndex = poke.side.n ^ playerIndex;

        const move = this.getMove(moveId);
        const actionIndex = IndexValueFromEnum(ActionsEnum, `move_${move.id}`);

        const edge = new Edge(this.player);
        edge.addMajorArg(argName, relativePlayerIndex);
        edge.setRelativeEdgeFeature(
            RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__ACTION_TOKEN,
            relativePlayerIndex,
            actionIndex,
        );
        this.addEdge(edge);
    }

    "|drag|"(args: Args["|drag|"]) {
        this.handleSwitch(args);
    }

    "|switch|"(args: Args["|switch|"]) {
        this.handleSwitch(args);
    }

    handleSwitch(args: Args["|switch|" | "|drag|"]) {
        const [argName, poke1Ident] = args;

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const poke = this.getPokemon(poke1Ident)!;
        const relativePlayerIndex = poke.side.n ^ playerIndex;

        const actionIndex = IndexValueFromEnum(
            ActionsEnum,
            `switch_${poke.species.baseSpecies.toLowerCase()}`,
        );

        const edge = new Edge(this.player);
        edge.addMajorArg(argName, relativePlayerIndex);
        edge.setRelativeEdgeFeature(
            RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__ACTION_TOKEN,
            relativePlayerIndex,
            actionIndex,
        );
        this.addEdge(edge);
    }

    "|cant|"(args: Args["|cant|"]) {
        const [argName, poke1Ident, conditionId, moveId] = args;

        const edge = new Edge(this.player);

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const poke = this.getPokemon(poke1Ident)!;
        const relativePlayerIndex = poke.side.n ^ playerIndex;

        if (moveId) {
            const move = this.getMove(moveId);
            const actionIndex = IndexValueFromEnum(
                ActionsEnum,
                `move_${move.id}`,
            );
            edge.setRelativeEdgeFeature(
                RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__ACTION_TOKEN,
                relativePlayerIndex,
                actionIndex,
            );
        }

        const condition = this.getCondition(conditionId);

        edge.addMajorArg(argName, relativePlayerIndex);
        edge.updateEdgeFromOf(condition, relativePlayerIndex);

        this.addEdge(edge);
    }

    "|faint|"(args: Args["|faint|"]) {
        const [argName, poke1Ident] = args;

        const edge = new Edge(this.player);

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const poke = this.getPokemon(poke1Ident)!;
        const relativePlayerIndex = poke.side.n ^ playerIndex;

        edge.addMajorArg(argName, relativePlayerIndex);
        this.addEdge(edge);
    }

    "|-fail|"(args: Args["|-fail|"], kwArgs: KWArgs["|-fail|"]) {
        const [argName, poke1Ident] = args;

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const poke = this.getPokemon(poke1Ident)!;
        const relativePlayerIndex = poke.side.n ^ playerIndex;

        const fromEffect = this.getCondition(kwArgs.from);

        this.edgeBuffer.updateLatestMinorArgs(argName, relativePlayerIndex);
        this.edgeBuffer.updateLatestEdgeFromOf(fromEffect, relativePlayerIndex);
    }

    "|-block|"(args: Args["|-block|"], kwArgs: KWArgs["|-block|"]) {
        const [argName, poke1Ident] = args;

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const poke = this.getPokemon(poke1Ident)!;
        const relativePlayerIndex = poke.side.n ^ playerIndex;

        const fromEffect = this.getCondition(kwArgs.from);

        this.edgeBuffer.updateLatestMinorArgs(argName, relativePlayerIndex);
        this.edgeBuffer.updateLatestEdgeFromOf(fromEffect, relativePlayerIndex);
    }

    "|-notarget|"(args: Args["|-notarget|"]) {
        const [argName, poke1Ident] = args;

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        if (poke1Ident !== undefined) {
            const poke = this.getPokemon(poke1Ident)!;
            const relativePlayerIndex = poke.side.n ^ playerIndex;
            this.edgeBuffer.updateLatestMinorArgs(argName, relativePlayerIndex);
        } else {
            for (const relativePlayerIndex of [0, 1]) {
                this.edgeBuffer.updateLatestMinorArgs(
                    argName,
                    relativePlayerIndex,
                );
            }
        }
    }

    "|-miss|"(args: Args["|-miss|"], kwArgs: KWArgs["|-miss|"]) {
        const [argName, poke1Ident, poke2Ident] = args;

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const poke = this.getPokemon(poke1Ident ?? poke2Ident)!;
        const relativePlayerIndex = poke.side.n ^ playerIndex;

        const fromEffect = this.getCondition(kwArgs.from);

        this.edgeBuffer.updateLatestMinorArgs(argName, relativePlayerIndex);
        this.edgeBuffer.updateLatestEdgeFromOf(fromEffect, relativePlayerIndex);
    }

    "|-damage|"(args: Args["|-damage|"], kwArgs: KWArgs["|-damage|"]) {
        const [argName, poke1Ident] = args;

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const poke = this.getPokemon(poke1Ident)!;
        const relativePlayerIndex = poke.side.n ^ playerIndex;
        const trueIdent = poke.originalIdent;

        if (kwArgs.from) {
            const fromEffect = this.getCondition(kwArgs.from);
            this.edgeBuffer.updateLatestEdgeFromOf(
                fromEffect,
                relativePlayerIndex,
            );
        }

        if (!this.prevHp.has(trueIdent)) {
            this.prevHp.set(trueIdent, 1);
        }
        const prevHp = this.prevHp.get(trueIdent) ?? 1;
        const currHp = poke.hp / poke.maxhp;
        const diffRatio = currHp - prevHp;
        this.prevHp.set(trueIdent, currHp);
        const addedDamageToken = Math.abs(
            Math.floor(MAX_16BIT_SIGNED * diffRatio),
        );

        this.edgeBuffer.updateLatestMinorArgs(argName, relativePlayerIndex);
        const currentDamageToken =
            this.edgeBuffer.getLatestRelativeEdgeFeature(
                RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__DAMAGE_RATIO,
                relativePlayerIndex,
            ) ?? 0;
        this.edgeBuffer.setLatestRelativeEdgeFeature(
            RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__DAMAGE_RATIO,
            relativePlayerIndex,
            Math.min(MAX_16BIT_SIGNED, currentDamageToken + addedDamageToken),
        );
    }

    "|-heal|"(args: Args["|-heal|"], kwArgs: KWArgs["|-heal|"]) {
        const [argName, poke1Ident] = args;

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const poke = this.getPokemon(poke1Ident)!;
        const relativePlayerIndex = poke.side.n ^ playerIndex;
        const trueIdent = poke.originalIdent;

        if (kwArgs.from) {
            const fromEffect = this.getCondition(kwArgs.from);
            this.edgeBuffer.updateLatestEdgeFromOf(
                fromEffect,
                relativePlayerIndex,
            );
        }

        if (!this.prevHp.has(trueIdent)) {
            this.prevHp.set(trueIdent, 1);
        }
        const prevHp = this.prevHp.get(trueIdent) ?? 1;
        const currHp = poke.hp / poke.maxhp;
        const diffRatio = currHp - prevHp;
        this.prevHp.set(trueIdent, currHp);
        const addedHealToken = Math.abs(
            Math.floor(MAX_16BIT_SIGNED * diffRatio),
        );

        this.edgeBuffer.updateLatestMinorArgs(argName, relativePlayerIndex);
        const currentHealToken =
            this.edgeBuffer.getLatestRelativeEdgeFeature(
                RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__HEAL_RATIO,
                relativePlayerIndex,
            ) ?? 0;
        this.edgeBuffer.setLatestRelativeEdgeFeature(
            RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__HEAL_RATIO,
            relativePlayerIndex,
            Math.min(MAX_16BIT_SIGNED, currentHealToken + addedHealToken),
        );
    }

    "|-status|"(args: Args["|-status|"], kwArgs: KWArgs["|-status|"]) {
        const [argName, poke1Ident, statusId] = args;

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const poke = this.getPokemon(poke1Ident)!;
        const relativePlayerIndex = poke.side.n ^ playerIndex;

        const fromEffect = this.getCondition(kwArgs.from);
        const statusToken = IndexValueFromEnum(StatusEnum, statusId);

        this.edgeBuffer.updateLatestMinorArgs(argName, relativePlayerIndex);
        this.edgeBuffer.updateLatestEdgeFromOf(fromEffect, relativePlayerIndex);
        this.edgeBuffer.setLatestRelativeEdgeFeature(
            RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__STATUS_TOKEN,
            relativePlayerIndex,
            statusToken,
        );
    }

    "|-curestatus|"(args: Args["|-curestatus|"]) {
        const [argName, poke1Ident, statusId] = args;

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const poke = this.getPokemon(poke1Ident)!;
        const relativePlayerIndex = poke.side.n ^ playerIndex;

        const statusToken = IndexValueFromEnum(StatusEnum, statusId);

        this.edgeBuffer.updateLatestMinorArgs(argName, relativePlayerIndex);
        this.edgeBuffer.setLatestRelativeEdgeFeature(
            RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__STATUS_TOKEN,
            relativePlayerIndex,
            statusToken,
        );
    }

    "|-cureteam|"(args: Args["|-cureteam|"]) {
        const [argName, poke1Ident] = args;

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const poke = this.getPokemon(poke1Ident)!;
        const relativePlayerIndex = poke.side.n ^ playerIndex;

        this.edgeBuffer.updateLatestMinorArgs(argName, relativePlayerIndex);
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

        const poke = this.getPokemon(poke1Ident)!;
        const relativePlayerIndex = poke.side.n ^ playerIndex;

        this.edgeBuffer.updateLatestMinorArgs(argName, relativePlayerIndex);
        this.edgeBuffer.setLatestRelativeEdgeFeature(
            featureIndex,
            relativePlayerIndex,
            parseInt(value),
        );
    }

    "|-unboost|"(args: Args["|-unboost|"]) {
        const [argName, poke1Ident, stat, value] = args;

        const featureIndex = EventHandler.getStatBoostEdgeFeatureIndex(stat);

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const poke = this.getPokemon(poke1Ident)!;
        const relativePlayerIndex = poke.side.n ^ playerIndex;

        this.edgeBuffer.updateLatestMinorArgs(argName, relativePlayerIndex);
        this.edgeBuffer.setLatestRelativeEdgeFeature(
            featureIndex,
            relativePlayerIndex,
            -parseInt(value),
        );
    }

    "|-setboost|"(args: Args["|-setboost|"], kwArgs: KWArgs["|-setboost|"]) {
        const [argName, poke1Ident, stat, value] = args;

        const featureIndex = EventHandler.getStatBoostEdgeFeatureIndex(stat);
        const fromEffect = this.getCondition(kwArgs.from);

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const poke = this.getPokemon(poke1Ident)!;
        const relativePlayerIndex = poke.side.n ^ playerIndex;

        this.edgeBuffer.updateLatestMinorArgs(argName, relativePlayerIndex);
        this.edgeBuffer.setLatestRelativeEdgeFeature(
            featureIndex,
            relativePlayerIndex,
            parseInt(value),
        );
        this.edgeBuffer.updateLatestEdgeFromOf(fromEffect, relativePlayerIndex);
    }

    "|-swapboost|"(args: Args["|-swapboost|"], kwArgs: KWArgs["|-swapboost|"]) {
        const [argName, poke1Ident] = args;

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const poke = this.getPokemon(poke1Ident)!;
        const relativePlayerIndex = poke.side.n ^ playerIndex;

        const fromEffect = this.getCondition(kwArgs.from);

        this.edgeBuffer.updateLatestMinorArgs(argName, relativePlayerIndex);
        this.edgeBuffer.updateLatestEdgeFromOf(fromEffect, relativePlayerIndex);
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

        const poke = this.getPokemon(poke1Ident)!;
        const relativePlayerIndex = poke.side.n ^ playerIndex;

        const fromEffect = this.getCondition(kwArgs.from);

        this.edgeBuffer.updateLatestMinorArgs(argName, relativePlayerIndex);
        this.edgeBuffer.updateLatestEdgeFromOf(fromEffect, relativePlayerIndex);
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

        const poke = this.getPokemon(poke1Ident)!;
        const relativePlayerIndex = poke.side.n ^ playerIndex;

        const fromEffect = this.getCondition(kwArgs.from);

        this.edgeBuffer.updateLatestMinorArgs(argName, relativePlayerIndex);
        this.edgeBuffer.updateLatestEdgeFromOf(fromEffect, relativePlayerIndex);
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

        const poke = this.getPokemon(poke1Ident)!;
        const relativePlayerIndex = poke.side.n ^ playerIndex;

        const fromEffect = this.getCondition(kwArgs.from);

        this.edgeBuffer.updateLatestMinorArgs(argName, relativePlayerIndex);
        this.edgeBuffer.updateLatestEdgeFromOf(fromEffect, relativePlayerIndex);
    }

    "|-copyboost|"() {}

    "|-weather|"(args: Args["|-weather|"]) {
        const [argName, weatherId] = args;

        const fromEffect = this.getCondition(weatherId);

        const weatherIndex =
            weatherId === "none"
                ? WeatherEnum.WEATHER_ENUM___NULL
                : IndexValueFromEnum(WeatherEnum, weatherId);
        for (const relativePlayerIndex of [0, 1]) {
            this.edgeBuffer.updateLatestMinorArgs(argName, relativePlayerIndex);
            this.edgeBuffer.updateLatestEdgeFromOf(
                fromEffect,
                relativePlayerIndex,
            );
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
        const relativePlayerIndex = side.n ^ playerIndex;

        const fromEffect = this.getCondition(conditionId);

        this.edgeBuffer.updateLatestMinorArgs(argName, relativePlayerIndex);
        this.edgeBuffer.updateLatestEdgeFromOf(fromEffect, relativePlayerIndex);
    }

    "|-sideend|"(args: Args["|-sideend|"]) {
        const [argName, sideId, conditionId] = args;

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const side = this.getSide(sideId);
        const relativePlayerIndex = side.n ^ playerIndex;

        const fromEffect = this.getCondition(conditionId);

        this.edgeBuffer.updateLatestMinorArgs(argName, relativePlayerIndex);
        this.edgeBuffer.updateLatestEdgeFromOf(fromEffect, relativePlayerIndex);
    }

    "|-swapsideconditions|"() {}

    "|-start|"(args: Args["|-start|"]) {
        const [argName, poke1Ident] = args;

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const poke = this.getPokemon(poke1Ident)!;
        const relativePlayerIndex = poke.side.n ^ playerIndex;

        this.edgeBuffer.updateLatestMinorArgs(argName, relativePlayerIndex);
    }

    "|-end|"(args: Args["|-end|"]) {
        const [argName, poke1Ident] = args;

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const poke = this.getPokemon(poke1Ident)!;
        const relativePlayerIndex = poke.side.n ^ playerIndex;

        this.edgeBuffer.updateLatestMinorArgs(argName, relativePlayerIndex);
    }

    "|-crit|"(args: Args["|-crit|"]) {
        const [argName, poke1Ident] = args;

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const poke = this.getPokemon(poke1Ident)!;
        const relativePlayerIndex = poke.side.n ^ playerIndex;

        this.edgeBuffer.updateLatestMinorArgs(argName, relativePlayerIndex);
    }

    "|-supereffective|"(args: Args["|-supereffective|"]) {
        const [argName, poke1Ident] = args;

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const poke = this.getPokemon(poke1Ident)!;
        const relativePlayerIndex = poke.side.n ^ playerIndex;

        this.edgeBuffer.updateLatestMinorArgs(argName, relativePlayerIndex);
    }

    "|-resisted|"(args: Args["|-resisted|"]) {
        const [argName, poke1Ident] = args;

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const poke = this.getPokemon(poke1Ident)!;
        const relativePlayerIndex = poke.side.n ^ playerIndex;

        this.edgeBuffer.updateLatestMinorArgs(argName, relativePlayerIndex);
    }

    "|-immune|"(args: Args["|-immune|"], kwArgs: KWArgs["|-immune|"]) {
        const [argName, poke1Ident] = args;

        const fromEffect = this.getCondition(kwArgs.from);

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const poke = this.getPokemon(poke1Ident)!;
        const relativePlayerIndex = poke.side.n ^ playerIndex;

        this.edgeBuffer.updateLatestMinorArgs(argName, relativePlayerIndex);
        this.edgeBuffer.updateLatestEdgeFromOf(fromEffect, relativePlayerIndex);
    }

    "|-item|"(args: Args["|-item|"], kwArgs: KWArgs["|-item|"]) {
        const [argName, poke1Ident, itemId] = args;

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const poke = this.getPokemon(poke1Ident)!;
        const relativePlayerIndex = poke.side.n ^ playerIndex;

        const itemIndex = IndexValueFromEnum(ItemsEnum, itemId);
        const fromEffect = this.getCondition(kwArgs.from);

        this.edgeBuffer.updateLatestMinorArgs(argName, relativePlayerIndex);
        this.edgeBuffer.updateLatestEdgeFromOf(fromEffect, relativePlayerIndex);
        this.edgeBuffer.setLatestRelativeEdgeFeature(
            RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__ITEM_TOKEN,
            relativePlayerIndex,
            itemIndex,
        );
    }

    "|-enditem|"(args: Args["|-enditem|"], kwArgs: KWArgs["|-enditem|"]) {
        const [argName, poke1Ident, itemId] = args;

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const poke = this.getPokemon(poke1Ident)!;
        const relativePlayerIndex = poke.side.n ^ playerIndex;

        const itemIndex = IndexValueFromEnum(ItemsEnum, itemId);
        const fromEffect = this.getCondition(kwArgs.from);

        this.edgeBuffer.updateLatestMinorArgs(argName, relativePlayerIndex);
        this.edgeBuffer.updateLatestEdgeFromOf(fromEffect, relativePlayerIndex);
        this.edgeBuffer.setLatestRelativeEdgeFeature(
            RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__ITEM_TOKEN,
            relativePlayerIndex,
            itemIndex,
        );
    }

    "|-ability|"(args: Args["|-ability|"], kwArgs: KWArgs["|-ability|"]) {
        const [argName, poke1Ident, abilityId] = args;

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const poke = this.getPokemon(poke1Ident)!;
        const relativePlayerIndex = poke.side.n ^ playerIndex;

        const abilityIndex = IndexValueFromEnum(AbilitiesEnum, abilityId);

        if (kwArgs.from) {
            const fromEffect = this.getCondition(kwArgs.from);
            this.edgeBuffer.updateLatestEdgeFromOf(
                fromEffect,
                relativePlayerIndex,
            );
        }

        this.edgeBuffer.updateLatestMinorArgs(argName, relativePlayerIndex);
        this.edgeBuffer.setLatestRelativeEdgeFeature(
            RelativeEdgeFeature.RELATIVE_EDGE_FEATURE__ABILITY_TOKEN,
            relativePlayerIndex,
            abilityIndex,
        );
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

        const poke = this.getPokemon(poke1Ident)!;
        const relativePlayerIndex = poke.side.n ^ playerIndex;

        const fromEffect = this.getCondition(kwArgs.from);

        this.edgeBuffer.updateLatestMinorArgs(argName, relativePlayerIndex);
        this.edgeBuffer.updateLatestEdgeFromOf(fromEffect, relativePlayerIndex);
    }

    "|-transform|"() {}

    "|-mega|"() {}

    "|-primal|"() {}

    "|-burst|"() {}

    "|-zpower|"() {}

    "|-zbroken|"() {}

    "|-activate|"(args: Args["|-activate|"], kwArgs: KWArgs["|-activate|"]) {
        const [argName, poke1Ident] = args;

        const fromEffect = this.getCondition(kwArgs.from);

        if (poke1Ident) {
            const playerIndex = this.player.getPlayerIndex();
            if (playerIndex === undefined) {
                throw new Error();
            }

            const poke = this.getPokemon(poke1Ident)!;
            const relativePlayerIndex = poke.side.n ^ playerIndex;

            this.edgeBuffer.updateLatestMinorArgs(argName, relativePlayerIndex);
            this.edgeBuffer.updateLatestEdgeFromOf(
                fromEffect,
                relativePlayerIndex,
            );
        }
    }

    "|-mustrecharge|"(args: Args["|-mustrecharge|"]) {
        const [argName, poke1Ident] = args;

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const poke = this.getPokemon(poke1Ident)!;
        const relativePlayerIndex = poke.side.n ^ playerIndex;

        this.edgeBuffer.updateLatestMinorArgs(argName, relativePlayerIndex);
    }

    "|-prepare|"(args: Args["|-prepare|"]) {
        const [argName, poke1Ident] = args;

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const poke = this.getPokemon(poke1Ident)!;
        const relativePlayerIndex = poke.side.n ^ playerIndex;

        this.edgeBuffer.updateLatestMinorArgs(argName, relativePlayerIndex);
    }

    "|-hitcount|"(args: Args["|-hitcount|"]) {
        const [argName, poke1Ident] = args;

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const poke = this.getPokemon(poke1Ident)!;
        const relativePlayerIndex = poke.side.n ^ playerIndex;

        this.edgeBuffer.updateLatestMinorArgs(argName, relativePlayerIndex);
    }

    "|done|"(args: Args["|done|"]) {
        const [argName] = args;

        const edge = new Edge(this.player);
        for (const relativePlayerIndex of [0, 1])
            edge.addMajorArg(argName, relativePlayerIndex);
        if (this.turnOrder > 0) {
            this.addEdge(edge);
        }
    }

    "|start|"() {
        this.turnOrder = 0;
    }

    "|turn|"(args: Args["|turn|"]) {
        const turnNum = (args.at(1) ?? "").toString();

        this.turnOrder = 0;
        this.turnNum = parseInt(turnNum);
    }

    reset() {
        this.prevHp = new Map();
        this.actives = new Map();
    }
}

export class StateHandler {
    player: Player;

    constructor(player: Player) {
        this.player = player;
    }

    static getLegalActions(request?: AnyObject | null): {
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

                for (let j = 1; j <= possibleMoves.length; j++) {
                    const currentMove = possibleMoves[j - 1];
                    if (currentMove.id === "struggle") {
                        isStruggling = true;
                    }
                    if (!currentMove.disabled) {
                        const moveIndex = j as 1 | 2 | 3 | 4;
                        legalActions.set(-1 + moveIndex, true);
                    }
                }

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
            assignActionBuffer(
                MovesetFeature.MOVESET_FEATURE__ACTION_ID,
                IndexValueFromEnum(ActionsEnum, `move_${action.id}`),
            );
            if ("pp" in action) {
                const ppUsed = (31 * (action.maxpp - action.pp)) / action.pp;
                assignActionBuffer(
                    MovesetFeature.MOVESET_FEATURE__PPUSED,
                    ppUsed,
                );
            }
            assignActionBuffer(
                MovesetFeature.MOVESET_FEATURE__MOVE_ID,
                IndexValueFromEnum(ActionsEnum, `move_${action.id}`),
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
            const species = member.species.baseSpecies.toLowerCase();
            assignActionBuffer(
                MovesetFeature.MOVESET_FEATURE__ACTION_ID,
                IndexValueFromEnum(ActionsEnum, `switch_${species}`),
            );
            assignActionBuffer(
                MovesetFeature.MOVESET_FEATURE__SPECIES_ID,
                IndexValueFromEnum(SpeciesEnum, species),
            );
        };

        for (const action of activeMoves) {
            pushMoveAction(action);
            assignActionBuffer(
                MovesetFeature.MOVESET_FEATURE__ACTION_TYPE,
                MovesetActionTypeEnum.MOVESET_ACTION_TYPE_ENUM__MOVE,
            );
            bufferOffset += numMoveFields;
        }
        bufferOffset += (4 - activeMoves.length) * numMoveFields;
        for (const action of switches) {
            pushSwitchAction(action);
            assignActionBuffer(
                MovesetFeature.MOVESET_FEATURE__ACTION_TYPE,
                MovesetActionTypeEnum.MOVESET_ACTION_TYPE_ENUM__SWITCH,
            );
            bufferOffset += numMoveFields;
        }
        bufferOffset += (6 - activeMoves.length) * numMoveFields;

        return new Uint8Array(actionBuffer.buffer);
    }

    getPrivateTeam(playerIndex: number): Int16Array {
        const teams: Int16Array[] = [];
        for (const side of [
            this.player.privateBattle.sides[playerIndex],
            this.player.privateBattle.sides[1 - playerIndex],
        ]) {
            const team: Int16Array[] = [];
            const sortedTeam = [...side.team].sort(
                (a, b) => +!a.isActive() - +!b.isActive(),
            );
            for (const member of sortedTeam) {
                team.push(getArrayFromPokemon(member, playerIndex));
            }
            for (
                let memberIndex = team.length;
                memberIndex < side.totalPokemon;
                memberIndex++
            ) {
                team.push(side.n ? unkPokemon1 : unkPokemon0);
            }
            for (
                let memberIndex = team.length;
                memberIndex < 6;
                memberIndex++
            ) {
                team.push(nullPokemon);
            }
            teams.push(...team);
        }
        return concatenateArrays(teams);
    }

    getPublicTeam(playerIndex: number): Int16Array {
        const side = this.player.publicBattle.sides[playerIndex];
        const team: Int16Array[] = [];
        for (const member of side.team) {
            team.push(getArrayFromPokemon(member, playerIndex));
        }
        for (
            let memberIndex = team.length;
            memberIndex < side.totalPokemon;
            memberIndex++
        ) {
            team.push(side.n ? unkPokemon1 : unkPokemon0);
        }
        for (let memberIndex = team.length; memberIndex < 6; memberIndex++) {
            team.push(nullPokemon);
        }
        return concatenateArrays(team);
    }

    getHistory(numHistory: number = NUM_HISTORY) {
        return this.player.eventHandler.edgeBuffer.getHistory(numHistory);
    }

    getRewards() {
        const rewards = new Rewards();

        const sides = this.player.publicBattle.sides;

        // Calculate alive totals for both sides
        const [aliveTotal1, aliveTotal2] = sides.map((side) => {
            const aliveInTeam = side.team.reduce(
                (count, pokemon) => count + (pokemon.fainted ? 0 : 1),
                0,
            );
            const remainingPokemon = side.totalPokemon - side.team.length;
            return aliveInTeam + remainingPokemon;
        });

        // Calculate HP totals for both sides
        const [hpTotal1, hpTotal2] = sides.map((side) => {
            const hpInTeam = side.team.reduce(
                (sum, pokemon) => sum + pokemon.hp / pokemon.maxhp,
                0,
            );
            const remainingPokemon = side.totalPokemon - side.team.length;
            return hpInTeam + remainingPokemon;
        });

        const { hpReward, faintedReward } = this.player.tracker.update1(
            this.player.publicBattle,
        );

        rewards.setHpReward(hpReward);
        rewards.setFaintedReward(faintedReward);

        if (!this.player.done) {
            return rewards;
        }

        let winReward = 0;

        if (!this.player.draw) {
            if (aliveTotal1 !== aliveTotal2) {
                // Determine winner based on alive totals
                winReward =
                    aliveTotal1 > aliveTotal2
                        ? 1 - aliveTotal2 / 6
                        : aliveTotal1 / 6 - 1;
            } else if (hpTotal1 !== hpTotal2) {
                // Determine winner based on HP totals
                winReward =
                    hpTotal1 > hpTotal2 ? 1 - hpTotal2 / 6 : hpTotal1 / 6 - 1;
            } else {
                /* empty */
            }
        }

        rewards.setWinReward(winReward);
        return rewards;
    }

    getHeuristics() {
        const heuristics = new Heuristics();

        const action = GetHeuristicAction({ player: this.player });
        const actionIndex = action.getValue();

        if (actionIndex < 0) {
            const { legalActions } = StateHandler.getLegalActions(
                this.player.privateBattle.request,
            );
            const validIndices = legalActions
                .toBinaryVector()
                .flatMap((val, ind) => (val ? [ind] : []));
            heuristics.setHeuristicAction(
                validIndices[Math.random() * validIndices.length],
            );
        } else {
            heuristics.setHeuristicAction(actionIndex);
        }

        return heuristics;
    }

    getInfo() {
        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error("Player index is undefined");
        }

        const info = new Info();
        info.setTs(performance.now());
        info.setGameId(this.player.gameId);
        info.setPlayerIndex(!!playerIndex);
        info.setTurn(this.player.privateBattle.turn);
        info.setDone(this.player.done);
        info.setDraw(this.player.draw);
        info.setRequestCount(this.player.requestCount);

        const worldStream = this.player.worldStream;
        if (worldStream !== null) {
            const world = worldStream.battle!;
            info.setSeed(hashArrayToInt32(world.prng.initialSeed));
        }

        const drawRatio = this.player.privateBattle.turn / DRAW_TURNS;
        info.setDrawRatio(Math.max(0, Math.min(1, drawRatio)));

        const rewards = this.getRewards();
        info.setRewards(rewards);

        // const heuristics = this.getHeuristics();
        // info.setHeuristics(heuristics);

        return info;
    }

    getState(numHistory: number = NUM_HISTORY): State {
        if (
            !this.player.offline &&
            !this.player.done &&
            this.player.getRequest() === undefined
        ) {
            throw new Error("Need Request");
        }

        const state = new State();
        const info = this.getInfo();
        state.setInfo(info);

        const { legalActions } = StateHandler.getLegalActions(
            this.player.privateBattle.request,
        );
        state.setLegalActions(legalActions.buffer);

        const history = this.getHistory(numHistory);
        // eslint-disable-next-line @typescript-eslint/no-unused-vars
        // const readableHistory = EdgeBuffer.toReadableHistory(history);
        state.setHistory(history);

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error("Player index is undefined");
        }

        const privateTeam = this.getPrivateTeam(playerIndex);
        state.setTeam(new Uint8Array(privateTeam.buffer));
        state.setMoveset(this.getMoveset());

        return state;
    }
}
