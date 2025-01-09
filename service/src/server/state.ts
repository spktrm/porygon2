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
    EffecttypesEnum,
    EffecttypesEnumMap,
    GendernameEnum,
    ItemeffecttypesEnum,
    ItemsEnum,
    MovesEnum,
    SideconditionEnumMap,
    SpeciesEnum,
    StatusEnum,
    WeatherEnum,
} from "../../protos/enums_pb";
import {
    FeatureAbsoluteEdge,
    FeatureAbsoluteEdgeMap,
    FeatureEntity,
    FeatureMoveset,
    FeatureRelativeEdge,
    FeatureRelativeEdgeMap,
    MovesetActionType,
} from "../../protos/features_pb";
import {
    MappingLookup,
    EnumKeyMapping,
    EnumMappings,
    Mappings,
    MoveIndex,
    numPokemonFields as numPokemonFeatures,
    numMovesetFields,
    numMoveFields,
    NUM_HISTORY,
    jsonDatum,
} from "./data";
import { NA, Pokemon, Side } from "@pkmn/client";
import { Ability, Item, Move, BoostID } from "@pkmn/dex-types";
import { ID } from "@pkmn/types";
import { Condition, Effect } from "@pkmn/data";
import { History } from "../../protos/history_pb";
import { OneDBoolean, TypedArray } from "./utils";
import { Player } from "./player";
import { DRAW_TURNS } from "./game";
import { GetMoveDamange } from "./baselines/max_dmg";
import { GetHeuristicAction } from "./baselines/heuristic";

type RemovePipes<T extends string> = T extends `|${infer U}|` ? U : T;
type MajorArgNames =
    | RemovePipes<BattleMajorArgName>
    | RemovePipes<BattleProgressArgName>;
type MinorArgNames = RemovePipes<BattleMinorArgName>;

const sanitizeKeyCache = new Map<string, string>();

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

function SanitizeKey(key: string): string {
    if (sanitizeKeyCache.has(key)) {
        return sanitizeKeyCache.get(key)!;
    }
    const sanitizedKey = key.replace(/\W/g, "").toLowerCase();
    sanitizeKeyCache.set(key, sanitizedKey);
    return sanitizedKey;
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

export function IndexValueFromEnum<T extends EnumMappings>(
    mappingType: Mappings,
    key: string,
): T[keyof T] {
    const mapping = MappingLookup[mappingType] as T;
    const enumMapping = EnumKeyMapping[mappingType];
    const sanitizedKey = SanitizeKey(key);
    const trueKey = enumMapping[sanitizedKey] as keyof T;
    const value = mapping[trueKey];
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
    data[FeatureEntity.ENTITY_SPECIES] = SpeciesEnum.SPECIES__UNK;
    data[FeatureEntity.ENTITY_ITEM] = ItemsEnum.ITEMS__UNK;
    data[FeatureEntity.ENTITY_ITEM_EFFECT] =
        ItemeffecttypesEnum.ITEMEFFECTTYPES__NULL;
    data[FeatureEntity.ENTITY_ABILITY] = AbilitiesEnum.ABILITIES__UNK;
    data[FeatureEntity.ENTITY_FAINTED] = 0;
    data[FeatureEntity.ENTITY_HP] = 100;
    data[FeatureEntity.ENTITY_MAXHP] = 100;
    data[FeatureEntity.ENTITY_STATUS] = StatusEnum.STATUS__NULL;
    data[FeatureEntity.ENTITY_MOVEID0] = MovesEnum.MOVES__UNK;
    data[FeatureEntity.ENTITY_MOVEID1] = MovesEnum.MOVES__UNK;
    data[FeatureEntity.ENTITY_MOVEID2] = MovesEnum.MOVES__UNK;
    data[FeatureEntity.ENTITY_MOVEID3] = MovesEnum.MOVES__UNK;
    data[FeatureEntity.ENTITY_HAS_STATUS] = 0;
    data[FeatureEntity.ENTITY_SIDE] = n ?? 0;
    data[FeatureEntity.ENTITY_HP] = 31;
    return data;
}

const unkPokemon0 = getUnkPokemon(0);
const unkPokemon1 = getUnkPokemon(1);

function getNullPokemon() {
    const data = getBlankPokemonArr();
    data[FeatureEntity.ENTITY_SPECIES] = SpeciesEnum.SPECIES__NULL;
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
            const idValue = IndexValueFromEnum<typeof ActionsEnum>(
                "Actions",
                `move_${id}`,
            );
            const correctUsed = ((isNaN(ppUsed) ? +!!ppUsed : ppUsed) * 5) / 8;

            moveIds.push(idValue);
            movePps.push(Math.floor((1023 * correctUsed) / maxPP));
        }
    }
    let remainingIndex: MoveIndex = moveSlots.length as MoveIndex;
    for (remainingIndex; remainingIndex < 4; remainingIndex++) {
        moveIds.push(ActionsEnum.ACTIONS_MOVE__UNK);
        movePps.push(0);
    }

    const dataArr = getBlankPokemonArr();
    dataArr[FeatureEntity.ENTITY_SPECIES] = IndexValueFromEnum<
        typeof SpeciesEnum
    >("Species", baseSpecies);
    dataArr[FeatureEntity.ENTITY_ITEM] = item
        ? IndexValueFromEnum<typeof ItemsEnum>("Items", item)
        : ItemsEnum.ITEMS__UNK;
    dataArr[FeatureEntity.ENTITY_ITEM_EFFECT] = itemEffect
        ? IndexValueFromEnum<typeof ItemeffecttypesEnum>(
              "ItemEffect",
              itemEffect,
          )
        : ItemeffecttypesEnum.ITEMEFFECTTYPES__NULL;

    const possibleAbilities = Object.values(pokemon.baseSpecies.abilities);
    if (ability) {
        const actualAbility = IndexValueFromEnum<typeof AbilitiesEnum>(
            "Ability",
            ability,
        );
        dataArr[FeatureEntity.ENTITY_ABILITY] = actualAbility;
    } else if (possibleAbilities.length === 1) {
        const onlyAbility = possibleAbilities[0]
            ? IndexValueFromEnum<typeof AbilitiesEnum>(
                  "Ability",
                  possibleAbilities[0],
              )
            : AbilitiesEnum.ABILITIES__UNK;
        dataArr[FeatureEntity.ENTITY_ABILITY] = onlyAbility;
    } else {
        dataArr[FeatureEntity.ENTITY_ABILITY] = AbilitiesEnum.ABILITIES__UNK;
    }

    dataArr[FeatureEntity.ENTITY_GENDER] = IndexValueFromEnum<
        typeof GendernameEnum
    >("Gender", pokemon.gender);
    dataArr[FeatureEntity.ENTITY_ACTIVE] = pokemon.isActive() ? 1 : 0;
    dataArr[FeatureEntity.ENTITY_FAINTED] = pokemon.fainted ? 1 : 0;
    dataArr[FeatureEntity.ENTITY_HP] = pokemon.hp;
    dataArr[FeatureEntity.ENTITY_MAXHP] = pokemon.maxhp;
    dataArr[FeatureEntity.ENTITY_HP_RATIO] = Math.floor(
        (31 * pokemon.hp) / pokemon.maxhp,
    );
    dataArr[FeatureEntity.ENTITY_STATUS] = pokemon.status
        ? IndexValueFromEnum<typeof StatusEnum>("Status", pokemon.status)
        : StatusEnum.STATUS__NULL;
    dataArr[FeatureEntity.ENTITY_HAS_STATUS] = pokemon.status ? 1 : 0;
    dataArr[FeatureEntity.ENTITY_TOXIC_TURNS] = pokemon.statusState.toxicTurns;
    dataArr[FeatureEntity.ENTITY_SLEEP_TURNS] = pokemon.statusState.sleepTurns;
    dataArr[FeatureEntity.ENTITY_BEING_CALLED_BACK] = pokemon.beingCalledBack
        ? 1
        : 0;
    dataArr[FeatureEntity.ENTITY_TRAPPED] = pokemon.trapped ? 1 : 0;
    dataArr[FeatureEntity.ENTITY_NEWLY_SWITCHED] = pokemon.newlySwitched
        ? 1
        : 0;
    dataArr[FeatureEntity.ENTITY_LEVEL] = pokemon.level;
    for (let moveIndex = 0; moveIndex < 4; moveIndex++) {
        dataArr[FeatureEntity[`ENTITY_MOVEID${moveIndex as MoveIndex}`]] =
            moveIds[moveIndex];
        dataArr[FeatureEntity[`ENTITY_MOVEPP${moveIndex as MoveIndex}`]] =
            movePps[moveIndex];
    }

    let volatiles = BigInt(0b0);
    for (const [key] of Object.entries({
        ...pokemon.volatiles,
        ...candidate.volatiles,
    })) {
        const index = IndexValueFromEnum("Volatilestatus", key);
        volatiles |= BigInt(1) << BigInt(index);
    }
    dataArr.set(bigIntToInt16Array(volatiles), FeatureEntity.ENTITY_VOLATILES0);

    dataArr[FeatureEntity.ENTITY_SIDE] = pokemon.side.n ^ playerIndex;

    dataArr[FeatureEntity.ENTITY_BOOST_ATK_VALUE] =
        (pokemon.boosts.atk ?? 0) + 6;
    dataArr[FeatureEntity.ENTITY_BOOST_DEF_VALUE] =
        (pokemon.boosts.def ?? 0) + 6;
    dataArr[FeatureEntity.ENTITY_BOOST_SPA_VALUE] =
        (pokemon.boosts.spa ?? 0) + 6;
    dataArr[FeatureEntity.ENTITY_BOOST_SPD_VALUE] =
        (pokemon.boosts.spd ?? 0) + 6;
    dataArr[FeatureEntity.ENTITY_BOOST_SPE_VALUE] =
        (pokemon.boosts.spe ?? 0) + 6;
    dataArr[FeatureEntity.ENTITY_BOOST_EVASION_VALUE] =
        (pokemon.boosts.evasion ?? 0) + 6;
    dataArr[FeatureEntity.ENTITY_BOOST_ACCURACY_VALUE] =
        (pokemon.boosts.accuracy ?? 0) + 6;

    let typeChanged = BigInt(0b0);
    const typechangeVolatile =
        pokemon.volatiles.typechange ?? candidate.volatiles.typechange;
    if (typechangeVolatile) {
        if (typechangeVolatile.apparentType) {
            for (const type of typechangeVolatile.apparentType.split("/")) {
                const index = IndexValueFromEnum("Types", type);
                typeChanged |= BigInt(1) << BigInt(index);
            }
        }
    }
    dataArr.set(
        bigIntToInt16Array(typeChanged),
        FeatureEntity.ENTITY_TYPECHANGE0,
    );

    return dataArr;
}

const numRelativeEdgeFeatures = Object.keys(FeatureRelativeEdge).length;
const numAbsoluteEdgeFeatures = Object.keys(FeatureAbsoluteEdge).length;

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
                FeatureRelativeEdge.EDGE_SIDECONDITIONS0;
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
            const featureIndex = IndexValueFromEnum<SideconditionEnumMap>(
                "Sidecondition",
                id,
            );
            sideConditionBuffer |= BigInt(1) << BigInt(featureIndex);
        }
        this.relativeEdgeData.set(
            bigIntToInt16Array(sideConditionBuffer),
            relativeEdgeOffset,
        );
        if (side.sideConditions.spikes) {
            this.setRelativeEdgeFeature(
                FeatureRelativeEdge.EDGE_SPIKES,
                side.n ^ playerIndex,
                side.sideConditions.spikes.level,
            );
        }
        if (side.sideConditions.toxicspikes) {
            this.setRelativeEdgeFeature(
                FeatureRelativeEdge.EDGE_TOXIC_SPIKES,
                side.n ^ playerIndex,
                side.sideConditions.toxicspikes.level,
            );
        }
    }

    updateFieldData() {
        const field = this.player.publicBattle.field;
        const weatherIndex = field.weatherState.id
            ? IndexValueFromEnum(
                  "Weather",
                  WEATHERS[field.weatherState.id as never],
              )
            : WeatherEnum.WEATHER__NULL;

        this.absoluteEdgeData[FeatureAbsoluteEdge.EDGE_WEATHER_ID] =
            weatherIndex;
        this.absoluteEdgeData[FeatureAbsoluteEdge.EDGE_WEATHER_MAX_DURATION] =
            field.weatherState.maxDuration;
        this.absoluteEdgeData[FeatureAbsoluteEdge.EDGE_WEATHER_MIN_DURATION] =
            field.weatherState.minDuration;
    }

    setRelativeEdgeFeature(
        featureIndex: FeatureRelativeEdgeMap[keyof FeatureRelativeEdgeMap],
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
        featureIndex: FeatureAbsoluteEdgeMap[keyof FeatureAbsoluteEdgeMap],

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
        featureIndex: FeatureRelativeEdgeMap[keyof FeatureRelativeEdgeMap],
        sideIndex: number,
    ) {
        const index =
            featureIndex + (sideIndex === 0 ? 0 : numRelativeEdgeFeatures);
        return this.relativeEdgeData.at(index);
    }

    addMajorArg(argName: MajorArgNames, sideIndex: number) {
        const index = IndexValueFromEnum("BattleMajorArg", argName);
        this.setRelativeEdgeFeature(
            FeatureRelativeEdge.EDGE_MAJOR_ARG,
            sideIndex,
            index,
        );
    }

    updateEdgeFromOf(effect: Partial<Effect>, sideIndex: number) {
        const { effectType } = effect;
        if (effectType) {
            const fromTypeToken =
                EffecttypesEnum[
                    `EFFECTTYPES_${effectType.toUpperCase()}` as keyof EffecttypesEnumMap
                ];
            this.setRelativeEdgeFeature(
                FeatureRelativeEdge.EDGE_FROM_TYPE_TOKEN,
                sideIndex,
                fromTypeToken,
            );
            const fromSourceToken = getEffectToken(effect);
            this.setRelativeEdgeFeature(
                FeatureRelativeEdge.EDGE_FROM_SOURCE_TOKEN,
                sideIndex,
                fromSourceToken,
            );
        }
    }
}

function getEffectToken(effect: Partial<Effect>): number {
    const { effectType, id } = effect;
    if (id) {
        const key = `${effectType}_${id}`;
        return IndexValueFromEnum("Effect", key);
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

        const MAX_TURNS = 1000;

        const maxEdges = NUM_HISTORY * MAX_TURNS;
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
            const featureIndex = IndexValueFromEnum<SideconditionEnumMap>(
                "Sidecondition",
                id,
            );
            sideConditionBuffer |= BigInt(1) << BigInt(featureIndex);
        }
        this.relativeEdgeData.set(
            bigIntToInt16Array(sideConditionBuffer),
            relativeEdgeOffset,
        );
        if (side.sideConditions.spikes) {
            this.setLatestRelativeEdgeFeature(
                FeatureRelativeEdge.EDGE_SPIKES,
                side.n ^ playerIndex,
                side.sideConditions.spikes.level,
            );
        }
        if (side.sideConditions.toxicspikes) {
            this.setLatestRelativeEdgeFeature(
                FeatureRelativeEdge.EDGE_TOXIC_SPIKES,
                side.n ^ playerIndex,
                side.sideConditions.toxicspikes.level,
            );
        }
    }

    updateLatestFieldData() {
        const field = this.player.publicBattle.field;
        const weatherIndex = field.weatherState.id
            ? IndexValueFromEnum(
                  "Weather",
                  WEATHERS[field.weatherState.id as never],
              )
            : WeatherEnum.WEATHER__NULL;

        this.absoluteEdgeData[
            this.prevAbsoluteEdgeCursor + FeatureAbsoluteEdge.EDGE_WEATHER_ID
        ] = weatherIndex;
        this.absoluteEdgeData[
            this.prevAbsoluteEdgeCursor +
                FeatureAbsoluteEdge.EDGE_WEATHER_MAX_DURATION
        ] = field.weatherState.maxDuration;
        this.absoluteEdgeData[
            this.prevAbsoluteEdgeCursor +
                FeatureAbsoluteEdge.EDGE_WEATHER_MIN_DURATION
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
                FeatureRelativeEdge.EDGE_SIDECONDITIONS0;
            this.updateLatestEntityData(side.active, entityOffset, playerIndex);
            this.updateLatestSideConditionData(
                side,
                sideConditionOffset,
                playerIndex,
            );
        }
    }

    setLatestRelativeEdgeFeature(
        featureIndex: FeatureRelativeEdgeMap[keyof FeatureRelativeEdgeMap],
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
        featureIndex: FeatureRelativeEdgeMap[keyof FeatureRelativeEdgeMap],
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

        const index = IndexValueFromEnum("BattleMinorArg", argName);
        const featureIndex = {
            0: FeatureRelativeEdge.EDGE_MINOR_ARG0,
            1: FeatureRelativeEdge.EDGE_MINOR_ARG1,
            2: FeatureRelativeEdge.EDGE_MINOR_ARG2,
            3: FeatureRelativeEdge.EDGE_MINOR_ARG3,
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
        const index = IndexValueFromEnum("BattleMajorArg", argName);
        this.setLatestRelativeEdgeFeature(
            FeatureRelativeEdge.EDGE_MAJOR_ARG,
            sideIndex,
            index,
        );
    }

    setLatestAbsoluteEdgeFeature(
        featureIndex: FeatureAbsoluteEdgeMap[keyof FeatureAbsoluteEdgeMap],
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
            const fromTypeToken =
                EffecttypesEnum[
                    `EFFECTTYPES_${effectType.toUpperCase()}` as keyof EffecttypesEnumMap
                ];
            this.setLatestRelativeEdgeFeature(
                FeatureRelativeEdge.EDGE_FROM_TYPE_TOKEN,
                sideIndex,
                fromTypeToken,
            );
            const fromSourceToken = getEffectToken(effect);
            this.setLatestRelativeEdgeFeature(
                FeatureRelativeEdge.EDGE_FROM_SOURCE_TOKEN,
                sideIndex,
                fromSourceToken,
            );
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
        history.setRelativeedges(
            new Uint8Array(
                this.relativeEdgeData.slice(
                    this.relativeEdgeCursor -
                        width * 2 * numRelativeEdgeFeatures,
                    this.relativeEdgeCursor,
                ).buffer,
            ),
        );
        history.setAbsoluteedge(
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
            history.getRelativeedges_asU8().buffer,
        );
        const historyAbsoluteEdges = new Int16Array(
            history.getAbsoluteedge_asU8().buffer,
        );

        const entityArrayToObject = (array: Int16Array) => {
            const volatilesFlat = array.slice(
                FeatureEntity.ENTITY_VOLATILES0,
                FeatureEntity.ENTITY_VOLATILES8 + 1,
            );
            const volatilesIndices = int16ArrayToBitIndices(volatilesFlat);

            const typechangeFlat = array.slice(
                FeatureEntity.ENTITY_TYPECHANGE0,
                FeatureEntity.ENTITY_TYPECHANGE1 + 1,
            );
            const typechangeIndices = int16ArrayToBitIndices(typechangeFlat);

            const moveIndicies = Array.from(
                array.slice(
                    FeatureEntity.ENTITY_MOVEID0,
                    FeatureEntity.ENTITY_MOVEID3 + 1,
                ),
            );

            return {
                species:
                    jsonDatum["species"][array[FeatureEntity.ENTITY_SPECIES]],
                item: jsonDatum["items"][array[FeatureEntity.ENTITY_ITEM]],
                ability:
                    jsonDatum["abilities"][array[FeatureEntity.ENTITY_ABILITY]],
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
                FeatureRelativeEdge.EDGE_MINOR_ARG0,
                FeatureRelativeEdge.EDGE_MINOR_ARG3 + 1,
            );
            const minorArgIndices = int16ArrayToBitIndices(minorArgsFlat);

            const sideConditionsFlat = array.slice(
                FeatureRelativeEdge.EDGE_SIDECONDITIONS0,
                FeatureRelativeEdge.EDGE_SIDECONDITIONS1 + 1,
            );
            const sideConditionIndices =
                int16ArrayToBitIndices(sideConditionsFlat);

            return {
                majorArg:
                    jsonDatum["battleMajorArgs"][
                        array[FeatureRelativeEdge.EDGE_MAJOR_ARG]
                    ],
                minorArgs: minorArgIndices.map(
                    (index) => jsonDatum["battleMinorArgs"][index],
                ),
                action: jsonDatum["Actions"][
                    array[FeatureRelativeEdge.EDGE_ACTION_TOKEN]
                ],
                damage: array[FeatureRelativeEdge.EDGE_DAMAGE_RATIO] / 31,
                heal: array[FeatureRelativeEdge.EDGE_HEAL_RATIO] / 31,
                sideConditions: sideConditionIndices.map(
                    (index) => jsonDatum["sideCondition"][index],
                ),
                from_source:
                    jsonDatum["Effect"][
                        array[FeatureRelativeEdge.EDGE_FROM_SOURCE_TOKEN]
                    ],
                boosts: {
                    EDGE_BOOST_ATK_VALUE:
                        array[FeatureRelativeEdge.EDGE_BOOST_ATK_VALUE],
                    EDGE_BOOST_DEF_VALUE:
                        array[FeatureRelativeEdge.EDGE_BOOST_DEF_VALUE],
                    EDGE_BOOST_SPA_VALUE:
                        array[FeatureRelativeEdge.EDGE_BOOST_SPA_VALUE],
                    EDGE_BOOST_SPD_VALUE:
                        array[FeatureRelativeEdge.EDGE_BOOST_SPD_VALUE],
                    EDGE_BOOST_SPE_VALUE:
                        array[FeatureRelativeEdge.EDGE_BOOST_SPE_VALUE],
                    EDGE_BOOST_ACCURACY_VALUE:
                        array[FeatureRelativeEdge.EDGE_BOOST_ACCURACY_VALUE],
                    EDGE_BOOST_EVASION_VALUE:
                        array[FeatureRelativeEdge.EDGE_BOOST_EVASION_VALUE],
                },
            };
        };

        const absoluteArrayToObject = (array: Int16Array) => {
            return {
                turnOrder: array[FeatureAbsoluteEdge.EDGE_TURN_ORDER_VALUE],
                requestCount: array[FeatureAbsoluteEdge.EDGE_REQUEST_COUNT],
                weatherId:
                    jsonDatum["weather"][
                        array[FeatureAbsoluteEdge.EDGE_WEATHER_ID]
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

    currHp: Map<string, number>;
    actives: Map<ID, PokemonIdent>;
    turnOrder: number;
    turnNum: number;

    edgeBuffer: EdgeBuffer;

    constructor(player: Player) {
        this.player = player;
        this.currHp = new Map();
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
            FeatureAbsoluteEdge.EDGE_REQUEST_COUNT,
            this.player.requestCount,
        );
        edge.setAbsoluteEdgeFeature(FeatureAbsoluteEdge.EDGE_VALID, 1);
        edge.setAbsoluteEdgeFeature(
            FeatureAbsoluteEdge.EDGE_INDEX,
            this.edgeBuffer.numEdges,
        );
        edge.setAbsoluteEdgeFeature(
            FeatureAbsoluteEdge.EDGE_TURN_ORDER_VALUE,
            this.turnOrder,
        );
        this.turnOrder += 1;
        edge.setAbsoluteEdgeFeature(
            FeatureAbsoluteEdge.EDGE_TURN_VALUE,
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
        const actionIndex = IndexValueFromEnum<typeof ActionsEnum>(
            "Actions",
            `move_${move.id}`,
        );

        const edge = new Edge(this.player);
        edge.addMajorArg(argName, relativePlayerIndex);
        edge.setRelativeEdgeFeature(
            FeatureRelativeEdge.EDGE_ACTION_TOKEN,
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

        const actionIndex = IndexValueFromEnum<typeof ActionsEnum>(
            "Actions",
            `switch_${poke.species.baseSpecies.toLowerCase()}`,
        );

        const edge = new Edge(this.player);
        edge.addMajorArg(argName, relativePlayerIndex);
        edge.setRelativeEdgeFeature(
            FeatureRelativeEdge.EDGE_ACTION_TOKEN,
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
            const actionIndex = IndexValueFromEnum<typeof ActionsEnum>(
                "Actions",
                `move_${move.id}`,
            );
            edge.setRelativeEdgeFeature(
                FeatureRelativeEdge.EDGE_ACTION_TOKEN,
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

        const poke = this.getPokemon(poke2Ident ?? poke1Ident)!;
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

        if (!this.currHp.has(trueIdent)) {
            this.currHp.set(trueIdent, 1);
        }
        const prevHp = this.currHp.get(trueIdent) ?? 1;
        const currHp = poke.hp / poke.maxhp;
        const diffRatio = currHp - prevHp;
        this.currHp.set(trueIdent, currHp);

        this.edgeBuffer.updateLatestMinorArgs(argName, relativePlayerIndex);
        this.edgeBuffer.setLatestRelativeEdgeFeature(
            FeatureRelativeEdge.EDGE_DAMAGE_RATIO,
            relativePlayerIndex,
            Math.abs(Math.floor(31 * diffRatio)),
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

        if (!this.currHp.has(trueIdent)) {
            this.currHp.set(trueIdent, 1);
        }
        const prevHp = this.currHp.get(trueIdent) ?? 1;
        const currHp = poke.hp / poke.maxhp;
        const diffRatio = currHp - prevHp;
        this.currHp.set(trueIdent, currHp);

        this.edgeBuffer.updateLatestMinorArgs(argName, relativePlayerIndex);
        this.edgeBuffer.setLatestRelativeEdgeFeature(
            FeatureRelativeEdge.EDGE_HEAL_RATIO,
            relativePlayerIndex,
            Math.abs(Math.floor(31 * diffRatio)),
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
        const statusToken = IndexValueFromEnum("Status", statusId);

        this.edgeBuffer.updateLatestMinorArgs(argName, relativePlayerIndex);
        this.edgeBuffer.updateLatestEdgeFromOf(fromEffect, relativePlayerIndex);
        this.edgeBuffer.setLatestRelativeEdgeFeature(
            FeatureRelativeEdge.EDGE_STATUS_TOKEN,
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

        const statusToken = IndexValueFromEnum("Status", statusId);

        this.edgeBuffer.updateLatestMinorArgs(argName, relativePlayerIndex);
        this.edgeBuffer.setLatestRelativeEdgeFeature(
            FeatureRelativeEdge.EDGE_STATUS_TOKEN,
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

    static getStatBoostEdgeFeatureIndex(
        stat: BoostID,
    ): FeatureRelativeEdgeMap[`EDGE_BOOST_${Uppercase<BoostID>}_VALUE`] {
        return FeatureRelativeEdge[
            `EDGE_BOOST_${stat.toLocaleUpperCase()}_VALUE` as `EDGE_BOOST_${Uppercase<BoostID>}_VALUE`
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
                ? WeatherEnum.WEATHER__NULL
                : IndexValueFromEnum("Weather", weatherId);
        for (const relativePlayerIndex of [0, 1]) {
            this.edgeBuffer.updateLatestMinorArgs(argName, relativePlayerIndex);
            this.edgeBuffer.updateLatestEdgeFromOf(
                fromEffect,
                relativePlayerIndex,
            );
        }
        this.edgeBuffer.setLatestAbsoluteEdgeFeature(
            FeatureAbsoluteEdge.EDGE_WEATHER_ID,
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

        const itemIndex = IndexValueFromEnum("Items", itemId);
        const fromEffect = this.getCondition(kwArgs.from);

        this.edgeBuffer.updateLatestMinorArgs(argName, relativePlayerIndex);
        this.edgeBuffer.updateLatestEdgeFromOf(fromEffect, relativePlayerIndex);
        this.edgeBuffer.setLatestRelativeEdgeFeature(
            FeatureRelativeEdge.EDGE_ITEM_TOKEN,
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

        const itemIndex = IndexValueFromEnum("Items", itemId);
        const fromEffect = this.getCondition(kwArgs.from);

        this.edgeBuffer.updateLatestMinorArgs(argName, relativePlayerIndex);
        this.edgeBuffer.updateLatestEdgeFromOf(fromEffect, relativePlayerIndex);
        this.edgeBuffer.setLatestRelativeEdgeFeature(
            FeatureRelativeEdge.EDGE_ITEM_TOKEN,
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

        const abilityIndex = IndexValueFromEnum("Ability", abilityId);
        const fromEffect = this.getCondition(kwArgs.from);

        this.edgeBuffer.updateLatestMinorArgs(argName, relativePlayerIndex);
        this.edgeBuffer.updateLatestEdgeFromOf(fromEffect, relativePlayerIndex);
        this.edgeBuffer.setLatestRelativeEdgeFeature(
            FeatureRelativeEdge.EDGE_ABILITY_TOKEN,
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
        this.currHp = new Map();
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

    getSideMoveset(side: Side, isStruggling: boolean) {
        const movesetArr = new Int16Array(numMovesetFields);

        let offset = 0;

        const active = side.active[0];

        const battle = this.player.privateBattle;
        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error("PlayerIndex is not defined");
        }

        const mySide = battle.sides[playerIndex];
        const oppSide = battle.sides[1 - playerIndex];

        const attacker = mySide.active[0];
        const defender = oppSide.active[0];
        let numMoves = 0;

        if (active !== null) {
            if (isStruggling) {
                movesetArr[offset + FeatureMoveset.MOVESET_ACTION_ID] =
                    IndexValueFromEnum<typeof ActionsEnum>(
                        "Actions",
                        `move_struggle`,
                    );
                movesetArr[offset + FeatureMoveset.MOVESET_ACTION_TYPE] =
                    MovesetActionType.MOVESET_ACTION_TYPE_MOVE;
                movesetArr[offset + FeatureMoveset.MOVESET_SIDE] =
                    side.n ^ playerIndex;
                movesetArr[offset + FeatureMoveset.MOVESET_MOVE_ID] =
                    IndexValueFromEnum("Move", "struggle");
                movesetArr[offset + FeatureMoveset.MOVESET_SPECIES_ID] =
                    SpeciesEnum.SPECIES__NULL;
                offset += numMoveFields;
                numMoves += 1;
            } else {
                const moveSlots = active.moveSlots.slice(0, 4);
                if (active.moveSlots.length > 4) {
                    /* empty */
                }
                for (const move of moveSlots) {
                    const { id, ppUsed } = move;
                    movesetArr[offset + FeatureMoveset.MOVESET_ACTION_ID] =
                        IndexValueFromEnum<typeof ActionsEnum>(
                            "Actions",
                            `move_${id}`,
                        );
                    movesetArr[offset + FeatureMoveset.MOVESET_ACTION_TYPE] =
                        MovesetActionType.MOVESET_ACTION_TYPE_MOVE;
                    movesetArr[offset + FeatureMoveset.MOVESET_PPUSED] = ppUsed;
                    movesetArr[offset + FeatureMoveset.MOVESET_SIDE] =
                        side.n ^ playerIndex;
                    if ("disabled" in move) {
                        movesetArr[offset + FeatureMoveset.MOVESET_LEGAL] =
                            move.disabled ? 0 : 1;
                    } else {
                        movesetArr[offset + FeatureMoveset.MOVESET_LEGAL] = 1;
                    }
                    if (attacker !== null && defender !== null) {
                        try {
                            const moveDamage = GetMoveDamange({
                                battle: this.player.privateBattle,
                                attacker,
                                defender,
                                moveId: id,
                            });
                            movesetArr[
                                offset + FeatureMoveset.MOVESET_EST_DAMAGE
                            ] = moveDamage;
                            // eslint-disable-next-line @typescript-eslint/no-unused-vars
                        } catch (err) {
                            /* empty */
                        }
                    } else {
                        movesetArr[
                            offset + FeatureMoveset.MOVESET_EST_DAMAGE
                        ] = 0;
                    }

                    movesetArr[offset + FeatureMoveset.MOVESET_MOVE_ID] =
                        IndexValueFromEnum("Move", id);
                    movesetArr[offset + FeatureMoveset.MOVESET_SPECIES_ID] =
                        SpeciesEnum.SPECIES__NULL;
                    offset += numMoveFields;
                    numMoves += 1;
                }
            }
            for (
                let remainingIndex = numMoves;
                remainingIndex < 4;
                remainingIndex++
            ) {
                movesetArr[offset + FeatureMoveset.MOVESET_ACTION_TYPE] =
                    MovesetActionType.MOVESET_ACTION_TYPE_MOVE;
                movesetArr[offset + FeatureMoveset.MOVESET_ACTION_ID] =
                    ActionsEnum.ACTIONS_MOVE__UNK;
                movesetArr[offset + FeatureMoveset.MOVESET_PPUSED] = 0;
                movesetArr[offset + FeatureMoveset.MOVESET_SIDE] = side.n;
                movesetArr[offset + FeatureMoveset.MOVESET_LEGAL] = 1;
                movesetArr[offset + FeatureMoveset.MOVESET_MOVE_ID] =
                    MovesEnum.MOVES__UNK;
                movesetArr[offset + FeatureMoveset.MOVESET_SPECIES_ID] =
                    SpeciesEnum.SPECIES__NULL;
                offset += numMoveFields;
            }
        } else {
            for (let remainingIndex = 0; remainingIndex < 4; remainingIndex++) {
                movesetArr[offset + FeatureMoveset.MOVESET_ACTION_TYPE] =
                    MovesetActionType.MOVESET_ACTION_TYPE_MOVE;
                movesetArr[offset + FeatureMoveset.MOVESET_ACTION_ID] =
                    ActionsEnum.ACTIONS_MOVE__NULL;
                movesetArr[offset + FeatureMoveset.MOVESET_PPUSED] = 0;
                movesetArr[offset + FeatureMoveset.MOVESET_SIDE] = side.n;
                movesetArr[offset + FeatureMoveset.MOVESET_LEGAL] = 0;
                movesetArr[offset + FeatureMoveset.MOVESET_MOVE_ID] =
                    MovesEnum.MOVES__NULL;
                movesetArr[offset + FeatureMoveset.MOVESET_SPECIES_ID] =
                    SpeciesEnum.SPECIES__NULL;
                offset += numMoveFields;
            }
        }

        for (
            let remainingIndex = 0;
            remainingIndex < side.totalPokemon;
            remainingIndex++
        ) {
            const pokemon = side.team[remainingIndex];
            if (pokemon === undefined) {
                movesetArr[offset + FeatureMoveset.MOVESET_ACTION_ID] =
                    ActionsEnum.ACTIONS_SWITCH__UNK;
                movesetArr[offset + FeatureMoveset.MOVESET_SPECIES_ID] =
                    SpeciesEnum.SPECIES__UNK;
                movesetArr[offset + FeatureMoveset.MOVESET_LEGAL] = 1;
            } else {
                const baseSpecies = pokemon.species.baseSpecies.toLowerCase();
                movesetArr[offset + FeatureMoveset.MOVESET_ACTION_ID] =
                    IndexValueFromEnum<typeof ActionsEnum>(
                        "Actions",
                        `switch_${baseSpecies}`,
                    );
                movesetArr[offset + FeatureMoveset.MOVESET_SPECIES_ID] =
                    IndexValueFromEnum<typeof SpeciesEnum>(
                        "Species",
                        baseSpecies,
                    );
                movesetArr[offset + FeatureMoveset.MOVESET_LEGAL] =
                    pokemon.isActive() || pokemon.fainted ? 0 : 1;
            }
            movesetArr[offset + FeatureMoveset.MOVESET_SIDE] = side.n;
            movesetArr[offset + FeatureMoveset.MOVESET_ACTION_TYPE] =
                MovesetActionType.MOVESET_ACTION_TYPE_SWITCH;
            movesetArr[offset + FeatureMoveset.MOVESET_PPUSED] = 0;
            movesetArr[offset + FeatureMoveset.MOVESET_MOVE_ID] =
                MovesEnum.MOVES__NULL;
            offset += numMoveFields;
        }

        for (
            let remainingIndex = side.totalPokemon;
            remainingIndex < 6;
            remainingIndex++
        ) {
            movesetArr[offset + FeatureMoveset.MOVESET_ACTION_TYPE] =
                MovesetActionType.MOVESET_ACTION_TYPE_SWITCH;
            movesetArr[offset + FeatureMoveset.MOVESET_ACTION_ID] =
                ActionsEnum.ACTIONS_SWITCH__NULL;
            movesetArr[offset + FeatureMoveset.MOVESET_PPUSED] = 0;
            movesetArr[offset + FeatureMoveset.MOVESET_SIDE] = side.n;
            movesetArr[offset + FeatureMoveset.MOVESET_LEGAL] = 0;
            movesetArr[offset + FeatureMoveset.MOVESET_MOVE_ID] =
                MovesEnum.MOVES__NULL;
            movesetArr[offset + FeatureMoveset.MOVESET_SPECIES_ID] =
                SpeciesEnum.SPECIES__NULL;
            offset += numMoveFields;
        }

        return movesetArr;
    }

    getMoveset(isStruggling: boolean): Uint8Array {
        const playerIndex = this.player.getPlayerIndex();
        const movesets = [];

        if (playerIndex !== undefined) {
            const mySide = this.player.privateBattle.sides[playerIndex];
            const oppSide = this.player.privateBattle.sides[1 - playerIndex];
            for (const [relativeSideIndex, side] of [
                mySide,
                oppSide,
            ].entries()) {
                const moveset = this.getSideMoveset(
                    side,
                    relativeSideIndex === 0 ? isStruggling : false,
                );
                movesets.push(moveset);
            }
        }

        return new Uint8Array(concatenateArrays(movesets).buffer);
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

        rewards.setHpreward(hpReward);
        rewards.setFaintedreward(faintedReward);

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

        rewards.setWinreward(winReward);
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
            heuristics.setHeuristicaction(
                validIndices[Math.random() * validIndices.length],
            );
        } else {
            heuristics.setHeuristicaction(actionIndex);
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
        info.setGameid(this.player.gameId);
        info.setPlayerindex(!!playerIndex);
        info.setTurn(this.player.privateBattle.turn);
        info.setDone(this.player.done);
        info.setDraw(this.player.draw);
        info.setRequestcount(this.player.requestCount);

        const worldStream = this.player.worldStream;
        if (worldStream !== null) {
            const world = worldStream.battle!;
            info.setSeed(hashArrayToInt32(world.prng.initialSeed));
        }

        const drawRatio = this.player.privateBattle.turn / DRAW_TURNS;
        info.setDrawratio(Math.max(0, Math.min(1, drawRatio)));

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

        const { legalActions, isStruggling } = StateHandler.getLegalActions(
            this.player.privateBattle.request,
        );
        state.setLegalactions(legalActions.buffer);

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
        state.setMoveset(this.getMoveset(isStruggling));

        return state;
    }
}
