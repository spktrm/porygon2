import { AnyObject } from "@pkmn/sim";
import {
    Args,
    BattleMajorArgName,
    BattleMinorArgName,
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
    EdgeTypes,
    FeatureEdge,
    FeatureEdgeMap,
    FeatureEntity,
    FeatureMoveset,
    FeatureWeather,
    MovesetActionType,
} from "../../protos/features_pb";
import {
    MappingLookup,
    EnumKeyMapping,
    EnumMappings,
    Mappings,
    MoveIndex,
    numPokemonFields,
    numMovesetFields,
    numMoveFields,
    numVolatiles,
    NUM_HISTORY,
    numSideConditions,
    numWeatherFields,
    numTypes,
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
type MajorArgNames = RemovePipes<BattleMajorArgName> | "turn";
type MinorArgNames = RemovePipes<BattleMinorArgName>;

const sanitizeKeyCache = new Map<string, string>();

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
    return new POKEMON_ARRAY_CONSTRUCTOR(numPokemonFields);
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

    const volatiles = new OneDBoolean(numVolatiles, POKEMON_ARRAY_CONSTRUCTOR);
    for (const [key] of Object.entries({
        ...pokemon.volatiles,
        ...candidate.volatiles,
    })) {
        const index = IndexValueFromEnum("Volatilestatus", key);
        volatiles.set(index, true);
    }
    dataArr.set(volatiles.buffer, FeatureEntity.ENTITY_VOLATILES0);

    dataArr[FeatureEntity.ENTITY_SIDE] = pokemon.side.n ^ playerIndex;

    dataArr[FeatureEntity.ENTITY_BOOST_ATK_VALUE] = pokemon.boosts.atk ?? 0;
    dataArr[FeatureEntity.ENTITY_BOOST_DEF_VALUE] = pokemon.boosts.def ?? 0;
    dataArr[FeatureEntity.ENTITY_BOOST_SPA_VALUE] = pokemon.boosts.spa ?? 0;
    dataArr[FeatureEntity.ENTITY_BOOST_SPD_VALUE] = pokemon.boosts.spd ?? 0;
    dataArr[FeatureEntity.ENTITY_BOOST_SPE_VALUE] = pokemon.boosts.spe ?? 0;
    dataArr[FeatureEntity.ENTITY_BOOST_EVASION_VALUE] =
        pokemon.boosts.evasion ?? 0;
    dataArr[FeatureEntity.ENTITY_BOOST_ACCURACY_VALUE] =
        pokemon.boosts.accuracy ?? 0;

    const typeChanged = new OneDBoolean(numTypes, POKEMON_ARRAY_CONSTRUCTOR);
    const typechangeVolatile =
        pokemon.volatiles.typechange ?? candidate.volatiles.typechange;
    if (typechangeVolatile) {
        if (typechangeVolatile.apparentType) {
            for (const type of typechangeVolatile.apparentType.split("/")) {
                const index = IndexValueFromEnum("Types", type);
                typeChanged.set(index, true);
            }
        }
    }
    dataArr.set(typeChanged.buffer, FeatureEntity.ENTITY_TYPECHANGE0);

    return dataArr;
}

const numEdgeFeatures = Object.keys(FeatureEdge).length;

class Edge {
    player: Player;

    entityData: Int16Array;
    edgeData: Int16Array;
    sideData: Uint8Array;
    fieldData: Uint8Array;

    constructor(player: Player) {
        this.player = player;

        this.edgeData = new Int16Array(numEdgeFeatures);
        this.entityData = new Int16Array(2 * numPokemonFields);
        this.sideData = new Uint8Array(2 * numSideConditions);
        this.fieldData = new Uint8Array(numWeatherFields);

        this.updateSideConditionData();
        this.updateFieldData();
    }

    clone() {
        const edge = new Edge(this.player);
        edge.edgeData.set(this.edgeData);
        edge.entityData.set(this.entityData);
        return edge;
    }

    setPoke1FromData(data: Int16Array) {
        this.entityData.set(data);
    }

    setPoke1(poke1: Pokemon | null) {
        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }
        const data = getArrayFromPokemon(poke1, playerIndex);
        this.setPoke1FromData(data);
        if (poke1 !== null) {
            this.setFeature(
                FeatureEdge.EDGE_AFFECTING_SIDE,
                poke1.side.n ^ playerIndex,
            );
        }
    }

    setPoke2FromData(data: Int16Array) {
        this.entityData.set(data, numPokemonFields);
    }

    setPoke2(poke2: Pokemon | null) {
        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }
        const data = getArrayFromPokemon(poke2, playerIndex);
        this.setPoke2FromData(data);
    }

    updateSideConditionData() {
        const playerIndex = this.player.getPlayerIndex()!;
        for (const side of this.player.publicBattle.sides) {
            const { n, sideConditions } = side;
            const sideconditionOffset =
                n === playerIndex ? numSideConditions : 0;
            for (const [id, { level }] of Object.entries(sideConditions)) {
                const featureIndex = IndexValueFromEnum<SideconditionEnumMap>(
                    "Sidecondition",
                    id,
                );
                this.sideData[featureIndex + sideconditionOffset] = level;
            }
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
        this.fieldData[FeatureWeather.WEATHER_ID] = weatherIndex;
        this.fieldData[FeatureWeather.MAX_DURATION] =
            field.weatherState.maxDuration;
        this.fieldData[FeatureWeather.MIN_DURATION] =
            field.weatherState.minDuration;
    }

    setFeature(index: FeatureEdgeMap[keyof FeatureEdgeMap], value: number) {
        if (index === undefined) {
            throw new Error("Index cannot be undefined");
        }
        if (value === undefined) {
            throw new Error("Value cannot be undefined");
        }
        this.edgeData[index] = value;
    }

    getFeature(index: FeatureEdgeMap[keyof FeatureEdgeMap]) {
        return this.edgeData[index];
    }

    addMinorArg(argName: MinorArgNames) {
        const index = IndexValueFromEnum("BattleMinorArg", argName);
        this.setFeature(FeatureEdge.MINOR_ARG, index);
    }

    addMajorArg(argName: MajorArgNames) {
        const index = IndexValueFromEnum("BattleMajorArg", argName);
        this.setFeature(FeatureEdge.MAJOR_ARG, index);
    }

    updateEdgeFromOf(effect: Partial<Effect>) {
        const { effectType } = effect;
        if (effectType) {
            const fromTypeToken =
                EffecttypesEnum[
                    `EFFECTTYPES_${effectType.toUpperCase()}` as keyof EffecttypesEnumMap
                ];
            this.setFeature(FeatureEdge.FROM_TYPE_TOKEN, fromTypeToken);
            const fromSourceToken = getEffectToken(effect);
            this.setFeature(FeatureEdge.FROM_SOURCE_TOKEN, fromSourceToken);
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
    entityData: Int16Array;
    edgeData: Int16Array;
    sideData: Uint8Array;
    fieldData: Uint8Array;

    entityCursor: number;
    edgeCursor: number;
    sideCursor: number;
    fieldCursor: number;

    numEdges: number;
    maxEdges: number;

    constructor() {
        const MAX_TURNS = 1000;

        const maxEdges = NUM_HISTORY * MAX_TURNS;
        this.maxEdges = maxEdges;

        this.edgeData = new Int16Array(maxEdges * numEdgeFeatures);
        this.entityData = new Int16Array(maxEdges * 2 * numPokemonFields);
        this.sideData = new Uint8Array(maxEdges * 2 * numSideConditions);
        this.fieldData = new Uint8Array(maxEdges * numWeatherFields);

        const cursorStart = maxEdges - NUM_HISTORY;
        this.edgeCursor = cursorStart * numEdgeFeatures;
        this.entityCursor = cursorStart * 2 * numPokemonFields;
        this.sideCursor = cursorStart * 2 * numSideConditions;
        this.fieldCursor = cursorStart * numWeatherFields;

        this.numEdges = 0;
    }

    addEdge(edge: Edge) {
        this.edgeData.set(edge.edgeData, this.edgeCursor);
        this.edgeCursor -= numEdgeFeatures;

        this.entityData.set(edge.entityData, this.entityCursor);
        this.entityCursor -= 2 * numPokemonFields;

        this.sideData.set(edge.sideData, this.sideCursor);
        this.sideCursor -= 2 * numSideConditions;

        this.fieldData.set(edge.fieldData, this.fieldCursor);
        this.fieldCursor -= numWeatherFields;

        this.numEdges += 1;
    }

    getHistory(numHistory: number = NUM_HISTORY) {
        const history = new History();
        const width = Math.max(1, Math.min(this.numEdges, numHistory));
        const trueWidth = width + 1;
        history.setEdges(
            new Uint8Array(
                this.edgeData.slice(
                    this.edgeCursor + numEdgeFeatures,
                    this.edgeCursor + trueWidth * numEdgeFeatures,
                ).buffer,
            ),
        );
        history.setEntities(
            new Uint8Array(
                this.entityData.slice(
                    this.entityCursor + 2 * numPokemonFields,
                    this.entityCursor + trueWidth * 2 * numPokemonFields,
                ).buffer,
            ),
        );
        history.setSideconditions(
            this.sideData.slice(
                this.sideCursor + 2 * numSideConditions,
                this.sideCursor + trueWidth * 2 * numSideConditions,
            ),
        );
        history.setField(
            this.fieldData.slice(
                this.fieldCursor + numWeatherFields,
                this.fieldCursor + trueWidth * numWeatherFields,
            ),
        );
        history.setLength(width);
        return history;
    }
}
export class EventHandler implements Protocol.Handler {
    readonly player: Player;

    currHp: Map<string, number>;
    actives: Map<ID, PokemonIdent>;
    actionWindow: { [k: number]: number[] };
    turnOrder: number;
    turnNum: number;

    majorEdgeBuffer: EdgeBuffer;
    minorEdgeBuffer: EdgeBuffer;

    constructor(player: Player) {
        this.player = player;
        this.currHp = new Map();
        this.actives = new Map();
        this.actionWindow = {
            0: [],
            1: [],
        };
        this.majorEdgeBuffer = new EdgeBuffer();
        this.minorEdgeBuffer = new EdgeBuffer();
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
        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex !== undefined) {
            edge.setFeature(FeatureEdge.PLAYER_ID, playerIndex);
        }
        edge.setFeature(FeatureEdge.REQUEST_COUNT, this.player.requestCount);
        edge.setFeature(FeatureEdge.EDGE_VALID, 1);
        edge.setFeature(FeatureEdge.EDGE_INDEX, this.majorEdgeBuffer.numEdges);
        edge.setFeature(FeatureEdge.TURN_ORDER_VALUE, this.turnOrder);
        this.turnOrder += 1;
        edge.setFeature(FeatureEdge.TURN_VALUE, this.turnNum);
        return edge;
    }

    addMajorEdge(edge: Edge) {
        const preprocessedEdge = this._preprocessEdge(edge);
        this.majorEdgeBuffer.addEdge(preprocessedEdge);
    }

    addMinorEdge(edge: Edge) {
        const preprocessedEdge = this._preprocessEdge(edge);
        this.minorEdgeBuffer.addEdge(preprocessedEdge);
    }

    "|move|"(args: Args["|move|"]) {
        const [argName, poke1Ident, moveId, poke2Ident] = args;

        const poke1 = this.getPokemon(poke1Ident)!;

        this.actionWindow[poke1.side.n].push(0);

        const move = this.getMove(moveId);
        const actionIndex = IndexValueFromEnum<typeof ActionsEnum>(
            "Actions",
            `move_${move.id}`,
        );

        const poke2 = this.getPokemon(poke2Ident as PokemonIdent);

        const edge = new Edge(this.player);
        edge.addMajorArg(argName);
        edge.setPoke1(poke1);
        edge.setPoke2(poke2);

        edge.setFeature(FeatureEdge.ACTION_TOKEN, actionIndex);
        edge.setFeature(FeatureEdge.EDGE_TYPE_TOKEN, EdgeTypes.MOVE_EDGE);

        this.addMajorEdge(edge);
    }

    "|drag|"(args: Args["|drag|"]) {
        this.handleSwitch(args);
    }

    "|switch|"(args: Args["|switch|"]) {
        this.handleSwitch(args);
    }

    handleSwitch(args: Args["|switch|" | "|drag|"]) {
        const [argName, poke1Ident] = args;

        const poke1 = this.getPokemon(poke1Ident)!;
        const actionIndex = IndexValueFromEnum<typeof ActionsEnum>(
            "Actions",
            `switch_${poke1.species.baseSpecies.toLowerCase()}`,
        );

        if (argName === "switch") {
            this.actionWindow[poke1.side.n].push(1);
        }

        const edge = new Edge(this.player);
        edge.addMajorArg(argName);
        edge.setPoke1(poke1);
        edge.setFeature(FeatureEdge.ACTION_TOKEN, actionIndex);

        edge.setFeature(FeatureEdge.EDGE_TYPE_TOKEN, EdgeTypes.SWITCH_EDGE);

        this.addMajorEdge(edge);
    }

    "|cant|"(args: Args["|cant|"]) {
        const [argName, poke1Ident, conditionId, moveId] = args;

        const edge = new Edge(this.player);

        const poke1 = this.getPokemon(poke1Ident)!;

        if (moveId) {
            const move = this.getMove(moveId);
            const actionIndex = IndexValueFromEnum<typeof ActionsEnum>(
                "Actions",
                `move_${move.id}`,
            );
            edge.setFeature(FeatureEdge.ACTION_TOKEN, actionIndex);
        }

        const condition = this.getCondition(conditionId);

        edge.addMajorArg(argName);
        edge.setPoke1(poke1);
        edge.setFeature(FeatureEdge.EDGE_TYPE_TOKEN, EdgeTypes.CANT_EDGE);
        edge.updateEdgeFromOf(condition);

        this.addMajorEdge(edge);
    }

    "|faint|"(args: Args["|faint|"]) {
        const [argName, poke1Ident] = args;

        const poke1 = this.getPokemon(poke1Ident)!;

        const edge = new Edge(this.player);

        edge.addMajorArg(argName);
        edge.setPoke1(poke1);
        edge.setFeature(FeatureEdge.EDGE_TYPE_TOKEN, EdgeTypes.EFFECT_EDGE);

        this.addMajorEdge(edge);
    }

    "|-fail|"(args: Args["|-fail|"], kwArgs: KWArgs["|-fail|"]) {
        const [argName, poke1Ident] = args;

        const poke1 = this.getPokemon(poke1Ident)!;

        const edge = new Edge(this.player);

        const fromEffect = this.getCondition(kwArgs.from);

        if (kwArgs.of) {
            const poke2 = this.getPokemon(kwArgs.of)!;
            edge.setPoke2(poke2);
        }

        edge.addMinorArg(argName);
        edge.setPoke1(poke1);
        edge.setFeature(FeatureEdge.EDGE_TYPE_TOKEN, EdgeTypes.EFFECT_EDGE);
        edge.updateEdgeFromOf(fromEffect);

        this.addMinorEdge(edge);
    }

    "|-block|"(args: Args["|-block|"], kwArgs: KWArgs["|-block|"]) {
        const [argName, poke1Ident] = args;

        const poke1 = this.getPokemon(poke1Ident)!;

        const edge = new Edge(this.player);

        const fromEffect = this.getCondition(kwArgs.from);

        if (kwArgs.of) {
            const poke2 = this.getPokemon(kwArgs.of)!;
            edge.setPoke2(poke2);
        }

        edge.addMinorArg(argName);
        edge.setPoke1(poke1);
        edge.setFeature(FeatureEdge.EDGE_TYPE_TOKEN, EdgeTypes.EFFECT_EDGE);
        edge.updateEdgeFromOf(fromEffect);

        this.addMinorEdge(edge);
    }

    "|-notarget|"(args: Args["|-notarget|"]) {
        const [argName, poke1Ident] = args;

        const edge = new Edge(this.player);

        if (poke1Ident) {
            const poke1 = this.getPokemon(poke1Ident)!;
            edge.setPoke1(poke1);
        }

        edge.addMinorArg(argName);
        edge.setFeature(FeatureEdge.EDGE_TYPE_TOKEN, EdgeTypes.EFFECT_EDGE);

        this.addMinorEdge(edge);
    }

    "|-miss|"(args: Args["|-miss|"], kwArgs: KWArgs["|-miss|"]) {
        const [argName, poke1Ident] = args;

        const poke1 = this.getPokemon(poke1Ident)!;
        const fromEffect = this.getCondition(kwArgs.from);

        const edge = new Edge(this.player);

        edge.addMinorArg(argName);
        edge.setPoke1(poke1);
        edge.setFeature(FeatureEdge.EDGE_TYPE_TOKEN, EdgeTypes.EFFECT_EDGE);
        edge.updateEdgeFromOf(fromEffect);

        this.addMinorEdge(edge);
    }

    "|-damage|"(args: Args["|-damage|"], kwArgs: KWArgs["|-damage|"]) {
        const [argName, poke1Ident] = args;

        const edge = new Edge(this.player);

        const poke1 = this.getPokemon(poke1Ident)!;
        const true1Ident = poke1.originalIdent;

        if (kwArgs.of) {
            const poke2 = this.getPokemon(kwArgs.of)!;
            edge.setPoke2(poke2);
        }

        if (kwArgs.from) {
            const fromEffect = this.getCondition(kwArgs.from);
            edge.updateEdgeFromOf(fromEffect);
        }

        if (!this.currHp.has(true1Ident)) {
            this.currHp.set(true1Ident, 1);
        }
        const prevHp = this.currHp.get(true1Ident) ?? 1;
        const currHp = poke1.hp / poke1.maxhp;
        const diffRatio = currHp - prevHp;
        this.currHp.set(true1Ident, currHp);

        edge.addMinorArg(argName);
        edge.setPoke1(poke1);
        edge.setFeature(
            FeatureEdge.DAMAGE_RATIO,
            Math.abs(Math.floor(31 * diffRatio)),
        );
        edge.setFeature(FeatureEdge.EDGE_TYPE_TOKEN, EdgeTypes.EFFECT_EDGE);

        this.addMinorEdge(edge);
    }

    "|-heal|"(args: Args["|-heal|"], kwArgs: KWArgs["|-heal|"]) {
        const [argName, poke1Ident] = args;

        const edge = new Edge(this.player);

        const poke1 = this.getPokemon(poke1Ident)!;
        const trueIdent = poke1.originalIdent;

        if (kwArgs.of) {
            const poke2 = this.getPokemon(kwArgs.of)!;
            edge.setPoke2(poke2);
        }

        const fromEffect = this.getCondition(kwArgs.from);

        if (!this.currHp.has(trueIdent)) {
            this.currHp.set(trueIdent, 1);
        }
        const prevHp = this.currHp.get(trueIdent) ?? 1;
        const currHp = poke1.hp / poke1.maxhp;
        const diffRatio = currHp - prevHp;
        this.currHp.set(trueIdent, currHp);

        edge.addMinorArg(argName);
        edge.setPoke1(poke1);
        edge.setFeature(
            FeatureEdge.HEAL_RATIO,
            Math.abs(Math.floor(31 * diffRatio)),
        );
        edge.setFeature(FeatureEdge.EDGE_TYPE_TOKEN, EdgeTypes.EFFECT_EDGE);
        edge.updateEdgeFromOf(fromEffect);

        this.addMinorEdge(edge);
    }

    "|-status|"(args: Args["|-status|"], kwArgs: KWArgs["|-status|"]) {
        const [argName, poke1Ident, statusId] = args;

        const fromEffect = this.getCondition(kwArgs.from);
        const statusToken = IndexValueFromEnum("Status", statusId);

        const poke1 = this.getPokemon(poke1Ident)!;

        const edge = new Edge(this.player);

        if (kwArgs.of) {
            const poke2 = this.getPokemon(kwArgs.of)!;
            edge.setPoke2(poke2);
        }

        edge.addMinorArg(argName);
        edge.updateEdgeFromOf(fromEffect);
        edge.setPoke1(poke1);
        edge.setFeature(FeatureEdge.EDGE_TYPE_TOKEN, EdgeTypes.EFFECT_EDGE);
        edge.setFeature(FeatureEdge.STATUS_TOKEN, statusToken);

        this.addMinorEdge(edge);
    }

    "|-curestatus|"(args: Args["|-curestatus|"]) {
        const [argName, poke1Ident, statusId] = args;

        const poke1 = this.getPokemon(poke1Ident)!;

        const statusIndex = IndexValueFromEnum("Status", statusId);

        const edge = new Edge(this.player);
        edge.addMinorArg(argName);
        edge.setPoke1(poke1);
        edge.setFeature(FeatureEdge.EDGE_TYPE_TOKEN, EdgeTypes.EFFECT_EDGE);
        edge.setFeature(FeatureEdge.STATUS_TOKEN, statusIndex);

        this.addMinorEdge(edge);
    }

    "|-cureteam|"(args: Args["|-cureteam|"]) {
        const [argName, poke1Ident] = args;

        const poke1 = this.getPokemon(poke1Ident)!;

        const edge = new Edge(this.player);
        edge.addMinorArg(argName);
        edge.setPoke1(poke1);
        edge.setFeature(FeatureEdge.EDGE_TYPE_TOKEN, EdgeTypes.EFFECT_EDGE);

        this.addMinorEdge(edge);
    }

    static getStatBoostEdgeFeatureIndex(
        stat: BoostID,
    ): FeatureEdgeMap[`BOOST_${Uppercase<BoostID>}_VALUE`] {
        return FeatureEdge[
            `BOOST_${stat.toLocaleUpperCase()}_VALUE` as `BOOST_${Uppercase<BoostID>}_VALUE`
        ];
    }

    "|-boost|"(args: Args["|-boost|"], kwArgs: KWArgs["|-boost|"]) {
        const [argName, poke1Ident, stat, value] = args;

        const poke1 = this.getPokemon(poke1Ident)!;

        const featureIndex = EventHandler.getStatBoostEdgeFeatureIndex(stat);

        const edge = new Edge(this.player);

        if (kwArgs.of) {
            const poke2 = this.getPokemon(kwArgs.of)!;
            edge.setPoke2(poke2);
        }

        edge.setPoke1(poke1);
        edge.setFeature(FeatureEdge.EDGE_TYPE_TOKEN, EdgeTypes.EFFECT_EDGE);
        edge.setFeature(featureIndex, parseInt(value));
        edge.addMinorArg(argName);

        this.addMinorEdge(edge);
    }

    "|-unboost|"(args: Args["|-unboost|"], kwArgs: KWArgs["|-unboost|"]) {
        const [argName, poke1Ident, stat, value] = args;

        const poke1 = this.getPokemon(poke1Ident)!;

        const featureIndex = EventHandler.getStatBoostEdgeFeatureIndex(stat);

        const edge = new Edge(this.player);

        if (kwArgs.of) {
            const poke2 = this.getPokemon(kwArgs.of)!;
            edge.setPoke2(poke2);
        }

        edge.setPoke1(poke1);
        edge.setFeature(FeatureEdge.EDGE_TYPE_TOKEN, EdgeTypes.EFFECT_EDGE);
        edge.setFeature(featureIndex, -parseInt(value));
        edge.addMinorArg(argName);

        this.addMinorEdge(edge);
    }

    "|-setboost|"(args: Args["|-setboost|"], kwArgs: KWArgs["|-setboost|"]) {
        const [argName, poke1Ident, stat, value] = args;

        const poke1 = this.getPokemon(poke1Ident)!;

        const featureIndex = EventHandler.getStatBoostEdgeFeatureIndex(stat);
        const fromEffect = this.getCondition(kwArgs.from);

        const edge = new Edge(this.player);

        if (kwArgs.of) {
            const poke2 = this.getPokemon(kwArgs.of)!;
            edge.setPoke2(poke2);
        }

        edge.setPoke1(poke1);
        edge.setFeature(FeatureEdge.EDGE_TYPE_TOKEN, EdgeTypes.EFFECT_EDGE);
        edge.setFeature(featureIndex, parseInt(value));
        edge.updateEdgeFromOf(fromEffect);
        edge.addMinorArg(argName);

        this.addMinorEdge(edge);
    }

    "|-swapboost|"(args: Args["|-swapboost|"], kwArgs: KWArgs["|-swapboost|"]) {
        const [argName, poke1Ident] = args;

        const edge = new Edge(this.player);

        const poke1 = this.getPokemon(poke1Ident)!;

        const fromEffect = this.getCondition(kwArgs.from);

        if (kwArgs.of) {
            const poke2 = this.getPokemon(kwArgs.of)!;
            edge.setPoke2(poke2);
        }

        edge.addMinorArg(argName);
        edge.setFeature(FeatureEdge.EDGE_TYPE_TOKEN, EdgeTypes.EFFECT_EDGE);
        edge.setPoke1(poke1);
        edge.updateEdgeFromOf(fromEffect);

        this.addMinorEdge(edge);
    }

    "|-invertboost|"(
        args: Args["|-invertboost|"],
        kwArgs: KWArgs["|-invertboost|"],
    ) {
        const [argName, poke1Ident] = args;

        const edge = new Edge(this.player);

        const poke1 = this.getPokemon(poke1Ident)!;

        const fromEffect = this.getCondition(kwArgs.from);

        if (kwArgs.of) {
            const poke2 = this.getPokemon(kwArgs.of)!;
            edge.setPoke2(poke2);
        }

        edge.addMinorArg(argName);
        edge.setFeature(FeatureEdge.EDGE_TYPE_TOKEN, EdgeTypes.EFFECT_EDGE);
        edge.setPoke1(poke1);
        edge.updateEdgeFromOf(fromEffect);

        this.addMinorEdge(edge);
    }

    "|-clearboost|"(
        args: Args["|-clearboost|"],
        kwArgs: KWArgs["|-clearboost|"],
    ) {
        const [argName, poke1Ident] = args;

        const edge = new Edge(this.player);

        const poke1 = this.getPokemon(poke1Ident)!;

        const fromEffect = this.getCondition(kwArgs.from);

        if (kwArgs.of) {
            const poke2 = this.getPokemon(kwArgs.of)!;
            edge.setPoke2(poke2);
        }

        edge.addMinorArg(argName);
        edge.setFeature(FeatureEdge.EDGE_TYPE_TOKEN, EdgeTypes.EFFECT_EDGE);
        edge.setPoke1(poke1);
        edge.updateEdgeFromOf(fromEffect);

        this.addMinorEdge(edge);
    }

    "|-clearnegativeboost|"(
        args: Args["|-clearnegativeboost|"],
        kwArgs: KWArgs["|-clearnegativeboost|"],
    ) {
        const [argName, poke1Ident] = args;

        const edge = new Edge(this.player);

        const poke1 = this.getPokemon(poke1Ident)!;

        const fromEffect = this.getCondition(kwArgs.from);

        if (kwArgs.of) {
            const poke2 = this.getPokemon(kwArgs.of)!;
            edge.setPoke2(poke2);
        }

        edge.addMinorArg(argName);
        edge.setFeature(FeatureEdge.EDGE_TYPE_TOKEN, EdgeTypes.EFFECT_EDGE);
        edge.setPoke1(poke1);
        edge.updateEdgeFromOf(fromEffect);

        this.addMinorEdge(edge);
    }

    "|-copyboost|"() {}

    "|-weather|"(args: Args["|-weather|"]) {
        const [argName, weatherId] = args;

        const fromEffect = this.getCondition(weatherId);

        const edge = new Edge(this.player);
        edge.addMinorArg(argName);
        edge.setFeature(FeatureEdge.EDGE_TYPE_TOKEN, EdgeTypes.EFFECT_EDGE);
        edge.setFeature(FeatureEdge.EDGE_AFFECTING_SIDE, 2);
        edge.updateEdgeFromOf(fromEffect);

        this.addMinorEdge(edge);
    }

    "|-fieldstart|"(
        args: Args["|-fieldstart|"],
        kwArgs: KWArgs["|-fieldstart|"],
    ) {
        const [argName] = args;

        const edge = new Edge(this.player);
        edge.addMinorArg(argName);

        if (kwArgs.of) {
            /* empty */
        }

        edge.setFeature(FeatureEdge.EDGE_TYPE_TOKEN, EdgeTypes.EFFECT_EDGE);
        edge.setFeature(FeatureEdge.EDGE_AFFECTING_SIDE, 2);

        this.addMinorEdge(edge);
    }

    "|-fieldend|"(args: Args["|-fieldend|"], kwArgs: KWArgs["|-fieldend|"]) {
        const [argName] = args;

        const edge = new Edge(this.player);
        edge.addMinorArg(argName);

        if (kwArgs.of) {
            /* empty */
        }

        edge.setFeature(FeatureEdge.EDGE_TYPE_TOKEN, EdgeTypes.EFFECT_EDGE);
        edge.setFeature(FeatureEdge.EDGE_AFFECTING_SIDE, 2);

        this.addMinorEdge(edge);
    }

    "|-sidestart|"(args: Args["|-sidestart|"]) {
        const [argName, sideId, conditionId] = args;

        const side = this.getSide(sideId);
        const fromEffect = this.getCondition(conditionId);

        const edge = new Edge(this.player);
        edge.addMinorArg(argName);
        edge.setFeature(FeatureEdge.EDGE_AFFECTING_SIDE, side.n);
        edge.setFeature(FeatureEdge.EDGE_TYPE_TOKEN, EdgeTypes.EFFECT_EDGE);
        edge.updateEdgeFromOf(fromEffect);
        this.addMinorEdge(edge);
    }

    "|-sideend|"(args: Args["|-sideend|"]) {
        const [argName, sideId, conditionId] = args;

        const side = this.getSide(sideId);
        const fromEffect = this.getCondition(conditionId);

        const edge = new Edge(this.player);
        edge.addMinorArg(argName);
        edge.setFeature(FeatureEdge.EDGE_AFFECTING_SIDE, side.n);
        edge.setFeature(FeatureEdge.EDGE_TYPE_TOKEN, EdgeTypes.EFFECT_EDGE);
        edge.updateEdgeFromOf(fromEffect);
        this.addMinorEdge(edge);
    }

    "|-swapsideconditions|"() {}

    "|-start|"(args: Args["|-start|"]) {
        const [argName, poke1Ident] = args;

        const poke1 = this.getPokemon(poke1Ident)!;

        const edge = new Edge(this.player);
        edge.addMinorArg(argName);
        edge.setPoke1(poke1);
        edge.setFeature(FeatureEdge.EDGE_TYPE_TOKEN, EdgeTypes.EFFECT_EDGE);

        this.addMinorEdge(edge);
    }

    "|-end|"(args: Args["|-end|"]) {
        const [argName, poke1Ident] = args;

        const poke1 = this.getPokemon(poke1Ident)!;

        const edge = new Edge(this.player);
        edge.addMinorArg(argName);
        edge.setPoke1(poke1);
        edge.setFeature(FeatureEdge.EDGE_TYPE_TOKEN, EdgeTypes.EFFECT_EDGE);

        this.addMinorEdge(edge);
    }

    "|-crit|"(args: Args["|-crit|"]) {
        const [argName, poke1Ident] = args;

        const poke1 = this.getPokemon(poke1Ident)!;

        const edge = new Edge(this.player);
        edge.addMinorArg(argName);
        edge.setPoke1(poke1);
        edge.setFeature(FeatureEdge.EDGE_TYPE_TOKEN, EdgeTypes.EFFECT_EDGE);

        this.addMinorEdge(edge);
    }

    "|-supereffective|"(args: Args["|-supereffective|"]) {
        const [argName, poke1Ident] = args;

        const poke1 = this.getPokemon(poke1Ident)!;

        const edge = new Edge(this.player);
        edge.addMinorArg(argName);
        edge.setPoke1(poke1);
        edge.setFeature(FeatureEdge.EDGE_TYPE_TOKEN, EdgeTypes.EFFECT_EDGE);

        this.addMinorEdge(edge);
    }

    "|-resisted|"(args: Args["|-resisted|"]) {
        const [argName, poke1Ident] = args;

        const poke1 = this.getPokemon(poke1Ident)!;

        const edge = new Edge(this.player);
        edge.addMinorArg(argName);
        edge.setPoke1(poke1);
        edge.setFeature(FeatureEdge.EDGE_TYPE_TOKEN, EdgeTypes.EFFECT_EDGE);

        this.addMinorEdge(edge);
    }

    "|-immune|"(args: Args["|-immune|"], kwArgs: KWArgs["|-immune|"]) {
        const [argName, poke1Ident] = args;

        const poke1 = this.getPokemon(poke1Ident)!;

        const edge = new Edge(this.player);

        const fromEffect = this.getCondition(kwArgs.from);

        if (kwArgs.of) {
            const poke2 = this.getPokemon(kwArgs.of)!;
            edge.setPoke2(poke2);
        }

        edge.addMinorArg(argName);
        edge.setPoke1(poke1);
        edge.setFeature(FeatureEdge.EDGE_TYPE_TOKEN, EdgeTypes.EFFECT_EDGE);
        edge.updateEdgeFromOf(fromEffect);

        this.addMinorEdge(edge);
    }

    "|-item|"(args: Args["|-item|"], kwArgs: KWArgs["|-item|"]) {
        const [argName, poke1Ident, itemId] = args;

        const poke1 = this.getPokemon(poke1Ident)!;

        const itemIndex = IndexValueFromEnum("Items", itemId);
        const fromEffect = this.getCondition(kwArgs.from);

        const edge = new Edge(this.player);

        if (kwArgs.of) {
            const poke2 = this.getPokemon(kwArgs.of)!;
            edge.setPoke2(poke2);
        }

        edge.setPoke1(poke1);
        edge.setFeature(FeatureEdge.ITEM_TOKEN, itemIndex);
        edge.setFeature(FeatureEdge.EDGE_TYPE_TOKEN, EdgeTypes.EFFECT_EDGE);
        edge.updateEdgeFromOf(fromEffect);
        edge.addMinorArg(argName);
        this.addMinorEdge(edge);
    }

    "|-enditem|"(args: Args["|-enditem|"], kwArgs: KWArgs["|-enditem|"]) {
        const [argName, poke1Ident, itemId] = args;

        const poke1 = this.getPokemon(poke1Ident)!;

        const itemIndex = IndexValueFromEnum("Items", itemId);
        const fromEffect = this.getCondition(kwArgs.from);

        const edge = new Edge(this.player);
        edge.setPoke1(poke1);
        edge.setFeature(FeatureEdge.ITEM_TOKEN, itemIndex);
        edge.updateEdgeFromOf(fromEffect);
        edge.setFeature(FeatureEdge.EDGE_TYPE_TOKEN, EdgeTypes.EFFECT_EDGE);
        edge.addMinorArg(argName);
        this.addMinorEdge(edge);
    }

    "|-ability|"(args: Args["|-ability|"], kwArgs: KWArgs["|-ability|"]) {
        const [argName, poke1Ident, abilityId] = args;

        const poke1 = this.getPokemon(poke1Ident)!;

        const abilityIndex = IndexValueFromEnum("Ability", abilityId);
        const fromEffect = this.getCondition(kwArgs.from);

        const edge = new Edge(this.player);
        edge.setPoke1(poke1);
        edge.setFeature(FeatureEdge.ABILITY_TOKEN, abilityIndex);
        edge.setFeature(FeatureEdge.EDGE_TYPE_TOKEN, EdgeTypes.EFFECT_EDGE);
        edge.updateEdgeFromOf(fromEffect);
        edge.addMinorArg(argName);
        this.addMinorEdge(edge);
    }

    "|-endability|"(
        args: Args["|-endability|"],
        kwArgs: KWArgs["|-endability|"],
    ) {
        const [argName, poke1Ident] = args;

        const poke1 = this.getPokemon(poke1Ident)!;

        const fromEffect = this.getCondition(kwArgs.from);

        const edge = new Edge(this.player);
        edge.setPoke1(poke1);
        edge.setFeature(FeatureEdge.EDGE_TYPE_TOKEN, EdgeTypes.EFFECT_EDGE);
        edge.updateEdgeFromOf(fromEffect);
        edge.addMinorArg(argName);
        this.addMinorEdge(edge);
    }

    "|-transform|"() {}

    "|-mega|"() {}

    "|-primal|"() {}

    "|-burst|"() {}

    "|-zpower|"() {}

    "|-zbroken|"() {}

    "|-activate|"(args: Args["|-activate|"], kwArgs: KWArgs["|-activate|"]) {
        const [argName, poke1Ident] = args;

        const edge = new Edge(this.player);

        if (poke1Ident) {
            const poke1 = this.getPokemon(poke1Ident)!;

            edge.setPoke1(poke1);
        }

        const fromEffect = this.getCondition(kwArgs.from);

        edge.updateEdgeFromOf(fromEffect);
        edge.addMinorArg(argName);
        edge.setFeature(FeatureEdge.EDGE_TYPE_TOKEN, EdgeTypes.EFFECT_EDGE);

        this.addMinorEdge(edge);
    }

    "|-mustrecharge|"(args: Args["|-mustrecharge|"]) {
        const [argName, poke1Ident] = args;
        const poke1 = this.getPokemon(poke1Ident)!;

        const edge = new Edge(this.player);

        edge.setPoke1(poke1);
        edge.setFeature(FeatureEdge.EDGE_TYPE_TOKEN, EdgeTypes.EFFECT_EDGE);
        edge.addMinorArg(argName);

        this.addMinorEdge(edge);
    }

    "|-prepare|"(args: Args["|-prepare|"]) {
        const [argName, poke1Ident] = args;
        const poke1 = this.getPokemon(poke1Ident)!;

        const edge = new Edge(this.player);

        edge.setPoke1(poke1);
        edge.setFeature(FeatureEdge.EDGE_TYPE_TOKEN, EdgeTypes.EFFECT_EDGE);
        edge.addMinorArg(argName);

        this.addMinorEdge(edge);
    }

    "|-hitcount|"(args: Args["|-hitcount|"]) {
        const [argName, poke1Ident] = args;
        const poke1 = this.getPokemon(poke1Ident)!;

        const edge = new Edge(this.player);

        edge.setPoke1(poke1);
        edge.setFeature(FeatureEdge.EDGE_TYPE_TOKEN, EdgeTypes.EFFECT_EDGE);
        edge.addMinorArg(argName);

        this.addMinorEdge(edge);
    }

    "|done|"() {}

    "|start|"() {
        const edge = new Edge(this.player);
        edge.addMajorArg("turn");
        edge.setFeature(FeatureEdge.EDGE_TYPE_TOKEN, EdgeTypes.EDGE_TYPE_START);

        this.turnOrder = 0;
        this.addMajorEdge(edge);
    }

    "|turn|"(args: Args["|turn|"]) {
        const [argName, turnNum] = args;

        const edge = new Edge(this.player);
        edge.addMajorArg(argName);
        edge.setFeature(FeatureEdge.EDGE_TYPE_TOKEN, EdgeTypes.EDGE_TYPE_START);

        this.turnOrder = 0;
        this.turnNum = parseInt(turnNum);
        this.addMajorEdge(edge);
    }

    reset() {
        this.currHp = new Map();
        this.actives = new Map();
        this.actionWindow = {
            0: [],
            1: [],
        };
    }
}

export class StateHandler {
    player: Player;

    constructor(player: Player) {
        this.player = player;
    }

    static getLegalActions(request?: AnyObject | null): OneDBoolean {
        const legalActions = new OneDBoolean(10, Uint8Array);

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
        return legalActions;
    }

    getSideMoveset(side: Side) {
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

        if (active !== null) {
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
                        movesetArr[offset + FeatureMoveset.MOVESET_EST_DAMAGE] =
                            moveDamage;
                        // eslint-disable-next-line @typescript-eslint/no-unused-vars
                    } catch (err) {
                        /* empty */
                    }
                } else {
                    movesetArr[offset + FeatureMoveset.MOVESET_EST_DAMAGE] = 0;
                }

                movesetArr[offset + FeatureMoveset.MOVESET_MOVE_ID] =
                    IndexValueFromEnum("Move", id);
                movesetArr[offset + FeatureMoveset.MOVESET_SPECIES_ID] =
                    SpeciesEnum.SPECIES__NULL;
                offset += numMoveFields;
            }
            for (
                let remainingIndex = moveSlots.length;
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

    getMoveset(): Uint8Array {
        const playerIndex = this.player.getPlayerIndex();
        const movesets = [];

        if (playerIndex !== undefined) {
            const mySide = this.player.privateBattle.sides[playerIndex];
            const oppSide = this.player.privateBattle.sides[1 - playerIndex];
            for (const side of [mySide, oppSide]) {
                const moveset = this.getSideMoveset(side);
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

    getMajorHistory(numHistory: number = NUM_HISTORY) {
        return this.player.eventHandler.majorEdgeBuffer.getHistory(numHistory);
    }

    getMinorHistory(numHistory: number = NUM_HISTORY) {
        return this.player.eventHandler.minorEdgeBuffer.getHistory(numHistory);
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
            const legalActions = StateHandler.getLegalActions(
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

        const legalActions = StateHandler.getLegalActions(
            this.player.privateBattle.request,
        );
        state.setLegalactions(legalActions.buffer);

        const majorHistory = this.getMajorHistory(numHistory);
        state.setMajorhistory(majorHistory);

        const minorHistory = this.getMinorHistory(numHistory);
        state.setMinorhistory(minorHistory);

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
