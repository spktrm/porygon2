import { AnyObject, Battle } from "@pkmn/sim";
import {
    Args,
    BattleMajorArgName,
    BattleMinorArgName,
    KWArgs,
    PokemonIdent,
    Protocol,
} from "@pkmn/protocol";
import { Info, State } from "../../protos/state_pb";
import {
    AbilitiesEnum,
    BattlemajorargsEnum,
    BattleminorargsEnum,
    EffectEnum,
    EffecttypesEnum,
    EffecttypesEnumMap,
    GendernameEnum,
    ItemeffecttypesEnum,
    ItemsEnum,
    MovesEnum,
    SideconditionEnumMap,
    SpeciesEnum,
    StatusEnum,
    VolatilestatusEnum,
    WeatherEnum,
} from "../../protos/enums_pb";
import {
    EdgeTypes,
    FeatureEdge,
    FeatureEdgeMap,
    FeatureEntity,
    FeatureMoveset,
    FeatureWeather,
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
    numBattleMinorArgs,
    numBattleMajorArgs,
    numVolatiles,
    NUM_HISTORY,
    numSideConditions,
    numPseudoweathers,
    numWeatherFields,
} from "./data";
import { Field, NA, Pokemon, Side } from "@pkmn/client";
import { OneDBoolean } from "./arr";
import { StreamHandler } from "./handler";
import { Ability, Item, Move, BoostID } from "@pkmn/dex-types";
import { ID } from "@pkmn/types";
import { Condition, Effect } from "@pkmn/data";
import { History } from "../../protos/history_pb";
import { getEvalAction } from "./eval";

type RemovePipes<T extends string> = T extends `|${infer U}|` ? U : T;
type MajorArgNames = RemovePipes<BattleMajorArgName>;
type MinorArgNames = RemovePipes<BattleMinorArgName>;

const featureEdgeSize = Object.keys(FeatureEdge).length;
const featureNodeSize = Object.keys(FeatureEntity).length;

const flatEdgeDataSize = 20 * featureEdgeSize;
const flatNodeDataSize = 12 * featureNodeSize;

const sanitizeKeyCache = new Map<string, string>();

function SanitizeKey(key: string): string {
    if (sanitizeKeyCache.has(key)) {
        return sanitizeKeyCache.get(key)!;
    }
    const sanitizedKey = key.replace(/\W/g, "").toLowerCase();
    sanitizeKeyCache.set(key, sanitizedKey);
    return sanitizedKey;
}

function IndexValueFromEnum<T extends EnumMappings>(
    mappingType: Mappings,
    key: string,
): T[keyof T] {
    const mapping = MappingLookup[mappingType] as T;
    const enumMapping = EnumKeyMapping[mappingType];
    const sanitizedKey = SanitizeKey(key);
    const trueKey = enumMapping[sanitizedKey] as keyof T;
    const value = mapping[trueKey];
    if (value === undefined) {
        throw new Error(`${key.toString()} not in mapping`);
    }
    return value;
}

type TypedArray =
    | Uint8Array
    | Int8Array
    | Uint16Array
    | Int16Array
    | Uint32Array
    | Int32Array
    | Float32Array
    | Float64Array;

function concatenateArrays<T extends TypedArray>(arrays: T[]): T {
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

function getBlankPokemonArr() {
    return new Int16Array(numPokemonFields);
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
    data[FeatureEntity.ENTITY_HP_TOKEN] = 1023;
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

function getArrayFromPokemon(candidate: Pokemon | null): Int16Array {
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

    const moveSlots = pokemon.moveSlots.slice(-4);
    const moveIds = [];
    const movePps = [];
    if (moveSlots) {
        for (let move of moveSlots) {
            const { id, ppUsed } = move;
            const maxPP = pokemon.side.battle.gens.dex.moves.get(id).pp;
            const idValue = IndexValueFromEnum<typeof MovesEnum>("Move", id);

            moveIds.push(idValue);
            movePps.push((1024 * (isNaN(ppUsed) ? +!!ppUsed : ppUsed)) / maxPP);
        }
    }
    let remainingIndex: MoveIndex = moveSlots.length as MoveIndex;
    for (remainingIndex; remainingIndex < 4; remainingIndex++) {
        moveIds.push(MovesEnum.MOVES__UNK);
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
    dataArr[FeatureEntity.ENTITY_ABILITY] = ability
        ? IndexValueFromEnum<typeof AbilitiesEnum>("Ability", ability)
        : AbilitiesEnum.ABILITIES__UNK;
    dataArr[FeatureEntity.ENTITY_GENDER] = IndexValueFromEnum<
        typeof GendernameEnum
    >("Gender", pokemon.gender);
    dataArr[FeatureEntity.ENTITY_ACTIVE] = pokemon.isActive() ? 1 : 0;
    dataArr[FeatureEntity.ENTITY_FAINTED] = pokemon.fainted ? 1 : 0;
    dataArr[FeatureEntity.ENTITY_HP] = pokemon.hp;
    dataArr[FeatureEntity.ENTITY_MAXHP] = pokemon.maxhp;
    dataArr[FeatureEntity.ENTITY_HP_TOKEN] = Math.floor(
        (1023 * pokemon.hp) / pokemon.maxhp,
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
    dataArr[FeatureEntity.ENTITY_TRAPPED] = !!pokemon.trapped ? 1 : 0;
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

    const volatiles = new OneDBoolean(numVolatiles, Int16Array);
    for (const [key, value] of Object.entries({
        ...pokemon.volatiles,
        ...candidate.volatiles,
    })) {
        const index = IndexValueFromEnum("Volatilestatus", key);
        volatiles.set(index, true);
    }
    dataArr.set(volatiles.buffer, FeatureEntity.ENTITY_VOLATILES0);

    dataArr[FeatureEntity.ENTITY_SIDE] = pokemon.side.n;

    dataArr[FeatureEntity.ENTITY_BOOST_ATK_VALUE] = pokemon.boosts.atk ?? 0;
    dataArr[FeatureEntity.ENTITY_BOOST_DEF_VALUE] = pokemon.boosts.def ?? 0;
    dataArr[FeatureEntity.ENTITY_BOOST_SPA_VALUE] = pokemon.boosts.spa ?? 0;
    dataArr[FeatureEntity.ENTITY_BOOST_SPD_VALUE] = pokemon.boosts.spd ?? 0;
    dataArr[FeatureEntity.ENTITY_BOOST_SPE_VALUE] = pokemon.boosts.spe ?? 0;
    dataArr[FeatureEntity.ENTITY_BOOST_EVASION_VALUE] =
        pokemon.boosts.evasion ?? 0;
    dataArr[FeatureEntity.ENTITY_BOOST_ACCURACY_VALUE] =
        pokemon.boosts.accuracy ?? 0;

    return dataArr;
}

const NumEdgeFeatures = Object.keys(FeatureEdge).length;

class Edge {
    data: Int16Array;

    constructor() {
        this.data = new Int16Array(NumEdgeFeatures);
        this.setFeature(FeatureEdge.POKE1_INDEX, -1);
        this.setFeature(FeatureEdge.POKE2_INDEX, -1);
    }

    clone() {
        const edge = new Edge();
        edge.data.set(this.data);
        return edge;
    }

    setFeature(index: FeatureEdgeMap[keyof FeatureEdgeMap], value: number) {
        if (index === undefined) {
            throw new Error("Index cannot be undefined");
        }
        if (value === undefined) {
            throw new Error("Value cannot be undefined");
        }
        this.data[index] = value;
    }

    getFeature(index: FeatureEdgeMap[keyof FeatureEdgeMap]) {
        return this.data[index];
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

    toVector() {
        const edgeTypeToken = this.getFeature(FeatureEdge.EDGE_TYPE_TOKEN);
        if (edgeTypeToken === EdgeTypes.EDGE_TYPE_NONE) {
            throw new Error();
        }
        return this.data;
    }

    toObject() {
        const featureIndicies = [
            FeatureEdge.MOVE_TOKEN,
            FeatureEdge.ABILITY_TOKEN,
            FeatureEdge.ITEM_TOKEN,
            FeatureEdge.EDGE_TYPE_TOKEN,
            FeatureEdge.FROM_TYPE_TOKEN,
            FeatureEdge.FROM_SOURCE_TOKEN,
            FeatureEdge.DAMAGE_TOKEN,
            FeatureEdge.TURN_ORDER_VALUE,
            FeatureEdge.STATUS_TOKEN,
            FeatureEdge.MAJOR_ARG,
            FeatureEdge.MINOR_ARG,
        ];
        return {
            ...Object.fromEntries(
                Object.entries(FeatureEdge)
                    .filter(([_, index]) => featureIndicies.includes(index))
                    .map(([key, featureIndex]) => {
                        const rawValue = this.getFeature(featureIndex);
                        if ([FeatureEdge.MOVE_TOKEN].includes(featureIndex)) {
                            return [key, Object.keys(MovesEnum).at(rawValue)];
                        }
                        if (
                            [FeatureEdge.ABILITY_TOKEN].includes(featureIndex)
                        ) {
                            return [
                                key,
                                Object.keys(AbilitiesEnum).at(rawValue),
                            ];
                        }
                        if ([FeatureEdge.ITEM_TOKEN].includes(featureIndex)) {
                            return [key, Object.keys(ItemsEnum).at(rawValue)];
                        }
                        if (
                            [FeatureEdge.EDGE_TYPE_TOKEN].includes(featureIndex)
                        ) {
                            return [key, Object.keys(EdgeTypes).at(rawValue)];
                        }
                        if (
                            [FeatureEdge.FROM_TYPE_TOKEN].includes(featureIndex)
                        ) {
                            return [
                                key,
                                Object.keys(EffecttypesEnum).at(rawValue),
                            ];
                        }
                        if ([FeatureEdge.MINOR_ARG].includes(featureIndex)) {
                            return [
                                key,
                                Object.keys(BattleminorargsEnum).at(rawValue),
                            ];
                        }
                        if ([FeatureEdge.MAJOR_ARG].includes(featureIndex)) {
                            return [
                                key,
                                Object.keys(BattlemajorargsEnum).at(rawValue),
                            ];
                        }
                        if (
                            [FeatureEdge.FROM_SOURCE_TOKEN].includes(
                                featureIndex,
                            )
                        ) {
                            const effectKeys = Object.keys(EffectEnum);
                            const valueIndex =
                                Object.values(EffectEnum).indexOf(rawValue);
                            const value = effectKeys.at(valueIndex);
                            return [key, value];
                        }
                        if ([FeatureEdge.DAMAGE_TOKEN].includes(featureIndex)) {
                            return [key, rawValue / 1023];
                        }
                        if ([FeatureEdge.STATUS_TOKEN].includes(featureIndex)) {
                            return [key, Object.keys(StatusEnum).at(rawValue)];
                        }
                        return [key, rawValue];
                    }),
            ),
            // MINORARGS: this.minorArgs
            //     .toBinaryVector()
            //     .flatMap((value, index) =>
            //         !!value ? Object.keys(BattleminorargsEnum).at(index) : [],
            //     ),
            // MAJORARGS: this.majorArgs
            //     .toBinaryVector()
            //     .flatMap((value, index) =>
            //         !!value ? Object.keys(BattlemajorargsEnum).at(index) : [],
            //     ),
            BOOSTS: Object.fromEntries(
                Object.entries(FeatureEdge)
                    .filter(([key, _]) => key.startsWith("BOOST"))
                    .map(([key, featureIndex]) => [
                        key,
                        this.getFeature(featureIndex),
                    ]),
            ),
        };
    }
}

class Node {
    nodeIndex: number;
    data: Int16Array;

    constructor(nodeIndex: number) {
        this.nodeIndex = nodeIndex;
        this.data = new Int16Array(featureNodeSize);
    }

    updateEntityData(entity: Pokemon) {
        const entityData = getArrayFromPokemon(entity);
        this.data.set(entityData);
    }

    toVector() {
        return this.data;
    }
}

class Turn {
    eventHandler: EventHandler;
    nodes: Map<PokemonIdent, number>;
    edges: Edge[];
    turn: number;
    order: number;

    edgeData: Int16Array;
    nodeData: Int16Array;
    sideData: Uint8Array;
    fieldData: Uint8Array;

    constructor(
        eventHandler: EventHandler,
        turn: number,
        init: boolean = false,
    ) {
        this.eventHandler = eventHandler;
        this.nodes = new Map();
        this.edges = [];
        this.turn = turn;
        this.order = 0;

        this.edgeData = new Int16Array(flatEdgeDataSize);
        this.nodeData = new Int16Array(flatNodeDataSize);
        this.sideData = new Uint8Array(2 * numSideConditions);
        this.fieldData = new Uint8Array(numWeatherFields);

        if (init) {
            this.init();
        }
    }

    private init() {
        const battle = this.eventHandler.handler.publicBattle;

        let offset = 0;
        for (const side of battle.sides) {
            const { team, totalPokemon, n } = side;
            for (const member of team) {
                this.nodeData.set(getArrayFromPokemon(member), offset);
                offset += numPokemonFields;
            }
            for (
                let memberIndex = team.length;
                memberIndex < totalPokemon - team.length;
                memberIndex++
            ) {
                this.nodeData.set(n ? unkPokemon1 : unkPokemon0, offset);
                offset += numPokemonFields;
            }
            for (
                let memberIndex = totalPokemon;
                memberIndex < 6;
                memberIndex++
            ) {
                this.nodeData.set(nullPokemon, offset);
                offset += numPokemonFields;
            }
            this.updateSideConditionData(side);
        }
    }

    updateSideConditionData(side: Side) {
        const playerIndex = this.eventHandler.handler.getPlayerIndex()!;
        const { n, sideConditions } = side;
        const sideconditionOffset = n === playerIndex ? numSideConditions : 0;
        for (const [
            id,
            { level, maxDuration, minDuration, name },
        ] of Object.entries(sideConditions)) {
            const featureIndex = IndexValueFromEnum<SideconditionEnumMap>(
                "Sidecondition",
                id,
            );
            this.sideData[featureIndex + sideconditionOffset] = level;
        }
    }

    updateFieldData(field: Field) {
        const weatherIndex = IndexValueFromEnum(
            "Weather",
            field.weatherState.id,
        );
        this.fieldData[FeatureWeather.WEATHER_ID] = weatherIndex;
        this.fieldData[FeatureWeather.MAX_DURATION] =
            field.weatherState.maxDuration;
        this.fieldData[FeatureWeather.MIN_DURATION] =
            field.weatherState.minDuration;
    }

    getNodeDataAtIndex(nodeIndex: number) {
        return this.nodeData.slice(
            nodeIndex * numPokemonFields,
            (nodeIndex + 1) * numPokemonFields,
        );
    }

    newTurn(): Turn {
        const turn = new Turn(this.eventHandler, this.turn);
        turn.nodes = new Map(this.nodes);
        turn.nodeData.set(this.nodeData);
        turn.updateFromBattle();
        return turn;
    }

    updateFromBattle() {
        const handler = this.eventHandler.handler;
        const battle = handler.publicBattle;
        this.turn = battle.turn;

        for (const [key, index] of this.nodes.entries()) {
            const updatedEntity = battle.getPokemon(key);
            if (updatedEntity) {
                const updatedEntityArray = getArrayFromPokemon(updatedEntity);
                this.nodeData.set(updatedEntityArray, index * numPokemonFields);
            } else {
                const updatedEntity = battle.getPokemon(
                    key.replace(/^([a-z])(\d+):/, "$1$2a:") as PokemonIdent,
                );
                if (updatedEntity) {
                    const updatedEntityArray =
                        getArrayFromPokemon(updatedEntity);
                    this.nodeData.set(
                        updatedEntityArray,
                        index * numPokemonFields,
                    );
                }
            }
        }
        for (const side of battle.sides) {
            this.updateSideConditionData(side);
        }
    }

    getNodeIndex(ident?: PokemonIdent | string) {
        return !!ident ? this.nodes.get(ident as PokemonIdent)! : 0;
    }

    addNode(entity: Pokemon | null) {
        if (entity !== null && !this.nodes.has(entity.originalIdent)) {
            const ident = entity.originalIdent;
            const playerIndex =
                this.eventHandler.handler.getPlayerIndex() as number;
            const isMe = entity.side.n === playerIndex;
            const index =
                entity.side.team.length - 1 + (!entity.side.n ? 0 : 6);
            this.nodes.set(ident, index);
        }
    }

    getNodeFromIdent(ident: PokemonIdent) {
        return this.nodes.get(ident);
    }

    appendEdge(edge: Edge) {
        edge.setFeature(FeatureEdge.TURN_ORDER_VALUE, this.order);
        this.edges.push(edge);
        this.order += 1;
        return edge;
    }

    getLatestEdge(): Edge {
        const latestEdge = this.edges.at(-1);
        if (latestEdge) {
            return latestEdge;
        } else {
            throw new Error("No Edges");
        }
    }

    private lockEdges() {
        let offset = 0;
        for (const edge of this.edges.slice(0, 20)) {
            const vector = edge.toVector();
            this.edgeData.set(vector, offset);
            offset += featureEdgeSize;
        }
    }

    lockTurn() {
        this.lockEdges();
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

export class EventHandler implements Protocol.Handler {
    readonly handler: StreamHandler;

    currHp: Map<string, number>;
    actives: Map<ID, PokemonIdent>;
    actionWindow: { [k: number]: number[] };
    lastKey: Protocol.ArgName | undefined;
    turns: Turn[];
    rewardQueue: number[];

    constructor(handler: StreamHandler) {
        this.handler = handler;
        this.currHp = new Map();
        this.actives = new Map();
        this.actionWindow = {
            0: [],
            1: [],
        };
        this.turns = [];
        this.rewardQueue = [];
        this.resetTurns();
    }

    getLatestTurn(): Turn {
        return this.turns.at(-1) as Turn;
    }

    getMove(ident?: string) {
        return this.handler.publicBattle.get("moves", ident) as Partial<Move> &
            NA;
    }

    getAbility(ident?: string) {
        return this.handler.publicBattle.get(
            "abilities",
            ident,
        ) as Partial<Ability> & NA;
    }

    getItem(ident: string) {
        return this.handler.publicBattle.get("items", ident) as Partial<Item> &
            NA;
    }

    getCondition(ident?: string) {
        return this.handler.publicBattle.get(
            "conditions",
            ident,
        ) as Partial<Condition>;
    }

    getPokemon(ident: PokemonIdent) {
        return this.handler.publicBattle.getPokemon(ident);
    }

    getSide(ident: Protocol.Side) {
        return this.handler.publicBattle.getSide(ident);
    }

    "|move|"(args: Args["|move|"], kwArgs: KWArgs["|move|"]) {
        const [argName, poke1Ident, moveId, poke2Ident] = args;

        const latestTurn = this.getLatestTurn();

        const poke1 = this.getPokemon(poke1Ident)!;
        const poke1Index = latestTurn.getNodeIndex(poke1.originalIdent);
        this.actionWindow[poke1.side.n].push(0);

        const move = this.getMove(moveId);
        const moveIndex = IndexValueFromEnum("Move", move.id);

        const poke2 = this.getPokemon(poke2Ident as PokemonIdent);
        const poke2Index = latestTurn.getNodeIndex(poke2?.originalIdent);

        const edge = new Edge();
        edge.addMajorArg(argName);
        edge.setFeature(FeatureEdge.POKE1_INDEX, poke1Index);
        edge.setFeature(FeatureEdge.EDGE_AFFECTING_SIDE, poke1.side.n);
        edge.setFeature(FeatureEdge.MOVE_TOKEN, moveIndex);
        edge.setFeature(FeatureEdge.POKE2_INDEX, poke2Index);
        edge.setFeature(FeatureEdge.EDGE_TYPE_TOKEN, EdgeTypes.MOVE_EDGE);

        latestTurn.appendEdge(edge);
    }

    "|drag|"(args: Args["|drag|"], kwArgs?: KWArgs["|drag|"]) {
        this.handleSwitch(args, kwArgs);
    }

    "|switch|"(args: Args["|switch|"], kwArgs?: KWArgs["|switch|"]) {
        this.handleSwitch(args, kwArgs);
    }

    handleSwitch(
        args: Args["|switch|" | "|drag|"],
        kwArgs?: KWArgs["|switch|" | "|drag|"],
    ) {
        const [argName, poke1Ident] = args;

        const latestTurn = this.getLatestTurn();

        const moveIndex = MovesEnum.MOVES__SWITCH;

        const poke1 = this.getPokemon(poke1Ident)!;
        if (argName === "switch") {
            this.actionWindow[poke1.side.n].push(1);
        }
        const trueIdent = poke1.originalIdent;
        if (!latestTurn.nodes.has(trueIdent)) {
            latestTurn.addNode(poke1);
        }
        const poke1Index = latestTurn.getNodeIndex(trueIdent);

        const opposite = poke1.side.foe.active[0];
        const lastPokemon = poke1.side.lastPokemon!;
        if (
            argName === "switch" &&
            opposite !== null &&
            !!lastPokemon &&
            !(lastPokemon.fainted ?? false)
        ) {
            const lastPokemonTurnsActive = this.getTurnsActive(lastPokemon);
            const oppTurnsActive = this.getTurnsActive(opposite);
            if (0 < oppTurnsActive && oppTurnsActive < lastPokemonTurnsActive) {
                this.rewardQueue.push(opposite.side.n ? -1 : 1);
            }
        }

        const edge = new Edge();
        edge.addMajorArg(argName);
        edge.setFeature(FeatureEdge.MOVE_TOKEN, moveIndex);
        edge.setFeature(FeatureEdge.POKE1_INDEX, poke1Index);
        edge.setFeature(FeatureEdge.EDGE_AFFECTING_SIDE, poke1.side.n);
        edge.setFeature(FeatureEdge.EDGE_TYPE_TOKEN, EdgeTypes.SWITCH_EDGE);

        latestTurn.appendEdge(edge);
    }

    "|cant|"(args: Args["|cant|"]) {
        const [argName, poke1Ident, conditionId, moveId] = args;

        const latestTurn = this.getLatestTurn();
        const edge = new Edge();

        const poke1 = this.getPokemon(poke1Ident)!;
        const poke1Index = latestTurn.getNodeIndex(poke1.originalIdent);

        if (!!moveId) {
            const move = this.getMove(moveId);
            const moveIndex = IndexValueFromEnum("Move", move.id);
            edge.setFeature(FeatureEdge.MOVE_TOKEN, moveIndex);
        }

        const condition = this.getCondition(conditionId);

        edge.addMajorArg(argName);
        edge.setFeature(FeatureEdge.POKE1_INDEX, poke1Index);
        edge.setFeature(FeatureEdge.EDGE_AFFECTING_SIDE, poke1.side.n);
        edge.setFeature(FeatureEdge.EDGE_TYPE_TOKEN, EdgeTypes.CANT_EDGE);
        edge.updateEdgeFromOf(condition);

        latestTurn.appendEdge(edge);
    }

    getTurnsActive(pokemon: Pokemon) {
        const latestTurn = this.getLatestTurn();
        const nodeIndex = latestTurn.getNodeIndex(pokemon.originalIdent);
        let turnsActive = 0;
        for (const turn of [...this.turns].reverse()) {
            const nodeData = turn.getNodeDataAtIndex(nodeIndex);
            const isActive = nodeData.at(FeatureEntity.ENTITY_ACTIVE)!;
            if (!isActive) {
                break;
            }
            turnsActive += isActive;
        }
        return turnsActive;
    }

    "|faint|"(args: Args["|faint|"]) {
        const [argName, poke1Ident] = args;

        const latestTurn = this.getLatestTurn();

        const poke1 = this.getPokemon(poke1Ident)!;
        const poke1Index = latestTurn.getNodeIndex(poke1.originalIdent);

        const edge = new Edge();

        const opposite = poke1.side.foe.active[0];
        if (opposite !== null) {
            const poke1TurnsActive = this.getTurnsActive(poke1);
            const oppTurnsActive = this.getTurnsActive(opposite);

            let rew = 0;

            if (
                0 < oppTurnsActive &&
                oppTurnsActive < poke1TurnsActive &&
                !(opposite.side.lastPokemon?.fainted ?? false)
            ) {
                rew +=
                    (oppTurnsActive <= 2
                        ? 1
                        : 1 / (oppTurnsActive - 1) ** 1.5) *
                    (opposite.side.n ? -1 : 1);
            }

            if (rew !== 0) {
                this.rewardQueue.push(rew);
            }
        }

        edge.addMajorArg(argName);
        edge.setFeature(FeatureEdge.POKE1_INDEX, poke1Index);
        edge.setFeature(FeatureEdge.EDGE_AFFECTING_SIDE, poke1.side.n);
        edge.setFeature(FeatureEdge.EDGE_TYPE_TOKEN, EdgeTypes.EFFECT_EDGE);

        latestTurn.appendEdge(edge);
    }

    "|-fail|"(args: Args["|-fail|"], kwArgs: KWArgs["|-fail|"]) {
        const [argName, poke1Ident] = args;

        const latestTurn = this.getLatestTurn();
        const poke1 = this.getPokemon(poke1Ident)!;
        const poke1Index = latestTurn.getNodeIndex(poke1.originalIdent);

        const edge = new Edge();

        const fromEffect = this.getCondition(kwArgs.from);

        if (!!kwArgs.of) {
            const poke2 = this.getPokemon(kwArgs.of)!;
            const true2Ident = poke2.originalIdent;
            const poke2Index = latestTurn.getNodeIndex(true2Ident);
            edge.setFeature(FeatureEdge.POKE2_INDEX, poke2Index);
        }

        edge.addMinorArg(argName);
        edge.setFeature(FeatureEdge.POKE1_INDEX, poke1Index);
        edge.setFeature(FeatureEdge.EDGE_AFFECTING_SIDE, poke1.side.n);
        edge.setFeature(FeatureEdge.EDGE_TYPE_TOKEN, EdgeTypes.EFFECT_EDGE);
        edge.updateEdgeFromOf(fromEffect);

        latestTurn.appendEdge(edge);
    }

    "|-block|"(args: Args["|-block|"], kwArgs: KWArgs["|-block|"]) {
        const [argName, poke1Ident] = args;

        const latestTurn = this.getLatestTurn();
        const poke1 = this.getPokemon(poke1Ident)!;
        const poke1Index = latestTurn.getNodeIndex(poke1.originalIdent);

        const edge = new Edge();

        const fromEffect = this.getCondition(kwArgs.from);

        if (!!kwArgs.of) {
            const poke2 = this.getPokemon(kwArgs.of)!;
            const true2Ident = poke2.originalIdent;
            const poke2Index = latestTurn.getNodeIndex(true2Ident);
            edge.setFeature(FeatureEdge.POKE2_INDEX, poke2Index);
        }

        edge.addMinorArg(argName);
        edge.setFeature(FeatureEdge.POKE1_INDEX, poke1Index);
        edge.setFeature(FeatureEdge.EDGE_AFFECTING_SIDE, poke1.side.n);
        edge.setFeature(FeatureEdge.EDGE_TYPE_TOKEN, EdgeTypes.EFFECT_EDGE);
        edge.updateEdgeFromOf(fromEffect);

        latestTurn.appendEdge(edge);
    }

    "|-notarget|"(args: Args["|-notarget|"]) {
        const [argName, poke1Ident] = args;

        const latestTurn = this.getLatestTurn();
        const edge = new Edge();

        if (poke1Ident) {
            const poke1 = this.getPokemon(poke1Ident)!;
            const poke1Index = latestTurn.getNodeIndex(poke1.originalIdent);
            edge.setFeature(FeatureEdge.POKE1_INDEX, poke1Index);
            edge.setFeature(FeatureEdge.EDGE_AFFECTING_SIDE, poke1.side.n);
        }

        edge.addMinorArg(argName);
        edge.setFeature(FeatureEdge.EDGE_TYPE_TOKEN, EdgeTypes.EFFECT_EDGE);

        latestTurn.appendEdge(edge);
    }

    "|-miss|"(args: Args["|-miss|"], kwArgs: KWArgs["|-miss|"]) {
        const [argName, poke1Ident] = args;

        const latestTurn = this.getLatestTurn();
        const poke1 = this.getPokemon(poke1Ident)!;
        const poke1Index = latestTurn.getNodeIndex(poke1.originalIdent);
        const fromEffect = this.getCondition(kwArgs.from);

        const edge = new Edge();

        edge.addMinorArg(argName);
        edge.setFeature(FeatureEdge.POKE1_INDEX, poke1Index);
        edge.setFeature(FeatureEdge.EDGE_AFFECTING_SIDE, poke1.side.n);
        edge.setFeature(FeatureEdge.EDGE_TYPE_TOKEN, EdgeTypes.EFFECT_EDGE);
        edge.updateEdgeFromOf(fromEffect);

        latestTurn.appendEdge(edge);
    }

    "|-damage|"(args: Args["|-damage|"], kwArgs: KWArgs["|-damage|"]) {
        const [argName, poke1Ident, hpStatus] = args;

        const latestTurn = this.getLatestTurn();

        const edge = new Edge();

        const poke1 = this.getPokemon(poke1Ident)!;
        const true1Ident = poke1.originalIdent;
        const poke1Index = latestTurn.getNodeIndex(true1Ident);

        if (!!kwArgs.of) {
            const poke2 = this.getPokemon(kwArgs.of)!;
            const true2Ident = poke2.originalIdent;
            const poke2Index = latestTurn.getNodeIndex(true2Ident);
            edge.setFeature(FeatureEdge.POKE2_INDEX, poke2Index);
        }

        if (!!kwArgs.from) {
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
        edge.setFeature(FeatureEdge.POKE1_INDEX, poke1Index);
        edge.setFeature(FeatureEdge.EDGE_AFFECTING_SIDE, poke1.side.n);
        edge.setFeature(FeatureEdge.DAMAGE_TOKEN, Math.floor(1023 * diffRatio));
        edge.setFeature(FeatureEdge.EDGE_TYPE_TOKEN, EdgeTypes.EFFECT_EDGE);

        latestTurn.appendEdge(edge);
    }

    "|-heal|"(args: Args["|-heal|"], kwArgs: KWArgs["|-heal|"]) {
        const [argName, poke1Ident, hpStatus] = args;

        const latestTurn = this.getLatestTurn();

        const edge = new Edge();

        const poke1 = this.getPokemon(poke1Ident)!;
        const trueIdent = poke1.originalIdent;
        const poke1Index = latestTurn.getNodeIndex(poke1.originalIdent);

        if (!!kwArgs.of) {
            const poke2 = this.getPokemon(kwArgs.of)!;
            const true2Ident = poke2.originalIdent;
            const poke2Index = latestTurn.getNodeIndex(true2Ident);
            edge.setFeature(FeatureEdge.POKE2_INDEX, poke2Index);
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
        edge.setFeature(FeatureEdge.POKE1_INDEX, poke1Index);
        edge.setFeature(FeatureEdge.EDGE_AFFECTING_SIDE, poke1.side.n);
        edge.setFeature(FeatureEdge.DAMAGE_TOKEN, Math.floor(1024 * diffRatio));
        edge.setFeature(FeatureEdge.EDGE_TYPE_TOKEN, EdgeTypes.EFFECT_EDGE);
        edge.updateEdgeFromOf(fromEffect);

        latestTurn.appendEdge(edge);
    }

    "|-status|"(args: Args["|-status|"], kwArgs: KWArgs["|-status|"]) {
        const [argName, poke1Ident, statusId] = args;
        const latestTurn = this.getLatestTurn();
        const fromEffect = this.getCondition(kwArgs.from);
        const statusToken = IndexValueFromEnum("Status", statusId);

        const poke1 = this.getPokemon(poke1Ident)!;
        const trueIdent = poke1.originalIdent;
        const poke1Index = latestTurn.getNodeIndex(trueIdent);

        const edge = new Edge();

        if (!!kwArgs.of) {
            const poke2 = this.getPokemon(kwArgs.of)!;
            const true2Ident = poke2.originalIdent;
            const poke2Index = latestTurn.getNodeIndex(true2Ident);
            edge.setFeature(FeatureEdge.POKE2_INDEX, poke2Index);
        }

        edge.addMinorArg(argName);
        edge.updateEdgeFromOf(fromEffect);
        edge.setFeature(FeatureEdge.POKE1_INDEX, poke1Index);
        edge.setFeature(FeatureEdge.EDGE_AFFECTING_SIDE, poke1.side.n);
        edge.setFeature(FeatureEdge.EDGE_TYPE_TOKEN, EdgeTypes.EFFECT_EDGE);
        edge.setFeature(FeatureEdge.STATUS_TOKEN, statusToken);

        latestTurn.appendEdge(edge);
    }

    "|-curestatus|"(args: Args["|-curestatus|"]) {
        const [argName, poke1Ident, statusId] = args;

        const poke1 = this.getPokemon(poke1Ident)!;
        const latestTurn = this.getLatestTurn();
        const poke1Index = latestTurn.getNodeIndex(poke1.originalIdent);
        const statusIndex = IndexValueFromEnum("Status", statusId);

        const edge = new Edge();
        edge.addMinorArg(argName);
        edge.setFeature(FeatureEdge.POKE1_INDEX, poke1Index);
        edge.setFeature(FeatureEdge.EDGE_AFFECTING_SIDE, poke1.side.n);
        edge.setFeature(FeatureEdge.EDGE_TYPE_TOKEN, EdgeTypes.EFFECT_EDGE);
        edge.setFeature(FeatureEdge.STATUS_TOKEN, statusIndex);

        latestTurn.appendEdge(edge);
    }

    "|-cureteam|"(args: Args["|-cureteam|"], kwArgs: KWArgs["|-cureteam|"]) {
        const [argName, poke1Ident] = args;

        const poke1 = this.getPokemon(poke1Ident)!;
        const latestTurn = this.getLatestTurn();
        const poke1Index = latestTurn.getNodeIndex(poke1.originalIdent);

        const edge = new Edge();
        edge.addMinorArg(argName);
        edge.setFeature(FeatureEdge.POKE1_INDEX, poke1Index);
        edge.setFeature(FeatureEdge.EDGE_AFFECTING_SIDE, poke1.side.n);
        edge.setFeature(FeatureEdge.EDGE_TYPE_TOKEN, EdgeTypes.EFFECT_EDGE);

        latestTurn.appendEdge(edge);
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

        const latestTurn = this.getLatestTurn();

        const poke1 = this.handler.publicBattle.getPokemon(poke1Ident)!;
        const poke1Index = latestTurn.getNodeIndex(poke1.originalIdent);
        const featureIndex = EventHandler.getStatBoostEdgeFeatureIndex(stat);

        const edge = new Edge();

        if (!!kwArgs.of) {
            const poke2 = this.getPokemon(kwArgs.of)!;
            const true2Ident = poke2.originalIdent;
            const poke2Index = latestTurn.getNodeIndex(true2Ident);
            edge.setFeature(FeatureEdge.POKE2_INDEX, poke2Index);
        }

        edge.setFeature(FeatureEdge.POKE1_INDEX, poke1Index);
        edge.setFeature(FeatureEdge.EDGE_AFFECTING_SIDE, poke1.side.n);
        edge.setFeature(FeatureEdge.EDGE_TYPE_TOKEN, EdgeTypes.EFFECT_EDGE);
        edge.setFeature(featureIndex, parseInt(value));
        edge.addMinorArg(argName);

        latestTurn.appendEdge(edge);
    }

    "|-unboost|"(args: Args["|-unboost|"], kwArgs: KWArgs["|-unboost|"]) {
        const [argName, poke1Ident, stat, value] = args;

        const latestTurn = this.getLatestTurn();

        const poke1 = this.handler.publicBattle.getPokemon(poke1Ident)!;
        const poke1Index = latestTurn.getNodeIndex(poke1.originalIdent);
        const featureIndex = EventHandler.getStatBoostEdgeFeatureIndex(stat);

        const edge = new Edge();

        if (!!kwArgs.of) {
            const poke2 = this.getPokemon(kwArgs.of)!;
            const true2Ident = poke2.originalIdent;
            const poke2Index = latestTurn.getNodeIndex(true2Ident);
            edge.setFeature(FeatureEdge.POKE2_INDEX, poke2Index);
        }

        edge.setFeature(FeatureEdge.POKE1_INDEX, poke1Index);
        edge.setFeature(FeatureEdge.EDGE_AFFECTING_SIDE, poke1.side.n);
        edge.setFeature(FeatureEdge.EDGE_TYPE_TOKEN, EdgeTypes.EFFECT_EDGE);
        edge.setFeature(featureIndex, -parseInt(value));
        edge.addMinorArg(argName);

        latestTurn.appendEdge(edge);
    }

    "|-setboost|"(args: Args["|-setboost|"], kwArgs: KWArgs["|-setboost|"]) {
        const [argName, poke1Ident, stat, value] = args;

        const latestTurn = this.getLatestTurn();

        const poke1 = this.handler.publicBattle.getPokemon(poke1Ident)!;
        const poke1Index = latestTurn.getNodeIndex(poke1.originalIdent);
        const featureIndex = EventHandler.getStatBoostEdgeFeatureIndex(stat);
        const fromEffect = this.getCondition(kwArgs.from);

        const edge = new Edge();

        if (!!kwArgs.of) {
            const poke2 = this.getPokemon(kwArgs.of)!;
            const true2Ident = poke2.originalIdent;
            const poke2Index = latestTurn.getNodeIndex(true2Ident);
            edge.setFeature(FeatureEdge.POKE2_INDEX, poke2Index);
        }

        edge.setFeature(FeatureEdge.POKE1_INDEX, poke1Index);
        edge.setFeature(FeatureEdge.EDGE_AFFECTING_SIDE, poke1.side.n);
        edge.setFeature(FeatureEdge.EDGE_TYPE_TOKEN, EdgeTypes.EFFECT_EDGE);
        edge.setFeature(featureIndex, parseInt(value));
        edge.updateEdgeFromOf(fromEffect);
        edge.addMinorArg(argName);

        latestTurn.appendEdge(edge);
    }

    "|-swapboost|"(args: Args["|-swapboost|"], kwArgs: KWArgs["|-swapboost|"]) {
        const [argName, poke1Ident] = args;

        const latestTurn = this.getLatestTurn();

        const edge = new Edge();

        const poke1 = this.getPokemon(poke1Ident)!;
        const poke1Index = latestTurn.getNodeIndex(poke1.originalIdent);

        const fromEffect = this.getCondition(kwArgs.from);

        if (!!kwArgs.of) {
            const poke2 = this.getPokemon(kwArgs.of)!;
            const true2Ident = poke2.originalIdent;
            const poke2Index = latestTurn.getNodeIndex(true2Ident);
            edge.setFeature(FeatureEdge.POKE2_INDEX, poke2Index);
        }

        edge.addMinorArg(argName);
        edge.setFeature(FeatureEdge.EDGE_TYPE_TOKEN, EdgeTypes.EFFECT_EDGE);
        edge.setFeature(FeatureEdge.POKE1_INDEX, poke1Index);
        edge.setFeature(FeatureEdge.EDGE_AFFECTING_SIDE, poke1.side.n);
        edge.updateEdgeFromOf(fromEffect);

        latestTurn.appendEdge(edge);
    }

    "|-invertboost|"(
        args: Args["|-invertboost|"],
        kwArgs: KWArgs["|-invertboost|"],
    ) {
        const [argName, poke1Ident] = args;

        const latestTurn = this.getLatestTurn();

        const edge = new Edge();

        const poke1 = this.getPokemon(poke1Ident)!;
        const poke1Index = latestTurn.getNodeIndex(poke1.originalIdent);

        const fromEffect = this.getCondition(kwArgs.from);

        if (!!kwArgs.of) {
            const poke2 = this.getPokemon(kwArgs.of)!;
            const true2Ident = poke2.originalIdent;
            const poke2Index = latestTurn.getNodeIndex(true2Ident);
            edge.setFeature(FeatureEdge.POKE2_INDEX, poke2Index);
        }

        edge.addMinorArg(argName);
        edge.setFeature(FeatureEdge.EDGE_TYPE_TOKEN, EdgeTypes.EFFECT_EDGE);
        edge.setFeature(FeatureEdge.POKE1_INDEX, poke1Index);
        edge.setFeature(FeatureEdge.EDGE_AFFECTING_SIDE, poke1.side.n);
        edge.updateEdgeFromOf(fromEffect);

        latestTurn.appendEdge(edge);
    }

    "|-clearboost|"(
        args: Args["|-clearboost|"],
        kwArgs: KWArgs["|-clearboost|"],
    ) {
        const [argName, poke1Ident] = args;

        const latestTurn = this.getLatestTurn();

        const edge = new Edge();

        const poke1 = this.getPokemon(poke1Ident)!;
        const poke1Index = latestTurn.getNodeIndex(poke1.originalIdent);

        const fromEffect = this.getCondition(kwArgs.from);

        if (!!kwArgs.of) {
            const poke2 = this.getPokemon(kwArgs.of)!;
            const true2Ident = poke2.originalIdent;
            const poke2Index = latestTurn.getNodeIndex(true2Ident);
            edge.setFeature(FeatureEdge.POKE2_INDEX, poke2Index);
        }

        edge.addMinorArg(argName);
        edge.setFeature(FeatureEdge.EDGE_TYPE_TOKEN, EdgeTypes.EFFECT_EDGE);
        edge.setFeature(FeatureEdge.POKE1_INDEX, poke1Index);
        edge.setFeature(FeatureEdge.EDGE_AFFECTING_SIDE, poke1.side.n);
        edge.updateEdgeFromOf(fromEffect);

        latestTurn.appendEdge(edge);
    }

    "|-clearnegativeboost|"(
        args: Args["|-clearnegativeboost|"],
        kwArgs: KWArgs["|-clearnegativeboost|"],
    ) {
        const [argName, poke1Ident] = args;

        const latestTurn = this.getLatestTurn();

        const edge = new Edge();

        const poke1 = this.getPokemon(poke1Ident)!;
        const poke1Index = latestTurn.getNodeIndex(poke1.originalIdent);

        const fromEffect = this.getCondition(kwArgs.from);

        if (!!kwArgs.of) {
            const poke2 = this.getPokemon(kwArgs.of)!;
            const true2Ident = poke2.originalIdent;
            const poke2Index = latestTurn.getNodeIndex(true2Ident);
            edge.setFeature(FeatureEdge.POKE2_INDEX, poke2Index);
        }

        edge.addMinorArg(argName);
        edge.setFeature(FeatureEdge.EDGE_TYPE_TOKEN, EdgeTypes.EFFECT_EDGE);
        edge.setFeature(FeatureEdge.POKE1_INDEX, poke1Index);
        edge.setFeature(FeatureEdge.EDGE_AFFECTING_SIDE, poke1.side.n);
        edge.updateEdgeFromOf(fromEffect);

        latestTurn.appendEdge(edge);
    }

    "|-copyboost|"(
        args: Args["|-copyboost|"],
        kwArgs: KWArgs["|-copyboost|"],
    ) {}

    "|-weather|"(args: Args["|-weather|"], kwArgs: KWArgs["|-weather|"]) {
        const [argName, weatherId] = args;
        const latestTurn = this.getLatestTurn();
        const fromEffect = this.getCondition(weatherId);

        const edge = new Edge();
        edge.addMinorArg(argName);
        edge.setFeature(FeatureEdge.EDGE_TYPE_TOKEN, EdgeTypes.EFFECT_EDGE);
        edge.setFeature(FeatureEdge.EDGE_AFFECTING_SIDE, 2);
        edge.updateEdgeFromOf(fromEffect);

        latestTurn.appendEdge(edge);
    }

    "|-fieldstart|"(
        args: Args["|-fieldstart|"],
        kwArgs: KWArgs["|-fieldstart|"],
    ) {
        const [argName, conditionId] = args;

        const latestTurn = this.getLatestTurn();
        const edge = new Edge();
        edge.addMinorArg(argName);

        if (kwArgs.of) {
        }

        edge.setFeature(FeatureEdge.EDGE_TYPE_TOKEN, EdgeTypes.EFFECT_EDGE);
        edge.setFeature(FeatureEdge.EDGE_AFFECTING_SIDE, 2);

        latestTurn.appendEdge(edge);
    }

    "|-fieldend|"(args: Args["|-fieldend|"], kwArgs: KWArgs["|-fieldend|"]) {
        const [argName, conditionId] = args;

        const latestTurn = this.getLatestTurn();
        const edge = new Edge();
        edge.addMinorArg(argName);

        if (kwArgs.of) {
        }

        edge.setFeature(FeatureEdge.EDGE_TYPE_TOKEN, EdgeTypes.EFFECT_EDGE);
        edge.setFeature(FeatureEdge.EDGE_AFFECTING_SIDE, 2);

        latestTurn.appendEdge(edge);
    }

    "|-sidestart|"(args: Args["|-sidestart|"], kwArgs: KWArgs["|-sidestart|"]) {
        const [argName, sideId, conditionId] = args;

        const latestTurn = this.getLatestTurn();
        const side = this.getSide(sideId);
        const fromEffect = this.getCondition(conditionId);

        const edge = new Edge();
        edge.addMinorArg(argName);
        edge.setFeature(FeatureEdge.EDGE_AFFECTING_SIDE, side.n);
        edge.setFeature(FeatureEdge.EDGE_TYPE_TOKEN, EdgeTypes.EFFECT_EDGE);
        edge.updateEdgeFromOf(fromEffect);
        latestTurn.appendEdge(edge);
    }

    "|-sideend|"(args: Args["|-sideend|"], kwArgs: KWArgs["|-sideend|"]) {
        const [argName, sideId, conditionId] = args;

        const latestTurn = this.getLatestTurn();
        const side = this.getSide(sideId);
        const fromEffect = this.getCondition(conditionId);

        const edge = new Edge();
        edge.addMinorArg(argName);
        edge.setFeature(FeatureEdge.EDGE_AFFECTING_SIDE, side.n);
        edge.setFeature(FeatureEdge.EDGE_TYPE_TOKEN, EdgeTypes.EFFECT_EDGE);
        edge.updateEdgeFromOf(fromEffect);
        latestTurn.appendEdge(edge);
    }

    "|-swapsideconditions|"(args: Args["|-swapsideconditions|"]) {}

    "|-start|"(args: Args["|-start|"], kwArgs: KWArgs["|-start|"]) {
        const [argName, poke1Ident, conditionId] = args;

        const poke1 = this.getPokemon(poke1Ident)!;
        const latestTurn = this.getLatestTurn();
        const poke1Index = latestTurn.getNodeIndex(poke1.originalIdent);

        const edge = new Edge();
        edge.addMinorArg(argName);
        edge.setFeature(FeatureEdge.POKE1_INDEX, poke1Index);
        edge.setFeature(FeatureEdge.EDGE_AFFECTING_SIDE, poke1.side.n);
        edge.setFeature(FeatureEdge.EDGE_TYPE_TOKEN, EdgeTypes.EFFECT_EDGE);

        latestTurn.appendEdge(edge);
    }

    "|-end|"(args: Args["|-end|"], kwArgs: KWArgs["|-end|"]) {
        const [argName, poke1Ident, conditionId] = args;

        const poke1 = this.getPokemon(poke1Ident)!;
        const latestTurn = this.getLatestTurn();
        const poke1Index = latestTurn.getNodeIndex(poke1.originalIdent);

        const edge = new Edge();
        edge.addMinorArg(argName);
        edge.setFeature(FeatureEdge.POKE1_INDEX, poke1Index);
        edge.setFeature(FeatureEdge.EDGE_AFFECTING_SIDE, poke1.side.n);
        edge.setFeature(FeatureEdge.EDGE_TYPE_TOKEN, EdgeTypes.EFFECT_EDGE);

        latestTurn.appendEdge(edge);
    }

    "|-crit|"(args: Args["|-crit|"]) {
        const [argName, poke1Ident] = args;

        const poke1 = this.getPokemon(poke1Ident)!;
        const latestTurn = this.getLatestTurn();
        const poke1Index = latestTurn.getNodeIndex(poke1.originalIdent);

        const edge = new Edge();
        edge.addMinorArg(argName);
        edge.setFeature(FeatureEdge.POKE1_INDEX, poke1Index);
        edge.setFeature(FeatureEdge.EDGE_AFFECTING_SIDE, poke1.side.n);
        edge.setFeature(FeatureEdge.EDGE_TYPE_TOKEN, EdgeTypes.EFFECT_EDGE);

        latestTurn.appendEdge(edge);
    }

    "|-supereffective|"(args: Args["|-supereffective|"]) {
        const [argName, poke1Ident] = args;

        const poke1 = this.getPokemon(poke1Ident)!;
        const latestTurn = this.getLatestTurn();
        const poke1Index = latestTurn.getNodeIndex(poke1.originalIdent);

        const edge = new Edge();
        edge.addMinorArg(argName);
        edge.setFeature(FeatureEdge.POKE1_INDEX, poke1Index);
        edge.setFeature(FeatureEdge.EDGE_AFFECTING_SIDE, poke1.side.n);
        edge.setFeature(FeatureEdge.EDGE_TYPE_TOKEN, EdgeTypes.EFFECT_EDGE);

        latestTurn.appendEdge(edge);
    }

    "|-resisted|"(args: Args["|-resisted|"]) {
        const [argName, poke1Ident] = args;

        const poke1 = this.getPokemon(poke1Ident)!;
        const latestTurn = this.getLatestTurn();
        const poke1Index = latestTurn.getNodeIndex(poke1.originalIdent);

        const edge = new Edge();
        edge.addMinorArg(argName);
        edge.setFeature(FeatureEdge.POKE1_INDEX, poke1Index);
        edge.setFeature(FeatureEdge.EDGE_AFFECTING_SIDE, poke1.side.n);
        edge.setFeature(FeatureEdge.EDGE_TYPE_TOKEN, EdgeTypes.EFFECT_EDGE);

        latestTurn.appendEdge(edge);
    }

    "|-immune|"(args: Args["|-immune|"], kwArgs: KWArgs["|-immune|"]) {
        const [argName, poke1Ident] = args;

        const poke1 = this.getPokemon(poke1Ident)!;
        const latestTurn = this.getLatestTurn();
        const poke1Index = latestTurn.getNodeIndex(poke1.originalIdent);

        const edge = new Edge();

        const fromEffect = this.getCondition(kwArgs.from);

        if (!!kwArgs.of) {
            const poke2 = this.getPokemon(kwArgs.of)!;
            const true2Ident = poke2.originalIdent;
            const poke2Index = latestTurn.getNodeIndex(true2Ident);
            edge.setFeature(FeatureEdge.POKE2_INDEX, poke2Index);
        }

        edge.addMinorArg(argName);
        edge.setFeature(FeatureEdge.POKE1_INDEX, poke1Index);
        edge.setFeature(FeatureEdge.EDGE_AFFECTING_SIDE, poke1.side.n);
        edge.setFeature(FeatureEdge.EDGE_TYPE_TOKEN, EdgeTypes.EFFECT_EDGE);
        edge.updateEdgeFromOf(fromEffect);

        latestTurn.appendEdge(edge);
    }

    "|-item|"(args: Args["|-item|"], kwArgs: KWArgs["|-item|"]) {
        const [argName, poke1Ident, itemId] = args;

        const latestTurn = this.getLatestTurn();

        const poke1 = this.getPokemon(poke1Ident)!;
        const poke1Index = latestTurn.getNodeIndex(poke1.originalIdent);
        const itemIndex = IndexValueFromEnum("Items", itemId);
        const fromEffect = this.getCondition(kwArgs.from);

        const edge = new Edge();

        if (!!kwArgs.of) {
            const poke2 = this.getPokemon(kwArgs.of)!;
            const true2Ident = poke2.originalIdent;
            const poke2Index = latestTurn.getNodeIndex(true2Ident);
            edge.setFeature(FeatureEdge.POKE2_INDEX, poke2Index);
        }

        edge.setFeature(FeatureEdge.POKE1_INDEX, poke1Index);
        edge.setFeature(FeatureEdge.EDGE_AFFECTING_SIDE, poke1.side.n);
        edge.setFeature(FeatureEdge.ITEM_TOKEN, itemIndex);
        edge.setFeature(FeatureEdge.EDGE_TYPE_TOKEN, EdgeTypes.EFFECT_EDGE);
        edge.updateEdgeFromOf(fromEffect);
        edge.addMinorArg(argName);
        latestTurn.appendEdge(edge);
    }

    "|-enditem|"(args: Args["|-enditem|"], kwArgs: KWArgs["|-enditem|"]) {
        const [argName, poke1Ident, itemId] = args;

        const latestTurn = this.getLatestTurn();

        const poke1 = this.getPokemon(poke1Ident)!;
        const poke1Index = latestTurn.getNodeIndex(poke1.originalIdent);
        const itemIndex = IndexValueFromEnum("Items", itemId);
        const fromEffect = this.getCondition(kwArgs.from);

        const edge = new Edge();
        edge.setFeature(FeatureEdge.POKE1_INDEX, poke1Index);
        edge.setFeature(FeatureEdge.EDGE_AFFECTING_SIDE, poke1.side.n);
        edge.setFeature(FeatureEdge.ITEM_TOKEN, itemIndex);
        edge.updateEdgeFromOf(fromEffect);
        edge.setFeature(FeatureEdge.EDGE_TYPE_TOKEN, EdgeTypes.EFFECT_EDGE);
        edge.addMinorArg(argName);
        latestTurn.appendEdge(edge);
    }

    "|-ability|"(args: Args["|-ability|"], kwArgs: KWArgs["|-ability|"]) {
        const [argName, poke1Ident, abilityId] = args;

        const latestTurn = this.getLatestTurn();

        const poke1 = this.getPokemon(poke1Ident)!;
        const poke1Index = latestTurn.getNodeIndex(poke1.originalIdent);
        const abilityIndex = IndexValueFromEnum("Ability", abilityId);
        const fromEffect = this.getCondition(kwArgs.from);

        const edge = new Edge();
        edge.setFeature(FeatureEdge.POKE1_INDEX, poke1Index);
        edge.setFeature(FeatureEdge.EDGE_AFFECTING_SIDE, poke1.side.n);
        edge.setFeature(FeatureEdge.ABILITY_TOKEN, abilityIndex);
        edge.setFeature(FeatureEdge.EDGE_TYPE_TOKEN, EdgeTypes.EFFECT_EDGE);
        edge.updateEdgeFromOf(fromEffect);
        edge.addMinorArg(argName);
        latestTurn.appendEdge(edge);
    }

    "|-endability|"(
        args: Args["|-endability|"],
        kwArgs: KWArgs["|-endability|"],
    ) {
        const [argName, poke1Ident] = args;

        const latestTurn = this.getLatestTurn();

        const poke1 = this.getPokemon(poke1Ident)!;
        const poke1Index = latestTurn.getNodeIndex(poke1.originalIdent);
        const fromEffect = this.getCondition(kwArgs.from);

        const edge = new Edge();
        edge.setFeature(FeatureEdge.POKE1_INDEX, poke1Index);
        edge.setFeature(FeatureEdge.EDGE_AFFECTING_SIDE, poke1.side.n);
        edge.setFeature(FeatureEdge.EDGE_TYPE_TOKEN, EdgeTypes.EFFECT_EDGE);
        edge.updateEdgeFromOf(fromEffect);
        edge.addMinorArg(argName);
        latestTurn.appendEdge(edge);
    }

    "|-transform|"(
        args: Args["|-transform|"],
        kwArgs: KWArgs["|-transform|"],
    ) {}

    "|-mega|"(args: Args["|-mega|"]) {}

    "|-primal|"(args: Args["|-primal|"]) {}

    "|-burst|"(args: Args["|-burst|"]) {}

    "|-zpower|"(args: Args["|-zpower|"]) {}

    "|-zbroken|"(args: Args["|-zbroken|"]) {}

    "|-activate|"(args: Args["|-activate|"], kwArgs: KWArgs["|-activate|"]) {
        const [argName, poke1Ident, conditionId, poke2Ident] = args;

        const latestTurn = this.getLatestTurn();
        const edge = new Edge();

        if (poke1Ident) {
            const poke1 = this.getPokemon(poke1Ident)!;
            const poke1INdex = latestTurn.getNodeIndex(poke1.originalIdent);
            edge.setFeature(FeatureEdge.EDGE_AFFECTING_SIDE, poke1.side.n);
            edge.setFeature(FeatureEdge.POKE1_INDEX, poke1INdex);
        }

        if (poke2Ident) {
            const poke2 = this.getPokemon(poke2Ident as PokemonIdent)!;
            const poke2Index = latestTurn.getNodeIndex(poke2.originalIdent);
            edge.setFeature(FeatureEdge.POKE2_INDEX, poke2Index);
        }

        const fromEffect = this.getCondition(kwArgs.from);

        edge.updateEdgeFromOf(fromEffect);
        edge.addMinorArg(argName);
        edge.setFeature(FeatureEdge.EDGE_TYPE_TOKEN, EdgeTypes.EFFECT_EDGE);

        latestTurn.appendEdge(edge);
    }

    "|-mustrecharge|"(args: Args["|-mustrecharge|"]) {
        const [argName, poke1Ident] = args;
        const poke1 = this.getPokemon(poke1Ident)!;
        const latestTurn = this.getLatestTurn();
        const poke1Index = latestTurn.getNodeIndex(poke1.originalIdent);

        const edge = new Edge();

        edge.setFeature(FeatureEdge.POKE1_INDEX, poke1Index);
        edge.setFeature(FeatureEdge.EDGE_AFFECTING_SIDE, poke1.side.n);
        edge.setFeature(FeatureEdge.EDGE_TYPE_TOKEN, EdgeTypes.EFFECT_EDGE);
        edge.addMinorArg(argName);

        latestTurn.appendEdge(edge);
    }

    "|-prepare|"(args: Args["|-prepare|"], kwArgs?: KWArgs["|-prepare|"]) {
        const [argName, poke1Ident] = args;
        const poke1 = this.getPokemon(poke1Ident)!;
        const latestTurn = this.getLatestTurn();
        const poke1Index = latestTurn.getNodeIndex(poke1.originalIdent);

        const edge = new Edge();

        edge.setFeature(FeatureEdge.POKE1_INDEX, poke1Index);
        edge.setFeature(FeatureEdge.EDGE_AFFECTING_SIDE, poke1.side.n);
        edge.setFeature(FeatureEdge.EDGE_TYPE_TOKEN, EdgeTypes.EFFECT_EDGE);
        edge.addMinorArg(argName);

        latestTurn.appendEdge(edge);
    }

    "|-hitcount|"(args: Args["|-hitcount|"], kwArgs?: KWArgs["|-hitcount|"]) {
        const [argName, poke1Ident, hitCount] = args;
        const poke1 = this.getPokemon(poke1Ident)!;
        const latestTurn = this.getLatestTurn();
        const poke1Index = latestTurn.getNodeIndex(poke1.originalIdent);

        const edge = new Edge();

        edge.setFeature(FeatureEdge.POKE1_INDEX, poke1Index);
        edge.setFeature(FeatureEdge.EDGE_AFFECTING_SIDE, poke1.side.n);
        edge.setFeature(FeatureEdge.EDGE_TYPE_TOKEN, EdgeTypes.EFFECT_EDGE);
        edge.addMinorArg(argName);

        latestTurn.appendEdge(edge);
    }

    "|done|"() {}

    "|turn|"(args: Args["|turn|"]) {
        const currentTurn = this.getLatestTurn();
        currentTurn.lockTurn();
        const nextTurn = currentTurn.newTurn();
        this.turns.push(nextTurn);
    }

    resetTurns() {
        const initTurn = new Turn(this, this.handler.publicBattle.turn, true);
        this.turns = [initTurn];
    }

    resetRewardQueue() {
        this.rewardQueue = [];
    }

    reset() {
        this.currHp = new Map();
        this.actives = new Map();
        this.actionWindow = {
            0: [],
            1: [],
        };
        this.turns = [];
        this.resetRewardQueue();
        this.resetTurns();
    }
}

class Scaler {
    readonly steepness: number;
    readonly denom: number;

    cache: { [k: number]: number };

    constructor(steepness: number = 1, range: number = 6) {
        this.steepness = steepness;
        this.denom = 0;
        for (let i = 1; i <= range; i++) {
            this.denom += Math.exp(i);
        }
        this.cache = {};
    }

    call(value: number) {
        if (!value) {
            return 0;
        }
        const cachedResult = this.cache[value];
        if (cachedResult !== undefined) {
            return cachedResult;
        }
        const result = Math.exp(value * this.steepness) / this.denom;
        this.cache[value] = result;
        return result;
    }
}

export class Tracker {
    hp: Map<string, number[]>;
    hpTotal: Map<number, number[]>;
    faintedTotal: Map<number, number[]>;
    scaler: Scaler;

    constructor() {
        this.hp = new Map();
        this.hpTotal = new Map();
        this.faintedTotal = new Map();

        this.scaler = new Scaler();
    }

    update(world: Battle) {
        for (const player of [world.p1, world.p2]) {
            if (!this.hpTotal.has(player.n)) {
                this.hpTotal.set(player.n, [0]);
                this.faintedTotal.set(player.n, [0]);
            }
            let hpTotal = 0;
            let faintedTotal = 0;
            for (const pokemon of player.pokemon) {
                const fullName = pokemon.fullname;
                const hpRatio = pokemon.hp / pokemon.maxhp;
                if (!this.hp.has(fullName)) {
                    this.hp.set(fullName, [1]);
                }
                this.hp.get(fullName)!.push(hpRatio);
                hpTotal += hpRatio;
                faintedTotal += +pokemon.fainted;
            }
            this.hpTotal.get(player.n)!.push(hpTotal);
            this.faintedTotal.get(player.n)!.push(faintedTotal);
        }
    }

    getHpChangeReward() {
        const p1HpTotal = this.hpTotal.get(0)!;
        const p2HpTotal = this.hpTotal.get(1)!;
        const [p1tm1, p2tm1] = [p1HpTotal.at(-2) ?? 6, p2HpTotal.at(-2) ?? 6];
        const [p1t, p2t] = [p1HpTotal.at(-1) ?? 6, p2HpTotal.at(-1) ?? 6];
        const score = p1t - p1tm1 - (p2t - p2tm1);
        return score;
    }

    getFaintedChangeReward() {
        const p1FaintedTotal = this.faintedTotal.get(0)!;
        const p2FaintedTotal = this.faintedTotal.get(1)!;
        const [p1t, p1tm1] = [
            this.scaler.call(p1FaintedTotal.at(-1) ?? 0),
            this.scaler.call(p1FaintedTotal.at(-2) ?? 0),
        ];
        const [p2t, p2tm1] = [
            this.scaler.call(p2FaintedTotal.at(-1) ?? 0),
            this.scaler.call(p2FaintedTotal.at(-2) ?? 0),
        ];
        let score = 0;
        if (p1t !== p1tm1) {
            score -= p1t;
        }
        if (p2t !== p2tm1) {
            score += p2t;
        }
        return score;
    }

    getRewardFromFinish(world: Battle, earlyFinish: boolean = false) {
        if (world.ended) {
            const winner = world.winner;
            if (winner === world.p1.name) {
                return 1;
            } else if (winner === world.p2.name) {
                return -1;
            }
        }
        // if (earlyFinish) {
        //     const [p1TotalHp, p2TotalHp] = world.sides.map(
        //         (side) =>
        //             side.pokemon.reduce(
        //                 (prev, curr) => prev + curr.hp / curr.maxhp,
        //                 0,
        //             ) / side.pokemon.length,
        //     );
        //     return p1TotalHp - p2TotalHp;
        // }
        return 0;
    }

    reset() {
        this.hp.clear();
        this.hpTotal.clear();
        this.faintedTotal.clear();
    }
}

export class StateHandler {
    handler: StreamHandler;

    constructor(handler: StreamHandler) {
        this.handler = handler;
    }

    static getLegalActions(request?: AnyObject | null): OneDBoolean {
        const legalActions = new OneDBoolean(10, Uint8Array);

        if (request === undefined || request === null) {
        } else {
            if (request.wait) {
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

    getMoveset(): Uint8Array {
        const movesetArr = new Int16Array(numMovesetFields);
        let offset = 0;

        const playerIndex = this.handler.getPlayerIndex();
        const PlaceMoveset = (side: Side) => {
            const active = side.active[0];

            if (active !== null) {
                const moveSlots = active.moveSlots.slice(0, 4);
                for (const move of moveSlots) {
                    const { id, ppUsed } = move;
                    movesetArr[offset + FeatureMoveset.MOVEID] =
                        id === undefined
                            ? MovesEnum.MOVES__UNK
                            : IndexValueFromEnum<typeof MovesEnum>("Move", id);
                    movesetArr[offset + FeatureMoveset.PPUSED] = ppUsed;
                    offset += numMoveFields;
                }
                for (
                    let remainingIndex = moveSlots.length;
                    remainingIndex < 4;
                    remainingIndex++
                ) {
                    movesetArr[offset + FeatureMoveset.MOVEID] =
                        MovesEnum.MOVES__UNK;
                    movesetArr[offset + FeatureMoveset.PPUSED] = 0;
                    offset += numMoveFields;
                }
            } else {
                offset += 4 * numMoveFields;
            }

            for (
                let remainingIndex = 0;
                remainingIndex < side.totalPokemon;
                remainingIndex++
            ) {
                movesetArr[offset + FeatureMoveset.MOVEID] =
                    MovesEnum.MOVES__SWITCH;
                movesetArr[offset + FeatureMoveset.PPUSED] = 0;
                offset += numMoveFields;
            }
            for (
                let remainingIndex = side.totalPokemon;
                remainingIndex < 6;
                remainingIndex++
            ) {
                movesetArr[offset + FeatureMoveset.MOVEID] =
                    MovesEnum.MOVES__NULL;
                movesetArr[offset + FeatureMoveset.PPUSED] = 0;
                offset += numMoveFields;
            }
        };
        if (playerIndex !== undefined) {
            const mySide = this.handler.privateBattle.sides[playerIndex];
            const oppSide = this.handler.privateBattle.sides[1 - playerIndex];

            for (const side of [mySide, oppSide]) {
                PlaceMoveset(side);
            }
        }

        return new Uint8Array(movesetArr.buffer);
    }

    getPrivateTeam(playerIndex: number): Int16Array {
        const teams: Int16Array[] = [];
        for (const side of [
            this.handler.privateBattle.sides[playerIndex],
            this.handler.privateBattle.sides[1 - playerIndex],
        ]) {
            const team: Int16Array[] = [];
            const sortedTeam = [...side.team].sort(
                (a, b) => +!a.isActive() - +!b.isActive(),
            );
            for (const member of sortedTeam) {
                team.push(getArrayFromPokemon(member));
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
        const side = this.handler.publicBattle.sides[playerIndex];
        const team: Int16Array[] = [];
        for (const member of side.team) {
            team.push(getArrayFromPokemon(member));
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
        const historyEdges: Int16Array[] = [];
        const historyNodes: Int16Array[] = [];
        const historySideConditions: Uint8Array[] = [];
        const historyField: Uint8Array[] = [];

        const finalSlice = this.handler.eventHandler.turns.slice(-numHistory);
        for (const { edgeData, nodeData, sideData, fieldData } of [
            ...finalSlice,
        ].reverse()) {
            historyEdges.push(edgeData);
            historyNodes.push(nodeData);
            historySideConditions.push(sideData);
            historyField.push(fieldData);
        }
        return {
            historyEdges: concatenateArrays(historyEdges),
            historyNodes: concatenateArrays(historyNodes),
            historySideConditions: concatenateArrays(historySideConditions),
            historyField: concatenateArrays(historyField),
            trueHistorySize: finalSlice.length,
        };
    }

    async getState(numHistory?: number): Promise<State> {
        const state = new State();

        const info = new Info();
        info.setGameid(this.handler.gameId);
        const playerIndex = this.handler.getPlayerIndex() as number;
        info.setPlayerindex(!!playerIndex);
        info.setTurn(this.handler.privateBattle.turn);

        const switchReward = this.handler.eventHandler.rewardQueue.reduce(
            (a, b) => a + b,
            0,
        );
        this.handler.eventHandler.resetRewardQueue();
        if (!!!playerIndex && !!switchReward) {
            info.setSwitchreward(switchReward);
        }

        // heuristic
        const heuristicAction = await getEvalAction(this.handler, 11);
        info.setHeuristicaction(heuristicAction.getIndex());

        // const heuristicDist = await GetSearchDistribution({
        //     handler: this.handler,
        // });
        // info.setHeuristicdist(new Uint8Array(heuristicDist.buffer));

        const legalActions = StateHandler.getLegalActions(
            this.handler.privateBattle.request,
        );
        state.setLegalactions(legalActions.buffer);

        state.setInfo(info);

        const {
            historyEdges,
            historyNodes,
            historySideConditions,
            historyField,
            trueHistorySize,
        } = this.getHistory(numHistory);
        const history = new History();
        history.setEdges(new Uint8Array(historyEdges.buffer));
        history.setNodes(new Uint8Array(historyNodes.buffer));
        history.setSideconditions(historySideConditions);
        history.setField(historyField);
        history.setNodes(new Uint8Array(historyNodes.buffer));
        history.setLength(trueHistorySize);
        state.setHistory(history);

        if (this.handler.getRequest()) {
            const privateTeam = this.getPrivateTeam(playerIndex);
            state.setTeam(new Uint8Array(privateTeam.buffer));
            state.setMoveset(this.getMoveset());
        }

        return state;
    }
}
