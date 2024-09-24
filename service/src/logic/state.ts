import { AnyObject } from "@pkmn/sim";
import {
    Args,
    BattleMajorArgName,
    BattleMinorArgName,
    KWArgs,
    PokemonIdent,
    Protocol,
} from "@pkmn/protocol";
import { Info, LegalActions, State } from "../../protos/state_pb";
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
    SpeciesEnum,
    StatusEnum,
} from "../../protos/enums_pb";
import {
    EdgeTypes,
    FeatureEdge,
    FeatureEdgeMap,
    FeatureEntity,
    FeatureMoveset,
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
} from "./data";
import { NA, Pokemon } from "@pkmn/client";
import { OneDBoolean } from "./arr";
import { StreamHandler } from "./handler";
import { Ability, Item, Move, BoostID } from "@pkmn/dex-types";
import { ID } from "@pkmn/types";
import { Condition, Effect } from "@pkmn/data";
import { History } from "../../protos/history_pb";

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
    return new Int32Array(numPokemonFields);
}

function getUnkPokemon() {
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
    return data;
}

const unkPokemon = getUnkPokemon();

function getNonePokemon() {
    const data = getBlankPokemonArr();
    data[FeatureEntity.ENTITY_SPECIES] = SpeciesEnum.SPECIES__NULL;
    return data;
}

const nonePokemon = getNonePokemon();

const NumEdgeFeatures = Object.keys(FeatureEdge).length;

class Edge {
    minorArgs: OneDBoolean<Int32Array>;
    majorArgs: OneDBoolean<Int32Array>;
    data: Int32Array;

    constructor() {
        this.minorArgs = new OneDBoolean(numBattleMinorArgs, Int32Array);
        this.majorArgs = new OneDBoolean(numBattleMajorArgs, Int32Array);
        this.data = new Int32Array(NumEdgeFeatures);
        this.setFeature(FeatureEdge.EDGE_TYPE_TOKEN, -1);
    }

    clone() {
        const edge = new Edge();
        edge.minorArgs.buffer.set(this.minorArgs.buffer);
        edge.majorArgs.buffer.set(this.majorArgs.buffer);
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
        if (index !== undefined) this.minorArgs.toggle(index);
    }

    addMajorArg(argName: MajorArgNames) {
        const index = IndexValueFromEnum("BattleMajorArg", argName);
        if (index !== undefined) this.majorArgs.toggle(index);
    }

    setMajorAndMinorArgs() {
        this.data.set(this.majorArgs.buffer, FeatureEdge.MAJOR_ARGS);
        this.data.set(this.minorArgs.buffer, FeatureEdge.MINOR_ARGS1);
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
                            return [key, rawValue / 1024];
                        }
                        if ([FeatureEdge.STATUS_TOKEN].includes(featureIndex)) {
                            return [key, Object.keys(StatusEnum).at(rawValue)];
                        }
                        return [key, rawValue];
                    }),
            ),
            MINORARGS: this.minorArgs
                .toBinaryVector()
                .flatMap((value, index) =>
                    !!value ? Object.keys(BattleminorargsEnum).at(index) : [],
                ),
            MAJORARGS: this.majorArgs
                .toBinaryVector()
                .flatMap((value, index) =>
                    !!value ? Object.keys(BattlemajorargsEnum).at(index) : [],
                ),
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
    data: Int32Array;

    constructor(nodeIndex: number) {
        this.nodeIndex = nodeIndex;
        this.data = new Int32Array(featureNodeSize);
    }

    updateEntityData(entity: Pokemon) {
        const entityData = getArrayFromPokemon(entity);
        this.data.set(entityData);
    }

    toVector() {
        return this.data;
    }
}

function getArrayFromPokemon(candidate: Pokemon | null): Int32Array {
    if (candidate === null) {
        return getNonePokemon();
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
            const idValue = IndexValueFromEnum<typeof MovesEnum>("Move", id);
            moveIds.push(idValue);
            movePps.push(isNaN(ppUsed) ? +!!ppUsed : ppUsed);
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

    return dataArr;
}

class Turn {
    eventHandler: EventHandler2;
    nodes: Map<PokemonIdent, Node>;
    edges: Edge[];
    turn: number;
    order: number;

    edgeData: Int32Array;
    nodeData: Int32Array;

    constructor(eventHandler: EventHandler2, turn: number) {
        this.eventHandler = eventHandler;
        this.nodes = new Map();
        this.edges = [];
        this.turn = turn;
        this.order = 0;

        this.edgeData = new Int32Array(flatEdgeDataSize);
        this.nodeData = new Int32Array(flatNodeDataSize);
    }

    newTurn(update: boolean = true): Turn {
        const turn = new Turn(this.eventHandler, this.turn);
        for (const [key, oldNode] of this.nodes.entries()) {
            const entityNode = new Node(oldNode.nodeIndex);
            turn.nodes.set(key, entityNode);
        }
        if (update) {
            this.updateFromBattle();
        }
        return turn;
    }

    updateFromBattle() {
        const handler = this.eventHandler.handler;
        const battle = handler.publicBattle;
        this.turn = battle.turn;

        for (const [key, emptyNode] of this.nodes.entries()) {
            const updatedEntity = battle.getPokemon(key);
            if (updatedEntity) {
                emptyNode.updateEntityData(updatedEntity);
            } else {
                const updatedEntity = battle.getPokemon(
                    key.replace(/^([a-z])(\d+):/, "$1$2a:") as PokemonIdent,
                );
                if (updatedEntity) {
                    emptyNode.updateEntityData(updatedEntity);
                }
            }
        }
    }

    getNodeIndex(ident?: PokemonIdent | string) {
        return !!ident ? this.nodes.get(ident as PokemonIdent)!.nodeIndex : 0;
    }

    addNode(entity: Pokemon | null) {
        if (entity !== null && !this.nodes.has(entity.originalIdent)) {
            const ident = entity.originalIdent;
            const playerIndex =
                this.eventHandler.handler.getPlayerIndex() as number;
            const isMe = entity.side.n === playerIndex;

            const entityNode = new Node(this.nodes.size + 1);
            entityNode.updateEntityData(entity);
            this.nodes.set(ident, entityNode);
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
        for (const edge of this.edges) {
            edge.setMajorAndMinorArgs();
            const vector = edge.toVector();
            this.edgeData.set(vector, offset);
            offset += featureEdgeSize;
        }
    }

    private lockNodes() {
        let offset = 0;
        for (const [_, node] of this.nodes.entries()) {
            const vector = node.toVector();
            this.nodeData.set(vector, offset);
            offset += featureNodeSize;
        }
    }

    lockTurn() {
        this.lockEdges();
        this.lockNodes();
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

export class EventHandler2 implements Protocol.Handler {
    readonly handler: StreamHandler;
    history: any[];
    currHp: Map<string, number>;
    actives: Map<ID, PokemonIdent>;
    lastKey: Protocol.ArgName | undefined;
    turns: Turn[];

    constructor(handler: StreamHandler) {
        this.handler = handler;
        this.history = [];
        this.currHp = new Map();
        this.actives = new Map();

        this.turns = [];
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

    "|move|"(args: Args["|move|"], kwArgs: KWArgs["|move|"]) {
        const [argName, poke1Ident, moveId, poke2Ident] = args;

        const latestTurn = this.getLatestTurn();

        const poke1 = this.getPokemon(poke1Ident)!;
        const poke1Index = latestTurn.getNodeIndex(poke1.originalIdent);

        const move = this.getMove(moveId);
        const moveIndex = IndexValueFromEnum("Move", move.id);

        const poke2 = this.getPokemon(poke2Ident as PokemonIdent);
        const poke2Index = latestTurn.getNodeIndex(poke2?.originalIdent);

        const edge = new Edge();
        edge.addMajorArg(argName);
        edge.setFeature(FeatureEdge.POKE1_INDEX, poke1Index);
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
        const trueIdent = poke1.originalIdent;
        if (!latestTurn.nodes.has(trueIdent)) {
            latestTurn.addNode(poke1);
        }
        const poke1Index = latestTurn.getNodeIndex(trueIdent);

        const edge = new Edge();
        edge.addMajorArg(argName);
        edge.setFeature(FeatureEdge.MOVE_TOKEN, moveIndex);
        edge.setFeature(FeatureEdge.POKE1_INDEX, poke1Index);
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
        edge.setFeature(FeatureEdge.EDGE_TYPE_TOKEN, EdgeTypes.CANT_EDGE);
        edge.updateEdgeFromOf(condition);

        latestTurn.appendEdge(edge);
    }

    "|faint|"(args: Args["|faint|"]) {
        const [argName, poke1Ident] = args;

        const latestTurn = this.getLatestTurn();

        const poke1 = this.getPokemon(poke1Ident)!;
        const poke1Index = latestTurn.getNodeIndex(poke1.originalIdent);

        const edge = new Edge();

        edge.addMajorArg(argName);
        edge.setFeature(FeatureEdge.POKE1_INDEX, poke1Index);
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
        edge.setFeature(FeatureEdge.EDGE_TYPE_TOKEN, EdgeTypes.EFFECT_EDGE);
        edge.updateEdgeFromOf(fromEffect);

        latestTurn.appendEdge(edge);
    }

    "|-notarget|"(args: Args["|-notarget|"]) {
        const [argName] = args;

        const latestTurn = this.getLatestTurn();

        const edge = new Edge();

        edge.addMinorArg(argName);
        edge.setFeature(FeatureEdge.EDGE_TYPE_TOKEN, EdgeTypes.EFFECT_EDGE);

        latestTurn.appendEdge(edge);
    }

    "|-miss|"(args: Args["|-miss|"], kwArgs: KWArgs["|-miss|"]) {
        const [argName, userIdent] = args;

        const latestTurn = this.getLatestTurn();
        const user = this.getPokemon(userIdent)!;
        const userIndex = latestTurn.getNodeIndex(user.originalIdent);
        const fromEffect = this.getCondition(kwArgs.from);

        const edge = new Edge();

        edge.addMinorArg(argName);
        edge.setFeature(FeatureEdge.POKE1_INDEX, userIndex);
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

        const fromEffect = this.getCondition(kwArgs.from);

        if (!this.currHp.has(true1Ident)) {
            this.currHp.set(true1Ident, 1);
        }
        const prevHp = this.currHp.get(true1Ident) ?? 1;
        const currHp = poke1.hp / poke1.maxhp;
        const diffRatio = currHp - prevHp;
        this.currHp.set(true1Ident, currHp);

        edge.addMinorArg(argName);
        edge.setFeature(FeatureEdge.POKE1_INDEX, poke1Index);
        edge.setFeature(FeatureEdge.DAMAGE_TOKEN, Math.floor(1024 * diffRatio));
        edge.setFeature(FeatureEdge.EDGE_TYPE_TOKEN, EdgeTypes.EFFECT_EDGE);
        edge.updateEdgeFromOf(fromEffect);

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
        edge.setFeature(FeatureEdge.EDGE_TYPE_TOKEN, EdgeTypes.EFFECT_EDGE);
        edge.setFeature(FeatureEdge.POKE1_INDEX, poke1Index);

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
        const featureIndex = EventHandler2.getStatBoostEdgeFeatureIndex(stat);

        const edge = new Edge();

        if (!!kwArgs.of) {
            const poke2 = this.getPokemon(kwArgs.of)!;
            const true2Ident = poke2.originalIdent;
            const poke2Index = latestTurn.getNodeIndex(true2Ident);
            edge.setFeature(FeatureEdge.POKE2_INDEX, poke2Index);
        }

        edge.setFeature(FeatureEdge.POKE1_INDEX, poke1Index);
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
        const featureIndex = EventHandler2.getStatBoostEdgeFeatureIndex(stat);

        const edge = new Edge();

        if (!!kwArgs.of) {
            const poke2 = this.getPokemon(kwArgs.of)!;
            const true2Ident = poke2.originalIdent;
            const poke2Index = latestTurn.getNodeIndex(true2Ident);
            edge.setFeature(FeatureEdge.POKE2_INDEX, poke2Index);
        }

        edge.setFeature(FeatureEdge.POKE1_INDEX, poke1Index);
        edge.setFeature(FeatureEdge.EDGE_TYPE_TOKEN, EdgeTypes.EFFECT_EDGE);
        edge.setFeature(featureIndex, -parseInt(value));
        edge.addMinorArg(argName);

        latestTurn.appendEdge(edge);
    }

    "|-setboost|"(args: Args["|-setboost|"], kwArgs: KWArgs["|-setboost|"]) {}

    "|-swapboost|"(
        args: Args["|-swapboost|"],
        kwArgs: KWArgs["|-swapboost|"],
    ) {}

    "|-invertboost|"(
        args: Args["|-invertboost|"],
        kwArgs: KWArgs["|-invertboost|"],
    ) {}

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
        edge.setFeature(FeatureEdge.POKE1_INDEX, poke1Index);
        edge.updateEdgeFromOf(fromEffect);

        latestTurn.appendEdge(edge);
    }

    "|-clearnegativeboost|"(
        args: Args["|-clearnegativeboost|"],
        kwArgs: KWArgs["|-clearnegativeboost|"],
    ) {}

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
        edge.updateEdgeFromOf(fromEffect);

        latestTurn.appendEdge(edge);
    }

    "|-fieldstart|"(
        args: Args["|-fieldstart|"],
        kwArgs: KWArgs["|-fieldstart|"],
    ) {}

    "|-fieldend|"(args: Args["|-fieldend|"], kwArgs: KWArgs["|-fieldend|"]) {}

    "|-sidestart|"(
        args: Args["|-sidestart|"],
        kwArgs: KWArgs["|-sidestart|"],
    ) {}

    "|-sideend|"(args: Args["|-sideend|"], kwArgs: KWArgs["|-sideend|"]) {}

    "|-swapsideconditions|"(args: Args["|-swapsideconditions|"]) {}

    "|-start|"(args: Args["|-start|"], kwArgs: KWArgs["|-start|"]) {
        const [argName, poke1Ident, conditionId] = args;

        const poke1 = this.getPokemon(poke1Ident)!;
        const latestTurn = this.getLatestTurn();
        const poke1Index = latestTurn.getNodeIndex(poke1.originalIdent);

        const edge = new Edge();
        edge.addMinorArg(argName);
        edge.setFeature(FeatureEdge.POKE1_INDEX, poke1Index);
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
        edge.setFeature(FeatureEdge.EDGE_TYPE_TOKEN, EdgeTypes.EFFECT_EDGE);

        latestTurn.appendEdge(edge);
    }

    "|-supereffective|"(args: Args["|-supereffective|"]) {
        const [argName, poke1Ident] = args;

        const target = this.getPokemon(poke1Ident)!;
        const latestTurn = this.getLatestTurn();
        const targetIndex = latestTurn.getNodeIndex(target.originalIdent);

        const edge = new Edge();
        edge.addMinorArg(argName);
        edge.setFeature(FeatureEdge.POKE1_INDEX, targetIndex);
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
        const [argName, sourceIdent, conditionId, targetIdent] = args;

        const latestTurn = this.getLatestTurn();
        const edge = new Edge();

        if (sourceIdent) {
            const source = this.getPokemon(sourceIdent)!;
            const sourceIndex = latestTurn.getNodeIndex(source.originalIdent);
            edge.setFeature(FeatureEdge.POKE1_INDEX, sourceIndex);
        }

        // if (targetIdent) {
        //     const target = this.getPokemon(targetIdent)!;
        //     const targetIndex = latestTurn.getNodeIndex(target.originalIdent);
        //     edge.setFeature(FeatureEdge.POKE1_INDEX, targetIndex);
        // }

        const fromEffect = this.getCondition(kwArgs.from);

        edge.updateEdgeFromOf(fromEffect);
        edge.addMinorArg(argName);
        edge.setFeature(FeatureEdge.EDGE_TYPE_TOKEN, EdgeTypes.EFFECT_EDGE);

        latestTurn.appendEdge(edge);
    }

    "|-mustrecharge|"(args: Args["|-mustrecharge|"]) {
        const [argName, userIdent] = args;
        const user = this.getPokemon(userIdent)!;
        const latestTurn = this.getLatestTurn();
        const sourceIndex = latestTurn.getNodeIndex(user.originalIdent);

        const edge = new Edge();

        edge.setFeature(FeatureEdge.POKE1_INDEX, sourceIndex);
        edge.setFeature(FeatureEdge.EDGE_TYPE_TOKEN, EdgeTypes.EFFECT_EDGE);
        edge.addMinorArg(argName);

        latestTurn.appendEdge(edge);
    }

    "|-prepare|"(args: Args["|-prepare|"], kwArgs?: KWArgs["|-prepare|"]) {
        const [argName, userIdent] = args;
        const user = this.getPokemon(userIdent)!;
        const latestTurn = this.getLatestTurn();
        const sourceIndex = latestTurn.getNodeIndex(user.originalIdent);

        const edge = new Edge();

        edge.setFeature(FeatureEdge.POKE1_INDEX, sourceIndex);
        edge.setFeature(FeatureEdge.EDGE_TYPE_TOKEN, EdgeTypes.EFFECT_EDGE);
        edge.addMinorArg(argName);

        latestTurn.appendEdge(edge);
    }

    "|-hitcount|"(args: Args["|-hitcount|"], kwArgs?: KWArgs["|-hitcount|"]) {
        const [argName, userIdent, hitCount] = args;
        const user = this.getPokemon(userIdent)!;
        const latestTurn = this.getLatestTurn();
        const sourceIndex = latestTurn.getNodeIndex(user.originalIdent);

        const edge = new Edge();

        edge.setFeature(FeatureEdge.POKE1_INDEX, sourceIndex);
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
        const initTurn = new Turn(this, this.handler.publicBattle.turn);
        this.turns = [initTurn];
    }

    reset() {
        this.history = [];
        this.resetTurns();
        this.currHp = new Map();
        this.actives = new Map();
    }
}

export class StateHandler {
    handler: StreamHandler;

    constructor(handler: StreamHandler) {
        this.handler = handler;
    }

    static getLegalActions(request?: AnyObject | null): LegalActions {
        const legalActions = new LegalActions();

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
                        legalActions[`setSwitch${switchIndex}`](true);
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
                        legalActions[`setMove${moveIndex}`](true);
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
                    legalActions[`setSwitch${switchIndex}`](true);
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
                    legalActions[`setSwitch${switchIndex}`](true);
                }
            }
        }
        return legalActions;
    }

    getMoveset(): Uint8Array {
        const movesetArr = new Int16Array(numMovesetFields);
        let offset = 0;

        const playerIndex = this.handler.getPlayerIndex();
        const PlaceMoveset = (member: Pokemon) => {
            const moveSlots = member.moveSlots.slice(0, 4);
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
            for (
                let remainingIndex = 0;
                remainingIndex < member.side.totalPokemon;
                remainingIndex++
            ) {
                movesetArr[offset + FeatureMoveset.MOVEID] =
                    MovesEnum.MOVES__SWITCH;
                movesetArr[offset + FeatureMoveset.PPUSED] = 0;
                offset += numMoveFields;
            }
            for (
                let remainingIndex = member.side.totalPokemon;
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

            for (const potentialSwitch of [mySide.team[0], oppSide.team[0]]) {
                PlaceMoveset(potentialSwitch);
            }
        }

        return new Uint8Array(movesetArr.buffer);
    }

    getPrivateTeam(playerIndex: number): Int32Array {
        const side = this.handler.privateBattle.sides[playerIndex];
        const team: Int32Array[] = [];
        for (const member of side.team) {
            team.push(getArrayFromPokemon(member));
        }
        for (
            let memberIndex = team.length;
            memberIndex < side.totalPokemon;
            memberIndex++
        ) {
            team.push(unkPokemon);
        }
        for (let memberIndex = team.length; memberIndex < 6; memberIndex++) {
            team.push(nonePokemon);
        }
        return concatenateArrays(team);
    }

    getPublicTeam(playerIndex: number): Int32Array {
        const side = this.handler.publicBattle.sides[playerIndex];
        const team: Int32Array[] = [];
        for (const member of side.team) {
            team.push(getArrayFromPokemon(member));
        }
        for (
            let memberIndex = team.length;
            memberIndex < side.totalPokemon;
            memberIndex++
        ) {
            team.push(unkPokemon);
        }
        for (let memberIndex = team.length; memberIndex < 6; memberIndex++) {
            team.push(nonePokemon);
        }
        return concatenateArrays(team);
    }

    getHistory(numHistory: number = 8) {
        const latestTurn = this.handler.eventHandler2.getLatestTurn();

        const latestTurnCopy = latestTurn.newTurn();
        latestTurnCopy.updateFromBattle();
        latestTurnCopy.lockTurn();

        const historyEdges: Int32Array[] = [latestTurnCopy.edgeData];
        const historyNodes: Int32Array[] = [latestTurnCopy.nodeData];

        for (const { edgeData, nodeData } of [
            ...this.handler.eventHandler2.turns
                .slice(-(numHistory + 1), -1)
                .reverse(),
        ]) {
            historyEdges.push(edgeData);
            historyNodes.push(nodeData);
        }
        return {
            historyEdges: concatenateArrays(historyEdges),
            historyNodes: concatenateArrays(historyNodes),
        };
    }

    async getState(): Promise<State> {
        const state = new State();

        const info = new Info();
        info.setGameid(this.handler.gameId);
        const playerIndex = this.handler.getPlayerIndex() as number;
        info.setPlayerindex(!!playerIndex);
        info.setTurn(this.handler.privateBattle.turn);

        // const heuristicAction = getEvalAction(this.handler, 11);
        // info.setHeuristicaction(heuristicAction.getIndex());

        // const heuristicDist = await GetSearchDistribution({
        //     handler: this.handler,
        // });
        // info.setHeuristicdist(new Uint8Array(heuristicDist.buffer));

        const legalActions = StateHandler.getLegalActions(
            this.handler.privateBattle.request,
        );
        state.setLegalactions(legalActions);

        state.setInfo(info);

        state.setTeam(new Uint8Array(this.getPrivateTeam(playerIndex).buffer));

        const { historyEdges, historyNodes } = this.getHistory();
        const history = new History();
        history.setEdges(new Uint8Array(historyEdges.buffer));
        history.setNodes(new Uint8Array(historyNodes.buffer));
        state.setHistory(history);

        state.setMoveset(this.getMoveset());

        return state;
    }
}
