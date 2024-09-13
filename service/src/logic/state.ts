import { AnyObject, StatID } from "@pkmn/sim";
import { Args, KWArgs, PokemonIdent, Protocol } from "@pkmn/protocol";
import { Info, LegalActions, State } from "../../protos/state_pb";
import {
    AbilitiesEnum,
    BoostsEnum,
    GendersEnum,
    ItemeffectEnum,
    ItemsEnum,
    MovesEnum,
    MovesEnumMap,
    SideconditionsEnum,
    SpeciesEnum,
    StatusesEnum,
    VolatilestatusEnum,
    WeathersEnum,
} from "../../protos/enums_pb";
import {
    ActionTypeEnum,
    ActionTypeEnumMap,
    History,
} from "../../protos/history_pb";
import {
    EdgeTypes,
    FeatureAdditionalInformation,
    FeatureAdditionalInformationMap,
    FeatureEdge,
    FeatureEdgeMap,
    FeatureEntity,
    FeatureMoveset,
    FeatureTurnContext,
    FeatureWeather,
} from "../../protos/features_pb";
import {
    MappingLookup,
    EnumKeyMapping,
    EnumMappings,
    Mappings,
    MoveIndex,
    numBoosts,
    numVolatiles,
    numSideConditions,
    numPseudoweathers,
    numPokemonFields,
    HistoryStep,
    SideObject,
    FieldObject,
    numWeatherFields,
    numTurnContextFields,
    sideIdMapping,
    numMovesetFields,
    numMoveFields,
    numAdditionalInformations,
    numBattleMinorArgs,
    numBattleMajorArgs,
} from "./data";
import { Pokemon, Side } from "@pkmn/client";
import { OneDBoolean } from "./arr";
import { StreamHandler } from "./handler";
import { Ability, Item, Move, BoostID, StatusName } from "@pkmn/dex-types";

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

function concatenateUint8Arrays(arrays: Uint8Array[]): Uint8Array {
    // Step 1: Calculate the total length
    let totalLength = 0;
    for (const arr of arrays) {
        totalLength += arr.length;
    }

    const result = new Uint8Array(totalLength);

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

function getUnkPokemon() {
    const data = getBlankPokemonArr();
    data[FeatureEntity.SPECIES] = SpeciesEnum.SPECIES_UNK;
    data[FeatureEntity.ITEM] = ItemsEnum.ITEMS_UNK;
    data[FeatureEntity.ITEM_EFFECT] = ItemeffectEnum.ITEMEFFECT_NULL;
    data[FeatureEntity.ABILITY] = AbilitiesEnum.ABILITIES_UNK;
    data[FeatureEntity.FAINTED] = 0;
    data[FeatureEntity.HP] = 100;
    data[FeatureEntity.MAXHP] = 100;
    data[FeatureEntity.STATUS] = StatusesEnum.STATUSES_NULL;
    data[FeatureEntity.MOVEID0] = MovesEnum.MOVES_UNK;
    data[FeatureEntity.MOVEID1] = MovesEnum.MOVES_UNK;
    data[FeatureEntity.MOVEID2] = MovesEnum.MOVES_UNK;
    data[FeatureEntity.MOVEID3] = MovesEnum.MOVES_UNK;
    data[FeatureEntity.HAS_STATUS] = 0;
    return new Uint8Array(data.buffer);
}

const unkPokemon = getUnkPokemon();

function getNonePokemon() {
    const data = getBlankPokemonArr();
    data[FeatureEntity.SPECIES] = SpeciesEnum.SPECIES_NONE;
    return new Uint8Array(data.buffer);
}

const nonePokemon = getNonePokemon();

export class EventHandler implements Protocol.Handler {
    readonly handler: StreamHandler;
    history: Array<HistoryStep>;
    moveCounter: number;
    switchCounter: number;
    historyMaxSize: number;
    hpRatios: { [k: string]: { turn: number; hpRatio: number }[] };
    numSwitches: { [k: string]: number };
    lastAction: [number, number];

    constructor(handler: StreamHandler, historyMaxSize: number = 8) {
        this.handler = handler;
        this.history = [];
        this.moveCounter = 0;
        this.switchCounter = 0;
        this.historyMaxSize = historyMaxSize;
        this.hpRatios = {};
        this.numSwitches = {};
        this.lastAction = [-1, -1];
    }

    getRecentDamage(ident: string) {
        const hpHistory = this.hpRatios[ident] ?? [];
        const { hpRatio: accum, turn: currentTurn } = hpHistory.at(-1) ?? {
            hpRatio: 1,
            turn: 1,
        };
        for (const { hpRatio, turn } of [...hpHistory].reverse()) {
            if (currentTurn !== turn) {
                return accum - hpRatio;
            }
        }
        return accum - (hpHistory.at(0)?.hpRatio ?? 1);
    }

    getPokemon(candidate: Pokemon | null): Uint8Array {
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
                const idValue = IndexValueFromEnum<typeof MovesEnum>(
                    "Moves",
                    id,
                );
                moveIds.push(idValue);
                movePps.push(isNaN(ppUsed) ? +!!ppUsed : ppUsed);
            }
        }
        let remainingIndex: MoveIndex = moveSlots.length as MoveIndex;
        for (remainingIndex; remainingIndex < 4; remainingIndex++) {
            moveIds.push(MovesEnum.MOVES_UNK);
            movePps.push(0);
        }

        const dataArr = getBlankPokemonArr();
        dataArr[FeatureEntity.SPECIES] = IndexValueFromEnum<typeof SpeciesEnum>(
            "Species",
            baseSpecies,
        );
        dataArr[FeatureEntity.ITEM] = item
            ? IndexValueFromEnum<typeof ItemsEnum>("Items", item)
            : ItemsEnum.ITEMS_UNK;
        dataArr[FeatureEntity.ITEM_EFFECT] = itemEffect
            ? IndexValueFromEnum<typeof ItemeffectEnum>(
                  "ItemEffects",
                  itemEffect,
              )
            : ItemeffectEnum.ITEMEFFECT_NULL;
        dataArr[FeatureEntity.ABILITY] = ability
            ? IndexValueFromEnum<typeof AbilitiesEnum>("Abilities", ability)
            : AbilitiesEnum.ABILITIES_UNK;
        dataArr[FeatureEntity.GENDER] = IndexValueFromEnum<typeof GendersEnum>(
            "Genders",
            pokemon.gender,
        );
        dataArr[FeatureEntity.ACTIVE] = pokemon.isActive() ? 1 : 0;
        dataArr[FeatureEntity.FAINTED] = pokemon.fainted ? 1 : 0;
        dataArr[FeatureEntity.HP] = pokemon.hp;
        dataArr[FeatureEntity.MAXHP] = pokemon.maxhp;
        dataArr[FeatureEntity.STATUS] = pokemon.status
            ? IndexValueFromEnum<typeof StatusesEnum>(
                  "Statuses",
                  pokemon.status,
              )
            : StatusesEnum.STATUSES_NULL;
        dataArr[FeatureEntity.HAS_STATUS] = pokemon.status ? 1 : 0;
        dataArr[FeatureEntity.TOXIC_TURNS] = pokemon.statusState.toxicTurns;
        dataArr[FeatureEntity.SLEEP_TURNS] = pokemon.statusState.sleepTurns;
        dataArr[FeatureEntity.BEING_CALLED_BACK] = pokemon.beingCalledBack
            ? 1
            : 0;
        dataArr[FeatureEntity.TRAPPED] = !!pokemon.trapped ? 1 : 0;
        dataArr[FeatureEntity.NEWLY_SWITCHED] = pokemon.newlySwitched ? 1 : 0;
        dataArr[FeatureEntity.LEVEL] = pokemon.level;
        for (let moveIndex = 0; moveIndex < 4; moveIndex++) {
            dataArr[FeatureEntity[`MOVEID${moveIndex as MoveIndex}`]] =
                moveIds[moveIndex];
            dataArr[FeatureEntity[`MOVEPP${moveIndex as MoveIndex}`]] =
                movePps[moveIndex];
        }

        if (!!!this.hpRatios[candidate.ident]) {
            this.hpRatios[candidate.ident] = [];
        }
        const hpRatio = candidate.hp / candidate.maxhp;
        if (hpRatio !== this.hpRatios[candidate.ident].at(-1)?.hpRatio)
            this.hpRatios[candidate.ident].push({
                turn: this.handler.publicBattle.turn,
                hpRatio,
            });

        return new Uint8Array(dataArr.buffer);
    }

    getSideAdditionalInformation(side: Side): Float32Array {
        const additionalInformationData = new Float32Array(
            numAdditionalInformations,
        );
        let numFainted = 0;
        let hpTotal = side.totalPokemon;

        additionalInformationData[FeatureAdditionalInformation.NUM_TYPES_UNK] =
            side.totalPokemon;

        additionalInformationData[FeatureAdditionalInformation.TOTAL_POKEMON] =
            side.totalPokemon;

        additionalInformationData[FeatureAdditionalInformation.WISHING] = +(
            side.wisher !== null
        );

        additionalInformationData[FeatureAdditionalInformation.MEMBER0_HP] = 1;
        additionalInformationData[FeatureAdditionalInformation.MEMBER1_HP] = 1;
        additionalInformationData[FeatureAdditionalInformation.MEMBER2_HP] = 1;
        additionalInformationData[FeatureAdditionalInformation.MEMBER3_HP] = 1;
        additionalInformationData[FeatureAdditionalInformation.MEMBER4_HP] = 1;
        additionalInformationData[FeatureAdditionalInformation.MEMBER5_HP] = 1;

        for (const [memberIndex, member] of side.team.entries()) {
            const hpRatio = member.hp / member.maxhp;
            additionalInformationData[
                FeatureAdditionalInformation[
                    `MEMBER${memberIndex}_HP` as keyof FeatureAdditionalInformationMap
                ]
            ] = hpRatio;

            additionalInformationData[
                FeatureAdditionalInformation.NUM_TYPES_UNK
            ] -= 1;

            numFainted += +member.fainted;
            hpTotal += hpRatio - 1;

            const numTypes = member.types.length;
            for (const type of member.types) {
                const upperCaseType = type.toUpperCase();
                const featureIndex =
                    FeatureAdditionalInformation[
                        `NUM_TYPES_${upperCaseType}` as keyof FeatureAdditionalInformationMap
                    ];
                additionalInformationData[featureIndex] += 1 / numTypes;
            }
        }

        additionalInformationData[FeatureAdditionalInformation.NUM_FAINTED] =
            numFainted;
        additionalInformationData[FeatureAdditionalInformation.HP_TOTAL] =
            hpTotal;
        return additionalInformationData;
    }

    getPublicSide(playerIndex: number): SideObject {
        const side = this.handler.publicBattle.sides[playerIndex];
        const active = side?.active[0];

        const boostsData = new Int8Array(numBoosts);
        const volatilesData = new Uint8Array(numVolatiles);
        const sideConditionsData = new Uint8Array(numSideConditions);

        const stateHandler = new StateHandler(this.handler);
        const teamArr = stateHandler.getPublicTeam(playerIndex);

        if (active) {
            for (const [stat, value] of Object.entries(active.boosts)) {
                const index = IndexValueFromEnum<typeof BoostsEnum>(
                    "Boosts",
                    stat,
                );
                boostsData[index] = value;
            }

            for (const [stat, state] of Object.entries(active.volatiles)) {
                const index = IndexValueFromEnum<typeof VolatilestatusEnum>(
                    "Volatilestatus",
                    stat,
                );
                const level = state.level ?? 1;
                volatilesData[index] = level;
            }

            for (const [stat, state] of Object.entries(side.sideConditions)) {
                const indexValue = IndexValueFromEnum<
                    typeof SideconditionsEnum
                >("Sideconditions", stat);
                const level = state.level ?? 1;
                sideConditionsData[indexValue] = level;
            }
        }

        const additionalInformationData =
            this.getSideAdditionalInformation(side);

        return {
            team: teamArr,
            boosts: new Uint8Array(boostsData.buffer),
            sideConditions: sideConditionsData,
            volatileStatus: volatilesData,
            additionalInformation: new Uint8Array(
                additionalInformationData.buffer,
            ),
        };
    }

    getPublicState(
        isMyTurn: 0 | 1,
        action: ActionTypeEnumMap[keyof ActionTypeEnumMap],
        move: MovesEnumMap[keyof MovesEnumMap],
    ): [SideObject, SideObject, FieldObject] {
        const playerIndex = this.handler.playerIndex as number;
        const battle = this.handler.publicBattle;
        const weather = battle.currentWeather().toString();

        const p1Side = this.getPublicSide(playerIndex);
        const p2Side = this.getPublicSide(1 - playerIndex);

        const weatherData = new Uint8Array(numWeatherFields);
        weatherData.fill(0);
        if (weather) {
            const weatherState = battle.field.weatherState;
            weatherData[FeatureWeather.WEATHER_ID] = IndexValueFromEnum<
                typeof WeathersEnum
            >("Weathers", weather);
            weatherData[FeatureWeather.MIN_DURATION] = weatherState.minDuration;
            weatherData[FeatureWeather.MAX_DURATION] = weatherState.maxDuration;
        } else {
            weatherData[FeatureWeather.WEATHER_ID] = WeathersEnum.WEATHERS_NULL;
        }

        // for (const [pseudoWeatherId, pseudoWeatherState] of Object.entries(
        //     battle.field.pseudoWeather,
        // )) {
        //     const pseudoweather = new PseudoWeatherPb();
        //     const { minDuration, maxDuration, level } = pseudoWeatherState;
        //     pseudoweather.setIndex(
        //         IndexValueFromEnum<typeof PseudoweatherEnum>(
        //             "PseudoWeathers",
        //             pseudoWeatherId,
        //         ),
        //     );
        //     pseudoweather.setMinduration(minDuration);
        //     pseudoweather.setMaxduration(maxDuration);
        //     step.addPseudoweather(pseudoweather);
        // }

        const turnContext = new Int16Array(numTurnContextFields);
        turnContext[FeatureTurnContext.VALID] = 1;
        turnContext[FeatureTurnContext.IS_MY_TURN] = isMyTurn;
        turnContext[FeatureTurnContext.ACTION] = action;
        turnContext[FeatureTurnContext.MOVE] = move;
        turnContext[FeatureTurnContext.SWITCH_COUNTER] = this.switchCounter;
        turnContext[FeatureTurnContext.MOVE_COUNTER] = this.moveCounter;
        turnContext[FeatureTurnContext.TURN] = this.handler.publicBattle.turn;

        if (isMyTurn) {
            this.lastAction = [action, move];
        }

        return [
            p1Side,
            p2Side,
            {
                weather: weatherData,
                turnContext: new Uint8Array(turnContext.buffer),
            },
        ];
    }

    addPublicState(
        isMyTurn: boolean,
        action: ActionTypeEnumMap[keyof ActionTypeEnumMap],
        move: MovesEnumMap[keyof MovesEnumMap],
    ) {
        const state = this.getPublicState(isMyTurn ? 1 : 0, action, move);
        this.history.push(state);
    }

    getUser(line: string): 0 | 1 {
        return sideIdMapping[line.slice(0, 2) as keyof typeof sideIdMapping];
    }

    isMyTurn(line: string): boolean {
        const playerIndex = this.handler.getPlayerIndex();
        const user = this.getUser(line);
        return playerIndex === user;
    }

    "|move|"(args: Args["|move|"], kwArgs: KWArgs["|move|"]) {
        const move = args[2];
        const isMyTurn = this.isMyTurn(args[1]);
        const moveIndex = move
            ? IndexValueFromEnum<typeof MovesEnum>("Moves", move)
            : MovesEnum.MOVES_NONE;
        this.addPublicState(isMyTurn, ActionTypeEnum.MOVE, moveIndex);
        this.moveCounter += 1;
    }

    "|switch|"(
        args: Args["|switch|" | "|drag|" | "|replace|"],
        kwArgs?: KWArgs["|switch|"],
    ) {
        const isMyTurn = this.isMyTurn(args[1]);
        this.addPublicState(
            isMyTurn,
            ActionTypeEnum.SWITCH,
            MovesEnum.MOVES_NONE,
        );
        const battle = this.handler.publicBattle;
        const pokemon = battle.getPokemon(args[1]);
        if (pokemon) {
            const ident = pokemon.ident;
            const count = this.numSwitches[ident] ?? 0;
            this.numSwitches[ident] = (count + 1) as number;
        }
        this.switchCounter += 1;
    }

    reset() {
        this.history = [];
        this.hpRatios = {};
        this.numSwitches = {};
        this.moveCounter = 0;
        this.switchCounter = 0;
    }

    resetTurn() {
        this.moveCounter = 0;
        this.switchCounter = 0;
    }
}

type StrippedArgNames = Protocol.ArgName extends `|${infer T}|` ? T : never;
type EdgeArgNames = StrippedArgNames | "switch" | "drag";

const NumEdgeFeatures = Object.keys(FeatureEdge).length;

class Edge {
    minorArgs: OneDBoolean<Int32Array>;
    majorArgs: OneDBoolean<Int32Array>;
    data: Int32Array;

    constructor() {
        this.minorArgs = new OneDBoolean(numBattleMinorArgs, Int32Array);
        this.majorArgs = new OneDBoolean(numBattleMajorArgs, Int32Array);
        this.data = new Int32Array(NumEdgeFeatures);
    }

    setFeature(index: number, value: number) {
        if (index === undefined) {
            throw new Error("Index cannot be undefined");
        }
        this.data[index] = value;
    }

    getFeature(index: number) {
        return this.data[index];
    }

    addMinorArg(argName: EdgeArgNames) {
        const index = IndexValueFromEnum("BattleMinorArgs", argName);
        this.minorArgs.toggle(index);
    }

    addMajorArg(argName: EdgeArgNames) {
        const index = IndexValueFromEnum("BattleMajorArgs", argName);
        this.majorArgs.toggle(index);
    }

    setMove(move: Move) {
        const index = IndexValueFromEnum("Moves", move.id);
        this.setFeature(FeatureEdge.MOVE_TOKEN, index);
    }

    setItem(item: Item) {
        const index = IndexValueFromEnum("Items", item.id);
        this.setFeature(FeatureEdge.ITEM_TOKEN, index);
    }

    setAbility(ability: Ability) {
        const index = IndexValueFromEnum("Abilities", ability.id);
        this.setFeature(FeatureEdge.ABILITY_TOKEN, index);
    }

    setStatus(status: StatusName) {
        const index = IndexValueFromEnum("Statuses", status);
        this.setFeature(FeatureEdge.STATUS_TOKEN, index);
    }

    toVector(): Uint8Array {
        const vector = new Uint8Array(this.data.buffer);
        vector.set(this.majorArgs.buffer, FeatureEdge.MAJOR_ARGS);
        vector.set(this.minorArgs.buffer, FeatureEdge.MINOR_ARGS1);
        return vector;
    }
}

type NodeType = "Entity" | "Side" | "Field";

type NodeArgs = {
    ident: string;
    nodeType?: NodeType;
    affectsMySide?: boolean;
    affectsOppSide?: boolean;
    nodeData?: Uint8Array;
};
class Node {
    ident: string;
    nodeType?: NodeType;
    affectsMySide?: boolean;
    affectsOppSide?: boolean;
    nodeData?: Uint8Array;

    constructor(args: NodeArgs) {
        const { ident, nodeType, affectsMySide, affectsOppSide, nodeData } =
            args;

        this.ident = ident;
        this.nodeType = nodeType;
        this.affectsMySide = affectsMySide;
        this.affectsOppSide = affectsOppSide;
        this.nodeData = nodeData;
    }
}

class EntityNode extends Node {
    entity: Pokemon;
    constructor(args: NodeArgs & { entity: Pokemon }) {
        super(args);

        this.entity = args.entity;
        this.nodeType = "Entity";
    }

    updateEntityData() {
        this.nodeData = getArrayFromPokemon(this.entity);
    }
}

class SideNode extends Node {
    constructor(args: NodeArgs) {
        super(args);
        this.nodeType = "Side";
    }
}

class FieldNode extends Node {
    constructor(args: NodeArgs) {
        super(args);
        this.nodeType = "Field";
    }
}

function getArrayFromPokemon(candidate: Pokemon | null): Uint8Array {
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
            const idValue = IndexValueFromEnum<typeof MovesEnum>("Moves", id);
            moveIds.push(idValue);
            movePps.push(isNaN(ppUsed) ? +!!ppUsed : ppUsed);
        }
    }
    let remainingIndex: MoveIndex = moveSlots.length as MoveIndex;
    for (remainingIndex; remainingIndex < 4; remainingIndex++) {
        moveIds.push(MovesEnum.MOVES_UNK);
        movePps.push(0);
    }

    const dataArr = getBlankPokemonArr();
    dataArr[FeatureEntity.SPECIES] = IndexValueFromEnum<typeof SpeciesEnum>(
        "Species",
        baseSpecies,
    );
    dataArr[FeatureEntity.ITEM] = item
        ? IndexValueFromEnum<typeof ItemsEnum>("Items", item)
        : ItemsEnum.ITEMS_UNK;
    dataArr[FeatureEntity.ITEM_EFFECT] = itemEffect
        ? IndexValueFromEnum<typeof ItemeffectEnum>("ItemEffects", itemEffect)
        : ItemeffectEnum.ITEMEFFECT_NULL;
    dataArr[FeatureEntity.ABILITY] = ability
        ? IndexValueFromEnum<typeof AbilitiesEnum>("Abilities", ability)
        : AbilitiesEnum.ABILITIES_UNK;
    dataArr[FeatureEntity.GENDER] = IndexValueFromEnum<typeof GendersEnum>(
        "Genders",
        pokemon.gender,
    );
    dataArr[FeatureEntity.ACTIVE] = pokemon.isActive() ? 1 : 0;
    dataArr[FeatureEntity.FAINTED] = pokemon.fainted ? 1 : 0;
    dataArr[FeatureEntity.HP] = pokemon.hp;
    dataArr[FeatureEntity.MAXHP] = pokemon.maxhp;
    dataArr[FeatureEntity.STATUS] = pokemon.status
        ? IndexValueFromEnum<typeof StatusesEnum>("Statuses", pokemon.status)
        : StatusesEnum.STATUSES_NULL;
    dataArr[FeatureEntity.HAS_STATUS] = pokemon.status ? 1 : 0;
    dataArr[FeatureEntity.TOXIC_TURNS] = pokemon.statusState.toxicTurns;
    dataArr[FeatureEntity.SLEEP_TURNS] = pokemon.statusState.sleepTurns;
    dataArr[FeatureEntity.BEING_CALLED_BACK] = pokemon.beingCalledBack ? 1 : 0;
    dataArr[FeatureEntity.TRAPPED] = !!pokemon.trapped ? 1 : 0;
    dataArr[FeatureEntity.NEWLY_SWITCHED] = pokemon.newlySwitched ? 1 : 0;
    dataArr[FeatureEntity.LEVEL] = pokemon.level;
    for (let moveIndex = 0; moveIndex < 4; moveIndex++) {
        dataArr[FeatureEntity[`MOVEID${moveIndex as MoveIndex}`]] =
            moveIds[moveIndex];
        dataArr[FeatureEntity[`MOVEPP${moveIndex as MoveIndex}`]] =
            movePps[moveIndex];
    }

    return new Uint8Array(dataArr.buffer);
}

class Turn {
    eventHandler: EventHandler2;

    entityNodes: Map<string, EntityNode>;
    sideNodes: Map<string, SideNode>;
    fieldNode: FieldNode;

    edges: Edge[];
    turn: number;
    order: number;

    constructor(eventHandler: EventHandler2, turn: number) {
        this.eventHandler = eventHandler;

        this.entityNodes = new Map();
        this.sideNodes = new Map();
        this.sideNodes.set(
            "me",
            new SideNode({
                ident: "me",
                affectsMySide: true,
                affectsOppSide: false,
            }),
        );
        this.sideNodes.set(
            "opp",
            new SideNode({
                ident: "opp",
                affectsMySide: false,
                affectsOppSide: true,
            }),
        );

        this.fieldNode = new FieldNode({
            ident: "field",
            affectsMySide: true,
            affectsOppSide: true,
        });

        this.edges = [];
        this.turn = turn;
        this.order = 0;
    }

    getNodeIndex(ident: string) {
        const entityNodeKeys = [...this.entityNodes.keys()];
        if (ident === "me") {
            return 12;
        } else if (ident === "opp") {
            return 13;
        } else if (ident === "field") {
            return 14;
        } else {
            const index = entityNodeKeys.indexOf(ident);
            if (index === undefined) {
                throw new Error("Index could not be found");
            }
            return index;
        }
    }

    addEntityNode(entity: Pokemon | null) {
        if (entity !== null && !this.entityNodes.has(entity.ident)) {
            const ident = entity.ident;
            const playerIndex =
                this.eventHandler.handler.getPlayerIndex() as number;
            const isMe = entity.side.n === playerIndex;
            this.entityNodes.set(
                ident,
                new EntityNode({
                    ident: entity.ident,
                    entity,
                    nodeData: getArrayFromPokemon(entity),
                    affectsMySide: isMe,
                    affectsOppSide: isMe,
                }),
            );
        }
    }

    getNodeFromIdent(ident: string) {
        switch (ident) {
            case "me":
            case "opp":
                return this.sideNodes.get(ident);
            case "field":
                return this.fieldNode;
            default:
                return this.entityNodes.get(ident);
        }
    }

    addSwitchEdge(
        argName: EdgeArgNames,
        sourceIdent: string | undefined,
        targetIdent: string,
    ) {
        const target = this.entityNodes.get(targetIdent);
        const sourceIndex = this.getNodeIndex(
            sourceIdent === undefined
                ? target?.affectsMySide
                    ? "me"
                    : "opp"
                : sourceIdent,
        );
        const targetIndex = this.getNodeIndex(targetIdent);

        const edge = new Edge();
        edge.setFeature(FeatureEdge.EDGE_TYPE_TOKEN, EdgeTypes.SWITCH_EDGE);
        edge.setFeature(FeatureEdge.SOURCE_INDEX, sourceIndex);
        edge.setFeature(FeatureEdge.TARGET_INDEX, targetIndex);
        edge.addMajorArg(argName);
        this.addEdge(edge);
        return edge;
    }

    addCantEdge(argName: EdgeArgNames, sourceIdent: string) {
        const sourceIndex = this.getNodeIndex(sourceIdent);
        const edge = new Edge();
        edge.setFeature(FeatureEdge.EDGE_TYPE_TOKEN, EdgeTypes.CANT_EDGE);
        edge.setFeature(FeatureEdge.SOURCE_INDEX, sourceIndex);
        edge.setFeature(FeatureEdge.TARGET_INDEX, sourceIndex);
        edge.addMajorArg(argName);
        return this.addEdge(edge);
    }

    addMoveEdge(sourceIdent: string, targetIdent: string) {
        const sourceIndex = this.getNodeIndex(sourceIdent);
        const targetIndex = this.getNodeIndex(targetIdent);
        const edge = new Edge();
        edge.setFeature(FeatureEdge.EDGE_TYPE_TOKEN, EdgeTypes.MOVE_EDGE);
        edge.setFeature(FeatureEdge.SOURCE_INDEX, sourceIndex);
        edge.setFeature(FeatureEdge.TARGET_INDEX, targetIndex);
        edge.addMajorArg("move");
        return this.addEdge(edge);
    }

    addEffectEdge(
        argName: EdgeArgNames,
        sourceIdent?: string,
        targetIdent?: string,
    ) {
        const edge = new Edge();
        edge.setFeature(FeatureEdge.EDGE_TYPE_TOKEN, EdgeTypes.EFFECT_EDGE);
        if (sourceIdent !== undefined) {
            const sourceIndex = this.getNodeIndex(sourceIdent);
            edge.setFeature(FeatureEdge.SOURCE_INDEX, sourceIndex);
        }
        if (targetIdent !== undefined) {
            const targetIndex = this.getNodeIndex(targetIdent);
            edge.setFeature(FeatureEdge.TARGET_INDEX, targetIndex);
        }
        edge.addMinorArg(argName);
        return this.addEdge(edge);
    }

    addEdge(edge: Edge) {
        edge.setFeature(FeatureEdge.TURN_ORDER_VALUE, this.order);
        this.edges.push(edge);
        this.order += 1;
        return edge;
    }

    getLatestEdge() {
        const latestEdge = this.edges.at(-1);
        if (latestEdge) {
            return latestEdge;
        } else {
            throw new Error("No Edges");
        }
    }
}

const fromTypes = new Set();
const fromSources = new Set();
const effects = new Set();
const activateEffects = new Set();

export class EventHandler2 implements Protocol.Handler {
    readonly handler: StreamHandler;
    history: any[];
    order: number;
    insideMove: boolean;
    currentObject: { [k: string]: any };
    currHp: Map<string, number>;
    actives: Map<string, string>;
    lastKey: Protocol.ArgName | undefined;
    turns: Turn[];

    constructor(handler: StreamHandler) {
        this.handler = handler;
        this.history = [];
        this.currentObject = {};
        this.currHp = new Map();
        this.actives = new Map();
        this.insideMove = false;
        this.order = 0;
        this.lastKey = undefined;

        this.turns = [];
        this.resetTurns();
    }

    getLatestTurn(): Turn {
        return this.turns.at(-1) as Turn;
    }

    storeLastKey(key: Protocol.ArgName) {
        if (this.lastKey?.startsWith("|-") && !key.startsWith("|-")) {
            this.process();
        }
        this.lastKey = key;
    }

    getFromOf(kwArgs: { [k: string]: any }) {
        const { from, of, upkeep } = kwArgs;
        let fromType: string | undefined = undefined,
            fromSource: string | undefined = undefined;
        if (from) {
            [fromType, fromSource] = (from ?? "").includes(": ")
                ? (from ?? "").split(": ")
                : ["effect", from];
        }
        return { fromType, fromSource, of, upkeep };
    }

    handleBase(argName: EdgeArgNames) {
        const latestTurn = this.getLatestTurn();
        const isMinorArg = argName.startsWith("-");
        if (latestTurn.edges.length === 0) {
            if (isMinorArg) {
                latestTurn.addEffectEdge(argName);
            }
        }
        const latestEdge = latestTurn.getLatestEdge();
        if (isMinorArg) {
            latestEdge.addMinorArg(argName);
        } else {
            latestEdge.addMajorArg(argName);
        }
        return { latestTurn, latestEdge };
    }

    getMoveFromDex(moveId: string) {
        return this.handler.privateBattle.gens.dex.moves.get(moveId);
    }

    getItemFromDex(itemId: string) {
        return this.handler.privateBattle.gens.dex.items.get(itemId);
    }

    getAbilityFromDex(abilityId: string) {
        return this.handler.privateBattle.gens.dex.abilities.get(abilityId);
    }

    updateEdgeFromOf(edge: Edge, kwArgs: { [k: string]: any }) {
        const { fromType, fromSource, of } = this.getFromOf(kwArgs);
        if (fromType) {
            fromTypes.add(fromType);
            // edge.setFeature(FeatureEdge.FROMTYPE, fromType);
        }
        if (fromSource) {
            fromSources.add(fromSource);
            // edge.setFeature(FeatureEdge.FROMTYPE, fromType);
        }
        switch (fromType) {
            case "move":
                if (fromSource) {
                    const move = this.getMoveFromDex(fromSource);
                    edge.setMove(move);
                }
                break;
            case "ability":
                if (fromSource) {
                    const ability = this.getAbilityFromDex(fromSource);
                    edge.setAbility(ability);
                }
                break;
            case "item":
                if (fromSource) {
                    const item = this.getItemFromDex(fromSource);
                    edge.setItem(item);
                }
                break;
            case "effect":
                effects.add(fromSource);
                // edge.setFeature(FeatureEdge.EFFECT, fromType);
                break;
        }
        return edge;
    }

    "|move|"(args: Args["|move|"], kwArgs: KWArgs["|move|"]) {
        const latestTurn = this.getLatestTurn();

        this.insideMove = true;

        const userIdent = args[1];
        const user = this.handler.privateBattle.getPokemon(
            userIdent,
        ) as Pokemon;
        const moveId = args[2];
        const actionType = "move";
        const targetIdent = args[3];
        const move = this.getMoveFromDex(moveId);

        this.currentObject.userIdent = userIdent;
        this.currentObject.target = move.target;
        this.currentObject.actionType = actionType;
        this.currentObject.moveId = moveId;

        const { fromType, fromSource, of } = this.getFromOf(kwArgs);
        this.currentObject.moveFromType = fromType;
        this.currentObject.moveFromSource = fromSource;
        this.currentObject.moveOf = of;

        let edges: Edge[] = [];
        if (targetIdent) {
            edges = [latestTurn.addMoveEdge(userIdent, targetIdent)];
        } else {
            switch (move.target) {
                case "normal":
                    edges = user.side.foe.active.flatMap((target) =>
                        target
                            ? latestTurn.addMoveEdge(userIdent, target.ident)
                            : [],
                    );
                    break;
                // case "adjacentAlly":
                //     break;
                // case "adjacentAllyOrSelf":
                //     break;
                // case "adjacentFoe":
                //     break;
                case "all":
                    edges = [latestTurn.addMoveEdge(userIdent, "field")];
                    break;
                // case "allAdjacent":
                //     break;
                // case "allAdjacentFoes":
                //     break;
                // case "allies":
                //     break;
                // case "allySide":
                //     break;
                // case "allyTeam":
                //     break;
                // case "any":
                //     break;
                case "foeSide":
                    edges = [latestTurn.addMoveEdge(userIdent, "opp")];
                    break;
                // case "randomNormal":
                //     break;
                // case "scripted":
                //     break;
                case "self":
                    edges = [latestTurn.addMoveEdge(userIdent, userIdent)];
                    break;
                default:
                    edges = user.side.foe.active.flatMap((target) =>
                        target
                            ? latestTurn.addMoveEdge(userIdent, target.ident)
                            : [],
                    );
                    break;
            }
            if (edges.length === 0) {
            }
        }
        edges.map((edge) => {
            edge.setMove(move);
        });
    }

    "|drag|"(args: Args["|drag|"], kwArgs?: KWArgs["|drag|"]) {
        this.handleSwitch(args, kwArgs);
    }

    "|switch|"(args: Args["|switch|"], kwArgs?: KWArgs["|switch|"]) {
        this.process();
        this.handleSwitch(args, kwArgs);
        this.process();
    }

    handleSwitch(
        args: Args["|switch|" | "|drag|"],
        kwArgs?: KWArgs["|switch|" | "|drag|"],
    ) {
        const latestTurn = this.getLatestTurn();

        const targetIdent = args[1];
        const target = this.handler.privateBattle.getPokemon(targetIdent);

        latestTurn.addEntityNode(target);

        let userIdent = undefined;
        if (target) {
            const sideId = target.side.id;

            if (this.actives.has(sideId)) {
                userIdent = this.actives.get(sideId);
            }

            this.actives.set(sideId, targetIdent);
        }

        latestTurn.addSwitchEdge(args[0], userIdent, targetIdent);
    }

    "|cant|"(args: Args["|cant|"], kwArgs: KWArgs["|cant|"]) {
        const userIdent = args[1];
        const latestTurn = this.getLatestTurn();
        latestTurn.addCantEdge(args[0], userIdent);
    }

    "|faint|"(args: Args["|faint|"]) {
        const { latestTurn, latestEdge } = this.handleBase(args[0]);
    }

    "|-fail|"(args: Args["|-fail|"], kwArgs: KWArgs["|-fail|"]) {
        const { latestTurn, latestEdge } = this.handleBase(args[0]);
    }

    "|-block|"(args: Args["|-block|"], kwArgs: KWArgs["|-block|"]) {
        const { latestTurn, latestEdge } = this.handleBase(args[0]);
    }

    "|-notarget|"(args: Args["|-notarget|"]) {
        const { latestTurn, latestEdge } = this.handleBase(args[0]);
    }

    "|-miss|"(args: Args["|-miss|"], kwArgs: KWArgs["|-miss|"]) {
        const { latestTurn, latestEdge } = this.handleBase(args[0]);
    }

    "|-damage|"(args: Args["|-damage|"], kwArgs: KWArgs["|-damage|"]) {
        const argName = args[0];

        const targetIdent = args[1];
        const target = this.handler.privateBattle.getPokemon(targetIdent);

        const latestTurn = this.getLatestTurn();
        const { fromType, fromSource, of } = this.getFromOf(kwArgs);
        if (fromType) {
            const effectEdge = latestTurn.addEffectEdge(
                argName,
                targetIdent,
                targetIdent,
            );
            this.updateEdgeFromOf(effectEdge, kwArgs);
        }
        const latestEdge = latestTurn.getLatestEdge();

        let diffRatio = 0;
        if (target) {
            if (!this.currHp.has(targetIdent)) {
                this.currHp.set(targetIdent, 1);
            }
            const prevHp = this.currHp.get(targetIdent) ?? 1;
            const currHp = target.hp / target.maxhp;
            diffRatio = currHp - prevHp;
            this.currHp.set(targetIdent, currHp);
        }

        latestEdge.setFeature(
            FeatureEdge.DAMAGE_TOKEN,
            Math.floor(1024 * diffRatio),
        );
    }

    "|-heal|"(args: Args["|-heal|"], kwArgs: KWArgs["|-heal|"]) {
        const argName = args[0];

        const targetIdent = args[1];
        const target = this.handler.privateBattle.getPokemon(targetIdent);

        const latestTurn = this.getLatestTurn();
        const { fromType, fromSource, of } = this.getFromOf(kwArgs);
        if (fromType) {
            const effectEdge = latestTurn.addEffectEdge(
                argName,
                targetIdent,
                targetIdent,
            );
            this.updateEdgeFromOf(effectEdge, kwArgs);
        }
        const latestEdge = latestTurn.getLatestEdge();

        let diffRatio = 0;
        if (target) {
            if (!this.currHp.has(targetIdent)) {
                this.currHp.set(targetIdent, 1);
            }
            const prevHp = this.currHp.get(targetIdent) ?? 1;
            const currHp = target.hp / target.maxhp;
            diffRatio = currHp - prevHp;
            this.currHp.set(targetIdent, currHp);
        }

        latestEdge.setFeature(
            FeatureEdge.DAMAGE_TOKEN,
            Math.floor(1024 * diffRatio),
        );
    }

    "|-status|"(args: Args["|-status|"], kwArgs: KWArgs["|-status|"]) {
        const targetIdent = args[1];
        const statusId = args[2];

        const { latestEdge } = this.handleBase(args[0]);
        latestEdge.setStatus(statusId);
    }

    "|-curestatus|"(
        args: Args["|-curestatus|"],
        kwArgs: KWArgs["|-curestatus|"],
    ) {
        const latestTurn = this.getLatestTurn();
        const latestEdge = latestTurn.addEffectEdge(args[0], args[1], args[1]);
        this.updateEdgeFromOf(latestEdge, kwArgs);
    }

    "|-cureteam|"(args: Args["|-cureteam|"], kwArgs: KWArgs["|-cureteam|"]) {
        this.currentObject.cureteam = true;

        const userIdent = args[1];
        const user = this.handler.privateBattle.getPokemon(userIdent);
        const latestTurn = this.getLatestTurn();

        if (user) {
            for (const member of user.side.team) {
                if (member.status) {
                    const latestEdge = latestTurn.addEffectEdge(
                        args[0],
                        userIdent,
                        member.ident,
                    );
                    latestEdge.addMinorArg(args[0]);
                    this.updateEdgeFromOf(latestEdge, kwArgs);
                }
            }
        }
    }

    static getStatBoostEdgeFeatureIndex(stat: BoostID): number {
        return FeatureEdge[
            `BOOST_${stat.toLocaleUpperCase()}_VALUE` as keyof FeatureEdgeMap
        ];
    }

    "|-boost|"(args: Args["|-boost|"], kwArgs: KWArgs["|-boost|"]) {
        const targetIdent = args[1];
        const stat = args[2] as BoostID;
        const value = args[3];

        const { latestTurn, latestEdge } = this.handleBase(args[0]);

        const targetIndex = latestTurn.getNodeIndex(targetIdent);
        latestEdge.setFeature(FeatureEdge.TARGET_INDEX, targetIndex);

        const featureIndex = EventHandler2.getStatBoostEdgeFeatureIndex(stat);
        latestEdge.setFeature(featureIndex, parseInt(value));
    }

    "|-unboost|"(args: Args["|-unboost|"], kwArgs: KWArgs["|-unboost|"]) {
        const targetIdent = args[1];
        const stat = args[2] as BoostID;
        const value = args[3];

        const { latestTurn, latestEdge } = this.handleBase(args[0]);

        const targetIndex = latestTurn.getNodeIndex(targetIdent);
        latestEdge.setFeature(FeatureEdge.TARGET_INDEX, targetIndex);

        const featureIndex = EventHandler2.getStatBoostEdgeFeatureIndex(stat);
        latestEdge.setFeature(featureIndex, -parseInt(value));
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
    ) {}

    "|-clearnegativeboost|"(
        args: Args["|-clearnegativeboost|"],
        kwArgs: KWArgs["|-clearnegativeboost|"],
    ) {}

    "|-copyboost|"(
        args: Args["|-copyboost|"],
        kwArgs: KWArgs["|-copyboost|"],
    ) {}

    "|-weather|"(args: Args["|-weather|"], kwArgs: KWArgs["|-weather|"]) {
        const weather = args[1];

        this.currentObject.weather = weather;

        const { fromType, fromSource, of, upkeep } = this.getFromOf(kwArgs);
        this.currentObject.weatherFromType = fromType;
        this.currentObject.weatherFromSource = fromSource;
        this.currentObject.weatherOf = of;
        this.currentObject.weatherUpKeep = upkeep;
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
        const { latestTurn, latestEdge } = this.handleBase(args[0]);
    }

    "|-end|"(args: Args["|-end|"], kwArgs: KWArgs["|-end|"]) {
        const { latestTurn, latestEdge } = this.handleBase(args[0]);
    }

    "|-crit|"(args: Args["|-crit|"]) {
        const { latestTurn, latestEdge } = this.handleBase(args[0]);
    }

    "|-supereffective|"(args: Args["|-supereffective|"]) {
        const { latestTurn, latestEdge } = this.handleBase(args[0]);
    }

    "|-resisted|"(args: Args["|-resisted|"]) {
        const { latestTurn, latestEdge } = this.handleBase(args[0]);
    }

    "|-immune|"(args: Args["|-immune|"], kwArgs: KWArgs["|-immune|"]) {
        const { latestTurn, latestEdge } = this.handleBase(args[0]);
    }

    "|-item|"(args: Args["|-item|"], kwArgs: KWArgs["|-item|"]) {}

    "|-enditem|"(args: Args["|-enditem|"], kwArgs: KWArgs["|-enditem|"]) {
        const userIdent = args[1];
        const itemId = args[2];
        const item = this.getItemFromDex(itemId);

        const latestTurn = this.getLatestTurn();
        const latestEdge = latestTurn.addEffectEdge(
            args[0],
            userIdent,
            userIdent,
        );
        latestEdge.setItem(item);
    }

    "|-ability|"(args: Args["|-ability|"], kwArgs: KWArgs["|-ability|"]) {
        const userIdent = args[1];
        const latestTurn = this.getLatestTurn();
        const latestEdge = latestTurn.addEffectEdge(
            args[0],
            userIdent,
            "field",
        );
        this.updateEdgeFromOf(latestEdge, kwArgs);
    }

    "|-endability|"(
        args: Args["|-endability|"],
        kwArgs: KWArgs["|-endability|"],
    ) {
        const userIdent = args[1];
        const latestTurn = this.getLatestTurn();
        const latestEdge = latestTurn.addEffectEdge(
            args[0],
            userIdent,
            "field",
        );
        this.updateEdgeFromOf(latestEdge, kwArgs);
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
        const userIdent = args[1];
        const latestTurn = this.getLatestTurn();
        const userIndex = latestTurn.getNodeIndex(userIdent);

        if (latestTurn.edges.length === 0) {
            const { fromType, fromSource, of } = this.getFromOf(kwArgs);
            const latestEdge = latestTurn.addEffectEdge(
                args[0],
                userIdent,
                of ?? userIdent,
            );
            switch (fromType) {
                case "move":
                    if (fromSource) {
                        const move = this.getMoveFromDex(fromSource);
                        latestEdge.setMove(move);
                    }
                    break;
                case "ability":
                    if (fromSource) {
                        const ability = this.getAbilityFromDex(fromSource);
                        latestEdge.setAbility(ability);
                    }
                    break;
                case "item":
                    if (fromSource) {
                        const item = this.getItemFromDex(fromSource);
                        latestEdge.setItem(item);
                    }
                    break;
                default:
                    break;
            }
            // latestEdge.setFeature(FeatureEdge.ACTIVATEEFFECT, args[2]);
            activateEffects.add(args[0]);
        } else {
            let latestEdge = latestTurn.getLatestEdge();
            const currentSourceIndex = latestEdge.getFeature(
                FeatureEdge.SOURCE_INDEX,
            );
            if (currentSourceIndex && userIndex !== currentSourceIndex) {
                latestTurn.addEffectEdge(args[0], userIdent, userIdent);
            }
            latestEdge = latestTurn.getLatestEdge();
            latestEdge.addMinorArg(args[0]);
            activateEffects.add(args[0]);
            // latestEdge.setFeature(FeatureEdge.ACTIVATEEFFECT, args[2]);
        }
    }

    "|-mustrecharge|"(
        args: Args["|-mustrecharge|"],
        kwArgs?: KWArgs["|-mustrecharge|"],
    ) {
        this.currentObject.mustRecharge = true;
    }

    "|-prepare|"(args: Args["|-prepare|"], kwArgs?: KWArgs["|-prepare|"]) {
        this.currentObject.preparing = true;
    }

    "|-hitcount|"(args: Args["|-hitcount|"], kwArgs?: KWArgs["|-hitcount|"]) {
        const hitCount = args[2];

        this.currentObject.hitcount = hitCount;
    }

    "|done|"() {
        this.insideMove = false;
        this.process();
    }

    "|turn|"(args: Args["|turn|"], kwArgs?: KWArgs["|turn|"]) {
        const currentTurn = this.getLatestTurn();
        const nextTurn = new Turn(this, this.handler.privateBattle.turn);
        nextTurn.entityNodes = new Map([...currentTurn.entityNodes.entries()]);
        for (const [key, value] of nextTurn.entityNodes.entries()) {
            const updatedEntity = this.handler.privateBattle.getPokemon(
                value.ident as PokemonIdent,
            );
            if (updatedEntity) {
                value.entity = updatedEntity;
                value.updateEntityData();
            }
            nextTurn.entityNodes.set(key, value);
        }
        this.turns.push(nextTurn);

        this.process(Math.max(0, this.handler.privateBattle.turn - 1));
        this.order = 0;
    }

    process(turn?: number, reset?: boolean) {
        if (Object.keys(this.currentObject).length > 0) {
            const actionType = this.currentObject.actionType;
            const user = this.handler.privateBattle.getPokemon(
                this.currentObject.userIdent,
            );
            const target = this.handler.privateBattle.getPokemon(
                this.currentObject.targetIdent,
            );

            switch (actionType) {
                case "move":
                    switch (this.currentObject.target) {
                        default:
                            this.currentObject.targetIdent =
                                this.currentObject.userIdent;
                    }
                    break;
                case "switch":
                    if (user === null && target) {
                        this.currentObject.targetIdent = `p${
                            target.side.n + 1
                        }`;
                    }
                    break;
            }

            this.currentObject.sender =
                this.currentObject.userIdent ?? this.currentObject.targetIdent;
            this.currentObject.reciever =
                this.currentObject.targetIdent ?? this.currentObject.userIdent;

            this.currentObject.order = this.order;
            this.currentObject.turn = +(
                turn ?? this.handler.privateBattle.turn
            );
            this.history.push(this.currentObject);
            this.order += 1;
        }
        if (reset ?? true) {
            this.currentObject = {};
        }
    }

    resetTurns() {
        const initTurn = new Turn(this, this.handler.privateBattle.turn);
        this.turns = [initTurn];
    }

    reset() {
        this.history = [];
        this.resetTurns();
        this.insideMove = false;
        this.currentObject = {};
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
                        ? MovesEnum.MOVES_UNK
                        : IndexValueFromEnum<typeof MovesEnum>("Moves", id);
                movesetArr[offset + FeatureMoveset.PPUSED] = ppUsed;
                offset += numMoveFields;
            }
            for (
                let remainingIndex = moveSlots.length;
                remainingIndex < 4;
                remainingIndex++
            ) {
                movesetArr[offset + FeatureMoveset.MOVEID] =
                    MovesEnum.MOVES_UNK;
                movesetArr[offset + FeatureMoveset.PPUSED] = 0;
                offset += numMoveFields;
            }
            for (
                let remainingIndex = 0;
                remainingIndex < member.side.totalPokemon;
                remainingIndex++
            ) {
                movesetArr[offset + FeatureMoveset.MOVEID] =
                    MovesEnum.MOVES_SWITCH;
                movesetArr[offset + FeatureMoveset.PPUSED] = 0;
                offset += numMoveFields;
            }
            for (
                let remainingIndex = member.side.totalPokemon;
                remainingIndex < 6;
                remainingIndex++
            ) {
                movesetArr[offset + FeatureMoveset.MOVEID] =
                    MovesEnum.MOVES_NONE;
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

    getPrivateTeam(playerIndex: number): Uint8Array {
        const side = this.handler.privateBattle.sides[playerIndex];
        const team = [];
        for (const member of side.team) {
            team.push(this.handler.eventHandler.getPokemon(member));
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
        return concatenateUint8Arrays(team);
    }

    getPublicTeam(playerIndex: number): Uint8Array {
        const side = this.handler.publicBattle.sides[playerIndex];
        const team = [];
        for (const member of side.team) {
            team.push(this.handler.eventHandler.getPokemon(member));
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
        return concatenateUint8Arrays(team);
    }

    constructHistory() {
        const activeArrs = [];
        const boostArrs = [];
        const sideConditionArrs = [];
        const volatileStatusArrs = [];
        const additionalInformationArrs = [];

        const weatherArrs = [];
        const turnContextArrs = [];

        const historySlice = this.handler.eventHandler.history.slice(
            -this.handler.eventHandler.historyMaxSize,
        );
        for (const historyStep of historySlice) {
            const [p1Side, p2Side, field] = historyStep;
            for (const side of [p1Side, p2Side]) {
                activeArrs.push(side.team);
                boostArrs.push(side.boosts);
                sideConditionArrs.push(side.sideConditions);
                volatileStatusArrs.push(side.volatileStatus);
                additionalInformationArrs.push(side.additionalInformation);
            }
            weatherArrs.push(field.weather);
            turnContextArrs.push(field.turnContext);
        }
        const history = new History();
        history.setActive(concatenateUint8Arrays(activeArrs));
        history.setBoosts(concatenateUint8Arrays(boostArrs));
        history.setSideconditions(concatenateUint8Arrays(sideConditionArrs));
        history.setVolatilestatus(concatenateUint8Arrays(volatileStatusArrs));
        history.setVolatilestatus(concatenateUint8Arrays(volatileStatusArrs));
        history.setAdditionalinformation(
            concatenateUint8Arrays(additionalInformationArrs),
        );
        history.setWeather(concatenateUint8Arrays(weatherArrs));
        history.setTurncontext(concatenateUint8Arrays(turnContextArrs));
        history.setLength(historySlice.length);
        return history;
    }

    getNumTurnsSinceSwitch(canMove: boolean, maxTurnsSinceSwitch: number = 10) {
        let numTurns = 1;
        while (numTurns < maxTurnsSinceSwitch) {
            const action = this.handler.actionLog.at(-numTurns);
            if (canMove && action && action.getIndex() >= 4) {
                break;
            }
            numTurns += 1;
        }
        return numTurns - 1;
    }

    async getState(): Promise<State> {
        const state = new State();

        const info = new Info();
        info.setGameid(this.handler.gameId);
        const playerIndex = this.handler.getPlayerIndex() as number;
        info.setPlayerindex(!!playerIndex);
        info.setTurn(this.handler.privateBattle.turn);
        info.setLastaction(this.handler.eventHandler.lastAction[0]);
        info.setLastmove(this.handler.eventHandler.lastAction[1]);

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

        const canMove = Object.values(legalActions.toObject())
            .slice(0, 4)
            .some((x) => !!x);
        info.setTurnssinceswitch(this.getNumTurnsSinceSwitch(canMove));
        state.setInfo(info);

        state.setHistory(this.constructHistory());
        this.handler.eventHandler.resetTurn();

        state.setTeam(this.getPrivateTeam(playerIndex));

        state.setMoveset(this.getMoveset());

        return state;
    }
}
