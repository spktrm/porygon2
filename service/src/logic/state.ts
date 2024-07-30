import { AnyObject } from "@pkmn/sim";
import { Args, KWArgs, Protocol } from "@pkmn/protocol";
import { Info, LegalActions, State } from "../../protos/state_pb";
import {
    AbilitiesEnum,
    BoostsEnum,
    GendersEnum,
    HyphenargsEnum,
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
    numHyphenArgs,
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
} from "./data";
import { Pokemon } from "@pkmn/client";
import { TwoDBoolArray } from "./arr";
import { StreamHandler } from "./handler";
import { evalActionMapping, getEvalAction, partial } from "./eval";
import { GetBestSwitchAction } from "./baselines/switcher";

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
    data[FeatureEntity.HP] = 100;
    data[FeatureEntity.MAXHP] = 100;
    data[FeatureEntity.STATUS] = StatusesEnum.STATUSES_NULL;
    data[FeatureEntity.MOVEID0] = MovesEnum.MOVES_UNK;
    data[FeatureEntity.MOVEID1] = MovesEnum.MOVES_UNK;
    data[FeatureEntity.MOVEID2] = MovesEnum.MOVES_UNK;
    data[FeatureEntity.MOVEID3] = MovesEnum.MOVES_UNK;
    return new Uint8Array(data.buffer);
}

const unkPokemon = getUnkPokemon();

function getNonePokemon() {
    const data = getBlankPokemonArr();
    data[FeatureEntity.SPECIES] = SpeciesEnum.SPECIES_NONE;
    return new Uint8Array(data.buffer);
}

export class EventHandler implements Protocol.Handler {
    readonly handler: StreamHandler;
    history: Array<HistoryStep>;
    moveCounter: number;
    switchCounter: number;
    historyMaxSize: number;
    hpRatios: { [k: string]: { turn: number; hpRatio: number }[] };

    constructor(handler: StreamHandler, historyMaxSize: number = 8) {
        this.handler = handler;
        this.history = [];
        this.moveCounter = 0;
        this.switchCounter = 0;
        this.historyMaxSize = historyMaxSize;
        this.hpRatios = {};
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

    getPublicSide(playerIndex: number): SideObject {
        const side = this.handler.publicBattle.sides[playerIndex];
        const active = side?.active[0];

        const boostsData = new Int8Array(numBoosts);
        const volatilesData = new Uint8Array(numVolatiles);
        const sideConditionsData = new Uint8Array(numSideConditions);
        const activeArr = this.getPokemon(active);

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
        const hyphenArgData = new TwoDBoolArray(numHyphenArgs, 1);
        const hyphenArgsArr = hyphenArgData.buffer;

        return {
            active: activeArr,
            boosts: new Uint8Array(boostsData.buffer),
            sideConditions: volatilesData,
            volatileStatus: sideConditionsData,
            hyphenArgs: hyphenArgsArr,
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

        const pseudoweatherData = new TwoDBoolArray(numPseudoweathers, 1);

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
        return [
            p1Side,
            p2Side,
            {
                weather: weatherData,
                pseudoweather: pseudoweatherData.buffer,
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
        this.switchCounter += 1;
    }

    handleHyphenLine(args: Protocol.ArgType, kwArgs?: {}) {
        const prevState = this.history.pop() as HistoryStep;
        const prevField = prevState[2];
        const prevturnContext = new Int16Array(prevField.turnContext.buffer);
        const newState = this.getPublicState(
            prevturnContext[FeatureTurnContext.IS_MY_TURN] as 0 | 1,
            prevturnContext[FeatureTurnContext.ACTION] as 0 | 1,
            prevturnContext[
                FeatureTurnContext.MOVE
            ] as MovesEnumMap[keyof MovesEnumMap],
        );

        const playerIndex = this.handler.getPlayerIndex();

        const users = [];
        if (args[1]) {
            const potentialUser = this.getUser(args[1] as string);
            users.push(potentialUser);
        } else {
            users.push(...[0, 1]);
        }

        for (const user of users) {
            const side = playerIndex === user ? prevState[0] : prevState[1];
            const hyphenArgKey = args[0].slice(1);
            const hyphenArgData = side.hyphenArgs as Uint8Array;
            const hyphenArgArray = new TwoDBoolArray(
                numHyphenArgs,
                1,
                hyphenArgData,
            );
            const hyphenIndex = IndexValueFromEnum<typeof HyphenargsEnum>(
                "Hyphenargs",
                hyphenArgKey,
            );
            hyphenArgArray.set(hyphenIndex, 1);
            side.hyphenArgs = hyphenArgArray.buffer;
            playerIndex === user ? (newState[0] = side) : (newState[1] = side);
        }

        this.history.push(newState);
    }

    reset() {
        this.history = [];
        this.hpRatios = {};
        this.moveCounter = 0;
        this.switchCounter = 0;
    }

    resetTurn() {
        this.moveCounter = 0;
        this.switchCounter = 0;
    }
}

export class StateHandler {
    handler: StreamHandler;
    constructor(handler: StreamHandler) {
        this.handler = handler;
    }

    getLegalActions(): LegalActions {
        const request = this.handler.privatebattle.request as AnyObject;
        const legalActions = new LegalActions();

        if (request === undefined) {
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
        const request = this.handler.getRequest() as AnyObject;
        const moves = (request?.active ?? [{}])[0].moves ?? [];
        const movesetArr = new Int16Array(numMovesetFields);
        let offset = 0;
        for (const move of moves) {
            const { id, pp, maxpp } = move;
            movesetArr[offset + FeatureMoveset.MOVEID] = IndexValueFromEnum<
                typeof MovesEnum
            >("Moves", id);
            movesetArr[offset + FeatureMoveset.PPLEFT] = pp;
            movesetArr[offset + FeatureMoveset.PPMAX] = maxpp;
            offset += numMoveFields;
        }
        for (
            let remainingIndex = moves.length;
            remainingIndex < 4;
            remainingIndex++
        ) {
            movesetArr[offset + FeatureMoveset.MOVEID] = MovesEnum.MOVES_NONE;
            movesetArr[offset + FeatureMoveset.PPLEFT] = 0;
            movesetArr[offset + FeatureMoveset.PPMAX] = 1;
            offset += numMoveFields;
        }
        return new Uint8Array(movesetArr.buffer);
    }

    getPrivateTeam(playerIndex: number): Uint8Array {
        const side = this.handler.privatebattle.sides[playerIndex];
        const team = [];
        for (const member of side.team) {
            team.push(this.handler.eventHandler.getPokemon(member));
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
            let memberIndex = side.team.length;
            memberIndex < 6;
            memberIndex++
        ) {
            team.push(unkPokemon);
        }
        return concatenateUint8Arrays(team);
    }

    constructHistory() {
        const activeArrs = [];
        const boostArrs = [];
        const sideConditionArrs = [];
        const volatileStatusArrs = [];
        const hyphenArgArrs = [];

        const weatherArrs = [];
        const pseudoweatherArrs = [];
        const turnContextArrs = [];

        const historySlice = this.handler.eventHandler.history.slice(
            -this.handler.eventHandler.historyMaxSize,
        );
        for (const historyStep of historySlice) {
            const [p1Side, p2Side, field] = historyStep;
            for (const side of [p1Side, p2Side]) {
                activeArrs.push(side.active);
                boostArrs.push(side.boosts);
                sideConditionArrs.push(side.sideConditions);
                volatileStatusArrs.push(side.volatileStatus);
                hyphenArgArrs.push(side.hyphenArgs);
            }
            weatherArrs.push(field.weather);
            pseudoweatherArrs.push(field.pseudoweather);
            turnContextArrs.push(field.turnContext);
        }
        const history = new History();
        history.setActive(concatenateUint8Arrays(activeArrs));
        history.setBoosts(concatenateUint8Arrays(boostArrs));
        history.setSideconditions(concatenateUint8Arrays(sideConditionArrs));
        history.setVolatilestatus(concatenateUint8Arrays(volatileStatusArrs));
        history.setHyphenargs(concatenateUint8Arrays(hyphenArgArrs));
        history.setWeather(concatenateUint8Arrays(weatherArrs));
        history.setPseudoweather(concatenateUint8Arrays(pseudoweatherArrs));
        history.setTurncontext(concatenateUint8Arrays(turnContextArrs));
        history.setLength(historySlice.length);
        return history;
    }

    getState(): State {
        const state = new State();

        const info = new Info();
        info.setGameid(this.handler.gameId);
        const playerIndex = this.handler.getPlayerIndex() as number;
        info.setPlayerindex(!!playerIndex);
        info.setTurn(this.handler.privatebattle.turn);

        const heuristicAction = getEvalAction(this.handler, 10);
        info.setHeuristicaction(heuristicAction.getIndex());

        state.setInfo(info);

        const legalActions = this.getLegalActions();
        state.setLegalactions(legalActions);

        state.setHistory(this.constructHistory());
        this.handler.eventHandler.resetTurn();

        state.setTeam(this.getPrivateTeam(playerIndex));
        state.setMypublic(this.getPublicTeam(playerIndex));
        state.setOpppublic(this.getPublicTeam(1 - playerIndex));

        state.setMoveset(this.getMoveset());

        return state;
    }
}
