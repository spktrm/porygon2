import { AnyObject } from "@pkmn/sim";
import { StreamHandler } from "./game";
import { Args, KWArgs, Protocol } from "@pkmn/protocol";
import { Info, LegalActions, State } from "../protos/state_pb";
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
} from "../protos/enums_pb";
import {
    ActionTypeEnum,
    ActionTypeEnumMap,
    History,
} from "../protos/history_pb";
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
} from "./data";
import { Pokemon } from "@pkmn/client";
import { TwoDBoolArray } from "./arr";

export const AllValidActions = new LegalActions();
AllValidActions.setMove1(true);
AllValidActions.setMove2(true);
AllValidActions.setMove3(true);
AllValidActions.setMove4(true);
AllValidActions.setSwitch1(true);
AllValidActions.setSwitch2(true);
AllValidActions.setSwitch3(true);
AllValidActions.setSwitch4(true);
AllValidActions.setSwitch5(true);
AllValidActions.setSwitch6(true);

const sanitizeKeyCache = new Map<string, string>();

const sideIdMapping: {
    [k in "p1" | "p2"]: 0 | 1;
} = {
    p1: 0,
    p2: 1,
};

interface SideObject {
    active: Uint8Array;
    boosts: Uint8Array;
    sideConditions: Uint8Array;
    volatileStatus: Uint8Array;
    hyphenArgs: Uint8Array;
}

interface FieldObject {
    weather: Uint8Array;
    pseudoweather: Uint8Array;
    turnContext: Uint8Array;
}

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
    const totalLength = arrays.reduce((sum, arr) => sum + arr.length, 0);

    // Step 2: Create a new Uint8Array with the total length
    const result = new Uint8Array(totalLength);

    // Step 3: Copy each array into the new array
    let offset = 0;
    arrays.forEach((arr) => {
        result.set(arr, offset);
        offset += arr.length;
    });

    return result;
}

type HistoryStep = [SideObject, SideObject, FieldObject];

export class EventHandler implements Protocol.Handler {
    readonly handler: StreamHandler;
    history: Array<HistoryStep>;
    moveCounter: number;
    switchCounter: number;
    historyMaxSize: number;

    constructor(handler: StreamHandler, historyMaxSize: number = 8) {
        this.handler = handler;
        this.history = [];
        this.moveCounter = 0;
        this.switchCounter = 0;
        this.historyMaxSize = historyMaxSize;
    }

    static getPokemon(pokemon: Pokemon | null): Uint8Array {
        if (pokemon === null) {
            const data = new Int32Array(19);
            data.fill(0);
            data[0] = SpeciesEnum.SPECIES_NONE;
            return new Uint8Array(data.buffer);
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

        const data = [
            IndexValueFromEnum<typeof SpeciesEnum>("Species", baseSpecies),
            item
                ? IndexValueFromEnum<typeof ItemsEnum>("Items", item)
                : ItemsEnum.ITEMS_UNK,
            itemEffect
                ? IndexValueFromEnum<typeof ItemeffectEnum>(
                      "ItemEffects",
                      itemEffect,
                  )
                : ItemeffectEnum.ITEMEFFECT_NULL,
            ability
                ? IndexValueFromEnum<typeof AbilitiesEnum>("Abilities", ability)
                : AbilitiesEnum.ABILITIES_UNK,
            IndexValueFromEnum<typeof GendersEnum>("Genders", pokemon.gender),
            pokemon.isActive() ? 1 : 0,
            pokemon.fainted ? 1 : 0,
            pokemon.hp,
            pokemon.maxhp,
            pokemon.status
                ? IndexValueFromEnum<typeof StatusesEnum>(
                      "Statuses",
                      pokemon.status,
                  )
                : StatusesEnum.STATUSES_NULL,
            pokemon.level,
            ...moveIds,
            ...movePps,
        ];
        return new Uint8Array(Int32Array.from(data).buffer);
    }

    getPublicSide(playerIndex: number): SideObject {
        const side = this.handler.publicBattle.sides[playerIndex];
        const active = side?.active[0];

        const boostsData = new Uint8Array(numBoosts);
        const volatilesData = new Uint8Array(numVolatiles);
        const sideConditionsData = new Uint8Array(numSideConditions);
        const activeArr = EventHandler.getPokemon(active);

        if (active) {
            for (const [stat, value] of Object.entries(active.boosts)) {
                const index = IndexValueFromEnum<typeof BoostsEnum>(
                    "Boosts",
                    stat,
                );
                boostsData[index] = value + 6;
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
            boosts: boostsData,
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

        const weatherData = new Uint8Array(3);
        weatherData.fill(0);
        if (weather) {
            const weatherState = battle.field.weatherState;
            weatherData[0] = IndexValueFromEnum<typeof WeathersEnum>(
                "Weathers",
                weather,
            );
            weatherData[1] = weatherState.minDuration;
            weatherData[2] = weatherState.maxDuration;
        } else {
            weatherData[0] = WeathersEnum.WEATHERS_NULL;
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

        return [
            p1Side,
            p2Side,
            {
                weather: weatherData,
                pseudoweather: pseudoweatherData.buffer,
                turnContext: new Uint8Array(
                    Int32Array.from([
                        isMyTurn,
                        action,
                        move,
                        this.switchCounter,
                        this.moveCounter,
                        this.handler.publicBattle.turn,
                    ]).buffer,
                ),
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
        const newState = this.getPublicState(
            prevState[2].turnContext[0] as 0 | 1,
            prevState[2].turnContext[1] as 0 | 1,
            prevState[2].turnContext[2] as MovesEnumMap[keyof MovesEnumMap],
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
        const moveset = [];
        for (const move of moves) {
            const { id, pp, maxpp } = move;
            moveset.push(IndexValueFromEnum<typeof MovesEnum>("Moves", id));
            moveset.push(pp);
            moveset.push(maxpp);
        }
        for (
            let remainingIndex = moves.length;
            remainingIndex < 4;
            remainingIndex++
        ) {
            moveset.push(MovesEnum.MOVES_NONE);
            moveset.push(0);
            moveset.push(1);
        }
        return new Uint8Array(Int32Array.from(moveset).buffer);
    }

    getTeam(): Uint8Array {
        const playerIndex = this.handler.getPlayerIndex() as number;
        const side = this.handler.privatebattle.sides[playerIndex];
        const team = [];
        for (const active of side.active) {
            team.push(EventHandler.getPokemon(active));
        }
        for (const member of side.team) {
            team.push(EventHandler.getPokemon(member));
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
        info.setPlayerindex(!!this.handler.getPlayerIndex());
        info.setTurn(this.handler.privatebattle.turn);
        state.setInfo(info);

        const legalActions = this.getLegalActions();
        state.setLegalactions(legalActions);

        state.setHistory(this.constructHistory());
        this.handler.eventHandler.resetTurn();

        state.setTeam(this.getTeam());
        state.setMoveset(this.getMoveset());

        return state;
    }
}
