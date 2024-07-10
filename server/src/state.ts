import { AnyObject } from "@pkmn/sim";
import { StreamHandler } from "./game";
import { Args, KWArgs, Protocol } from "@pkmn/protocol";

import { Info, LegalActions, State } from "../protos/state_pb";
import { Pokemon as PokemonPb } from "../protos/pokemon_pb";
import {
    AbilitiesEnum,
    BoostsEnum,
    GendersEnum,
    HyphenargsEnum,
    ItemeffectEnum,
    ItemsEnum,
    MovesEnum,
    PseudoweatherEnum,
    SideconditionsEnum,
    SpeciesEnum,
    StatusesEnum,
    VolatilestatusEnum,
    WeathersEnum,
} from "../protos/enums_pb";
import {
    HistoryStep,
    HistorySide,
    ActionTypeEnum,
    ActionTypeEnumMap,
    Boost,
    Volatilestatus,
    Sidecondition,
    HyphenArg,
    Weather as WeatherPb,
    PseudoWeather as PseudoWeatherPb,
} from "../protos/history_pb";
import {
    MappingLookup,
    EnumKeyMapping,
    EnumMappings,
    Mappings,
    MoveIndex,
} from "./data";
import { Pokemon } from "@pkmn/client";

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

    static getPokemon(pokemon: Pokemon): PokemonPb {
        const pb = new PokemonPb();
        const baseSpecies = pokemon.species.baseSpecies.toLowerCase();
        pb.setSpecies(
            IndexValueFromEnum<typeof SpeciesEnum>("Species", baseSpecies),
        );

        pb.setGender(
            IndexValueFromEnum<typeof GendersEnum>("Genders", pokemon.gender),
        );

        pb.setFainted(pokemon.fainted);
        pb.setHp(pokemon.hp);
        pb.setMaxhp(pokemon.maxhp);

        pb.setStatus(
            pokemon.status
                ? IndexValueFromEnum<typeof StatusesEnum>(
                      "Statuses",
                      pokemon.status,
                  )
                : StatusesEnum.STATUSES_NULL,
        );

        pb.setLevel(pokemon.level);

        const item = pokemon.item ?? pokemon.lastItem;
        pb.setItem(
            item
                ? IndexValueFromEnum<typeof ItemsEnum>("Items", item)
                : ItemsEnum.ITEMS_UNK,
        );

        const itemEffect = pokemon.itemEffect ?? pokemon.lastItemEffect;
        pb.setItemeffect(
            itemEffect
                ? IndexValueFromEnum<typeof ItemeffectEnum>(
                      "ItemEffects",
                      itemEffect,
                  )
                : ItemeffectEnum.ITEMEFFECT_NULL,
        );

        const ability = pokemon.ability;
        pb.setAbility(
            ability
                ? IndexValueFromEnum<typeof AbilitiesEnum>("Abilities", ability)
                : AbilitiesEnum.ABILITIES_UNK,
        );

        const moveSlots = pokemon.moveSlots.slice(-4);
        if (moveSlots) {
            for (let [moveIndex, move] of moveSlots.entries()) {
                const { id, ppUsed } = move;
                pb[`setMoveid${moveIndex as MoveIndex}`](
                    IndexValueFromEnum<typeof MovesEnum>("Moves", id),
                );
                pb[`setPpused${moveIndex as MoveIndex}`](ppUsed);
            }
        }
        let remainingIndex: MoveIndex = moveSlots.length as MoveIndex;
        for (remainingIndex; remainingIndex < 4; remainingIndex++) {
            pb[`setMoveid${remainingIndex}`](MovesEnum.MOVES_UNK);
        }

        return pb;
    }

    getPublicSide(playerIndex: number): HistorySide {
        const side = this.handler.publicBattle.sides[playerIndex];
        const active = side?.active[0];
        const historySide = new HistorySide();

        if (active) {
            const pb = EventHandler.getPokemon(active);
            historySide.setActive(pb);

            for (const [stat, value] of Object.entries(active.boosts)) {
                const boost = new Boost();
                boost.setIndex(
                    IndexValueFromEnum<typeof BoostsEnum>("Boosts", stat),
                );
                boost.setValue(value);
                historySide.addBoosts(boost);
            }

            for (const [stat, state] of Object.entries(active.volatiles)) {
                const volatileStatus = new Volatilestatus();
                const indexValue = IndexValueFromEnum<
                    typeof VolatilestatusEnum
                >("Volatilestatus", stat);
                volatileStatus.setIndex(indexValue);
                volatileStatus.setValue(state.level ?? 1);
                historySide.addVolatilestatus(volatileStatus);
            }

            for (const [stat, state] of Object.entries(side.sideConditions)) {
                const sideCondition = new Sidecondition();
                const indexValue = IndexValueFromEnum<
                    typeof SideconditionsEnum
                >("Sideconditions", stat);
                sideCondition.setIndex(indexValue);
                sideCondition.setValue(state.level ?? 1);
                historySide.addSideconditions(sideCondition);
            }
        } else {
            const pb = new PokemonPb();
            pb.setSpecies(SpeciesEnum.SPECIES_NONE);
            historySide.setActive(pb);
        }

        return historySide;
    }

    getPublicState(
        action?: ActionTypeEnumMap[keyof ActionTypeEnumMap],
        move?: string,
    ): HistoryStep {
        const playerIndex = this.handler.playerIndex as number;
        const battle = this.handler.publicBattle;
        const weather = battle.currentWeather().toString();
        const step = new HistoryStep();
        step.setP1(this.getPublicSide(playerIndex));
        step.setP2(this.getPublicSide(1 - playerIndex));
        if (action) {
            step.setAction(action);
        }
        step.setMove(
            move
                ? IndexValueFromEnum<typeof MovesEnum>("Moves", move)
                : MovesEnum.MOVES_NONE,
        );

        const weatherpb = new WeatherPb();
        if (weather) {
            const weatherState = battle.field.weatherState;
            weatherpb.setIndex(
                IndexValueFromEnum<typeof WeathersEnum>("Weathers", weather),
            );
            weatherpb.setMinduration(weatherState.minDuration);
            weatherpb.setMaxduration(weatherState.maxDuration);
        }
        step.setWeather(weatherpb);

        for (const [pseudoWeatherId, pseudoWeatherState] of Object.entries(
            battle.field.pseudoWeather,
        )) {
            const pseudoweather = new PseudoWeatherPb();
            const { minDuration, maxDuration, level } = pseudoWeatherState;
            pseudoweather.setIndex(
                IndexValueFromEnum<typeof PseudoweatherEnum>(
                    "PseudoWeathers",
                    pseudoWeatherId,
                ),
            );
            pseudoweather.setMinduration(minDuration);
            pseudoweather.setMaxduration(maxDuration);
            step.addPseudoweather(pseudoweather);
        }
        step.setSwitchcounter(this.switchCounter);
        step.setMovecounter(this.moveCounter);
        step.setTurn(this.handler.publicBattle.turn);
        return step;
    }

    addPublicState(
        isMyTurn: boolean,
        action: ActionTypeEnumMap[keyof ActionTypeEnumMap],
        move?: string,
    ) {
        const state = this.getPublicState(action, move);
        state.setIsmyturn(isMyTurn);
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
        this.addPublicState(isMyTurn, ActionTypeEnum.MOVE, move);
        this.moveCounter += 1;
    }

    "|switch|"(
        args: Args["|switch|" | "|drag|" | "|replace|"],
        kwArgs?: KWArgs["|switch|"],
    ) {
        const isMyTurn = this.isMyTurn(args[1]);
        this.addPublicState(isMyTurn, ActionTypeEnum.SWITCH);
        this.switchCounter += 1;
    }

    handleHyphenLine(args: Protocol.ArgType, kwArgs?: {}) {
        const prevState = this.history.pop() as HistoryStep;
        const newState = this.getPublicState();
        const prevAction = prevState.getAction();
        const playerIndex = this.handler.getPlayerIndex();

        const users = [];
        if (args[1]) {
            const potentialUser = this.getUser(args[1] as string);
            users.push(potentialUser);
        } else {
            users.push(...[0, 1]);
        }
        for (const user of users) {
            const side = (
                playerIndex === user ? prevState.getP1() : prevState.getP2()
            ) as HistorySide;

            const hyphenArg = new HyphenArg();
            const hyphenArgKey = args[0].slice(1);
            hyphenArg.setIndex(
                IndexValueFromEnum<typeof HyphenargsEnum>(
                    "Hyphenargs",
                    hyphenArgKey,
                ),
            );
            hyphenArg.setValue(true);
            side.addHyphenargs(hyphenArg);

            playerIndex === user ? newState.setP1(side) : newState.setP2(side);
        }

        if (prevAction) {
            newState.setAction(prevAction);
        }
        const prevMove = prevState.getMove();
        if (prevMove) {
            newState.setMove(prevMove);
        }
        const prevUser = prevState.getIsmyturn();
        if (prevUser !== undefined) {
            newState.setIsmyturn(prevUser);
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

    getState(): State {
        const state = new State();

        const info = new Info();
        info.setGameid(this.handler.gameId);
        info.setPlayerindex(!!this.handler.getPlayerIndex());
        info.setTurn(this.handler.privatebattle.turn);
        state.setInfo(info);

        const legalActions = this.getLegalActions();
        state.setLegalactions(legalActions);

        for (const historyStep of this.handler.eventHandler.history.slice(
            -this.handler.eventHandler.historyMaxSize,
        ))
            state.addHistory(historyStep);
        this.handler.eventHandler.resetTurn();
        return state;
    }
}
