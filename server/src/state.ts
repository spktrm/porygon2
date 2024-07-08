import { AnyObject } from "@pkmn/sim";
import { StreamHandler } from "./game";
import { Args, KWArgs, Protocol } from "@pkmn/protocol";

import { Info, LegalActions, State } from "../protos/state_pb";
import { Pokemon } from "../protos/pokemon_pb";
import {
    AbilitiesEnum,
    AbilitiesEnumMap,
    BoostsEnum,
    BoostsEnumMap,
    HyphenargsEnum,
    HyphenargsEnumMap,
    ItemsEnum,
    ItemsEnumMap,
    MovesEnum,
    MovesEnumMap,
    SideconditionsEnum,
    SideconditionsEnumMap,
    SpeciesEnum,
    SpeciesEnumMap,
    VolatilestatusEnum,
    VolatilestatusEnumMap,
    WeathersEnum,
    WeathersEnumMap,
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
} from "../protos/history_pb";
import {
    BoostsMessage,
    SideconditionsMessage,
    VolatilestatusMessage,
    PseudoweatherMessage,
    HyphenargsMessage,
} from "../protos/messages_pb";

type SettersOf<T> = {
    [K in keyof T as K extends `set${string}` ? K : never]: T[K] extends (
        ...args: any[]
    ) => any
        ? T[K]
        : never;
};

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

    getPublicSide(playerIndex: number): HistorySide {
        const side = this.handler.publicBattle.sides[playerIndex];
        const active = side?.active[0];
        const historySide = new HistorySide();

        if (active) {
            const activeProto = new Pokemon();

            const species = active.species;
            const speciesKey =
                `SPECIES_${species.baseSpecies.toUpperCase()}` as keyof SpeciesEnumMap;
            activeProto.setSpecies(SpeciesEnum[speciesKey]);

            const item = active.item;
            const itemKey = (
                item ? `ITEMS_${item.toUpperCase()}` : `ITEMS_UNK`
            ) as keyof ItemsEnumMap;
            activeProto.setItem(ItemsEnum[itemKey]);

            const ability = active.ability;
            const abilityKey = (
                ability ? `ABILITIES_${ability.toUpperCase()}` : `ABILITIES_UNK`
            ) as keyof AbilitiesEnumMap;
            activeProto.setAbility(AbilitiesEnum[abilityKey]);

            for (const [moveIndex_, move] of active.moveSlots
                .slice(-4)
                .entries()) {
                const { id, ppUsed } = move;
                const moveKey =
                    `MOVES_${id.toUpperCase()}` as keyof MovesEnumMap;
                const moveIndex = (moveIndex_ + 1) as 1 | 2 | 3 | 4;
                activeProto[`setMove${moveIndex}id`](MovesEnum[moveKey]);
                activeProto[`setPp${moveIndex}used`](ppUsed);
            }

            historySide.setActive(activeProto);

            for (const [stat, value] of Object.entries(active.boosts)) {
                const boost = new Boost();
                boost.setIndex(
                    BoostsEnum[
                        `BOOSTS_${stat.toUpperCase()}` as keyof BoostsEnumMap
                    ]
                );
                boost.setValue(value);
                historySide.addBoosts(boost);
            }

            for (const [stat, state] of Object.entries(active.volatiles)) {
                const volatileStatus = new Volatilestatus();
                const indexValue =
                    VolatilestatusEnum[
                        `VOLATILESTATUS_${stat.toUpperCase()}` as keyof VolatilestatusEnumMap
                    ];
                if (indexValue === undefined) {
                    throw new Error();
                }
                volatileStatus.setIndex(indexValue);
                volatileStatus.setValue(state.level ?? 1);
                historySide.addVolatilestatus(volatileStatus);
            }

            for (const [stat, state] of Object.entries(side.sideConditions)) {
                const sideCondition = new Sidecondition();
                const indexValue =
                    SideconditionsEnum[
                        `SIDECONDITIONS_${stat.toUpperCase()}` as keyof SideconditionsEnumMap
                    ];
                if (indexValue === undefined) {
                    throw new Error();
                }
                sideCondition.setIndex(indexValue);
                sideCondition.setValue(state.level ?? 1);
                historySide.addSideconditions(sideCondition);
            }
        }

        return historySide;
    }

    getPublicState(
        action?: ActionTypeEnumMap[keyof ActionTypeEnumMap],
        move?: MovesEnumMap[keyof MovesEnumMap]
    ): HistoryStep {
        const playerIndex = this.handler.playerIndex as number;
        const battle = this.handler.publicBattle;
        const weather = battle.currentWeather().toString().toUpperCase();
        const step = new HistoryStep();
        step.setP1(this.getPublicSide(playerIndex));
        step.setP2(this.getPublicSide(1 - playerIndex));
        if (action) {
            step.setAction(action);
        }
        step.setMove(move ?? MovesEnum.MOVES_NONE);
        step.setWeather(
            WeathersEnum[
                `WEATHERS_${
                    weather ? weather : "NULL"
                }` as keyof WeathersEnumMap
            ]
        );
        const pseudoweather = new PseudoweatherMessage();
        for (const [pseudoWeatherId, pseudoWeatherState] of Object.entries(
            battle.field.pseudoWeather
        )) {
            const { minDuration, maxDuration, level } = pseudoWeatherState;
        }
        step.setPseudoweather(pseudoweather);
        return step;
    }

    addPublicState(
        action: ActionTypeEnumMap[keyof ActionTypeEnumMap],
        move?: MovesEnumMap[keyof MovesEnumMap]
    ) {
        const state = this.getPublicState(action, move);
        this.history.push(state);
    }

    "|move|"(args: Args["|move|"], kwArgs: KWArgs["|move|"]) {
        const move = args[2];
        const MovesKey = `MOVES_${move.toUpperCase()}` as keyof MovesEnumMap;
        this.addPublicState(ActionTypeEnum.MOVE, MovesEnum[MovesKey]);
        this.moveCounter += 1;
    }

    "|switch|"(
        args: Args["|switch|" | "|drag|" | "|replace|"],
        kwArgs?: KWArgs["|switch|"]
    ) {
        this.addPublicState(ActionTypeEnum.SWITCH);
        this.switchCounter += 1;
    }

    handleHyphenLine(args: Protocol.ArgType, kwArgs?: {}) {
        const prevState = this.history.pop() as HistoryStep;
        const newState = this.getPublicState();
        const prevAction = prevState.getAction();
        const playerIndex = this.handler.playerIndex as 0 | 1;

        const users = [];
        if (args[1]) {
            const potentialUser = (parseInt((args[1] as string).slice(1, 2)) -
                1) as 0 | 1;
            users.push(potentialUser);
        } else {
            users.push(...[0, 1]);
        }
        for (const user of users) {
            const side = (
                playerIndex === user ? prevState.getP1() : prevState.getP2()
            ) as HistorySide;

            const hyphenArg = new HyphenArg();
            const hyphenArgKey = `HYPHENARGS_${args[0].slice(1).toUpperCase()}`;
            hyphenArg.setIndex(
                HyphenargsEnum[hyphenArgKey as keyof HyphenargsEnumMap]
            );
            hyphenArg.setValue(true);
            side.addHyphenargs(hyphenArg);

            playerIndex === user ? newState.setP1(side) : newState.setP2(side);
        }

        if (prevAction) {
            newState.setAction(prevAction);
        }
        const prevMove = prevState?.getMove();
        if (prevMove) {
            newState.setMove(prevMove);
        }
        this.history.push(newState);
    }

    reset() {
        this.history = [];
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
            -this.handler.eventHandler.historyMaxSize
        ))
            state.addHistory(historyStep);

        return state;
    }
}
