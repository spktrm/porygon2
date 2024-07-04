import { AnyObject } from "@pkmn/sim";
import { StreamHandler } from "./game";
import { Args, KWArgs, Protocol } from "@pkmn/protocol";

import { Info, LegalActions, State } from "../protos/state_pb";
import { Move, Pokemon } from "../protos/pokemon_pb";
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
    SpeciesEnum,
    SpeciesEnumMap,
} from "../protos/enums_pb";
import {
    HistoryStep,
    HistorySide,
    ActionTypeEnum,
    ActionTypeEnumMap,
    Boost,
} from "../protos/history_pb";

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

    constructor(handler: StreamHandler, historyMaxSize: number = 5) {
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

            for (const move of active.moveSlots) {
                const { id, ppUsed } = move;
                const moveSlot = new Move();
                const moveKey =
                    `MOVES_${id.toUpperCase()}` as keyof MovesEnumMap;
                moveSlot.setMoveid(MovesEnum[moveKey]);
                moveSlot.setPpused(ppUsed);
                activeProto.addMoveset(moveSlot);
            }

            historySide.setActive(activeProto);

            for (const [stat, value] of Object.entries(active.boosts)) {
                const boostsKey =
                    `BOOSTS_${stat.toUpperCase()}` as keyof BoostsEnumMap;
                const boost = new Boost();
                boost.setStat(BoostsEnum[boostsKey]);
                boost.setValue(value);
                historySide.addBoosts(boost);
            }
        }

        return historySide;
    }

    getPublicState(
        action?: ActionTypeEnumMap[keyof ActionTypeEnumMap],
        move?: MovesEnumMap[keyof MovesEnumMap]
    ): HistoryStep {
        const playerIndex = this.handler.playerIndex as number;
        const step = new HistoryStep();
        step.setP1(this.getPublicSide(playerIndex));
        step.setP2(this.getPublicSide(1 - playerIndex));
        if (action) {
            step.setAction(action);
        }
        step.setMove(move ?? MovesEnum.MOVES_NONE);
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
        const prevState = this.history.pop();
        const newState = this.getPublicState();
        const prevAction = prevState?.getAction();
        const hyphenArgs = prevState?.getHyphenargsList();
        if (hyphenArgs) {
            const hyphenArgKey = `HYPHENARGS_${args[0]
                .slice(1)
                .toUpperCase()}` as keyof HyphenargsEnumMap;
            newState.setHyphenargsList(hyphenArgs);
            newState.addHyphenargs(HyphenargsEnum[hyphenArgKey]);
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
