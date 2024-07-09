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
    ItemeffectEnumMap,
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
import { PseudoweatherMessage } from "../protos/messages_pb";
import { MappingLookup, EnumKeyMapping, Mappings } from "./data";

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

function SanitizeKey(key: string): string {
    return key.replace(/[^\w]|_/g, "").toLowerCase();
}

function IndexValueFromEnum<
    T extends
        | SpeciesEnumMap
        | ItemsEnumMap
        | ItemeffectEnumMap
        | MovesEnumMap
        | AbilitiesEnumMap
        | BoostsEnumMap
        | VolatilestatusEnumMap
        | SideconditionsEnumMap
        | WeathersEnumMap
        | HyphenargsEnumMap
>(mappingType: Mappings, key: string): T[keyof T] {
    const mapping = MappingLookup[mappingType] as T;
    const enumMapping = EnumKeyMapping[mappingType];
    const sanitizedKey = SanitizeKey(key);
    const trueKey = enumMapping[sanitizedKey] as keyof T;
    const value = mapping[trueKey];
    if (value === undefined) {
        console.error(`${key.toString()} not in mapping`);
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

    getPublicSide(playerIndex: number): HistorySide {
        const side = this.handler.publicBattle.sides[playerIndex];
        const active = side?.active[0];
        const historySide = new HistorySide();

        if (active) {
            const activeProto = new Pokemon();

            activeProto.setSpecies(
                IndexValueFromEnum<SpeciesEnumMap>(
                    "Species",
                    active.species.baseSpecies.toLowerCase()
                )
            );

            const item = active.item;
            const itemEffect = active.itemEffect ?? active.lastItemEffect;
            activeProto.setItem(
                IndexValueFromEnum<ItemsEnumMap>(
                    "Items",
                    item === "" ? "unk" : item
                )
            );
            activeProto.setItemeffect(
                IndexValueFromEnum<ItemeffectEnumMap>(
                    "ItemEffects",
                    itemEffect === "" ? "unk" : itemEffect
                )
            );

            const ability = active.ability;
            const abilityKey = ability ? ability : "unk";
            activeProto.setAbility(
                IndexValueFromEnum<AbilitiesEnumMap>("Abilities", abilityKey)
            );

            for (const [moveIndex_, move] of active.moveSlots
                .slice(-4)
                .entries()) {
                const { id, ppUsed } = move;
                const moveIndex = (moveIndex_ + 1) as 1 | 2 | 3 | 4;
                activeProto[`setMove${moveIndex}id`](
                    IndexValueFromEnum<MovesEnumMap>("Moves", id)
                );
                activeProto[`setPp${moveIndex}used`](ppUsed);
            }

            historySide.setActive(activeProto);

            for (const [stat, value] of Object.entries(active.boosts)) {
                const boost = new Boost();
                boost.setIndex(
                    IndexValueFromEnum<BoostsEnumMap>("Boosts", stat)
                );
                boost.setValue(value);
                historySide.addBoosts(boost);
            }

            for (const [stat, state] of Object.entries(active.volatiles)) {
                const volatileStatus = new Volatilestatus();
                const indexValue = IndexValueFromEnum<VolatilestatusEnumMap>(
                    "Volatilestatus",
                    stat
                );
                volatileStatus.setIndex(indexValue);
                volatileStatus.setValue(state.level ?? 1);
                historySide.addVolatilestatus(volatileStatus);
            }

            for (const [stat, state] of Object.entries(side.sideConditions)) {
                const sideCondition = new Sidecondition();
                const indexValue = IndexValueFromEnum<SideconditionsEnumMap>(
                    "Sideconditions",
                    stat
                );
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
        const weather = battle.currentWeather().toString();
        const step = new HistoryStep();
        step.setP1(this.getPublicSide(playerIndex));
        step.setP2(this.getPublicSide(1 - playerIndex));
        if (action) {
            step.setAction(action);
        }
        step.setMove(move ?? MovesEnum.MOVES_NONE);
        step.setWeather(
            IndexValueFromEnum<WeathersEnumMap>(
                "Weathers",
                weather ? weather : "null"
            )
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
        this.addPublicState(
            ActionTypeEnum.MOVE,
            IndexValueFromEnum<MovesEnumMap>("Moves", move)
        );
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
            const hyphenArgKey = args[0].slice(1);
            hyphenArg.setIndex(
                IndexValueFromEnum<HyphenargsEnumMap>(
                    "Hyphenargs",
                    hyphenArgKey
                )
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
