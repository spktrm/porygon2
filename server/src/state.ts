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
} from "../protos/history_pb";
import { PseudoweatherMessage } from "../protos/messages_pb";
import { MappingLookup, EnumKeyMapping, EnumMappings, Mappings } from "./data";
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

function SanitizeKey(key: string): string {
    return key.replace(/[^\w]|_/g, "").toLowerCase();
}

function IndexValueFromEnum<T extends EnumMappings>(
    mappingType: Mappings,
    key: string
): T[keyof T] {
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

    static getPokemon(pokemon: Pokemon): PokemonPb {
        const pb = new PokemonPb();
        pb.setSpecies(
            IndexValueFromEnum<typeof SpeciesEnum>(
                "Species",
                pokemon.species.baseSpecies.toLowerCase()
            )
        );

        pb.setGender(
            IndexValueFromEnum<typeof GendersEnum>("Genders", pokemon.gender)
        );

        pb.setFainted(pokemon.fainted);
        pb.setHp(pokemon.hp);
        pb.setMaxhp(pokemon.maxhp);

        pb.setStatus(
            pokemon.status
                ? IndexValueFromEnum<typeof StatusesEnum>(
                      "Statuses",
                      pokemon.status
                  )
                : StatusesEnum.STATUSES_NULL
        );

        pb.setLevel(pokemon.level);

        const item = pokemon.item;
        const itemEffect = pokemon.itemEffect ?? pokemon.lastItemEffect;
        pb.setItem(
            IndexValueFromEnum<typeof ItemsEnum>(
                "Items",
                item === "" ? "unk" : item
            )
        );
        pb.setItemeffect(
            IndexValueFromEnum<typeof ItemeffectEnum>(
                "ItemEffects",
                itemEffect === "" ? "unk" : itemEffect
            )
        );

        const ability = pokemon.ability;
        const abilityKey = ability ? ability : "unk";
        pb.setAbility(
            IndexValueFromEnum<typeof AbilitiesEnum>("Abilities", abilityKey)
        );

        for (const [moveIndex_, move] of pokemon.moveSlots
            .slice(-4)
            .entries()) {
            const { id, ppUsed } = move;
            const moveIndex = (moveIndex_ + 1) as 1 | 2 | 3 | 4;
            pb[`setMove${moveIndex}id`](
                IndexValueFromEnum<typeof MovesEnum>("Moves", id)
            );
            pb[`setPp${moveIndex}used`](ppUsed);
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
                    IndexValueFromEnum<typeof BoostsEnum>("Boosts", stat)
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
        move?: string
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
                : MovesEnum.MOVES_NONE
        );
        step.setWeather(
            IndexValueFromEnum<typeof WeathersEnum>(
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
        move?: string
    ) {
        const state = this.getPublicState(action, move);
        this.history.push(state);
    }

    "|move|"(args: Args["|move|"], kwArgs: KWArgs["|move|"]) {
        const move = args[2];
        this.addPublicState(ActionTypeEnum.MOVE, move);
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
                IndexValueFromEnum<typeof HyphenargsEnum>(
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
