import { Battle, Pokemon, Side } from "@pkmn/client";

type Scalar = string | number | boolean;
type EffectState = Pokemon["volatiles"][string];

export interface SerialisedBattle {
    turn: number;
    gameType: string;
    tier: string;
    lastMove: string;
    kickingInactive: number | string;
    totalTimeLeft: number;
    graceTimeLeft: number;
    field: ReturnType<typeof serialiseField>;
    sides: ReturnType<typeof serialiseSide>[];
    request: unknown;
    log: string[];
}

/**
 * An ALLOWLIST, not a cycle-stripping walk. `species`, `types`, `moves`,
 * `ident`, `position`, `teraType` and `isTerastallized` are prototype getters
 * on @pkmn/client's Pokemon, and JSON.stringify skips prototype getters — a
 * walk that only removed the back-references would drop them without a trace.
 * Own scalar fields are the one generic copy: they can carry neither a cycle
 * nor a datastore.
 */
function scalarFields(source: object): Record<string, Scalar> {
    const fields: Record<string, Scalar> = {};
    for (const [key, value] of Object.entries(source)) {
        if (
            typeof value === "string" ||
            typeof value === "number" ||
            typeof value === "boolean"
        ) {
            fields[key] = value;
        }
    }
    return fields;
}

function scalarTable(table: {
    [id: string]: object;
}): Record<string, Record<string, Scalar>> {
    const entries: Record<string, Record<string, Scalar>> = {};
    for (const [id, state] of Object.entries(table)) {
        entries[id] = scalarFields(state);
    }
    return entries;
}

function serialiseEffect(effect: EffectState) {
    const fields: Record<string, Scalar> = scalarFields(effect);
    if (effect.pokemon) {
        fields.pokemon = effect.pokemon.ident;
    }
    return fields;
}

function serialiseVolatiles(volatiles: Pokemon["volatiles"]) {
    const entries: Record<string, Record<string, Scalar>> = {};
    for (const [id, effect] of Object.entries(volatiles)) {
        entries[id] = serialiseEffect(effect);
    }
    return entries;
}

function serialisePokemon(pokemon: Pokemon) {
    const species = pokemon.species;
    return {
        ident: pokemon.ident,
        name: pokemon.name,
        details: pokemon.details,
        slot: pokemon.slot,
        level: pokemon.level,
        gender: pokemon.gender,
        species: {
            name: species.name,
            types: species.types,
            baseStats: { ...species.baseStats },
        },
        speciesForme: pokemon.speciesForme,
        types: pokemon.types,
        addedType: pokemon.addedType,
        teraType: pokemon.teraType,
        terastallized: pokemon.terastallized,
        isTerastallized: pokemon.isTerastallized,
        active: pokemon.isActive(),
        hp: pokemon.hp,
        maxhp: pokemon.maxhp,
        hpcolor: pokemon.hpcolor,
        status: pokemon.status,
        statusState: { ...pokemon.statusState },
        fainted: pokemon.fainted,
        boosts: { ...pokemon.boosts },
        volatiles: serialiseVolatiles(pokemon.volatiles),
        ability: pokemon.ability,
        baseAbility: pokemon.baseAbility,
        item: pokemon.item,
        itemEffect: pokemon.itemEffect,
        lastItem: pokemon.lastItem,
        lastItemEffect: pokemon.lastItemEffect,
        moveSlots: pokemon.moveSlots.map((moveSlot) => scalarFields(moveSlot)),
        lastMove: pokemon.lastMove,
        movesUsedWhileActive: [...pokemon.movesUsedWhileActive],
        newlySwitched: pokemon.newlySwitched,
        timesAttacked: pokemon.timesAttacked,
        trapped: pokemon.trapped,
        maybeTrapped: pokemon.maybeTrapped,
    };
}

function teamIndex(side: Side, pokemon: Pokemon | null): number | null {
    if (pokemon === null) {
        return null;
    }
    return side.team.indexOf(pokemon);
}

function serialiseSide(side: Side) {
    return {
        id: side.id,
        name: side.name,
        totalPokemon: side.totalPokemon,
        faints: side.faints,
        sideConditions: scalarTable(side.sideConditions),
        team: side.team.map(serialisePokemon),
        active: side.active.map((pokemon) => teamIndex(side, pokemon)),
        wisher: teamIndex(side, side.wisher),
    };
}

function serialiseField(battle: Battle) {
    const field = battle.field;
    return {
        weather: field.weather,
        weatherState: scalarFields(field.weatherState),
        terrain: field.terrain,
        terrainState: scalarFields(field.terrainState),
        pseudoWeather: scalarTable(field.pseudoWeather),
    };
}

/**
 * `battle` must be the player's OWN client battle (`privateBattle`) and `log`
 * that player's own protocol lines: the result is what goes to an external
 * decision-maker, and its information set has to be a ladder player's.
 */
export function serialiseBattle(
    battle: Battle,
    log: readonly string[],
): SerialisedBattle {
    return {
        turn: battle.turn,
        gameType: battle.gameType,
        tier: battle.tier,
        lastMove: battle.lastMove,
        kickingInactive: battle.kickingInactive,
        totalTimeLeft: battle.totalTimeLeft,
        graceTimeLeft: battle.graceTimeLeft,
        field: serialiseField(battle),
        sides: battle.sides.map(serialiseSide),
        request: battle.request,
        log: [...log],
    };
}
