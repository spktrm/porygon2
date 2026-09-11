/** A serialisable information boundary: public client state + our own request. */
import type { Pokemon } from "@pkmn/client";
import type { AnyObject } from "@pkmn/sim";
import type { TrainablePlayerAI } from "../runner";

export interface ObservedPokemon {
    species: string;
    level: number;
    hp: number;
    status: string;
    boosts: Record<string, number>;
    active: boolean;
    moves: string[];
    ppUsed: Record<string, number>;
    ability?: string;
    item?: string;
    teraType?: string;
    terastallized?: string;
    stats?: Record<string, number>;
    maxhp?: number;
    lastMove: string;
}

export interface ObservedBattle {
    own: ObservedPokemon[];
    opponent: ObservedPokemon[];
    opponentSize: number;
    ownConditions: Record<string, number>;
    opponentConditions: Record<string, number>;
    weather?: string;
    terrain?: string;
    turn: number;
    request: AnyObject;
}

export function captureObservation(
    player: Pick<
        TrainablePlayerAI,
        "publicBattle" | "getRequest" | "getPlayerIndex"
    >,
): ObservedBattle {
    const view = player.publicBattle;
    const request = player.getRequest() as AnyObject;
    const sideIndex = player.getPlayerIndex();
    if (
        sideIndex === undefined ||
        !request?.active ||
        request.teamPreview ||
        request.forceSwitch
    )
        throw new Error("unsupported_request");
    if (view.gen.num !== 9 || view.gameType !== "singles")
        throw new Error("unsupported_format");
    // These fields carry state/duration information not faithfully reconstructed yet.
    if (Object.keys(view.field.pseudoWeather).length)
        throw new Error("unsupported_field_condition");
    const ownSide = view.sides[sideIndex];
    const opponentSide = view.sides[1 - sideIndex];
    function publicMon(mon: Pokemon): ObservedPokemon {
        if (
            Object.keys(mon.volatiles).length ||
            mon.illusion ||
            mon.addedType ||
            mon.terastallized === "Stellar"
        )
            throw new Error("unsupported_pokemon_state");
        if (mon.status === "slp" || mon.status === "tox")
            throw new Error("unsupported_status_timer");
        const ppUsed: Record<string, number> = {};
        for (const move of mon.moveSlots) ppUsed[move.id] = move.ppUsed;
        const observed: ObservedPokemon = {
            species: mon.speciesForme,
            level: mon.level,
            hp: mon.hp / Math.max(1, mon.maxhp),
            status: mon.status ?? "",
            boosts: { ...mon.boosts },
            active: mon.isActive(),
            moves: mon.moveSlots
                .filter((move) => !move.virtual)
                .map((move) => move.id),
            ppUsed,
            lastMove: mon.lastMove,
            terastallized: mon.terastallized,
        };
        if (mon.ability) observed.ability = mon.ability;
        if (mon.item) observed.item = mon.item;
        else if (mon.lastItem && mon.lastItemEffect) observed.item = "";
        return observed;
    }
    const own = request.side.pokemon.map((entry: AnyObject) => {
        const name = entry.ident.split(": ").slice(1).join(": ");
        const visible = ownSide.team.find((mon) => mon.name === name);
        let observed: ObservedPokemon;
        if (visible) observed = publicMon(visible);
        else {
            observed = {
                species: entry.details.split(",")[0],
                level: 100,
                hp: 1,
                status: "",
                boosts: {},
                active: false,
                moves: [],
                ppUsed: {},
                lastMove: "",
            };
        }
        const level = entry.details.match(/, L(\d+)/);
        if (level) observed.level = Number(level[1]);
        observed.species = entry.details.split(",")[0];
        observed.moves = [...entry.moves];
        observed.ability = entry.ability ?? entry.baseAbility;
        observed.item = entry.item;
        observed.teraType = entry.teraType;
        if (entry.terastallized) observed.terastallized = entry.terastallized;
        observed.stats = { ...entry.stats };
        observed.active = Boolean(entry.active);
        const status = entry.condition.split(" ")[1];
        if (status && status !== "fnt") observed.status = status;
        if (observed.status === "slp" || observed.status === "tox")
            throw new Error("unsupported_status_timer");
        const hp = entry.condition.match(/^(\d+)\/(\d+)/);
        if (hp) {
            observed.maxhp = Number(hp[2]);
            observed.hp = Number(hp[1]) / observed.maxhp;
        } else if (entry.condition.startsWith("0")) observed.hp = 0;
        else throw new Error("unsupported_hp");
        return observed;
    });
    function conditions(side: typeof ownSide) {
        const result: Record<string, number> = {};
        for (const [condition, details] of Object.entries(
            side.sideConditions,
        )) {
            // Permanent entry hazards are reconstructible; timer-dependent screens are not.
            if (
                !["stealthrock", "spikes", "toxicspikes", "stickyweb"].includes(
                    condition,
                )
            )
                throw new Error("unsupported_side_condition");
            result[condition] = details.level;
        }
        return result;
    }
    if (view.field.weather || view.field.terrain)
        throw new Error("unsupported_weather_or_terrain");
    return {
        own,
        opponent: opponentSide.team.map(publicMon),
        opponentSize: opponentSide.totalPokemon,
        ownConditions: conditions(ownSide),
        opponentConditions: conditions(opponentSide),
        turn: view.turn,
        request: JSON.parse(JSON.stringify(request)),
    };
}
