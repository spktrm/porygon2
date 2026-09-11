/** Approximate sampled simulator worlds, constructed without the live simulator. */
import { AnyObject, Battle, Dex, PokemonSet, PRNG, PRNGSeed } from "@pkmn/sim";
import { TeamGenerators } from "@pkmn/randoms";
import { ObservedBattle, ObservedPokemon } from "./mcts_observation";
import { searchPriors, SearchPriorName } from "./search_priors";
import { SearchState } from "./mcts";
import { getSearchPotential, SearchPotential } from "./search_potentials";

export function normaliseChoice(choice: string): string {
    // Only the singles adapter calls this: its sole opposing target is implicit.
    return choice.replace(/^(move \d+) -?\d+(?= |$)/, "$1").trim();
}

export function simulatorActions(battle: Battle, sideIndex: 0 | 1): string[] {
    const side = battle.sides[sideIndex];
    const request = side.activeRequest as AnyObject | null;
    if (!request || request.wait) return ["wait"];
    if (request.teamPreview) return ["default"];
    const actions: string[] = [];
    if (request.active) {
        const active = request.active[0];
        for (let slot = 0; slot < active.moves.length; slot++) {
            const move = active.moves[slot];
            if (move.disabled || move.pp === 0) continue;
            actions.push(`move ${slot + 1}`);
            if (active.canTerastallize)
                actions.push(`move ${slot + 1} terastallize`);
        }
        if (!actions.length) actions.push("move 1"); // The simulator's Struggle request.
    }
    if (request.forceSwitch || !request.active?.[0]?.trapped) {
        for (let slot = 0; slot < side.pokemon.length; slot++) {
            const candidate = side.pokemon[slot];
            if (!candidate.isActive && candidate.hp > 0 && !candidate.fainted)
                actions.push(`switch ${slot + 1}`);
        }
    }
    if (!actions.length) return ["default"];
    return actions;
}

export function setPriorSpecies(name: string): string {
    const species = Dex.species.get(name);
    const base = Dex.species.get(species.baseSpecies);
    if (base.cosmeticFormes?.includes(species.name)) return base.name;
    // Antique has identical battle mechanics, but is an otherForme in the Dex.
    if (species.id === "polteageistantique") return "Polteageist";
    return name;
}

function sampleSet(
    observed: ObservedPokemon,
    generator: AnyObject,
    own: boolean,
): PokemonSet {
    let sampled: AnyObject;
    if (own) {
        sampled = {
            species: observed.species,
            moves: [...observed.moves],
            ability: observed.ability,
            item: observed.item ?? "",
            level: observed.level,
            nature: "Serious",
            evs: { hp: 85, atk: 85, def: 85, spa: 85, spd: 85, spe: 85 },
        };
    } else {
        try {
            sampled = generator.randomSet(setPriorSpecies(observed.species));
        } catch {
            throw new Error("unsupported_species_prior");
        }
        // Keep revealed facts. Unrevealed slots are a random-battle set prior,
        // not a posterior conditioned on every observed damage roll.
        const known = [
            ...new Set(observed.moves.map((move) => Dex.toID(move))),
        ];
        if (known.length > 4) throw new Error("unsupported_revealed_moves");
        const remaining = sampled.moves.filter(
            (move: string) => !known.includes(Dex.toID(move)),
        );
        sampled.moves = [...known, ...remaining].slice(0, 4);
    }
    sampled.species = observed.species;
    sampled.name = observed.species;
    sampled.level = observed.level;
    if (observed.ability !== undefined) sampled.ability = observed.ability;
    if (observed.item !== undefined) sampled.item = observed.item;
    if (observed.teraType) sampled.teraType = observed.teraType;
    if (observed.terastallized) sampled.teraType = observed.terastallized;
    return sampled as PokemonSet;
}

export function buildSampledBattle(
    observation: ObservedBattle,
    seed: PRNGSeed,
): Battle {
    const generator = TeamGenerators.getTeamGenerator(
        "gen9randombattle",
        seed,
    ) as AnyObject;
    if (
        observation.own.filter((mon) => mon.active).length !== 1 ||
        observation.opponent.filter((mon) => mon.active).length !== 1
    )
        throw new Error("unsupported_active_layout");
    if (
        observation.opponentSize < observation.opponent.length ||
        observation.opponentSize > 6
    )
        throw new Error("unsupported_roster_size");
    // Preserve the real request's own roster order: switch N must retain meaning.
    if (!observation.own[0].active)
        throw new Error("unsupported_own_roster_order");
    const opposing = [...observation.opponent].sort(
        (left, right) => Number(right.active) - Number(left.active),
    );
    const seen = new Set(
        opposing.map((mon) => Dex.species.get(mon.species).baseSpecies),
    );
    const opponentSets = opposing.map((mon) =>
        sampleSet(mon, generator, false),
    );
    for (
        let attempt = 0;
        opponentSets.length < observation.opponentSize && attempt < 16;
        attempt++
    ) {
        for (const candidate of generator.getTeam() as PokemonSet[]) {
            const identity = Dex.species.get(candidate.species).baseSpecies;
            if (seen.has(identity)) continue;
            seen.add(identity);
            opponentSets.push(candidate);
            if (opponentSets.length === observation.opponentSize) break;
        }
    }
    if (opponentSets.length !== observation.opponentSize)
        throw new Error("opponent_sampling_failed");
    const ownSets = observation.own.map((mon) =>
        sampleSet(mon, generator, true),
    );
    const battle = new Battle({
        formatid: "gen9customgame" as never,
        seed,
        p1: { name: "search-own", team: ownSets },
        p2: { name: "search-opponent", team: opponentSets },
    });
    try {
        if (battle.requestState === "teampreview") {
            battle.choose("p1", "default");
            battle.choose("p2", "default");
        }
        // Synthetic switch-in events are not observations: overwrite their effects.
        battle.field.clearWeather();
        battle.field.clearTerrain();
        battle.field.pseudoWeather = {};
        for (const sideIndex of [0, 1] as const) {
            let source: ObservedPokemon[];
            let hazards: Record<string, number>;
            if (sideIndex === 0) {
                source = observation.own;
                hazards = observation.ownConditions;
            } else {
                source = opposing;
                hazards = observation.opponentConditions;
            }
            const side = battle.sides[sideIndex];
            side.sideConditions = {};
            const usedTera = source.some((mon) => Boolean(mon.terastallized));
            for (let index = 0; index < side.pokemon.length; index++) {
                const mon = side.pokemon[index];
                const observed = source[index];
                mon.volatiles = {};
                mon.boosts = {
                    atk: 0,
                    def: 0,
                    spa: 0,
                    spd: 0,
                    spe: 0,
                    accuracy: 0,
                    evasion: 0,
                };
                mon.status = "" as never;
                mon.statusState = {} as never;
                if (usedTera) mon.canTerastallize = false;
                if (!observed) continue;
                if (observed.maxhp) {
                    mon.maxhp = observed.maxhp;
                    mon.baseMaxhp = observed.maxhp;
                }
                mon.hp = Math.round(mon.maxhp * observed.hp);
                mon.fainted = observed.hp === 0;
                Object.assign(mon.boosts, observed.boosts);
                if (observed.stats) {
                    Object.assign(mon.storedStats, observed.stats);
                    Object.assign(mon.baseStoredStats, observed.stats);
                }
                if (observed.status)
                    mon.setStatus(observed.status, mon, undefined, true);
                if (observed.terastallized)
                    mon.terastallized = observed.terastallized;
                if (observed.lastMove)
                    mon.lastMove = battle.dex.getActiveMove(observed.lastMove);
                for (const move of mon.moveSlots)
                    move.pp = Math.max(
                        0,
                        move.maxpp - (observed.ppUsed[move.id] ?? 0),
                    );
            }
            side.pokemonLeft = side.pokemon.filter(
                (mon) => !mon.fainted,
            ).length;
            for (const [hazard, layers] of Object.entries(hazards)) {
                for (let layer = 0; layer < Math.max(1, layers); layer++)
                    side.addSideCondition(hazard, side.active[0]);
            }
        }
        battle.turn = observation.turn;
        const ownActive = battle.sides[0].active[0];
        const rootMoves = observation.request.active[0].moves;
        for (let index = 0; index < ownActive.moveSlots.length; index++) {
            const requested = rootMoves[index];
            if (requested && typeof requested.pp === "number")
                ownActive.moveSlots[index].pp = requested.pp;
        }
        battle.makeRequest("move");
        return battle;
    } catch (error) {
        battle.destroy();
        throw error;
    }
}

export class SimulatorSearchState implements SearchState {
    private depth = 0;
    constructor(
        readonly battle: Battle,
        private rootActions?: string[],
        private potential: SearchPotential = getSearchPotential("original"),
        private prior: SearchPriorName = "uniform",
    ) {}
    priors(side: 0 | 1, actions: string[]): number[] {
        return searchPriors(this.battle, side, actions, this.prior);
    }
    actions(side: 0 | 1): string[] {
        if (side === 0 && this.depth === 0 && this.rootActions)
            return [...this.rootActions];
        return simulatorActions(this.battle, side);
    }
    advance(ownAction: string, opponentAction: string): void {
        // Select both before choose() can resolve. Neither selection sees the other.
        if (ownAction !== "wait" && !this.battle.choose("p1", ownAction))
            throw Object.assign(new Error("sample_rejected_own_choice"), {
                choiceDetail: this.battle.sides[0].choice.error,
            });
        if (
            opponentAction !== "wait" &&
            !this.battle.choose("p2", opponentAction)
        )
            throw Object.assign(new Error("sample_rejected_opponent_choice"), {
                choiceDetail: this.battle.sides[1].choice.error,
            });
        this.depth++;
    }
    terminalValue(): number | undefined {
        if (!this.battle.ended) return undefined;
        if (this.battle.winner === this.battle.sides[0].name) return 1;
        if (this.battle.winner === this.battle.sides[1].name) return -1;
        return 0;
    }
    evaluate(): number {
        const terminal = this.terminalValue();
        if (terminal !== undefined) return terminal;
        return this.potential(this.battle);
    }
    dispose(): void {
        this.battle.destroy();
    }
}

export function makeRootSampler(
    observation: ObservedBattle,
    rootActions: string[],
    random: PRNG,
    determinisations = 3,
    potential: SearchPotential = getSearchPotential("original"),
    prior: SearchPriorName = "uniform",
): () => SearchState {
    const snapshots: string[] = [];
    return () => {
        const seed: PRNGSeed = `${random.random(65536)},${random.random(65536)},${random.random(65536)},${random.random(65536)}`;
        if (snapshots.length < determinisations) {
            const generated = buildSampledBattle(observation, seed);
            try {
                // The simulator serialiser aliases its log array. Freeze to a string
                // so every rollout gets fresh logs and state, not a shared object.
                snapshots.push(JSON.stringify(generated.toJSON()));
            } finally {
                generated.destroy();
            }
        }
        const snapshot = snapshots[random.random(snapshots.length)];
        const battle = Battle.fromJSON(snapshot);
        battle.resetRNG(seed);
        return new SimulatorSearchState(battle, rootActions, potential, prior);
    };
}
