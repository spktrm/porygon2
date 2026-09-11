import { describe, expect, test } from "vitest";
import { Battle as ClientBattle } from "@pkmn/client";
import { Generations } from "@pkmn/data";
import { Dex as ClientDex } from "@pkmn/dex";
import { Battle, PRNG, Teams } from "@pkmn/sim";
import { searchMcts, SearchState } from "./mcts";
import { evaluatePosition, PotentialPokemon } from "./position_potential";
import {
    buildSampledBattle,
    makeRootSampler,
    SimulatorSearchState,
    simulatorActions,
    setPriorSpecies,
} from "./mcts_simulator";
import { captureObservation, ObservedBattle } from "./mcts_observation";

function random(seed = "1,2,3,4" as const) {
    const generator = new PRNG(seed);
    return () => generator.random();
}

class MatrixState implements SearchState {
    result: number | undefined;
    constructor(
        private payoff: Record<string, Record<string, number>>,
        private disposed: () => void = () => {},
    ) {}
    actions(side: 0 | 1): string[] {
        if (side === 0) return Object.keys(this.payoff);
        return Object.keys(Object.values(this.payoff)[0]);
    }
    advance(own: string, opponent: string) {
        this.result = this.payoff[own][opponent];
    }
    terminalValue() {
        return this.result;
    }
    evaluate(): number {
        throw new Error("Terminal outcome must bypass heuristic evaluation");
    }
    dispose() {
        this.disposed();
    }
}

function observation(): ObservedBattle {
    return {
        own: [
            {
                species: "Pikachu",
                level: 100,
                hp: 0.75,
                maxhp: 200,
                stats: { atk: 200, def: 150, spa: 300, spd: 150, spe: 250 },
                status: "",
                boosts: { spa: 1 },
                active: true,
                moves: ["thunderbolt", "protect"],
                ppUsed: {},
                ability: "static",
                item: "",
                teraType: "Electric",
                lastMove: "",
            },
        ],
        opponent: [
            {
                species: "Blastoise",
                level: 80,
                hp: 1,
                status: "",
                boosts: {},
                active: true,
                moves: ["surf"],
                ppUsed: {},
                lastMove: "surf",
            },
        ],
        opponentSize: 1,
        ownConditions: {},
        opponentConditions: {},
        turn: 8,
        request: {
            active: [
                {
                    moves: [
                        { id: "thunderbolt", pp: 12 },
                        { id: "protect", pp: 8 },
                    ],
                },
            ],
        },
    };
}

function pokemon(kind: string, hp = 1, active = true): PotentialPokemon {
    return { hp, active, offensiveTypes: [kind], defensiveTypes: [kind] };
}

describe("service-only potential MCTS", () => {
    test("set priors canonicalise cosmetic identities without collapsing battle formes", () => {
        expect(setPriorSpecies("Florges-Yellow")).toBe("Florges");
        expect(setPriorSpecies("Minior-Orange")).toBe("Minior");
        expect(setPriorSpecies("Polteageist-Antique")).toBe("Polteageist");
        expect(setPriorSpecies("Raichu-Alola")).toBe("Raichu-Alola");
        expect(setPriorSpecies("Minior-Meteor")).toBe("Minior-Meteor");
    });

    test.each(["Florges-Yellow", "Minior-Orange", "Polteageist-Antique"])(
        "fainted %s does not block a sampled world",
        (species) => {
            const observed = observation();
            observed.opponent.push({
                ...observed.opponent[0],
                species,
                active: false,
                hp: 0,
                moves: [],
                lastMove: "",
            });
            observed.opponentSize = 2;
            const battle = buildSampledBattle(observed, "1,2,3,4");
            try {
                expect(battle.sides[1].pokemon[1].set.species).toBe(species);
                expect(battle.sides[1].pokemon[1].hp).toBe(0);
                expect(battle.sides[1].pokemon[1].fainted).toBe(true);
            } finally {
                battle.destroy();
            }
        },
    );

    test("learns a terminal win and actually explores the losing alternative", () => {
        let disposals = 0;
        const result = searchMcts(
            () =>
                new MatrixState(
                    { win: { reply: 1 }, lose: { reply: -1 } },
                    () => disposals++,
                ),
            { iterations: 128, maxDepth: 2, random: random() },
        );
        expect(result.action).toBe("win");
        expect(
            result.root.find((arm) => arm.action === "lose")!.visits,
        ).toBeGreaterThan(0);
        expect(disposals).toBe(128);
    });
    test("opponent learns to punish a trap instead of being treated as random", () => {
        const result = searchMcts(
            () =>
                new MatrixState({
                    safe: { punish: 0.3, cooperate: 0.3 },
                    trap: { punish: -1, cooperate: 1 },
                }),
            { iterations: 2000, maxDepth: 1, random: random() },
        );
        expect(result.action).toBe("safe");
    });
    test("reproducible seeds and deadline bound iterations without skipping cleanup", () => {
        const run = () =>
            searchMcts(
                () =>
                    new MatrixState({
                        first: { reply: 0 },
                        second: { reply: 0 },
                    }),
                { iterations: 32, maxDepth: 1, random: random() },
            );
        expect(run()).toEqual(run());
        let clock = 0;
        let disposed = 0;
        const result = searchMcts(
            () => new MatrixState({ first: { reply: 0 } }, () => disposed++),
            {
                iterations: 100,
                maxDepth: 1,
                maxMillis: 1,
                now: () => clock++,
                random: random(),
            },
        );
        expect(result.iterations).toBe(1);
        expect(disposed).toBe(1);
    });
    test("depth-limited leaves use potential, but terminal outcomes override it", () => {
        class LeafState implements SearchState {
            selected = "";
            actions() {
                return ["good", "bad"];
            }
            advance(own: string) {
                this.selected = own;
            }
            terminalValue() {
                return undefined;
            }
            evaluate() {
                if (this.selected === "good") return 0.7;
                return -0.7;
            }
            dispose() {}
        }
        const result = searchMcts(() => new LeafState(), {
            iterations: 128,
            maxDepth: 1,
            random: random(),
        });
        expect(result.action).toBe("good");
    });
    test("potential matches the Python singles fixture and is symmetric in doubles", () => {
        const own = [
            pokemon("Fire", 0.5),
            ...Array.from({ length: 5 }, () => pokemon("Normal", 1, false)),
        ];
        const opponent = [
            pokemon("Grass"),
            ...Array.from({ length: 5 }, () => pokemon("Normal", 1, false)),
        ];
        expect(evaluatePosition(own, opponent)).toBeCloseTo(
            Math.tanh(
                (-0.5 * 0.6309440873897149 + 2 * 0.04568841302462828) / 2,
            ),
            12,
        );
        const ownDoubles = [pokemon("Fire", 0.5), pokemon("Water")];
        const opposingDoubles = [pokemon("Grass"), pokemon("Electric")];
        expect(evaluatePosition(ownDoubles, opposingDoubles)).toBeCloseTo(
            -evaluatePosition(opposingDoubles, ownDoubles),
            12,
        );
        expect(
            evaluatePosition([...ownDoubles].reverse(), opposingDoubles),
        ).toBe(evaluatePosition(ownDoubles, opposingDoubles));
    });
    test("sampled worlds retain own facts and revealed opponent moves without mutating observations", () => {
        const observed = observation();
        const before = JSON.stringify(observed);
        const battle = buildSampledBattle(observed, "1,2,3,4");
        try {
            expect(battle.turn).toBe(8);
            expect(battle.sides[0].active[0].hp).toBe(150);
            expect(battle.sides[0].active[0].storedStats.spa).toBe(300);
            expect(battle.sides[0].active[0].boosts.spa).toBe(1);
            expect(battle.sides[0].active[0].moveSlots[0].pp).toBe(12);
            expect(battle.sides[1].active[0].moves).toContain("surf");
            expect(JSON.stringify(observed)).toBe(before);
        } finally {
            battle.destroy();
        }
    });
    test("real simulator search chooses legal actions and never changes its sampled root", () => {
        const observed = observation();
        const generator = new PRNG("1,2,3,4");
        const result = searchMcts(
            makeRootSampler(observed, ["move 1", "move 2"], generator, 2),
            { iterations: 24, maxDepth: 3, random: () => generator.random() },
        );
        expect(["move 1", "move 2"]).toContain(result.action);
        expect(result.iterations).toBe(24);
        expect(observed.own[0].hp).toBe(0.75);
    });
    test("rollout clones cannot append to the cached root log or mutate its HP", () => {
        const sample = makeRootSampler(
            observation(),
            ["move 1", "move 2"],
            new PRNG("1,2,3,4"),
            1,
        );
        const original = sample() as SimulatorSearchState;
        const rootLogLength = original.battle.log.length;
        original.dispose();
        for (let iteration = 0; iteration < 80; iteration++) {
            const state = sample() as SimulatorSearchState;
            try {
                expect(state.battle.log.length).toBe(rootLogLength);
                expect(state.battle.sides[0].active[0].hp).toBe(150);
                state.advance("move 1", simulatorActions(state.battle, 1)[0]);
                expect(state.battle.log.length).toBeGreaterThan(rootLogLength);
            } finally {
                state.dispose();
            }
        }
    });
    test("a forced replacement and a terminal outcome are handled inside rollouts", () => {
        const battle = new Battle({
            formatid: "gen9customgame" as never,
            seed: "1,2,3,4",
            p1: {
                name: "own",
                team: Teams.pack([
                    {
                        species: "Miraidon",
                        moves: ["electrodrift"],
                        level: 100,
                    },
                ] as never),
            },
            p2: {
                name: "opp",
                team: Teams.pack([
                    { species: "Magikarp", moves: ["splash"], level: 1 },
                    { species: "Magikarp", moves: ["splash"], level: 1 },
                ] as never),
            },
        });
        const state = new SimulatorSearchState(battle);
        try {
            battle.choose("p1", "default");
            battle.choose("p2", "default");
            state.advance("move 1", "move 1");
            expect(simulatorActions(battle, 0)).toEqual(["wait"]);
            expect(simulatorActions(battle, 1)).toContain("switch 2");
            state.advance("wait", "switch 2");
            state.advance("move 1", "move 1");
            expect(state.terminalValue()).toBe(1);
            expect(state.evaluate()).toBe(1);
        } finally {
            state.dispose();
        }
    });
    test("successful capture cannot read live hidden state and public changes remain visible", () => {
        const view = new ClientBattle(new Generations(ClientDex));
        for (const line of [
            "|gen|9",
            "|gametype|singles",
            "|player|p1|Own",
            "|player|p2|Opp",
            "|teamsize|p1|1",
            "|teamsize|p2|1",
            "|start",
            "|switch|p1a: Pikachu|Pikachu, L100|75/100",
            "|switch|p2a: Blastoise|Blastoise, L80|100/100",
            "|turn|8",
        ])
            view.add(line as never);
        const request = {
            active: observation().request.active,
            side: {
                pokemon: [
                    {
                        ident: "p1: Pikachu",
                        details: "Pikachu, L100",
                        condition: "150/200",
                        active: true,
                        moves: ["thunderbolt", "protect"],
                        stats: {
                            atk: 200,
                            def: 150,
                            spa: 300,
                            spd: 150,
                            spe: 250,
                        },
                        baseAbility: "static",
                        item: "",
                        teraType: "Electric",
                    },
                ],
            },
        };
        const guarded = {
            publicBattle: view,
            getRequest: () => request,
            getPlayerIndex: () => 0,
            get opponent(): never {
                throw new Error("live opponent read");
            },
            get privateBattle(): never {
                throw new Error("private client bypass");
            },
            get liveSimulator(): never {
                throw new Error("live simulator read");
            },
        };
        try {
            const captured = captureObservation(guarded as never);
            expect(captured.own[0].hp).toBe(0.75);
            expect(captured.opponent[0].moves).toEqual([]);
            const again = captureObservation(guarded as never);
            expect(again).toEqual(captured);
            view.add("|-damage|p2a: Blastoise|50/100" as never);
            const changed = captureObservation(guarded as never);
            expect(changed.opponent[0].hp).toBe(0.5);
            expect(changed).not.toEqual(captured);
            const first = buildSampledBattle(captured, "1,2,3,4");
            const second = buildSampledBattle(again, "1,2,3,4");
            try {
                expect(first.toJSON()).toEqual(second.toJSON());
            } finally {
                first.destroy();
                second.destroy();
            }
        } finally {
            view.destroy();
        }
    });
    test("public-only capture refuses unsupported requests before reading hidden state", () => {
        const forbidden = () => {
            throw new Error("hidden data was accessed");
        };
        const player = {
            publicBattle: { gen: { num: 9 }, gameType: "doubles" },
            getRequest: () => ({ active: [{}] }),
            getPlayerIndex: () => 0,
            get opponent() {
                return forbidden();
            },
            get privateBattle() {
                return forbidden();
            },
        };
        expect(() => captureObservation(player as never)).toThrow(
            "unsupported_format",
        );
    });
});
