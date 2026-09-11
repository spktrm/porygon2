import { expect, test } from "vitest";
import { observedPotential, resolvedReply } from "./payoff_audit";
import { ObservedBattle, ObservedPokemon } from "./mcts_observation";

test("audit uses only an opponent action that became public", () => {
    const trace = [
        "|move|p1a: Own|Surf|p2a: Opp",
        "|-terastallize|p2a: Opp|Steel",
        "|move|p2a: Opp|Iron Head|p1a: Own",
    ];
    expect(resolvedReply(trace, 0)).toEqual({
        kind: "move",
        identity: "ironhead",
        tera: true,
    });
    expect(
        resolvedReply(
            [
                "|cant|p2a: Opp|flinch",
                "|switch|p2a: Next|Blastoise, L80|100/100",
            ],
            0,
        ),
    ).toBeUndefined();
    expect(
        resolvedReply(
            ["|faint|p2a: Opp", "|switch|p2a: Next|Blastoise, L80|100/100"],
            0,
        ),
    ).toBeUndefined();
    expect(
        resolvedReply(
            ["|move|p2a: Opp|Surf|p1a: Own|[from]move: Sleep Talk"],
            0,
        ),
    ).toBeUndefined();
    expect(
        resolvedReply(["|switch|p1a: Next|Blastoise, L80|100/100"], 1),
    ).toEqual({ kind: "switch", identity: "blastoise", tera: false });
});

test("public potential includes unrevealed healthy reserves and responds to actual HP", () => {
    const mon: ObservedPokemon = {
        species: "Blastoise",
        level: 80,
        hp: 1,
        status: "",
        boosts: {},
        active: true,
        moves: [],
        ppUsed: {},
        lastMove: "",
    };
    const own = Array.from({ length: 6 }, (_, index) => ({
        ...mon,
        active: index === 0,
    }));
    const observation: ObservedBattle = {
        own,
        opponent: [{ ...mon }],
        opponentSize: 6,
        ownConditions: {},
        opponentConditions: {},
        turn: 2,
        request: {},
    };
    expect(observedPotential(observation)).toBeCloseTo(0);
    own[0].hp = 0.5;
    expect(observedPotential(observation)).toBeLessThan(0);
    own[0].hp = 1;
    observation.opponent[0].hp = 0.5;
    expect(observedPotential(observation)).toBeGreaterThan(0);
});
