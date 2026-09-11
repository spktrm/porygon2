import { Battle, Teams, Dex } from "@pkmn/sim";
import { searchPriors } from "./search_priors";
import { describe, expect, test } from "vitest";
import {
    damagePressure,
    getSearchPotential,
    SEARCH_POTENTIAL_NAMES,
} from "./search_potentials";

function battleFixture() {
    const battle = new Battle({
        formatid: Dex.toID("gen9customgame"),
        seed: "1,2,3,4",
    });
    const team = Teams.import(
        "Blastoise\nAbility: Torrent\nModest Nature\n- Surf\n- Ice Beam\n- Shell Smash",
    )!;
    battle.setPlayer("p1", { name: "own", team });
    battle.setPlayer("p2", { name: "opponent", team });
    battle.choose("p1", "default");
    battle.choose("p2", "default");
    return battle;
}

describe("search-only potentials", () => {
    for (const name of SEARCH_POTENTIAL_NAMES) {
        test(`${name} is bounded, symmetric, health-sensitive and does not mutate simulation`, () => {
            const battle = battleFixture();
            try {
                const evaluate = getSearchPotential(name);
                expect(evaluate(battle)).toBeCloseTo(0);
                battle.sides[1].pokemon[0].hp = Math.floor(
                    battle.sides[1].pokemon[0].hp / 2,
                );
                const before = JSON.stringify(battle.toJSON());
                const advantage = evaluate(battle);
                expect(advantage).toBeGreaterThan(0);
                expect(advantage).toBeLessThan(1);
                expect(JSON.stringify(battle.toJSON())).toBe(before);
                [battle.sides[0], battle.sides[1]] = [
                    battle.sides[1],
                    battle.sides[0],
                ];
                expect(evaluate(battle)).toBeCloseTo(-advantage);
            } finally {
                battle.destroy();
            }
        });
    }
    test("strategic evaluator values useful setup independently of material", () => {
        const battle = battleFixture();
        try {
            battle.sides[0].active[0].boosts.spa = 2;
            expect(getSearchPotential("material")(battle)).toBeCloseTo(0);
            expect(getSearchPotential("strategic")(battle)).toBeGreaterThan(0);
        } finally {
            battle.destroy();
        }
    });
    test("damage proxy recognises revealed immunity and exhausted PP", () => {
        const battle = battleFixture();
        try {
            const attacker = battle.sides[0].active[0];
            const defender = battle.sides[1].active[0];
            attacker.moveSlots = attacker.moveSlots.slice(0, 1);
            expect(damagePressure(attacker, defender)).toBeGreaterThan(0);
            defender.ability = battle.dex.toID("waterabsorb");
            expect(damagePressure(attacker, defender)).toBe(0);
            defender.ability = battle.dex.toID("torrent");
            attacker.moveSlots[0].pp = 0;
            expect(damagePressure(attacker, defender)).toBe(0);
        } finally {
            battle.destroy();
        }
    });
});

test("tactical priors are positive, normalised, action-sensitive and pure", () => {
    const battle = battleFixture();
    try {
        const before = JSON.stringify(battle.toJSON());
        const legal = ["move 1", "move 2", "move 3"];
        const prior = searchPriors(battle, 0, legal, "tactical");
        expect(
            prior.reduce((total, probability) => total + probability, 0),
        ).toBeCloseTo(1);
        for (const probability of prior) expect(probability).toBeGreaterThan(0);
        expect(new Set(prior).size).toBeGreaterThan(1);
        expect(searchPriors(battle, 0, legal, "uniform")).toEqual([
            1 / 3,
            1 / 3,
            1 / 3,
        ]);
        expect(JSON.stringify(battle.toJSON())).toBe(before);
    } finally {
        battle.destroy();
    }
});
