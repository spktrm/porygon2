import { Pokemon } from "@pkmn/client";
import fs from "fs";
import path from "path";
import { describe, expect, test } from "vitest";
import {
    evaluatePosition,
    positionFeatures,
    PotentialPokemon,
    publicHpFraction,
} from "./position_potential";

interface FixtureMon {
    hp: number;
    active: boolean;
    offensive: string[];
    defensive: string[];
}

interface FixturePosition {
    name: string;
    own: (FixtureMon | null)[];
    opponent: (FixtureMon | null)[];
    features: { hp_balance: number; alive_balance: number; matchup: number };
    potential: number;
}

const fixture: { positions: FixturePosition[] } = JSON.parse(
    fs.readFileSync(
        path.resolve(
            __dirname,
            "../../../rl/offline/position_potential_fixture.json",
        ),
        "utf8",
    ),
);

function team(mons: (FixtureMon | null)[]): PotentialPokemon[] {
    return mons.map((mon) => {
        if (mon === null) {
            return {
                hp: 1,
                active: false,
                offensiveTypes: [],
                defensiveTypes: [],
            };
        }
        return {
            hp: mon.hp,
            active: mon.active,
            offensiveTypes: mon.offensive,
            defensiveTypes: mon.defensive,
        };
    });
}

function pokemon(kind: string, hp = 1, active = true): PotentialPokemon {
    return { hp, active, offensiveTypes: [kind], defensiveTypes: [kind] };
}

function member(hp: number, maxhp: number, fainted = false): Pokemon {
    return { hp, maxhp, fainted } as Pokemon;
}

describe("position potential", () => {
    test("reproduces the Python reference on the shared fixture", () => {
        expect(fixture.positions.length).toBeGreaterThan(0);
        for (const position of fixture.positions) {
            const own = team(position.own);
            const opponent = team(position.opponent);
            const features = positionFeatures(own, opponent);
            expect(features.hpBalance, position.name).toBeCloseTo(
                position.features.hp_balance,
                12,
            );
            expect(features.aliveBalance, position.name).toBeCloseTo(
                position.features.alive_balance,
                12,
            );
            expect(features.matchup, position.name).toBeCloseTo(
                position.features.matchup,
                12,
            );
            expect(evaluatePosition(own, opponent), position.name).toBeCloseTo(
                position.potential,
                12,
            );
        }
    });

    test("player swap negates and slot order does not matter in doubles", () => {
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

    test("own exact HP reads as the percentage the opponent is shown", () => {
        // Our stream holds 150/301 exactly; the opponent's shows
        // ceil(100 * 150 / 301) = 50 of 100 (sim/pokemon.ts getHealth).
        const ours = member(150, 301);
        const theirs = member(50, 100);
        expect(publicHpFraction(ours)).toBe(publicHpFraction(theirs));
        // Positive control: the exact fraction is NOT what they see, so
        // reading it would break the harness's antisymmetry.
        expect(ours.hp / ours.maxhp).not.toBe(publicHpFraction(theirs));
        // The sim never shows 100 for a damaged mon.
        expect(publicHpFraction(member(300, 301))).toBe(0.99);
        expect(publicHpFraction(member(99, 99))).toBe(1);
        expect(publicHpFraction(member(0, 301, true))).toBe(0);
        // Revealed without an HP report yet reads full (getHpRatio).
        expect(publicHpFraction(member(0, 0))).toBe(1);
    });
});
