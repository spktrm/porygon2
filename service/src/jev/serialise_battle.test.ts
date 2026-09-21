import { expect, test } from "vitest";
import { BattleStreams, PRNGSeed, Teams } from "@pkmn/sim";
import { TrainablePlayerAI } from "../server/runner";
import { DecisionSource, playDecisions } from "../agents/decision";
import { SerialisedBattle, serialiseBattle } from "./serialise_battle";

const CAPTURE_DECISION = 6;
const FORCE_TIE_TURN = 12;
const BACK_REFERENCE_KEYS = [
    "gens",
    "gen",
    "dex",
    "battle",
    "side",
    "foe",
    "ally",
];

interface Capture {
    serialised: SerialisedBattle;
    logLength: number;
    rawStringifyThrew: boolean;
    spreadCopyKeys: string[];
    opponentRequest: {
        side: { pokemon: { details: string; moves: string[] }[] };
    };
}

function lowestLegalCell(player: TrainablePlayerAI): number {
    if (player.legalChoiceByCell.size === 0) {
        return 0;
    }
    return Math.min(...player.legalChoiceByCell.keys());
}

async function captureMidGame(): Promise<Capture> {
    const streams = BattleStreams.getPlayerStreams(
        new BattleStreams.BattleStream(),
    );
    const first = new TrainablePlayerAI("capture-first", streams.p1);
    const second = new TrainablePlayerAI("capture-second", streams.p2);
    first.opponent = second;
    second.opponent = first;

    let capture: Capture | undefined;
    let decisions = 0;
    const capturing: DecisionSource<null> = async (player) => {
        decisions += 1;
        if (decisions === CAPTURE_DECISION) {
            let rawStringifyThrew = false;
            try {
                JSON.stringify(player.privateBattle);
            } catch {
                rawStringifyThrew = true;
            }
            capture = {
                serialised: JSON.parse(
                    JSON.stringify(
                        serialiseBattle(player.privateBattle, player.log),
                    ),
                ),
                logLength: player.log.length,
                rawStringifyThrew,
                spreadCopyKeys: Object.keys({
                    ...player.privateBattle.p1.team[0],
                }),
                opponentRequest: JSON.parse(
                    JSON.stringify(second.privateBattle.request),
                ),
            };
        }
        return { cell: lowestLegalCell(player), meta: null };
    };
    const following: DecisionSource<null> = async (player) => {
        return { cell: lowestLegalCell(player), meta: null };
    };

    const battleSeed: PRNGSeed = "9713,1,831,719";
    const loops = [
        first.start(),
        second.start(),
        playDecisions(first, capturing),
        playDecisions(second, following),
    ];
    await streams.omniscient.write(
        `>start ${JSON.stringify({ formatid: "gen9randombattle", seed: battleSeed })}\n>player p1 ${JSON.stringify(
            {
                name: first.userName,
                team: Teams.pack(
                    Teams.generate("gen9randombattle", {
                        seed: "9713,1,101,201",
                    }),
                ),
            },
        )}\n>player p2 ${JSON.stringify({
            name: second.userName,
            team: Teams.pack(
                Teams.generate("gen9randombattle", { seed: "9713,1,102,202" }),
            ),
        })}`,
    );
    let tied = false;
    for await (const chunk of streams.omniscient) {
        for (const line of chunk.split("\n")) {
            if (
                line.startsWith("|turn|") &&
                Number(line.split("|")[2]) >= FORCE_TIE_TURN &&
                !tied
            ) {
                tied = true;
                await streams.omniscient.write(">forcetie");
            }
        }
    }
    await Promise.all(loops);
    if (capture === undefined) {
        throw new Error(
            `battle ended before decision ${CAPTURE_DECISION} (${decisions} made)`,
        );
    }
    return capture;
}

const captured = captureMidGame();

function collectKeys(value: unknown, keys: Set<string>) {
    if (Array.isArray(value)) {
        for (const entry of value) {
            collectKeys(entry, keys);
        }
    } else if (value !== null && typeof value === "object") {
        for (const [key, entry] of Object.entries(value)) {
            keys.add(key);
            collectKeys(entry, keys);
        }
    }
}

function toId(name: string): string {
    return name.toLowerCase().replace(/[^a-z0-9]/g, "");
}

test("the battle serialises mid-game with both sides, the request and the full log", async () => {
    const { serialised, logLength, rawStringifyThrew } = await captured;
    expect(serialised.sides).toHaveLength(2);
    for (const side of serialised.sides) {
        expect(side.team.length).toBeGreaterThan(0);
        expect(side.active[0]).not.toBeNull();
    }
    expect(serialised.request).toBeTruthy();
    expect(serialised.log).toHaveLength(logLength);
    expect(serialised.log.some((line) => line.startsWith("|turn|"))).toBe(true);

    // Positive control: the battle this was built from does not survive a
    // plain stringify, so the assertions above are not true of any object.
    expect(rawStringifyThrew).toBe(true);
});

test("no back-reference or datastore key survives, and the payload stays small", async () => {
    const { serialised } = await captured;
    // The request is the simulator's own JSON and legitimately has a `side`.
    const keys = new Set<string>();
    collectKeys({ ...serialised, request: null }, keys);
    for (const key of BACK_REFERENCE_KEYS) {
        expect(keys.has(key)).toBe(false);
    }
    const payloadChars = JSON.stringify(serialised).length;
    expect(payloadChars).toBeLessThan(100_000);
    console.log(
        "serialised battle size",
        JSON.stringify({
            turn: serialised.turn,
            payloadChars,
            logChars: serialised.log.join("\n").length,
            requestChars: JSON.stringify(serialised.request).length,
        }),
    );
});

test("prototype getters are emitted", async () => {
    const { serialised, spreadCopyKeys } = await captured;
    const pokemon = serialised.sides[0].team[0];
    expect(pokemon.species.name.length).toBeGreaterThan(0);
    expect(pokemon.species.baseStats.hp).toBeGreaterThan(0);
    expect(pokemon.types.length).toBeGreaterThan(0);
    expect(pokemon.ident.startsWith("p1")).toBe(true);

    // Positive control: a copy of the same Pokemon's own properties — what a
    // cycle-stripping walk would have kept — carries none of them.
    expect(spreadCopyKeys).toContain("hp");
    for (const getter of ["species", "types", "ident", "teraType"]) {
        expect(spreadCopyKeys).not.toContain(getter);
    }
});

test("the opponent's hidden team and moves never reach the payload", async () => {
    const { serialised, opponentRequest } = await captured;
    const payload = JSON.stringify(serialised);
    const publicLog = serialised.log.join("\n");

    const hiddenSpecies = opponentRequest.side.pokemon
        .map((pokemon) => pokemon.details.split(",")[0])
        .filter((species) => !publicLog.includes(species));
    // Positive control: the opponent does hold species this player has not
    // seen, so the absence below is not vacuous.
    expect(hiddenSpecies.length).toBeGreaterThan(0);
    for (const species of hiddenSpecies) {
        expect(payload.includes(species)).toBe(false);
    }

    const revealedMoves = new Set(
        serialised.log
            .filter((line) => line.startsWith("|move|p2"))
            .map((line) => toId(line.split("|")[3])),
    );
    const opponentSide = serialised.sides[1];
    for (const pokemon of opponentSide.team) {
        for (const moveSlot of pokemon.moveSlots) {
            expect(revealedMoves.has(String(moveSlot.id))).toBe(true);
        }
    }
    const heldMoves = opponentRequest.side.pokemon.flatMap(
        (pokemon) => pokemon.moves,
    );
    expect(heldMoves.some((move) => !revealedMoves.has(move))).toBe(true);
});
