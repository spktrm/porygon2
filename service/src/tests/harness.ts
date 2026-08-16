import { createBattle, TrainablePlayerAI } from "../server/runner";
import { InfoFeature } from "../../protos/features_pb";
import { EnvironmentState, StepRequest } from "../../protos/service_pb";
import {
    EdgeBuffer,
    generateTeamFromArray,
    getSampleTeam,
    StateHandler,
} from "../server/state";
import { Teams } from "@pkmn/sim";
import { TeamGenerators } from "@pkmn/randoms";
import { GetRandomAction } from "../server/baselines/random";
import { numEvals } from "../server/eval";

Teams.setGeneratorFactory(TeamGenerators);

/**
 * Asserts the slot alignment the RL world model relies on. Edges carry a
 * stable entity index (ENTITY_EDGE_FEATURE__ENTITY_IDX, revelation order
 * across both sides) while the public team buffers are per-side and
 * re-sorted actives-first every state. INFO_FEATURE__PUBLIC_ORDER_* is the
 * per-state permutation between the two: publicOrder[row] = stable slot of
 * the pokemon in that public-team row (-1 for unrevealed fillers).
 *
 * The model keys 12 recurrent states by the stable slot and residual-injects
 * state publicOrder[row] onto public-team token row, so this checks that
 * every edge's slot maps through the permutation to a current team row
 * describing the same pokemon.
 *
 * Caveat: compared by (species, side), so a forme change between the
 * snapshot and the current state can produce a false positive on species.
 *
 * Illusion: on |replace| the EventHandler retroactively remaps edges
 * recorded since the disguised pokemon's switch-in from the disguise's slot
 * to the true pokemon's slot (rewriting snapshot species to match), so
 * disguised periods must also satisfy this check once revealed.
 */
function assertSlotAlignment(
    publicOrder: Int16Array,
    readableHistory: ReturnType<typeof EdgeBuffer.toReadableHistory>,
    readablePublicTeam: ReturnType<typeof StateHandler.toReadablePublic>,
    readableRevealedTeam: ReturnType<typeof StateHandler.toReadableRevealed>,
) {
    const slotToRow = new Map<number, number>();
    publicOrder.forEach((slot, row) => {
        if (slot < 0) {
            return;
        }
        if (slotToRow.has(slot)) {
            throw new Error(
                `Slot alignment: PUBLIC_ORDER maps slot ${slot} to both ` +
                    `row ${slotToRow.get(slot)} and row ${row}`,
            );
        }
        slotToRow.set(slot, row);
    });

    for (const [stepIndex, step] of readableHistory.entries()) {
        for (const [memberIndex, edge] of step.edges.entries()) {
            const slot = edge.entityIdx;
            if (slot < 0 || slot > 11) {
                throw new Error(
                    `Slot alignment: entityIdx ${slot} out of [0, 11] ` +
                        `at history step ${stepIndex}, member ${memberIndex}`,
                );
            }
            const row = slotToRow.get(slot);
            if (row === undefined) {
                throw new Error(
                    `Slot alignment: edge entityIdx ${slot} at history step ` +
                        `${stepIndex}, member ${memberIndex} has no ` +
                        `PUBLIC_ORDER row in the current state`,
                );
            }
            const snapshotSpecies = step.revealed[memberIndex].species;
            const snapshotSide = step.public[memberIndex].side;
            const currentSpecies = readableRevealedTeam[row].species;
            const currentSide = readablePublicTeam[row].side;
            if (
                snapshotSpecies !== currentSpecies ||
                snapshotSide !== currentSide
            ) {
                throw new Error(
                    `Slot alignment violated at history step ${stepIndex}, ` +
                        `member ${memberIndex}: edge entityIdx ${slot} -> ` +
                        `current team row ${row}, snapshot ` +
                        `(${snapshotSpecies}, side ${snapshotSide}) vs row ` +
                        `(${currentSpecies}, side ${currentSide})`,
                );
            }
        }
    }
}

export function bytesEqual(a: Uint8Array, b: Uint8Array): boolean {
    if (a.length !== b.length) {
        return false;
    }
    for (let i = 0; i < a.length; i++) {
        if (a[i] !== b[i]) {
            return false;
        }
    }
    return true;
}

export async function playerController(player: TrainablePlayerAI) {
    let historyLength = 0,
        packedHistoryLength = 0,
        stateCount = 0;
    // opp_private_team invariants: the sheet may be empty for a prefix of
    // states (the opponent's first request may not have landed yet), but
    // once populated it must be FROZEN — byte-identical in every later
    // state. Cross-player equality (my sheet of the opponent == the
    // opponent's own first-state private team) is asserted in runBattle.
    let firstPrivateTeam: Uint8Array | undefined;
    let oppPrivateSheet: Uint8Array | undefined;
    while (true) {
        // Only the stream read is guarded: a closed stream ends the loop
        // cleanly, while invariant violations below THROW upward so the
        // harness (soak loop or vitest) registers a failure instead of a
        // console line.
        let state: EnvironmentState;
        try {
            state = await player.receiveEnvironmentState();
        } catch (error) {
            console.error(error);
            break;
        }
        {
            const info = new Int16Array(state.getInfo_asU8().buffer);
            const done = info[InfoFeature.INFO_FEATURE__DONE];
            if (done) {
                break;
            }
            stateCount += 1;

            if (firstPrivateTeam === undefined) {
                firstPrivateTeam = state.getPrivateTeam_asU8().slice();
            }
            const oppSheet = state.getOppPrivateTeam_asU8();
            const oppSheetEmpty = oppSheet.every((byte) => byte === 0);
            if (oppPrivateSheet === undefined) {
                if (!oppSheetEmpty) {
                    oppPrivateSheet = oppSheet.slice();
                }
            } else if (!bytesEqual(oppSheet, oppPrivateSheet)) {
                throw new Error(
                    "opp_private_team changed mid-game — it must stay " +
                        "frozen at the opponent's first-request team sheet",
                );
            }

            const readableHistory = EdgeBuffer.toReadableHistory({
                historyEntityPublicCacheBuffer:
                    state.getHistoryEntityPublicCache_asU8(),
                historyEntityRevealedCacheBuffer:
                    state.getHistoryEntityRevealedCache_asU8(),
                historyEntityEdgeCacheBuffer:
                    state.getHistoryEntityEdgeCache_asU8(),
                historyFieldBuffer: state.getHistoryField_asU8(),
                historyLength: state.getHistoryLength(),
            });
            historyLength = Math.max(historyLength, state.getHistoryLength());
            packedHistoryLength = Math.max(
                packedHistoryLength,
                state.getHistoryPackedLength(),
            );

            // eslint-disable-next-line @typescript-eslint/no-unused-vars
            const readablePrivateTeam = StateHandler.toReadablePrivate(
                state.getPrivateTeam_asU8(),
            );
            const readablePublicTeam = StateHandler.toReadablePublic(
                state.getPublicTeam_asU8(),
            );
            const readableRevealedTeam = StateHandler.toReadableRevealed(
                state.getRevealedTeam_asU8(),
            );

            const publicOrder = info.slice(
                InfoFeature.INFO_FEATURE__PUBLIC_ORDER_0,
                InfoFeature.INFO_FEATURE__PUBLIC_ORDER_11 + 1,
            );
            assertSlotAlignment(
                publicOrder,
                readableHistory,
                readablePublicTeam,
                readableRevealedTeam,
            );
            // eslint-disable-next-line @typescript-eslint/no-unused-vars
            const readableMoveset = StateHandler.toReadableMoveset(
                state.getMyMoveset_asU8(),
            );

            const request = player.getRequest();
            if (!request) {
                throw new Error("No request available");
            }

            // A request is pending, so we need to choose an action.
            const stepRequest = new StepRequest();

            const action = GetRandomAction({ player });

            stepRequest.setAction(action);
            stepRequest.setRqid(state.getRqid());
            player.submitStepRequest(stepRequest);
        }
    }
    return {
        historyLength,
        packedHistoryLength,
        stateCount,
        firstPrivateTeam,
        oppPrivateSheet,
    };
}

export const testFormats = [
    "gen9randomdoublesbattle",
    "gen9vgc2026regf",
    "gen9ou",
    "gen9randombattle",
];
const testPackedTeamArray = [
    0, 397, 40, 289, 850, 364, 857, 639, 11, 5, 58, 2, 0, 23, 23, 17, 4, 23, 0,
    132, 141, 118, 103, 825, 58, 725, 17, 4, 12, 63, 3, 0, 3, 11, 13, 20, 0,
    550, 81, 264, 385, 680, 90, 729, 17, 4, 1, 14, 0, 18, 32, 51, 9, 9, 0, 563,
    180, 259, 635, 228, 872, 676, 6, 5, 55, 0, 40, 9, 12, 0, 10, 9, 0, 976, 254,
    286, 850, 531, 935, 685, 14, 4, 22, 36, 1, 42, 13, 8, 8, 7, 0, 584, 373,
    199, 925, 880, 808, 530, 15, 6, 0, 26, 2, 16, 48, 2, 8, 7,
];

function generateTeamFromStratgies(strategies: string[]) {
    const stratIdx = Math.floor(Math.random() * strategies.length);
    return strategies[stratIdx];
}

export async function runBattle(
    options: { smogonFormat?: string; controlledOpponent?: boolean } = {},
) {
    console.log("Creating battle...");

    const smogonFormat =
        options.smogonFormat ??
        testFormats[Math.floor(Math.random() * testFormats.length)];
    const teamGenerationStrategies = [
        generateTeamFromArray(testPackedTeamArray),
        getSampleTeam("gen9ou", "Zoroark"),
    ];
    if (smogonFormat === "gen9randombattle") {
        teamGenerationStrategies.push(Teams.pack(Teams.generate(smogonFormat)));
    }

    const evalIndex = Math.floor(Math.random() * numEvals);
    const battleOptions = {
        p1Name: "Bot1",
        p2Name: options.controlledOpponent
            ? "Bot2"
            : `baseline-eval-heuristic:${evalIndex}`,
        p1team:
            Math.random() < 0.75 && smogonFormat.includes("randombattle")
                ? Teams.pack(Teams.generate(smogonFormat))
                : getSampleTeam("gen9ou", "Zoroark"),
        p2team: generateTeamFromStratgies(teamGenerationStrategies),
        smogonFormat,
    };
    const { p1, p2 } = createBattle(battleOptions, false);
    const players = [p1];
    if (!battleOptions.p2Name.startsWith("baseline-")) {
        players.push(p2);
    }

    console.log("Starting asynchronous player controllers...");
    let results: Awaited<ReturnType<typeof playerController>>[] = [];

    try {
        // Create a promise for each player's control loop.
        const promises = [];
        promises.push(playerController(p1));
        if (!battleOptions.p2Name.startsWith("baseline-")) {
            const p2Promise = playerController(p2);
            promises.push(p2Promise);
        }

        // Wait for both player loops to complete. This happens when the battle ends.
        results = await Promise.all(promises);

        // Privileged-critic invariants. A player that saw two or more
        // states must have received a populated opponent sheet (a
        // single-state game can legitimately end before the opponent's
        // first request lands — observed at ~0.1% of soak battles); when
        // both sides are controlled, each side's sheet must equal the
        // other side's OWN first-state private team byte-for-byte (same
        // encoder, same frozen first request).
        for (const result of results) {
            if (result.oppPrivateSheet === undefined && result.stateCount >= 2) {
                throw new Error(
                    "opp_private_team never populated over a full battle",
                );
            }
        }
        if (results.length === 2) {
            const [r1, r2] = results;
            if (
                r1.oppPrivateSheet !== undefined &&
                !bytesEqual(r1.oppPrivateSheet, r2.firstPrivateTeam!)
            ) {
                throw new Error(
                    "p1's opp_private_team != p2's first private_team",
                );
            }
            if (
                r2.oppPrivateSheet !== undefined &&
                !bytesEqual(r2.oppPrivateSheet, r1.firstPrivateTeam!)
            ) {
                throw new Error(
                    "p2's opp_private_team != p1's first private_team",
                );
            }
        }

        console.log("\nBattle has concluded.");
    } finally {
        // Ensure players are properly cleaned up regardless of outcome.
        console.log("Destroying player instances.");
        for (const player of players) {
            if (player) {
                player.destroy();
            }
        }
    }
    return results;
}

