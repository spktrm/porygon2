import { createBattle, TrainablePlayerAI } from "../server/runner";
import { InfoFeature } from "../../protos/features_pb";
import {
    ActionMask,
    ActionRequestKind,
    EnvironmentState,
    StepRequest,
} from "../../protos/service_pb";
import {
    MOVE_SLOT_INDICES,
    RESERVE_SLOT_INDICES,
    TARGET_SLOT_INDICES,
} from "../server/data";
import {
    EdgeBuffer,
    generateTeamFromArray,
    getSampleTeam,
    StateHandler,
} from "../server/state";
import { AnyObject, Teams } from "@pkmn/sim";
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

/**
 * The private-side shape contract. Until 2026-08-25 these two buffers were
 * decoded here only to feed the frozen-opponent-sheet invariants; when those
 * were deleted with the privileged critic the decodes stayed behind an
 * eslint-disable, i.e. a decode-does-not-throw smoke test wearing the costume
 * of an assertion. This is the replacement: the encoder's own shape contract,
 * which is what the python decoder (`rl/environment/utils.process_state`)
 * reshapes against and will throw on if it ever drifts.
 *
 * Deliberately NOT cross-checked against the action mask here: the moveset ->
 * legal-action correspondence runs straight through the doubles slot-alignment
 * surface, which is a known-open defect, so that assertion would fail for
 * reasons unrelated to what it claims to test.
 */
function assertPrivateSideShape(
    privateTeam: ReturnType<typeof StateHandler.toReadablePrivate>,
    moveset: ReturnType<typeof StateHandler.toReadableMoveset>,
) {
    if (privateTeam.length !== 6) {
        throw new Error(
            `private_team decoded to ${privateTeam.length} rows, expected 6`,
        );
    }
    if (moveset.length === 0 || moveset.length % 4 !== 0) {
        throw new Error(
            `my_moveset decoded to ${moveset.length} moves, expected a ` +
                `non-zero multiple of 4 (one 4-move block per active slot)`,
        );
    }
}

const MAX_RATIO_TOKEN_HARNESS = 16384;

/**
 * The private TRUTH CHANNEL (2026-08-31): a private row's condition must
 * equal what the REQUEST says about that mon -- an independent parse of the
 * request's condition string ("288/288", "0 fnt", "150/288 tox"), never the
 * public row, which under a my-side Illusion blends two mons' histories
 * until |replace| remaps. Battles with organic Illusion (Zoroark is in the
 * random sets) make the soak itself the positive control: any accidental
 * public-sourcing of these columns diverges from the request there and this
 * throws.
 *
 * Also asserts the alignment key: a row's entityIdxPlusOne, when present,
 * must point at a stable entity index that appears in MY side's
 * PUBLIC_ORDER permutation -- i.e. the tag connects to a real public row.
 * KNOWN ~0.1% false-positive class (2 in 1791 soak battles): a my-side
 * Illusion mon's own index attaches to no public row until |replace| --
 * the wire is CORRECT there (the hidden mon has no public identity yet,
 * the tag reads as absent model-side), and vitest's retry: 2 absorbs it
 * like the rest of the Illusion family.
 */
function assertPrivateTruthChannel(
    privateTeam: ReturnType<typeof StateHandler.toReadablePrivate>,
    request: AnyObject,
    publicOrder: Int16Array,
    channel: "mine" | "opponent" = "mine",
) {
    const requestPokemon = request?.side?.pokemon as
        | { condition: string; ident: string }[]
        | undefined;
    if (!requestPokemon) {
        return;
    }
    const sideSlots = new Set<number>();
    // My side's public rows are the first half of PUBLIC_ORDER, the
    // opponent's the second half -- the channel picks which half the
    // alignment keys must land in.
    const halfLength = publicOrder.length / 2;
    let rowStart = 0;
    if (channel === "opponent") {
        rowStart = halfLength;
    }
    for (let row = rowStart; row < rowStart + halfLength; row++) {
        if (publicOrder[row] >= 0) {
            sideSlots.add(publicOrder[row]);
        }
    }
    for (const [j, member] of requestPokemon.entries()) {
        const row = privateTeam[j];
        if (row === undefined) {
            break;
        }
        const condition = member.condition;
        const fainted = condition.endsWith(" fnt");
        let expectedRatio = 0;
        let statusToken: string | undefined = undefined;
        if (!fainted) {
            const [hpPart, rest] = condition.split("/");
            const restParts = (rest ?? "").split(" ");
            const maxHp = parseInt(restParts[0]);
            statusToken = restParts[1];
            expectedRatio = Math.floor(
                (MAX_RATIO_TOKEN_HARNESS * parseInt(hpPart)) / maxHp,
            );
        }
        if (row.fainted !== fainted) {
            throw new Error(
                `private row ${j}: fainted ${row.fainted} but the request ` +
                    `says "${condition}"`,
            );
        }
        if (!fainted && row.hpRatio !== expectedRatio) {
            throw new Error(
                `private row ${j}: hpRatio ${row.hpRatio} but the request ` +
                    `says "${condition}" (expected ${expectedRatio})`,
            );
        }
        if (!fainted && row.hasStatus !== (statusToken !== undefined)) {
            throw new Error(
                `private row ${j}: hasStatus ${row.hasStatus} but the ` +
                    `request says "${condition}"`,
            );
        }
        if (row.entityIdxPlusOne > 0) {
            const stableIdx = row.entityIdxPlusOne - 1;
            if (!sideSlots.has(stableIdx)) {
                throw new Error(
                    `${channel} private row ${j}: entity idx ${stableIdx} ` +
                        `is not in the ${channel} half of PUBLIC_ORDER -- ` +
                        `the alignment key points at no public row`,
                );
            }
        }
    }
}

/**
 * The wire mask must name exactly the cells the decoder can answer.
 *
 * This mirrors `_cells_from_structured_mask` in `rl/environment/utils.py` --
 * deliberately a SECOND implementation, because the thing under test is that
 * the two languages agree, and a shared helper could only prove itself
 * self-consistent. If this drifts from the python one the assertion below
 * fires, which is the point. The offsets are derived HERE from the slot-list
 * lengths, independently of data.ts's exported block constants, for the same
 * reason.
 */
function cellsFromStructuredMask(mask: ActionMask): Set<number> {
    const cells = new Set<number>();
    const kind = mask.getKind();
    if (
        kind === ActionRequestKind.ACTION_REQUEST_KIND___UNSPECIFIED ||
        kind === ActionRequestKind.ACTION_REQUEST_KIND__WAIT
    ) {
        return cells;
    }
    const numTargets = TARGET_SLOT_INDICES.length;
    const moveOffset = RESERVE_SLOT_INDICES.length;
    const otherOffset = moveOffset + MOVE_SLOT_INDICES.length * numTargets;

    const switchSlots = mask.getSwitchSlots();
    for (let j = 0; j < RESERVE_SLOT_INDICES.length; j++) {
        if ((switchSlots >> j) & 1) {
            cells.add(j);
        }
    }
    for (let moveBit = 0; moveBit < MOVE_SLOT_INDICES.length; moveBit++) {
        const targets = mask.getMoveTargetsList()[moveBit];
        for (let targetBit = 0; targetBit < numTargets; targetBit++) {
            if ((targets >> targetBit) & 1) {
                cells.add(moveOffset + moveBit * numTargets + targetBit);
            }
        }
    }
    const otherSrcs = mask.getOtherSrcs();
    for (let slotBit = 0; slotBit < numTargets; slotBit++) {
        if ((otherSrcs >> slotBit) & 1) {
            cells.add(otherOffset + slotBit);
        }
    }
    return cells;
}

/**
 * The mask <-> decoder agreement invariant (2026-08-29).
 *
 * Every cell the wire says is legal must have a Showdown choice string, and
 * every cell that has one must be on the wire. Before the cell -> choice map
 * these were two independently written code paths and they had drifted three
 * ways -- a hardcoded " terastallize" suffix, a move-target lookup through a
 * different move list under Dynamax, and up to 7 duplicate team-preview cells
 * for one choice. None of it was caught, because nothing asserted the two
 * halves agreed and the sim's rejections were logged rather than counted.
 */
function assertMaskMatchesDecoder(
    mask: ActionMask,
    choiceByCell: Map<number, string>,
) {
    const kind = mask.getKind();
    const carriesChoice =
        kind !== ActionRequestKind.ACTION_REQUEST_KIND___UNSPECIFIED &&
        kind !== ActionRequestKind.ACTION_REQUEST_KIND__WAIT;
    if (carriesChoice && choiceByCell.size === 0) {
        throw new Error(
            `action mask kind ${kind} legalised nothing; an empty mask makes ` +
                `the random baseline pick a uniformly ILLEGAL cell`,
        );
    }
    const onWire = cellsFromStructuredMask(mask);
    for (const cell of onWire) {
        if (!choiceByCell.has(cell)) {
            throw new Error(
                `cell ${cell} is legal on the wire but the decoder has no ` +
                    `choice string for it`,
            );
        }
    }
    for (const cell of choiceByCell.keys()) {
        if (!onWire.has(cell)) {
            throw new Error(
                `cell ${cell} decodes to "${choiceByCell.get(cell)}" but is ` +
                    `not legal on the wire`,
            );
        }
    }
}

export async function playerController(player: TrainablePlayerAI) {
    let historyLength = 0,
        packedHistoryLength = 0,
        stateCount = 0,
        rewriteCount = 0;
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
            // `history_rewrite_count` (2026-09-02): the service's count of
            // in-place rewrites of rows it already handed out. Read on
            // EVERY state, the done state included, so a |replace| that
            // lands after the last decision still reaches the wire; it must
            // never go backwards within a game, and the wire can only lag
            // the buffer (rows rewritten after this build are counted there
            // first), never lead it.
            const wireRewriteCount = state.getHistoryRewriteCount();
            if (wireRewriteCount < rewriteCount) {
                throw new Error(
                    `history_rewrite_count went backwards: ` +
                        `${rewriteCount} -> ${wireRewriteCount}`,
                );
            }
            const bufferRewriteCount =
                player.eventHandler.edgeBuffer.rewriteCount;
            if (wireRewriteCount > bufferRewriteCount) {
                throw new Error(
                    `history_rewrite_count ${wireRewriteCount} on the wire ` +
                        `exceeds the EdgeBuffer's ${bufferRewriteCount}`,
                );
            }
            rewriteCount = wireRewriteCount;

            const info = new Int16Array(state.getInfo_asU8().buffer);
            const done = info[InfoFeature.INFO_FEATURE__DONE];
            if (done) {
                break;
            }
            stateCount += 1;

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
            const readableMoveset = StateHandler.toReadableMoveset(
                state.getMyMoveset_asU8(),
            );
            assertPrivateSideShape(readablePrivateTeam, readableMoveset);

            const truthRequest = player.getRequest();
            if (truthRequest) {
                assertPrivateTruthChannel(
                    readablePrivateTeam,
                    truthRequest as AnyObject,
                    publicOrder,
                );
            }

            // The OPPONENT truth channel (2026-09-01). The opponent's live
            // request can move between build and this check (their loop is
            // an independent async consumer), so the compare runs against
            // `lastSerialisedOppRequest` -- the EXACT object the build
            // serialised, race-free by construction (the client replaces
            // its request wholesale per |request| line, so the held
            // reference is a stable snapshot). Undefined snapshot == the
            // build wrote the all-zero degrade, which the shape of the
            // buffer must then agree with.
            const readableOppPrivateTeam = StateHandler.toReadablePrivate(
                state.getOppPrivateTeam_asU8(),
            );
            if (readableOppPrivateTeam.length !== 6) {
                throw new Error(
                    `opp_private_team decoded to ` +
                        `${readableOppPrivateTeam.length} rows, expected 6`,
                );
            }
            // The all-zero degrade: every row reads hpRatio 0 AND fainted
            // false, a combination no real request produces (a living mon
            // has hp, a fainted one has the flag).
            const oppRowsEmpty = readableOppPrivateTeam.every(
                (row) => row.hpRatio === 0 && !row.fainted,
            );
            const oppSnapshot = player.lastSerialisedOppRequest;
            if (oppSnapshot === undefined && !oppRowsEmpty) {
                throw new Error(
                    "opp_private_team is populated but the build recorded " +
                        "no serialised opponent request",
                );
            }
            if (oppSnapshot !== undefined) {
                if (oppRowsEmpty) {
                    throw new Error(
                        "the build serialised an opponent request but " +
                            "opp_private_team decoded as the all-zero degrade",
                    );
                }
                assertPrivateTruthChannel(
                    readableOppPrivateTeam,
                    oppSnapshot,
                    publicOrder,
                    "opponent",
                );
            }

            assertMaskMatchesDecoder(
                state.getStructuredActionMask()!,
                player.legalChoiceByCell,
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
    if (player.invalidChoiceCount > 0) {
        throw new Error(
            `${player.invalidChoiceCount} choice(s) were rejected by the sim; ` +
                `a legal mask cell decoded to something Showdown would not take`,
        );
    }
    return {
        historyLength,
        packedHistoryLength,
        stateCount,
        rewriteCount,
        // The count of |replace| handler calls that reached a remap -- the
        // positive control's ground truth for `rewriteCount` above.
        bufferRewriteCount: player.eventHandler.edgeBuffer.rewriteCount,
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
