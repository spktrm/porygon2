import { createBattle, TrainablePlayerAI } from "../server/runner";
import { InfoFeature } from "../../protos/features_pb";
import { StepRequest } from "../../protos/service_pb";
import {
    EdgeBuffer,
    generateTeamFromArray,
    getSampleTeam,
    StateHandler,
} from "../server/state";
import { Teams } from "@pkmn/sim";
import { TeamGenerators } from "@pkmn/randoms";
import { GetRandomAction } from "../server/baselines/random";

Teams.setGeneratorFactory(TeamGenerators);

async function playerController(player: TrainablePlayerAI) {
    let historyLength = 0,
        packedHistoryLength = 0;
    while (true) {
        try {
            const state = await player.receiveEnvironmentState();

            const info = new Int16Array(state.getInfo_asU8().buffer);
            const done = info[InfoFeature.INFO_FEATURE__DONE];
            if (done) {
                break;
            }

            // eslint-disable-next-line @typescript-eslint/no-unused-vars
            const readableHistory = EdgeBuffer.toReadableHistory({
                historyEntityPublicBuffer: state.getHistoryEntityPublic_asU8(),
                historyEntityRevealedBuffer:
                    state.getHistoryEntityRevealed_asU8(),
                historyEntityEdgesBuffer: state.getHistoryEntityEdges_asU8(),
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
            // eslint-disable-next-line @typescript-eslint/no-unused-vars
            const readablePublicTeam = StateHandler.toReadablePublic(
                state.getPublicTeam_asU8(),
            );
            // eslint-disable-next-line @typescript-eslint/no-unused-vars
            const readableRevealedTeam = StateHandler.toReadableRevealed(
                state.getRevealedTeam_asU8(),
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
        } catch (error) {
            // This can happen if the stream closes unexpectedly.
            console.error(error);
            break;
        }
    }
    return { historyLength, packedHistoryLength };
}

const testFormats = [
    // "gen9randomdoublesbattle",
    "gen9vgc2026regf",
    // "gen9championsvgc2026regma",
    // "gen9ou",
    // "gen9randombattle",
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

async function runBattle() {
    console.log("Creating battle...");

    const smogonFormat =
        testFormats[Math.floor(Math.random() * testFormats.length)];
    const teamGenerationStrategies = [
        generateTeamFromArray(testPackedTeamArray),
        getSampleTeam("gen9ou", "Zoroark"),
    ];
    if (smogonFormat === "gen9randombattle") {
        teamGenerationStrategies.push(Teams.pack(Teams.generate(smogonFormat)));
    }

    const battleOptions = {
        p1Name: "Bot1",
        p2Name: `baseline-eval-heuristic:0`,
        p1team: getSampleTeam("gen9ou", "Zoroark"),
        p2team: generateTeamFromStratgies(teamGenerationStrategies),
        smogonFormat,
    };
    const { p1, p2 } = createBattle(battleOptions, false);
    const players = [p1];
    if (!battleOptions.p2Name.startsWith("baseline-")) {
        players.push(p2);
    }

    console.log("Starting asynchronous player controllers...");
    let results: { historyLength: number; packedHistoryLength: number }[] = [];

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
    } catch (error) {
        console.error("An error occurred during the battle:", error);
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

async function main() {
    let historyLength = 0;
    let packedHistoryLength = 0;
    while (true) {
        const results = await runBattle();
        historyLength = Math.max(
            historyLength,
            ...results.map((r) => r.historyLength),
        );
        packedHistoryLength = Math.max(
            packedHistoryLength,
            ...results.map((r) => r.packedHistoryLength),
        );
        console.log(historyLength, packedHistoryLength);
    }
}

main().catch((error) => {
    console.error("An error occurred in the main execution:", error);
});
