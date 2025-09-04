import { createBattle, TrainablePlayerAI } from "../server/runner";
import { InfoFeature } from "../../protos/features_pb";
import { StepRequest } from "../../protos/service_pb";
import {
    EdgeBuffer,
    generateTeamFromIndices,
    StateHandler,
} from "../server/state";
import { OneDBoolean } from "../server/utils";
import { numActionMaskFeatures, numPackedSetFeatures } from "../server/data";
import { Teams } from "@pkmn/sim";
import { TeamGenerators } from "@pkmn/randoms";
import { actionMaskToRandomAction } from "../server/baselines/random";

Teams.setGeneratorFactory(TeamGenerators);

async function playerController(player: TrainablePlayerAI) {
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
                historyEntityNodesBuffer: state.getHistoryEntityNodes_asU8(),
                historyEntityEdgesBuffer: state.getHistoryEntityEdges_asU8(),
                historyFieldBuffer: state.getHistoryField_asU8(),
                historyLength: state.getHistoryLength(),
            });

            // eslint-disable-next-line @typescript-eslint/no-unused-vars
            const readablePrivateTeam = StateHandler.toReadableTeam(
                state.getPrivateTeam_asU8(),
            );
            // eslint-disable-next-line @typescript-eslint/no-unused-vars
            const readablePublicTeam = StateHandler.toReadableTeam(
                state.getPublicTeam_asU8(),
            );

            // eslint-disable-next-line @typescript-eslint/no-unused-vars
            const readableMoveset = StateHandler.toReadableMoveset(
                state.getMoveset_asU8(),
            );

            const request = player.getRequest();
            if (!request) {
                throw new Error("No request available");
            }

            // A request is pending, so we need to choose an action.
            const stepRequest = new StepRequest();

            const actionMask = new OneDBoolean(numActionMaskFeatures);
            actionMask.setBuffer(state.getActionMask_asU8());
            const randomAction = actionMaskToRandomAction(actionMask);

            stepRequest.setAction(randomAction);
            stepRequest.setRqid(state.getRqid());
            player.submitStepRequest(stepRequest);
        } catch (error) {
            // This can happen if the stream closes unexpectedly.
            console.error(error);
            break;
        }
    }
}

const fakeTeam1 = [
    0, 27, 8, 65, 239, 665, 444, 835, 5, 0, 128, 84, 0, 124, 104, 64, 31, 31,
    31, 31, 31, 31, 0, 5, 0, 0, 0, 372, 110, 170, 236, 176, 280, 612, 19, 0,
    100, 64, 12, 116, 132, 80, 31, 31, 31, 31, 31, 31, 0, 17, 0, 0, 0, 578, 200,
    199, 661, 651, 642, 783, 23, 0, 180, 12, 16, 44, 180, 76, 31, 31, 31, 31,
    31, 31, 0, 12, 0, 0, 0, 574, 200, 240, 480, 928, 929, 118, 24, 0, 80, 108,
    132, 64, 56, 68, 31, 31, 31, 31, 31, 31, 0, 11, 0, 0, 0, 427, 133, 193, 625,
    239, 15, 417, 9, 0, 140, 104, 52, 20, 16, 176, 31, 31, 31, 31, 31, 31, 0,
    17, 0, 0, 0, 1422, 539, 115, 484, 800, 267, 915, 3, 0, 0, 96, 128, 72, 80,
    128, 31, 31, 31, 31, 31, 31, 0, 18, 0, 0,
];

const fakeTeam2 = [
    0, 83, 19, 130, 478, 931, 778, 234, 22, 0, 160, 80, 76, 20, 108, 60, 31, 31,
    31, 31, 31, 31, 0, 17, 0, 0, 0, 215, 55, 145, 639, 564, 152, 769, 18, 0,
    140, 56, 108, 60, 0, 140, 31, 31, 31, 31, 31, 31, 0, 11, 0, 0, 0, 469, 160,
    110, 693, 832, 298, 212, 28, 0, 92, 128, 32, 136, 24, 100, 31, 31, 31, 31,
    31, 31, 0, 19, 0, 0, 0, 1084, 351, 171, 420, 782, 855, 850, 27, 0, 32, 140,
    16, 72, 116, 136, 31, 31, 31, 31, 31, 31, 0, 2, 0, 0, 0, 20, 214, 14, 782,
    243, 176, 42, 28, 0, 272, 96, 56, 12, 24, 52, 31, 31, 31, 31, 31, 31, 0, 10,
    0, 0, 0, 1186, 371, 248, 717, 309, 855, 680, 25, 0, 32, 152, 4, 140, 76,
    104, 31, 31, 31, 31, 31, 31, 0, 3, 0, 0,
];

async function runBattle() {
    console.log("Creating battle...");

    const format = "gen9ou";
    const battleOptions = {
        p1Name: "Bot1",
        p2Name: `baseline-4`,
        // p1team: null,
        // p2team: null,
        // smogonFormat: "gen9randombattle",
        p1team: generateTeamFromIndices(format, fakeTeam1),
        p2team: generateTeamFromIndices(format, fakeTeam2),
        smogonFormat: format,
    };
    const { p1, p2 } = createBattle(battleOptions, false);
    const players = [p1];
    if (!battleOptions.p2Name.startsWith("baseline-")) {
        players.push(p2);
    }

    console.log("Starting asynchronous player controllers...");

    try {
        // Create a promise for each player's control loop.
        const promises = [];
        promises.push(playerController(p1));
        if (!battleOptions.p2Name.startsWith("baseline-")) {
            const p2Promise = playerController(p2);
            promises.push(p2Promise);
        }

        // Wait for both player loops to complete. This happens when the battle ends.
        await Promise.all(promises);

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
}

async function main() {
    while (true) {
        await runBattle();
    }
}

main().catch((error) => {
    console.error("An error occurred in the main execution:", error);
});
