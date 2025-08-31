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

const fakeTeam = [
    ...Array(numPackedSetFeatures).fill(0),
    ...Array(numPackedSetFeatures).fill(0),
    ...Array(numPackedSetFeatures).fill(0),
    ...Array(numPackedSetFeatures).fill(0),
    ...Array(numPackedSetFeatures).fill(0),
    ...Array(numPackedSetFeatures).fill(0),
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
        p1team: generateTeamFromIndices(format, fakeTeam),
        p2team: generateTeamFromIndices(format, fakeTeam),
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
