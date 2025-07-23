import { createBattle, TrainablePlayerAI } from "../server/runner";
import { InfoFeature } from "../../protos/features_pb";
import { StepRequest } from "../../protos/service_pb";
import { EdgeBuffer } from "../server/state";

async function playerController(player: TrainablePlayerAI) {
    while (true) {
        try {
            const state = await player.recieveEnvironmentState();

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

            // A request is pending, so we need to choose an action.

            const stepRequest = new StepRequest();
            stepRequest.setAction(-1);
            stepRequest.setRqid(state.getRqid());
            player.submitStepRequest(stepRequest);
        } catch (error) {
            // This can happen if the stream closes unexpectedly.
            console.error(error);
            break;
        }
    }
}

async function runBattle() {
    console.log("Creating battle...");
    const names = { p1Name: "Bot1", p2Name: "Bot2" };
    const { p1, p2 } = createBattle(names, true);

    console.log("Starting asynchronous player controllers...");

    try {
        // Create a promise for each player's control loop.
        const p1Promise = playerController(p1);
        const p2Promise = playerController(p2);

        // Wait for both player loops to complete. This happens when the battle ends.
        await Promise.all([p1Promise, p2Promise]);

        console.log("\nBattle has concluded.");
    } catch (error) {
        console.error("An error occurred during the battle:", error);
    } finally {
        // Ensure players are properly cleaned up regardless of outcome.
        console.log("Destroying player instances.");
        p1.destroy();
        p2.destroy();
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
