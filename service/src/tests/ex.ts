import * as fs from "fs";
import * as path from "path";
import { createBattle, TrainablePlayerAI } from "../server/runner";
import { EnvironmentState, StepRequest } from "../../protos/service_pb";
import { InfoFeature } from "../../protos/features_pb";

async function playerController(
    player: TrainablePlayerAI,
    playerName: string,
    stateTracker: { lastState: EnvironmentState | null },
) {
    console.log(`${playerName}: Controller started.`);
    // The loop will continue as long as the player's stream is open.
    // The `receiveEnvironmentResponse` will resolve when a request is available.
    while (true) {
        try {
            const state = await player.receiveEnvironmentState();
            stateTracker.lastState = state; // Update the shared last state

            const info = new Int16Array(state.getInfo_asU8().buffer);
            const done = info[InfoFeature.INFO_FEATURE__DONE];
            if (done) {
                console.log(
                    `${playerName}: Received 'done' state. Exiting loop.`,
                );
                break;
            }

            // A request is pending, so we need to choose an action.

            const stepRequest = new StepRequest();
            stepRequest.setAction(-1);
            stepRequest.setRqid(state.getRqid());
            player.submitStepRequest(stepRequest);
        } catch (error) {
            // This can happen if the stream closes unexpectedly.
            console.error(`${playerName}: Error in controller loop:`, error);
            break;
        }
    }
    console.log(`${playerName}: Controller finished.`);
}

async function runBattle() {
    console.log("Creating battle...");
    const { p1, p2 } = createBattle({ p1Name: "Bot1", p2Name: "Bot2" });

    // This object will be shared between the two player controllers
    // to keep track of the most recent game state.
    const stateTracker = { lastState: null as EnvironmentState | null };

    console.log("Starting asynchronous player controllers...");

    try {
        // Create a promise for each player's control loop.
        const p1Promise = playerController(p1, "P1", stateTracker);
        const p2Promise = playerController(p2, "P2", stateTracker);

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

    // Save the very last state that was recorded.
    if (stateTracker.lastState) {
        const filePath = path.join(__dirname, "../../../rl/environment/ex.bin");
        console.log(`Saving latest environment response to ${filePath}`);
        try {
            fs.writeFileSync(
                filePath,
                stateTracker.lastState.serializeBinary(),
            );
            console.log("File saved successfully.");
        } catch (error) {
            console.error("Failed to save the environment state:", error);
        }
    } else {
        console.log("No environment state was generated to save.");
    }
}

// Execute the battle run
runBattle().catch((error) => {
    console.error("Unhandled error in runBattle:", error);
});
