import * as fs from "fs";
import * as path from "path";
import { createBattle, TrainablePlayerAI } from "../server/runner";
import { EnvironmentTrajectory, StepRequest } from "../../protos/service_pb";
import { InfoFeature } from "../../protos/features_pb";
import { GetRandomAction } from "../server/baselines/random";
import { getSampleTeam } from "../server/state";

async function playerController(player: TrainablePlayerAI, playerName: string) {
    console.log(`${playerName}: Controller started.`);
    // The loop will continue as long as the player's stream is open.
    // The `receiveEnvironmentResponse` will resolve when a request is available.

    const trajectory = new EnvironmentTrajectory();

    while (true) {
        try {
            const state = await player.receiveEnvironmentState();
            trajectory.addStates(state);

            const info = new Int16Array(state.getInfo_asU8().buffer);
            const done = info[InfoFeature.INFO_FEATURE__DONE];
            if (done) {
                console.log(
                    `${playerName}: Received 'done' state. Exiting loop.`,
                );
                break;
            }

            // A request is pending, so we need to choose an action.
            const randomAction = GetRandomAction({ player });

            const stepRequest = new StepRequest();
            stepRequest.setAction(randomAction);
            stepRequest.setRqid(state.getRqid());
            player.submitStepRequest(stepRequest);
        } catch (error) {
            // This can happen if the stream closes unexpectedly.
            console.error(`${playerName}: Error in controller loop:`, error);
            break;
        }
    }
    console.log(`${playerName}: Controller finished.`);

    return trajectory;
}

async function runBattle() {
    console.log("Creating battle...");
    const { p1, p2 } = createBattle({
        p1Name: "Bot1",
        p2Name: "Bot2",

        p1team: getSampleTeam("gen9ou"),
        p2team: getSampleTeam("gen9ou"),
        // smogonFormat: "gen9ou",
        // smogonFormat: "gen9randomdoublesbattle",
        smogonFormat: "gen9vgc2025regibo3",
    });

    console.log("Starting asynchronous player controllers...");
    let trajectories: EnvironmentTrajectory[] = [];

    try {
        // Create a promise for each player's control loop.
        const p1Promise = playerController(p1, "P1");
        const p2Promise = playerController(p2, "P2");

        // Wait for both player loops to complete. This happens when the battle ends.
        trajectories = await Promise.all([p1Promise, p2Promise]);

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
    const filePath = path.join(__dirname, "../../../rl/environment/ex.bin");
    console.log(`Saving latest environment response to ${filePath}`);
    const data = trajectories[0].serializeBinary();
    fs.writeFile(filePath, data, (err) => {
        if (err) {
            console.error("Failed to save the environment state:", err);
        }
        console.log("File saved successfully.");
    });
}

// Execute the battle run
runBattle().catch((error) => {
    console.error("Unhandled error in runBattle:", error);
});
