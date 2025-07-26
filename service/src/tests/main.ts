import { createBattle, TrainablePlayerAI } from "../server/runner";
import { InfoFeature, MovesetFeature } from "../../protos/features_pb";
import { StepRequest } from "../../protos/service_pb";
import { EdgeBuffer } from "../server/state";
import { OneDBoolean } from "../server/utils";
import { numMoveFeatures } from "../server/data";
import { Protocol } from "@pkmn/protocol";

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

            const request = player.getRequest();
            if (!request) {
                throw new Error("No request available");
            }
            // const actives = (request?.active ??
            //     []) as Protocol.MoveRequest["active"];
            const switches = (request?.side?.pokemon ??
                []) as Protocol.Request.SideInfo["pokemon"];

            const myActions = new Int16Array(state.getMyActions_asU8().buffer);
            const numMoves = myActions.length / numMoveFeatures;
            for (let i = 0; i < numMoves; i++) {
                const action = myActions.slice(
                    i * numMoveFeatures,
                    (i + 1) * numMoveFeatures,
                );
                const entityIndex =
                    action[MovesetFeature.MOVESET_FEATURE__ENTITY_IDX];
                if (i < 4) {
                    if (entityIndex !== 0) {
                        throw new Error(
                            `Unexpected entity index ${entityIndex} for action ${i}`,
                        );
                    }
                } else {
                    if (switches.length > 0) {
                        const { pokemon, index: switchIndex } =
                            player.eventHandler.getPokemon(
                                switches[i - 4].ident,
                                false,
                            );
                        if (entityIndex !== switchIndex) {
                            throw new Error(
                                `Unexpected entity index ${entityIndex} for switch action ${i}`,
                            );
                        }
                    }
                }
            }

            // A request is pending, so we need to choose an action.
            const stepRequest = new StepRequest();

            const legalActions = new OneDBoolean(10);
            legalActions.setBuffer(state.getLegalActions_asU8());
            const legalIndices = legalActions
                .toBinaryVector()
                .flatMap((value, index) => (value > 0 ? [index] : []));
            const randomIndex =
                legalIndices[Math.floor(Math.random() * legalIndices.length)];

            stepRequest.setAction(randomIndex);
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
    const { p1, p2 } = createBattle(names, false);

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
