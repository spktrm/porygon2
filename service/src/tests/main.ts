import { createBattle, TrainablePlayerAI } from "../server/runner";
import { InfoFeature, MovesetFeature } from "../../protos/features_pb";
import { StepRequest } from "../../protos/service_pb";
import { EdgeBuffer, generateTeamFromFormat } from "../server/state";
import { OneDBoolean } from "../server/utils";
import { numActionMaskFeatures, numMoveFeatures } from "../server/data";
import { Protocol } from "@pkmn/protocol";
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

            const request = player.getRequest();
            if (!request) {
                throw new Error("No request available");
            }
            // const actives = (request?.active ??
            //     []) as Protocol.MoveRequest["active"];
            const switches = (request?.side?.pokemon ??
                []) as Protocol.Request.SideInfo["pokemon"];

            const myMoveset = new Int16Array(state.getMoveset_asU8().buffer);
            const numMoves = myMoveset.length / numMoveFeatures;
            for (let i = 0; i < numMoves; i++) {
                const action = myMoveset.slice(
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
                        const { index: switchIndex } =
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

async function runBattle() {
    console.log("Creating battle...");

    const format = "gen3ou";
    const names = {
        p1Name: "Bot1",
        p2Name: `baseline-4`,
        p1team: generateTeamFromFormat(format),
        p2team: generateTeamFromFormat(format),
    };
    const { p1, p2 } = createBattle(names, false);
    const players = [p1];
    if (!names.p2Name.startsWith("baseline-")) {
        players.push(p2);
    }

    console.log("Starting asynchronous player controllers...");

    try {
        // Create a promise for each player's control loop.
        const promises = [];
        promises.push(playerController(p1));
        if (!names.p2Name.startsWith("baseline-")) {
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
