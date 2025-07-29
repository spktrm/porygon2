import { createBattle, TrainablePlayerAI } from "../server/runner";
import {
    InfoFeature,
    MovesetFeature,
    TeamSetFeature,
} from "../../protos/features_pb";
import { StepRequest } from "../../protos/service_pb";
import {
    EdgeBuffer,
    IndexValueFromEnum,
    teamBytesToPackedString,
} from "../server/state";
import { OneDBoolean } from "../server/utils";
import { numMoveFeatures, numTeamSetFeatures } from "../server/data";
import { Protocol } from "@pkmn/protocol";
import { Teams } from "@pkmn/sim";
import { TeamGenerators } from "@pkmn/randoms";
import {
    AbilitiesEnum,
    ItemsEnum,
    MovesEnum,
    NaturesEnum,
    SpeciesEnum,
} from "../../protos/enums_pb";
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

function createFakeTeamBytes(): Uint8Array {
    const team = Teams.generate("gen3randombattle");
    const teamBytes = new Int16Array(numTeamSetFeatures * 6);
    for (const [i, pokemonSet] of team.entries()) {
        const offset = i * numTeamSetFeatures;

        const [moveId0, moveId1, moveId2, moveId3] = pokemonSet.moves;

        teamBytes[offset + TeamSetFeature.TEAM_SET_FEATURE__SPECIES] =
            IndexValueFromEnum(SpeciesEnum, pokemonSet.name);
        teamBytes[offset + TeamSetFeature.TEAM_SET_FEATURE__ITEM] =
            IndexValueFromEnum(ItemsEnum, pokemonSet.item);
        teamBytes[offset + TeamSetFeature.TEAM_SET_FEATURE__ABILITY] =
            IndexValueFromEnum(AbilitiesEnum, pokemonSet.ability);
        teamBytes[offset + TeamSetFeature.TEAM_SET_FEATURE__MOVEID0] = moveId0
            ? IndexValueFromEnum(MovesEnum, moveId0)
            : MovesEnum.MOVES_ENUM___NULL;
        teamBytes[offset + TeamSetFeature.TEAM_SET_FEATURE__MOVEID1] = moveId1
            ? IndexValueFromEnum(MovesEnum, moveId1)
            : MovesEnum.MOVES_ENUM___NULL;
        teamBytes[offset + TeamSetFeature.TEAM_SET_FEATURE__MOVEID2] = moveId2
            ? IndexValueFromEnum(MovesEnum, moveId2)
            : MovesEnum.MOVES_ENUM___NULL;
        teamBytes[offset + TeamSetFeature.TEAM_SET_FEATURE__MOVEID3] = moveId3
            ? IndexValueFromEnum(MovesEnum, moveId3)
            : MovesEnum.MOVES_ENUM___NULL;
        teamBytes[offset + TeamSetFeature.TEAM_SET_FEATURE__NATURE] =
            pokemonSet.nature
                ? IndexValueFromEnum(NaturesEnum, pokemonSet.nature)
                : NaturesEnum.NATURES_ENUM__SERIOUS;
        teamBytes[offset + TeamSetFeature.TEAM_SET_FEATURE__HAPPINESS] =
            pokemonSet.happiness ?? 255;
        teamBytes[offset + TeamSetFeature.TEAM_SET_FEATURE__LEVEL] =
            pokemonSet.level ?? 100;
        teamBytes[offset + TeamSetFeature.TEAM_SET_FEATURE__EV_HP] =
            pokemonSet.evs.hp;
        teamBytes[offset + TeamSetFeature.TEAM_SET_FEATURE__EV_ATK] =
            pokemonSet.evs.atk;
        teamBytes[offset + TeamSetFeature.TEAM_SET_FEATURE__EV_DEF] =
            pokemonSet.evs.def;
        teamBytes[offset + TeamSetFeature.TEAM_SET_FEATURE__EV_SPA] =
            pokemonSet.evs.spa;
        teamBytes[offset + TeamSetFeature.TEAM_SET_FEATURE__EV_SPD] =
            pokemonSet.evs.spd;
        teamBytes[offset + TeamSetFeature.TEAM_SET_FEATURE__EV_SPE] =
            pokemonSet.evs.spe;
    }
    return new Uint8Array(teamBytes.buffer);
}

async function runBattle() {
    console.log("Creating battle...");

    const team1Bytes = createFakeTeamBytes();
    const team2Bytes = createFakeTeamBytes();

    const names = {
        p1Name: "Bot1",
        p2Name: "Bot2",
        p1team: teamBytesToPackedString(new Int16Array(team1Bytes.buffer)),
        p2team: teamBytesToPackedString(new Int16Array(team2Bytes.buffer)),
    };
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
