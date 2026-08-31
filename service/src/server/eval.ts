import { GetRandomAction } from "./baselines/random";
import { GetSimpleHeuristicAction } from "./baselines/simple_heuristic";
import { TrainablePlayerAI } from "./runner";
import { Action } from "../../protos/service_pb";

export type EvalFuncArgs = {
    player: TrainablePlayerAI;
};

export type EvalActionFnType = (args: EvalFuncArgs) => Action;

export const evalActionMapping: EvalActionFnType[] = [
    GetRandomAction, // Random - 0
    ({ player }) => {
        const request = player.getRequest();
        if (!request) {
            throw new Error("No request available for default action.");
        }

        const action = new Action();
        // Not a real cell: the decoder answers "default" when the request
        // carries no choice map, and throws otherwise -- exactly what the
        // (DEFAULT, DEFAULT) pseudo-cell did on the grid.
        action.setCell(-1);

        return action;
    }, // Default - 1
    GetSimpleHeuristicAction, // Type-aware max-damage + switching - 2
];

export const numEvals = evalActionMapping.length;
