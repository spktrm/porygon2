import { GetRandomAction } from "./baselines/random";
import { GetMaxDamageAction } from "./baselines/max_dmg";
import { GetHeuristicAction } from "./baselines/heuristic";
import { TrainablePlayerAI } from "./runner";
import { GetKaizoPlusAction } from "./baselines/kaizo_plus";
import { Action } from "../../protos/service_pb";
import { ActionType } from "../../protos/features_pb";

export type EvalFuncArgs = {
    player: TrainablePlayerAI;
};

export type EvalActionFnType = (args: EvalFuncArgs) => Action;

export const evalActionMapping: EvalActionFnType[] = [
    GetRandomAction, // Random - 0
    () => {
        const action = new Action();
        action.setActionType(ActionType.ACTION_TYPE__DEFAULT);
        return action;
    }, // Default - 1
    GetMaxDamageAction,
    GetHeuristicAction,
    // GetSearchAction,
    GetKaizoPlusAction,
];

export const numEvals = evalActionMapping.length;
