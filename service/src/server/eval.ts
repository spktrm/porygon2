import { GetRandomAction } from "./baselines/random";
// import { GetMaxDamageAction } from "./baselines/max_dmg";
// import { GetHeuristicAction } from "./baselines/heuristic";
// import { GetKaizoPlusAction } from "./baselines/kaizo_plus";
import { TrainablePlayerAI } from "./runner";
import { Action, ActionEnum } from "../../protos/service_pb";

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
        action.setSrc(ActionEnum.ACTION_ENUM__DEFAULT);
        action.setTgt(ActionEnum.ACTION_ENUM__DEFAULT);

        return action;
    }, // Default - 1
    // GetMaxDamageAction,
    // GetHeuristicAction,
    // GetSearchAction,
    // GetKaizoPlusAction,
];

export const numEvals = evalActionMapping.length;
