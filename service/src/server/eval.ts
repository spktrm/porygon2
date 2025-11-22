import { GetRandomAction } from "./baselines/random";
// import { GetMaxDamageAction } from "./baselines/max_dmg";
// import { GetHeuristicAction } from "./baselines/heuristic";
// import { GetKaizoPlusAction } from "./baselines/kaizo_plus";
import { TrainablePlayerAI } from "./runner";
import { Action, ActionEnum } from "../../protos/service_pb";

export type EvalFuncArgs = {
    player: TrainablePlayerAI;
};

export type EvalActionFnType = (args: EvalFuncArgs) => Action[];

export const evalActionMapping: EvalActionFnType[] = [
    GetRandomAction, // Random - 0
    ({ player }) => {
        const request = player.getRequest();
        if (!request) {
            throw new Error("No request available for default action.");
        }
        const numActive = player.privateBattle.gameType.includes("doubles")
            ? 2
            : 1;
        const actions: Action[] = [];
        for (let i = 0; i < numActive; i++) {
            const action = new Action();
            action.setAction(ActionEnum.ACTION_ENUM__DEFAULT);
            actions.push(action);
        }
        return actions;
    }, // Default - 1
    // GetMaxDamageAction,
    // GetHeuristicAction,
    // GetSearchAction,
    // GetKaizoPlusAction,
];

export const numEvals = evalActionMapping.length;
