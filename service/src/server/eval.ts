import { GetRandomAction } from "./baselines/random";
import { GetMaxDamageAction } from "./baselines/max_dmg";
import { GetHeuristicAction } from "./baselines/heuristic";
import { TrainablePlayerAI } from "./runner";

export type EvalFuncArgs = {
    player: TrainablePlayerAI;
};

export type EvalAction = {
    actionIndex?: number;
    actionString?: string;
};

export type EvalActionFnType = (args: EvalFuncArgs) => EvalAction;

export const evalActionMapping: EvalActionFnType[] = [
    GetRandomAction, // Random - 0
    () => {
        return { actionIndex: -1 };
    }, // Default - 1
    GetMaxDamageAction,
    GetHeuristicAction,
    // GetSearchAction,
];

export const numEvals = Object.keys(evalActionMapping).length;
