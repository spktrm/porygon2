import { Action } from "../../protos/action_pb";
import { StreamHandler } from "./handler";
import { GetMaxDamageAction } from "./baselines/max_dmg";
import { GetBestSwitchAction } from "./baselines/switcher";
import { GetSearchAction } from "./baselines/search";
import { Battle as World } from "@pkmn/sim";
import { GetRandomAction } from "./baselines/random";

export type evalFuncArgs = {
    handler: StreamHandler;
    world?: World | null;
    [k: string]: any;
};

export type EvalActionFnType = (args: evalFuncArgs) => number;

export function partial(
    fn: EvalActionFnType,
    presetArgs: { [k: string]: any },
): (remainingArgs: evalFuncArgs) => number {
    return function (remainingArgs: evalFuncArgs): number {
        const allArgs = { ...presetArgs, ...remainingArgs };
        return fn(allArgs);
    };
}

export const evalActionMapping: EvalActionFnType[] = [
    // Random
    GetRandomAction,
    // Default
    () => {
        return -1;
    },
    GetMaxDamageAction,
    partial(GetBestSwitchAction, { switchThreshold: 1.6 }),
    partial(GetBestSwitchAction, { switchThreshold: 1.2 }),
    partial(GetBestSwitchAction, { switchThreshold: 0.8 }),
    partial(GetBestSwitchAction, { switchThreshold: 0.4 }),
    partial(GetBestSwitchAction, { switchThreshold: 0.2 }),
    partial(GetBestSwitchAction, { switchThreshold: 0 }),
    partial(GetBestSwitchAction, { switchThreshold: -0.2 }),
    partial(GetBestSwitchAction, { switchThreshold: -0.4 }),
    partial(GetBestSwitchAction, { switchThreshold: -0.8 }),
    partial(GetBestSwitchAction, { switchThreshold: -1.2 }),
    partial(GetBestSwitchAction, { switchThreshold: -1.6 }),
    GetSearchAction,
];

export const numEvals = Object.keys(evalActionMapping).length;

export function getEvalAction(
    handler: StreamHandler,
    evalIndex: number,
): Action {
    let action = new Action();

    const evalFunc = evalActionMapping[evalIndex];
    const actionIndex = evalFunc({ handler });

    action.setIndex(actionIndex);
    if (actionIndex < 0) {
        action.setText("default");
    }

    return action;
}
