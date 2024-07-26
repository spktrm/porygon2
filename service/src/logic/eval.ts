import { Action } from "../../protos/action_pb";
import { StreamHandler } from "./handler";
import { chooseRandom } from "./utils";
import { GetMaxDamageAction } from "./baselines/max_dmg";
import { GetBestSwitchAction } from "./baselines/switcher";

export type evalFuncArgs = {
    handler: StreamHandler;
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
    ({ handler }) => {
        const state = handler.getState();
        const legalActions = state.getLegalactions();
        if (legalActions) {
            const randomIndex = chooseRandom(legalActions);
            return randomIndex;
        } else {
            return -1;
        }
    },
    // Default
    () => {
        return -1;
    },
    GetMaxDamageAction,
    partial(GetBestSwitchAction, { switchThreshold: 0 }),
    partial(GetBestSwitchAction, { switchThreshold: -1 }),
    partial(GetBestSwitchAction, { switchThreshold: -2 }),
    partial(GetBestSwitchAction, { switchThreshold: -3 }),
    partial(GetBestSwitchAction, { switchThreshold: -4 }),
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
