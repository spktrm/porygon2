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

const evalActionMapping: {
    [k: number]: EvalActionFnType;
} = {
    // Random
    0: ({ handler }) => {
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
    1: () => {
        return -1;
    },
    2: GetMaxDamageAction,
    3: partial(GetBestSwitchAction, { switchThreshold: 0 }),
    4: partial(GetBestSwitchAction, { switchThreshold: -1 }),
    5: partial(GetBestSwitchAction, { switchThreshold: -2 }),
    6: partial(GetBestSwitchAction, { switchThreshold: -3 }),
    7: partial(GetBestSwitchAction, { switchThreshold: -4 }),
};

export const numEvals = Object.keys(evalActionMapping).length;

export function getEvalAction(handler: StreamHandler): Action {
    let action = new Action();

    const evalIndex = (handler.gameId %
        numEvals) as keyof typeof evalActionMapping;
    const evalFunc = evalActionMapping[evalIndex];
    const actionIndex = evalFunc({ handler, action });

    action.setIndex(actionIndex);
    if (actionIndex < 0) {
        action.setText("default");
    }

    return action;
}
