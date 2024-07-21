import { Action } from "../../protos/action_pb";
import { StreamHandler } from "./handler";
import { chooseRandom } from "./utils";
import { GetMaxDamageAction } from "./baselines/max_dmg";
import { GetBestSwitchAction } from "./baselines/switcher";

export type evalFuncArgs = {
    handler: StreamHandler;
    action: Action;
    [k: string]: any;
};

export type EvalActionFnType = (args: evalFuncArgs) => Action;

function partial(
    fn: EvalActionFnType,
    presetArgs: { [k: string]: any },
): (remainingArgs: evalFuncArgs) => Action {
    return function (remainingArgs: evalFuncArgs): Action {
        const allArgs = { ...presetArgs, ...remainingArgs };
        return fn(allArgs);
    };
}

const evalActionMapping: {
    [k: number]: EvalActionFnType;
} = {
    // Random
    0: ({ handler, action }) => {
        const state = handler.getState();
        const legalActions = state.getLegalactions();
        if (legalActions) {
            const randomIndex = chooseRandom(legalActions);
            action.setIndex(randomIndex);
        } else {
            action.setIndex(-1);
            action.setText("default");
        }
        return action;
    },
    // Default
    1: ({ action }) => {
        action.setIndex(-1);
        action.setText("default");
        return action;
    },
    2: GetMaxDamageAction,
    3: partial(GetBestSwitchAction, { switchThreshold: 0 }),
    4: partial(GetBestSwitchAction, { switchThreshold: -1 }),
    5: partial(GetBestSwitchAction, { switchThreshold: -2 }),
};

export const numEvals = Object.keys(evalActionMapping).length;

export function getEvalAction(handler: StreamHandler): Action {
    let action = new Action();

    const evalIndex = (handler.gameId %
        numEvals) as keyof typeof evalActionMapping;
    const evalFunc = evalActionMapping[evalIndex];

    return evalFunc({ handler, action });
}
