import { AnyObject } from "@pkmn/sim";
import { Action } from "../../protos/action_pb";
import { StreamHandler } from "./handler";
import { chooseRandom } from "./utils";
import { GetMaxDamageAction } from "./baselines/max_dmg";
import { GetBestSwitchAction } from "./baselines/switcher";

type evalFuncArgs = {
    handler: StreamHandler;
    action: Action;
};

export type EvalActionFnType = (args: evalFuncArgs) => Action;

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
    3: GetBestSwitchAction,
};

export const numEvals = Object.keys(evalActionMapping).length;

export function getEvalAction(handler: StreamHandler): Action {
    let action = new Action();

    const evalIndex = (handler.gameId %
        numEvals) as keyof typeof evalActionMapping;
    const evalFunc = evalActionMapping[evalIndex];

    return evalFunc({ handler, action });
}
