import { Action } from "../../protos/action_pb";
import { State } from "../../protos/state_pb";
import { StreamHandler } from "./handler";
import { chooseRandom } from "./utils";

type evalFuncArgs = {
    handler: StreamHandler;
    action: Action;
};

const evalActionMapping: {
    [k: number]: (args: evalFuncArgs) => Action;
} = {
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
    1: ({ action }) => {
        action.setIndex(-1);
        action.setText("default");
        return action;
    },
    2: ({ handler, action }) => {
        action.setIndex(-1);
        action.setText("default");
        return action;
    },
};

const numEvals = Object.keys(evalActionMapping).length;

export function getEvalAction(handler: StreamHandler): Action {
    let action = new Action();

    const evalIndex = (handler.gameId %
        numEvals) as keyof typeof evalActionMapping;
    const evalFunc = evalActionMapping[evalIndex];
    action = evalFunc({ handler, action });

    action.setIndex(-1);
    action.setText("default");
    return action;
}
