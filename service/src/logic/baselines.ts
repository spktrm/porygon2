import { Action } from "../../protos/action_pb";
import { State } from "../../protos/state_pb";
import { chooseRandom } from "./utils";

const evalActionMapping = {
    0: (state: State, action: Action) => {
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
    1: (state: State, action: Action) => {
        action.setIndex(-1);
        action.setText("default");
        return action;
    },
};

const numEvals = Object.keys(evalActionMapping).length;

export function getEvalAction(state: State): Action {
    const action = new Action();
    const info = state.getInfo();
    if (info) {
        const gameId = info.getGameid() % numEvals;
        const evalFunc =
            evalActionMapping[gameId as keyof typeof evalActionMapping];
        return evalFunc(state, action);
    }
    action.setIndex(-1);
    action.setText("default");
    return action;
}
