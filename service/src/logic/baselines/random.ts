import { EvalActionFnType } from "../eval";
import { chooseRandom } from "../utils";

export const GetRandomAction: EvalActionFnType = ({ handler }) => {
    const state = handler.getState();
    const legalActions = state.getLegalactions();
    if (legalActions) {
        const randomIndex = chooseRandom(legalActions);
        return randomIndex;
    } else {
        return -1;
    }
};
