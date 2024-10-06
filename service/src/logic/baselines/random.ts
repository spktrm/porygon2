import { EvalActionFnType } from "../eval";
import { StateHandler } from "../state";
import { chooseRandom, getLegalActionIndices } from "../utils";

export const GetRandomAction: EvalActionFnType = async ({ handler }) => {
    const legalActions = StateHandler.getLegalActions(
        handler.privateBattle.request,
    );
    if (legalActions) {
        const legalIndices = getLegalActionIndices(
            legalActions.toBinaryVector(),
        );
        return legalIndices[Math.floor(Math.random() * legalIndices.length)];
    } else {
        return -1;
    }
};
