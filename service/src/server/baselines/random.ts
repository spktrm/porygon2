import { Action } from "../../../protos/service_pb";
import { EvalActionFnType } from "../eval";
import { StateHandler } from "../state";

function getRandomOneIndex(arr: number[]): number {
    // Collect indices where the element is 1
    const oneIndices: number[] = [];

    // Loop through the array to find indices of 1s
    for (let i = 0; i < arr.length; i++) {
        if (arr[i] === 1) {
            oneIndices.push(i);
        }
    }

    // If there are no ones, return undefined
    if (oneIndices.length === 0) {
        return Math.floor(Math.random() * arr.length);
    }

    // Pick a random index from the oneIndices array
    const randomIndex = Math.floor(Math.random() * oneIndices.length);
    return oneIndices[randomIndex];
}

export const GetRandomAction: EvalActionFnType = ({ player }) => {
    const { legalActions } = StateHandler.getLegalActions(
        player.privateBattle.request,
    );
    const legalIndices = legalActions.toBinaryVector();
    const randIndex = getRandomOneIndex(legalIndices);
    const action = new Action();
    action.setValue(randIndex);
    return action;
};
