import { EvalActionFnType } from "../eval";
import { StateHandler } from "../state";
import { OneDBoolean } from "../utils";
import { Action, ActionEnum, ActionEnumMap } from "../../../protos/service_pb";
import { AnyObject } from "@pkmn/sim";
import { numActionFeatures } from "../data";

export function getRandomOneIndex(arr: number[]): number {
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

export function actionMaskToRandomAction(actionMask: OneDBoolean): Action {
    const actionBinary = actionMask.toBinaryVector();

    const action = new Action();

    const randomAction = getRandomOneIndex(actionBinary);

    const srcIndex = Math.floor(randomAction / numActionFeatures);
    const tgtIndex = randomAction % numActionFeatures;

    action.setSrc(srcIndex as ActionEnumMap[keyof typeof ActionEnum]);
    action.setTgt(tgtIndex as ActionEnumMap[keyof typeof ActionEnum]);

    return action;
}

export const GetRandomAction: EvalActionFnType = ({ player }) => {
    const request = player.privateBattle.request as
        | AnyObject
        | null
        | undefined;

    const stateHandler = new StateHandler(player);
    const { actionMask } = stateHandler.getActionMask({
        request,
        format: player.privateBattle.gameType,
    });

    return actionMaskToRandomAction(actionMask);
};
