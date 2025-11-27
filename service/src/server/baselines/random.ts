import { ActionType } from "../../../protos/features_pb";
import { EvalActionFnType } from "../eval";
import { StateHandler } from "../state";
import { OneDBoolean } from "../utils";
import { Action, ActionEnum, WildCardEnum } from "../../../protos/service_pb";
import { AnyObject } from "@pkmn/sim";

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

export function actionMaskToRandomAction(
    actionMask: OneDBoolean,
    wildCardMask: OneDBoolean,
): Action {
    const actionBinary = actionMask.toBinaryVector();
    const wildCardBinary = wildCardMask.toBinaryVector();

    const action = new Action();

    const randomAction = getRandomOneIndex(actionBinary);
    action.setAction(
        randomAction as (typeof ActionEnum)[keyof typeof ActionEnum],
    );

    const canTerastallize =
        !!wildCardBinary[WildCardEnum.WILD_CARD_ENUM__CAN_TERA];
    if (canTerastallize && Math.random() < 0.3) {
        action.setWildcard(WildCardEnum.WILD_CARD_ENUM__CAN_TERA);
    } else {
        action.setWildcard(WildCardEnum.WILD_CARD_ENUM__CAN_NORMAL);
    }

    return action;
}

export const GetRandomAction: EvalActionFnType = ({ player }) => {
    const request = player.privateBattle.request as
        | AnyObject
        | null
        | undefined;

    const numActive = player.privateBattle.gameType.includes("doubles") ? 2 : 1;
    const { actionMask, wildCardMask } = StateHandler.getActionMask({
        request,
        format: player.privateBattle.gameType,
    });

    const actionList = [];
    const actionMaskSplits = actionMask.split(numActive);
    const wildCardMaskSplits = wildCardMask.split(numActive);

    for (let i = 0; i < numActive; i++) {
        const actionSplit_i = actionMaskSplits[i];
        const wildCardSplit_i = wildCardMaskSplits[i];
        for (const chosen of actionList) {
            const chosenActionIndex = chosen.getAction();
            if (
                ActionEnum.ACTION_ENUM__SWITCH_1 <= chosenActionIndex &&
                chosenActionIndex <= ActionEnum.ACTION_ENUM__SWITCH_6
            ) {
                actionSplit_i.set(chosenActionIndex, false); // Mark this switch as unavailable
            }
            if (actionSplit_i.sum() === 0) {
                actionSplit_i.set(ActionEnum.ACTION_ENUM__PASS, true); // If no actions left, set PASS to true
                break;
            }
            const chosenWildCard = chosen.getWildcard();
            if (chosenWildCard === WildCardEnum.WILD_CARD_ENUM__CAN_TERA) {
                wildCardSplit_i.set(chosenWildCard, false); // Mark this wildcard as unavailable
            }
        }
        const action = actionMaskToRandomAction(actionSplit_i, wildCardSplit_i);
        actionList.push(action);
    }

    return actionList;
};
