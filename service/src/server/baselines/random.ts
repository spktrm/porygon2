import { ActionMaskFeature, ActionType } from "../../../protos/features_pb";
import { EvalActionFnType } from "../eval";
import { StateHandler } from "../state";
import { OneDBoolean } from "../utils";
import { Action } from "../../../protos/service_pb";

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

    const actionTypeMask = [
        actionBinary[ActionMaskFeature.ACTION_MASK_FEATURE__CAN_MOVE],
        actionBinary[ActionMaskFeature.ACTION_MASK_FEATURE__CAN_SWITCH],
        actionBinary[ActionMaskFeature.ACTION_MASK_FEATURE__CAN_TEAMPREVIEW],
    ];
    const randomActionType = getRandomOneIndex(actionTypeMask);
    if (randomActionType === 0) {
        action.setActionType(ActionType.ACTION_TYPE__MOVE);
    } else if (randomActionType === 1) {
        action.setActionType(ActionType.ACTION_TYPE__SWITCH);
    } else if (randomActionType === 2) {
        action.setActionType(ActionType.ACTION_TYPE__TEAMPREVIEW);
    } else {
        throw new Error(
            `Invalid action type index: ${randomActionType}. Expected 0 or 1.`,
        );
    }

    const moveMask = [
        actionBinary[ActionMaskFeature.ACTION_MASK_FEATURE__MOVE_SLOT_1],
        actionBinary[ActionMaskFeature.ACTION_MASK_FEATURE__MOVE_SLOT_2],
        actionBinary[ActionMaskFeature.ACTION_MASK_FEATURE__MOVE_SLOT_3],
        actionBinary[ActionMaskFeature.ACTION_MASK_FEATURE__MOVE_SLOT_4],
    ];
    action.setMoveSlot(getRandomOneIndex(moveMask));

    const switchMask = [
        actionBinary[ActionMaskFeature.ACTION_MASK_FEATURE__SWITCH_SLOT_1],
        actionBinary[ActionMaskFeature.ACTION_MASK_FEATURE__SWITCH_SLOT_2],
        actionBinary[ActionMaskFeature.ACTION_MASK_FEATURE__SWITCH_SLOT_3],
        actionBinary[ActionMaskFeature.ACTION_MASK_FEATURE__SWITCH_SLOT_4],
        actionBinary[ActionMaskFeature.ACTION_MASK_FEATURE__SWITCH_SLOT_5],
        actionBinary[ActionMaskFeature.ACTION_MASK_FEATURE__SWITCH_SLOT_6],
    ];
    action.setSwitchSlot(getRandomOneIndex(switchMask));

    const canTerastallize =
        !!actionBinary[ActionMaskFeature.ACTION_MASK_FEATURE__CAN_TERA];
    action.setShouldTera(canTerastallize && Math.random() < 0.3);

    return action;
}

export const GetRandomAction: EvalActionFnType = ({ player }) => {
    const { actionMask } = StateHandler.getActionMask(
        player.privateBattle.request,
    );
    return actionMaskToRandomAction(actionMask);
};
