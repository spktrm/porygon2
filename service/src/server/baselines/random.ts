import { EvalActionFnType } from "../eval";
import { StateHandler } from "../state";
import { Action } from "../../../protos/service_pb";
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

export function legalCellsToRandomAction(legalCells: boolean[]): Action {
    const action = new Action();
    action.setCell(
        getRandomOneIndex(legalCells.map((legal) => (legal ? 1 : 0))),
    );
    return action;
}

export const GetRandomAction: EvalActionFnType = ({ player }) => {
    const request = player.privateBattle.request as
        | AnyObject
        | null
        | undefined;

    const playerIndex = player.getPlayerIndex();
    if (playerIndex === undefined) {
        throw new Error("Player index is undefined");
    }
    const stateHandler = new StateHandler(player);
    const allyActive = player.publicBattle.sides[playerIndex].active;
    const enemyActive = player.publicBattle.sides[1 - playerIndex].active;
    const { legalCells } = stateHandler.getActionMask({
        request,
        allyActive,
        enemyActive,
    });

    return legalCellsToRandomAction(legalCells);
};
