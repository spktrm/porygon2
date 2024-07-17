import { AnyObject } from "@pkmn/sim";
import { EvalActionFnType } from "../eval";

export const GetBestSwitchAction: EvalActionFnType = ({ handler, action }) => {
    const request = handler.privatebattle.request as AnyObject;
    const active = request.active ?? [];
    const moves = active[0]?.moves;
    if (moves !== undefined) {
        const moveData: number[] = moves.map(
            ({ id, disabled }: { id: string; disabled: boolean }) => {
                const trueId = id.startsWith("return") ? "return" : id;
                return disabled
                    ? -100
                    : handler.privatebattle.gens.dex.moves.get(trueId)
                          .basePower;
            },
        );
        const indexOfLargestNum = moveData.reduce(
            (maxIndex, currentElement, currentIndex, arr) => {
                return currentElement > arr[maxIndex] ? currentIndex : maxIndex;
            },
            0,
        );
        action.setIndex(indexOfLargestNum);
    } else {
        action.setIndex(-1);
        action.setText("default");
    }

    return action;
};
