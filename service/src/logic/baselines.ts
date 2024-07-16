import { AnyObject } from "@pkmn/sim";
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
                    return currentElement > arr[maxIndex]
                        ? currentIndex
                        : maxIndex;
                },
                0,
            );
            action.setIndex(indexOfLargestNum);
        } else {
            action.setIndex(-1);
            action.setText("default");
        }

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
