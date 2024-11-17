import { Player } from "./player";
import { EVAL_GAME_ID_OFFSET } from "./data";
import { GetRandomAction } from "./baselines/random";
import { GetMaxDamageAction } from "./baselines/max_dmg";
import { Action } from "../../protos/servicev2_pb";
import { GetHeuristicAction } from "./baselines/heuristic";

export type evalFuncArgs = {
    player: Player;
};

export type EvalActionFnType = (args: evalFuncArgs) => Action;

export const evalActionMapping: EvalActionFnType[] = [
    GetRandomAction, // Random - 0
    () => {
        const action = new Action();
        action.setValue(-1);
        return action;
    }, // Default - 1
    GetMaxDamageAction,
    GetHeuristicAction,
];

export const numEvals = Object.keys(evalActionMapping).length;

export function getEvalAction(player: Player): Action {
    if (!!player.playerId) {
        console.error("Evaluation playerId should be undefined");
    }
    return evalActionMapping[player.gameId - EVAL_GAME_ID_OFFSET]({ player });
}
