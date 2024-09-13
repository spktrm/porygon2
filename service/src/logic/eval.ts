import { Action } from "../../protos/action_pb";
import { StreamHandler } from "./handler";
import { GetMaxDamageAction } from "./baselines/max_dmg";
import { GetBestSwitchAction } from "./baselines/switcher";
import { Battle as World } from "@pkmn/sim";
import { GetRandomAction } from "./baselines/random";

export type evalFuncArgs = {
    handler: StreamHandler;
    world?: World | null;
    [k: string]: any;
};

export type EvalActionFnType = (args: evalFuncArgs) => Promise<number>;

export function partial(
    fn: EvalActionFnType,
    presetArgs: { [k: string]: any },
): EvalActionFnType {
    return function (remainingArgs: evalFuncArgs): Promise<number> {
        const allArgs = { ...presetArgs, ...remainingArgs };
        return fn(allArgs);
    };
}

export const evalActionMapping: EvalActionFnType[] = [
    GetRandomAction, // Random - 0
    async () => {
        return -1;
    }, // Default - 1
    GetMaxDamageAction,
    partial(GetBestSwitchAction, { switchThreshold: 1.6 }), // - 2
    partial(GetBestSwitchAction, { switchThreshold: 1.2 }), // - 3
    partial(GetBestSwitchAction, { switchThreshold: 0.8 }), // - 4
    partial(GetBestSwitchAction, { switchThreshold: 0.4 }), // - 5
    partial(GetBestSwitchAction, { switchThreshold: 0.2 }), // - 6
    partial(GetBestSwitchAction, { switchThreshold: 0 }), // - 7
    partial(GetBestSwitchAction, { switchThreshold: -0.2 }), // - 8
    partial(GetBestSwitchAction, { switchThreshold: -0.4 }), // - 9
    partial(GetBestSwitchAction, { switchThreshold: -0.8 }), // - 10
    partial(GetBestSwitchAction, { switchThreshold: -1.2 }), // - 11
    partial(GetBestSwitchAction, { switchThreshold: -1.6 }), // - 12
    // GetSearchAction,
];

export const numEvals = Object.keys(evalActionMapping).length;

export async function getEvalAction(
    handler: StreamHandler,
    evalIndex: number,
): Promise<Action> {
    let action = new Action();

    const evalFunc = evalActionMapping[evalIndex];
    const actionIndex = await evalFunc({ handler });

    action.setIndex(actionIndex);
    if (actionIndex < 0) {
        action.setText("default");
    }

    return action;
}
