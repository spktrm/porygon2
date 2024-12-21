import { Action } from "../../../protos/servicev2_pb";
import { LLMState } from "../../../protos/llm_pb";
import { EvalActionFnType } from "../eval";
import { StateHandler } from "../state";

export const GetLLMAction: EvalActionFnType = ({ player }) => {
    const legalMask = StateHandler.getLegalActions(player.privateBattle.request)
        .toBinaryVector()
        .map((x) => !!x);

    const llmState = new LLMState();
    llmState.setLegalmaskList(legalMask);
    llmState.setRequest(JSON.stringify(player.privateBattle.request));
    llmState.setLog(JSON.stringify(player.log));

    const action = new Action();
    return action;
};
