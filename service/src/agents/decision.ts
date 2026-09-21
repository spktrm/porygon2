import { TrainablePlayerAI } from "../server/runner";
import {
    Action as ProtoAction,
    EnvironmentState,
    StepRequest,
} from "../../protos/service_pb";

export interface Decision<Meta> {
    cell: number;
    meta: Meta;
}

/**
 * `player.legalChoiceByCell` is current when a source runs: the runner
 * assigns it while building the state it then enqueues.
 */
export type DecisionSource<Meta> = (
    player: TrainablePlayerAI,
    state: EnvironmentState,
) => Promise<Decision<Meta>>;

export interface CheckpointMeta {
    vWin: number;
    logProb: number;
    entropy: number;
}

export function checkpointSource(
    rlServerUrl: string,
): DecisionSource<CheckpointMeta> {
    return async (_player, state) => {
        const response = await fetch(`${rlServerUrl}/step`, {
            method: "POST",
            body: state.serializeBinary(),
        });
        if (!response.ok) {
            throw new Error(
                `RL server /step returned ${response.status}: ${await response.text()}`,
            );
        }
        const body = await response.json();
        if (!Number.isInteger(body.cell)) {
            throw new Error(`RL server /step returned cell ${body.cell}`);
        }
        return {
            cell: body.cell,
            meta: {
                vWin: body.v_win,
                logProb: body.log_prob,
                entropy: body.entropy,
            },
        };
    };
}

export function submitCell(
    player: TrainablePlayerAI,
    state: EnvironmentState,
    cell: number,
) {
    const protoAction = new ProtoAction();
    protoAction.setCell(cell);
    const stepRequest = new StepRequest();
    stepRequest.setAction(protoAction);
    stepRequest.setRqid(state.getRqid());
    player.submitStepRequest(stepRequest);
}

/**
 * Decisions go through the runner's queue rather than an override of
 * `getChoice`, which is what keeps the player's `actionCells` record right.
 */
export async function playDecisions<Meta>(
    player: TrainablePlayerAI,
    decide: DecisionSource<Meta>,
    onDecision?: (decision: Decision<Meta>) => void,
): Promise<void> {
    while (true) {
        const state = await player.receiveEnvironmentState();
        if (player.done) {
            break;
        }
        const decision = await decide(player, state);
        if (onDecision !== undefined) {
            onDecision(decision);
        }
        submitCell(player, state, decision.cell);
    }
}
