import { evalActionMapping, getEvalAction } from "../logic/eval";
import { port } from "./utils";
import { Game } from "../server/game";
import { Action } from "../../protos/action_pb";
import { writeFileSync } from "fs";

const tasks = [];

const evals = [
    ...evalActionMapping,
    // "extra"
];

// Split tasks among workers
for (const [evalIndex1, _] of evals.entries()) {
    for (const [evalIndex2, _] of evals.entries()) {
        for (let i = 0; i < 10; i++) {
            tasks.push({ evalIndex1, evalIndex2, i });
        }
    }
}

async function ActionFromResponse(response: Response): Promise<Action> {
    const { pi, v, action: actionIndex, prev_pi } = await response.json();
    const action = new Action();
    action.setIndex(actionIndex);
    return action;
}

const concurrentLimit = 2; // Adjust this value based on your memory constraints

async function processGame(
    evalIndex1: number,
    evalIndex2: number,
    i: number,
): Promise<number> {
    const game = new Game({
        port: port,
        gameId: 0,
    });
    game.handlers[0].sendFn = async (state) => {
        const jobKey = game.queueSystem.createJob();
        state.setKey(jobKey);
        if (evalIndex1 >= evalActionMapping.length) {
            const response = await fetch("http://127.0.0.1:8080/predict", {
                method: "POST",
                body: state.serializeBinary(),
            });
            const action = await ActionFromResponse(response);
            game.queueSystem.submitResult(jobKey, action);
        } else {
            const action = await getEvalAction(game.handlers[0], evalIndex1);
            game.queueSystem.submitResult(jobKey, action);
        }
        return jobKey;
    };
    game.handlers[1].sendFn = async (state) => {
        const jobKey = game.queueSystem.createJob();
        state.setKey(jobKey);
        if (evalIndex2 >= evalActionMapping.length) {
            const response = await fetch("http://127.0.0.1:8080/predict", {
                method: "POST",
                body: state.serializeBinary(),
            });
            const action = await ActionFromResponse(response);
            game.queueSystem.submitResult(jobKey, action);
        } else {
            const action = await getEvalAction(game.handlers[1], evalIndex2);
            game.queueSystem.submitResult(jobKey, action);
        }
        return jobKey;
    };
    await game.run();
    const r1 = game.getRewardFromHpDiff();
    return r1;
}

async function processChunk(
    chunk: { evalIndex1: number; evalIndex2: number; i: number }[],
) {
    const results = [];
    for (const { evalIndex1, evalIndex2, i } of chunk) {
        const chunkResults = [];
        for (let j = 0; j < concurrentLimit; j++) {
            chunkResults.push(processGame(evalIndex1, evalIndex2, i));
        }
        results.push(...[await Promise.all(chunkResults)]);
        console.log(results.flat().length / tasks.length / concurrentLimit);
    }
    writeFileSync("../viz/matchupResults.json", JSON.stringify(results.flat()));
}

processChunk(tasks);
