import { Worker, isMainThread, parentPort, workerData } from "worker_threads";
import { writeFileSync } from "fs";
import { evalActionMapping, getEvalAction } from "../logic/eval";
import { port } from "./utils";
import { Game } from "../server/game";
import { Action } from "../../protos/action_pb";

const numWorkers = require("os").cpus().length; // Number of CPU cores
const tasks = [];

const evals = [...evalActionMapping, "extra"];

// Split tasks among workers
for (const [evalIndex1, _] of evals.entries()) {
    for (const [evalIndex2, _] of evals.entries()) {
        for (let i = 0; i < 100; i++) {
            tasks.push({ evalIndex1, evalIndex2, i });
        }
    }
}

async function ActionFromResponse(response: Response): Promise<Action> {
    const { pi, v, action: actionIndex, prev_pi } = await response.json();
    // console.log(pi);
    // console.log(v);
    const action = new Action();
    action.setIndex(actionIndex);
    return action;
}

if (isMainThread) {
    const chunkSize = Math.ceil(tasks.length / numWorkers);
    const promises: Promise<number[]>[] = [];

    for (let i = 0; i < numWorkers; i++) {
        const chunk = tasks.slice(i * chunkSize, (i + 1) * chunkSize);
        promises.push(
            new Promise((resolve, reject) => {
                const worker = new Worker(__filename, {
                    workerData: chunk,
                });

                worker.on("message", resolve);
                worker.on("error", reject);
                worker.on("exit", (code) => {
                    if (code !== 0)
                        reject(
                            new Error(`Worker stopped with exit code ${code}`),
                        );
                });
            }),
        );
    }

    Promise.all(promises)
        .then((results) => {
            const combinedResults = results.flat();
            writeFileSync(
                "../viz/matchupResults.json",
                JSON.stringify(combinedResults),
            );
        })
        .catch((err) => {
            console.error(err);
        });
} else {
    // Worker thread: process assigned tasks
    const chunk = workerData;
    const results: number[] = [];
    const concurrentLimit = 1; // Adjust this value based on your memory constraints

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
                const action = getEvalAction(game.handlers[0], evalIndex1);
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
                const action = getEvalAction(game.handlers[1], evalIndex2);
                game.queueSystem.submitResult(jobKey, action);
            }
            return jobKey;
        };
        await game.run();
        const [r1, r2] = game.getRewardFromHpDiff(0);
        console.log(evalIndex1, evalIndex2, i);
        return r1;
    }

    async function processChunk(
        chunk: { evalIndex1: number; evalIndex2: number; i: number }[],
    ) {
        for (const { evalIndex1, evalIndex2, i } of chunk) {
            const chunkResults = [];
            for (let j = 0; j < concurrentLimit; j++) {
                chunkResults.push(processGame(evalIndex1, evalIndex2, i));
            }
            const resultsBatch = await Promise.all(chunkResults);
            results.push(...resultsBatch);
        }
    }

    processChunk(chunk).then(() => parentPort?.postMessage(results));
}
