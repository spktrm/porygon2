import assert from "assert";

import { Action } from "../../protos/action_pb";
import { State } from "../../protos/state_pb";
import { port } from "./utils";
import { Game } from "../server/game";
import { getEvalAction, numEvals } from "../logic/eval";

const totalTest = 10000;
// const bar = new ProgressBar(":bar", { total: totalTest });

async function runGame(game: Game) {
    await game.run();

    assert(game.done);
    assert(game.queueSystem.allDone());

    game.reset();
    const isTraining = Math.random() < 0.5;
    if (Math.random() < 0.5) {
        game = new Game({
            port: port,
            gameId: 0,
        });
        for (const playerIndex of [0, 1]) {
            if (Math.random() < 0.5) {
                game.handlers[playerIndex].sendFn = async (state) => {
                    const jobKey = game.queueSystem.createJob();
                    state.setKey(jobKey);
                    const action = getEvalAction(
                        game.handlers[playerIndex],
                        Math.floor(Math.random() * numEvals),
                    );
                    game.queueSystem.submitResult(jobKey, action);
                    return jobKey;
                };
            }
        }
    }
    return game;
}

function assertTrajectory(trajectory: State[]) {
    let prevTurn = undefined;
    for (const state of trajectory) {
        const stateObject = state.toObject();

        const currentTurn = stateObject?.info?.turn;
        assert(!!currentTurn >= !!prevTurn);
        prevTurn = currentTurn;
    }
    return;
}

async function main(verbose: boolean = false) {
    let game = new Game({
        port: port,
        gameId: 0,
    });

    let trajectory: State[] = [];
    let prevT = Date.now();
    let currT = Date.now();
    let tSum = 0;
    let n = 0;

    port.postMessage = (buffer) => {
        const state = State.deserializeBinary(buffer);
        const key = state.getKey();
        trajectory.push(state);

        const info = state.getInfo();
        const legalActions = state.getLegalactions();
        if (info && legalActions) {
            const done = info.getDone();

            if (!done) {
                const action = new Action();
                action.setKey(key);
                // const randomIndex = chooseRandom(legalActions);
                action.setIndex(-1);
                action.setText("default");

                game.queueSystem.submitResult(key, action);
            } else {
                const wasTie = game.world?.winner === "" || game.tied;
                if (verbose) {
                    console.log(game.world?.log.slice(-10));
                }
                const [r1, r2] = [
                    info.getPlayeronereward(),
                    info.getPlayertworeward(),
                ];
                if (wasTie) {
                    assert(r1 === 0);
                    assert(r2 === 0);
                } else {
                    assert(Math.abs(r1) === 1);
                    assert(Math.abs(r2) === 1);
                }
                assert(r1 + r2 === 0);
            }
            n += 1;
            currT = Date.now();
            const diff = currT - prevT;
            tSum += diff;
            prevT = currT;
        }
    };

    let rateWindow = [];
    for (let runIdx = 1; runIdx <= totalTest; runIdx++) {
        game = await runGame(game);
        const rate = (1000 * n) / tSum;
        rateWindow.push(rate);
        if (rateWindow.length > 16) {
            rateWindow.shift();
        }
        console.log(
            game.gameId,
            runIdx,
            rateWindow.reduce((a, b) => a + b) / rateWindow.length,
        );
        prevT = Date.now();
        currT = Date.now();
        tSum = 0;
        n = 0;
        assertTrajectory(trajectory);
        trajectory = [];
        // bar.tick();
        // console.log(runIdx);
    }
}

main(false);
