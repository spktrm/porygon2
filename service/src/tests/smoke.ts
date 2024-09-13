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
    return game;
}

function assertTrajectory(game: Game, trajectory: State[]) {
    let prevTurn = 0;

    let winRewardSum = 0;
    let hpRewardSum = 0;

    for (const state of trajectory) {
        const stateObject = state.toObject();

        const currentTurn = stateObject?.info?.turn ?? 0;
        assert(currentTurn >= prevTurn);
        prevTurn = currentTurn;

        winRewardSum += stateObject.info?.winreward ?? 0;
        hpRewardSum += stateObject.info?.hpreward ?? 0;
    }

    const winner = game.getWinner();
    if (winner === game.world?.p1.name) {
        assert(winRewardSum === 1);
    } else if (winner === game.world?.p2.name) {
        assert(winRewardSum === -1);
    } else {
        if (game.earlyFinish) {
            assert(winRewardSum === 0);
        }
    }

    if (winRewardSum !== 0) {
        assert(Math.sign(hpRewardSum) === Math.sign(winRewardSum));
    }

    if (trajectory.length < 5) {
        throw Error();
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

    port.postMessage = async (buffer) => {
        const state = State.deserializeBinary(buffer);
        const key = state.getKey();
        trajectory.push(state);

        const info = state.getInfo();
        const legalActions = state.getLegalactions();

        if (info && legalActions) {
            const playerIndex = info.getPlayerindex();
            const done = info.getDone();

            if (!done) {
                const action = await getEvalAction(
                    game.handlers[+playerIndex],
                    Math.floor(Math.random() * numEvals),
                );

                game.queueSystem.submitResult(key, action);
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
        assertTrajectory(game, trajectory);
        trajectory = [];
        game.reset();
    }
}

main(false);
