import assert from "assert";

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
    const accum = {
        winRewardSum: 0,
        hpRewardSum: 0,
        faintedRewardSum: 0,
        switchRewardSum: 0,
        longevityRewardSum: 0,
    };
    trajectory.map((state) => {
        const info = state.toObject().info!;
        accum.winRewardSum += info.winreward;
        accum.hpRewardSum += info.hpreward;
        accum.faintedRewardSum += info.faintedreward;
        accum.switchRewardSum += info.switchreward;
        accum.longevityRewardSum += info.longevityreward;
    });

    trajectory.reduce((prev, curr) => {
        if (prev.toObject().info!.ts > curr.toObject().info!.ts) {
            throw new Error();
        }
        return curr;
    });

    const winner = game.getWinner();

    if (winner === game.world?.p1.name) {
        if (accum.winRewardSum !== 1) {
            throw new Error();
        }
    } else if (winner === game.world?.p2.name) {
        if (accum.winRewardSum !== -1) {
            throw new Error();
        }
    } else {
        if (game.earlyFinish) {
            if (accum.winRewardSum === 0) {
                throw new Error();
            }
        }
    }

    if (accum.faintedRewardSum < -6 || 6 < accum.faintedRewardSum) {
        throw new Error();
    }

    const rtol = 1e-3;
    if (accum.hpRewardSum + rtol < -6 || 6 < accum.hpRewardSum - rtol) {
        throw new Error();
    }

    if (accum.faintedRewardSum < -6 || 6 < accum.faintedRewardSum) {
        throw new Error();
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
                const evalIndex = Math.floor(Math.random() * 2);
                const action = await getEvalAction(
                    game.handlers[+playerIndex],
                    evalIndex,
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
