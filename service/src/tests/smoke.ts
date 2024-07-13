import assert from "assert";

import { Action } from "../../protos/action_pb";
import { port } from "./utils";
import { State } from "../../protos/state_pb";
import { chooseRandom } from "../logic/utils";
import { Game } from "../server/game";

const totalTest = 10000;
// const bar = new ProgressBar(":bar", { total: totalTest });

async function runGame(game: Game) {
    await game.run();

    assert(game.done);
    assert(game.queues.p1.size() === 0);
    assert(game.queues.p2.size() === 0);

    game.reset();
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
    let game = new Game({ port: port, isTraining: true, gameId: 0 });

    let trajectory: State[] = [];
    let prevT = Date.now();
    let currT = Date.now();
    let tSum = 0;
    let n = 0;

    port.postMessage = (buffer) => {
        const state = State.deserializeBinary(buffer);
        trajectory.push(state);

        const info = state.getInfo();
        const legalActions = state.getLegalactions();
        if (info && legalActions) {
            const done = info.getDone();

            if (!done) {
                const playerIndex = info.getPlayerindex();
                const action = new Action();
                action.setPlayerindex(playerIndex);
                const randomIndex = chooseRandom(legalActions);
                action.setIndex(randomIndex);
                game.queues[`p${playerIndex ? 2 : 1}`].enqueue(action);
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
