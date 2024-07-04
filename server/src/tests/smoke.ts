import assert from "assert";

import { Action } from "../../protos/action_pb";
import { Game } from "../game";
import { port } from "./utils";
import { State } from "../../protos/state_pb";
import { chooseRandom } from "../utils";

async function runGame(game: Game) {
    await game.run();

    assert(game.dones === 2);
    assert(game.queues.p1.size() === 0);
    assert(game.queues.p2.size() === 0);

    game.reset();
    return game;
}

async function main() {
    let game = new Game({ port: port, gameId: 0 });
    port.postMessage = (buffer) => {
        const state = State.deserializeBinary(buffer);
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
                const wasTie = game.world?.winner === "";
                console.log(game.world?.log.slice(-10));
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
        }
    };
    for (let runIdx = 0; runIdx < 1000; runIdx++) {
        game = await runGame(game);
    }
}

main();
