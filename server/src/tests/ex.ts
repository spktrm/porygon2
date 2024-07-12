import assert from "assert";

import { Action } from "../../protos/action_pb";
import { Game } from "../game";
import { port } from "./utils";
import { State } from "../../protos/state_pb";
import { chooseRandom } from "../utils";
import { writeFileSync } from "fs";
import { exit } from "process";

async function runGame(game: Game) {
    await game.run();

    assert(game.done);
    assert(game.queues.p1.size() === 0);
    assert(game.queues.p2.size() === 0);

    game.reset();
    return game;
}

async function main(verbose: boolean = false) {
    let game = new Game({ port: port, gameId: 0 });

    port.postMessage = (buffer) => {
        const state = State.deserializeBinary(buffer);
        const info = state.getInfo();
        const turn = info?.getTurn() ?? 0;
        const legalActions = state.getLegalactions();

        if (info && legalActions) {
            if (turn > 10) {
                writeFileSync("../rlenv/ex", buffer);
                exit(0);
            } else {
                const playerIndex = info.getPlayerindex();
                const action = new Action();
                action.setPlayerindex(playerIndex);
                const randomIndex = chooseRandom(legalActions);
                action.setIndex(randomIndex);
                game.queues[`p${playerIndex ? 2 : 1}`].enqueue(action);
            }
        }
    };

    game = await runGame(game);
}

main(false);
