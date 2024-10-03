import assert from "assert";

import { Action } from "../../protos/action_pb";
import { port } from "./utils";
import { State } from "../../protos/state_pb";
import { chooseRandom } from "../logic/utils";
import { writeFileSync } from "fs";
import { exit } from "process";
import { Game } from "../server/game";

async function runGame(game: Game) {
    await game.run();

    assert(game.done);

    game.reset();
    return game;
}

async function main(verbose: boolean = false) {
    let game = new Game({ port: port, gameId: 0 });

    port.postMessage = (buffer) => {
        const state = State.deserializeBinary(buffer);
        const key = state.getKey();
        const info = state.getInfo()!;
        const turn = info.getTurn();
        const playerIndex = info.getPlayerindex();
        const legalActions = state.getLegalactions();

        if (info && legalActions) {
            if (turn > 10 && playerIndex === true) {
                writeFileSync("../rlenv/ex", buffer);
                exit(0);
            } else {
                const playerIndex = info.getPlayerindex();
                const action = new Action();
                action.setKey(key);
                const randomIndex = chooseRandom(legalActions);
                action.setIndex(randomIndex);
                game.queueSystem.submitResult(key, action);
            }
        }
    };

    game = await runGame(game);
}

main(false);
