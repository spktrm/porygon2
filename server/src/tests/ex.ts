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
        writeFileSync("../rlenv/ex", buffer);
        exit(0);
    };

    game = await runGame(game);
}

main(false);
