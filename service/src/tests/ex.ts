import { MessagePort } from "worker_threads";
import { Game } from "../server/game";
import { Action, GameState } from "../../protos/servicev2_pb";
import { AsyncQueue } from "../server/utils";
import { writeFileSync } from "fs";
import { exit } from "process";

async function worker(gameId: number, playerIds: number[]) {
    const queue = new AsyncQueue<GameState>();

    const port = {
        postMessage: (stateBuffer: Buffer) => {
            const gameState = GameState.deserializeBinary(stateBuffer);
            queue.put(gameState);
        },
    } as MessagePort;

    const game = new Game(gameId, 0, port);

    for (const playerId of playerIds) {
        game.addPlayerId(playerId);
    }

    game.reset();
    game.reset();

    while (true) {
        const gameState = await queue.get();
        const rqid = gameState.getRqid();
        if (rqid >= 0) {
            const action = new Action();
            action.setValue(-1);
            game.tasks.submitResult(rqid, action);
        } else {
            writeFileSync("../rlenv/ex", gameState.getState_asU8());
            game.reset();
            exit(0);
        }
    }
}

async function main() {
    await worker(0, [0, 1]);
}

main();
