import { MessagePort } from "worker_threads";
import { Game } from "../server/game";
import { Action, GameState } from "../../protos/servicev2_pb";
import { AsyncQueue } from "../server/utils";
import { writeFileSync } from "fs";
import { exit } from "process";
import { State } from "../../protos/state_pb";

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

    (async () => {
        let i = 0;
        while (true) {
            const gameState = await queue.get();
            const rqid = gameState.getRqid();
            if (rqid >= 0) {
                const action = new Action();
                action.setValue(-1);
                game.tasks.submitResult(rqid, action);

                if (i >= 5) {
                    writeFileSync("../rlenvv2/ex", gameState.getState_asU8());
                    exit(0);
                } else {
                    i += 1;
                }
            } else {
                game.reset();
            }
        }
    })();
}

function main() {
    for (const { gameId, playerIds } of [{ gameId: 0, playerIds: [0, 1] }]) {
        worker(gameId, playerIds);
    }
}

main();
