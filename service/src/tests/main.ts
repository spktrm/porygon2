import { MessagePort } from "worker_threads";
import { Game } from "../server/game";
import { Action, GameState } from "../../protos/servicev2_pb";
import { AsyncQueue } from "../server/utils";
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
        while (true) {
            const gameState = await queue.get();
            const rqid = gameState.getRqid();
            if (rqid >= 0) {
                const action = new Action();
                action.setValue(-1);
                game.tasks.submitResult(rqid, action);
            } else {
                const state = State.deserializeBinary(
                    gameState.getState_asU8(),
                );
                const info = state.getInfo()!;
                const reward = info.getWinreward();
                game.reset();
            }
        }
    })();
}

function main() {
    for (const { gameId, playerIds } of [
        // { gameId: 0, playerIds: [0, 1] },
        // { gameId: 1, playerIds: [2, 3] },
        // { gameId: 10000, playerIds: [10000] },
        // { gameId: 10001, playerIds: [10001] },
        // { gameId: 10002, playerIds: [10002] },
        { gameId: 10003, playerIds: [10003] },
    ]) {
        worker(gameId, playerIds);
    }
}

main();
