import { MessagePort } from "worker_threads";
import { Game } from "../server/game";
import { Action, GameState } from "../../protos/servicev2_pb";
import { AsyncQueue } from "../server/utils";
import { State } from "../../protos/state_pb";
// import { State } from "../../protos/state_pb";

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

    let infos = [];
    let rewardsCounts = [0, 0];
    let maxMajorEdges = 0;
    let maxMinorEdges = 0;

    while (true) {
        const gameState = await queue.get();
        const rqid = gameState.getRqid();

        const state = State.deserializeBinary(gameState.getState_asU8());
        const info = state.getInfo()!;
        infos.push(info);

        if (rqid >= 0) {
            const action = new Action();
            action.setValue(-1);
            game.tasks.submitResult(rqid, action);
        } else {
            // eslint-disable-next-line @typescript-eslint/no-unused-vars
            const rewards = info.getRewards()!;
            infos.map((info) => {
                const rewards = info.getRewards()!;
                const playerIndex = info.getPlayerindex();
                rewardsCounts[+playerIndex] += rewards.getFaintedreward();
            });

            const player = (game.players ?? [])[0];
            if (player !== undefined) {
                const numMajorEdges =
                    player.eventHandler.majorEdgeBuffer.numEdges;
                const numMinorEdges =
                    player.eventHandler.minorEdgeBuffer.numEdges;

                maxMajorEdges = Math.max(maxMajorEdges, numMajorEdges);
                maxMinorEdges = Math.max(maxMinorEdges, numMinorEdges);

                console.log(maxMajorEdges);
                console.log(maxMinorEdges);
                console.log("");
            }

            game.reset();
            infos = [];
            rewardsCounts = [0, 0];
        }
    }
}

async function main() {
    for (const { gameId, playerIds } of [
        { gameId: 0, playerIds: [0, 1] },
        // { gameId: 1, playerIds: [2, 3] },
        // { gameId: 10000, playerIds: [10000] },
        // { gameId: 10001, playerIds: [10001] },
        // { gameId: 10002, playerIds: [10002] },
        // { gameId: 10003, playerIds: [10003] },
    ]) {
        await worker(gameId, playerIds);
    }
}

main();
