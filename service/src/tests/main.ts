import { MessagePort } from "worker_threads";
import { Game } from "../server/game";
import { Action, GameState } from "../../protos/service_pb";
import { AsyncQueue, OneDBoolean } from "../server/utils";
import { Rewards, State } from "../../protos/state_pb";
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

    while (true) {
        const gameState = await queue.get();
        const rqid = gameState.getRqid();

        const state = State.deserializeBinary(gameState.getState_asU8());
        const info = state.getInfo()!;
        infos.push(info);

        if (rqid >= 0) {
            const legalActions = state.getLegalActions_asU8();
            const vector = new OneDBoolean(10);
            vector.setBuffer(legalActions);
            const binaryArray = vector.toBinaryVector();

            const availableIndices = binaryArray
                .map((value, index) => (value === 1 ? index : -1))
                .filter((index) => index !== -1);
            const randomIndex =
                availableIndices[
                    Math.floor(Math.random() * availableIndices.length)
                ];
            const randomAction = randomIndex !== undefined ? randomIndex : -1;

            const action = new Action();
            action.setValue(randomAction);
            game.tasks.submitResult(rqid, action);
        } else {
            // eslint-disable-next-line @typescript-eslint/no-unused-vars
            const rewards = info.getRewards() ?? new Rewards();
            infos.map((info) => {
                const rewards = info.getRewards() ?? new Rewards();
                const playerIndex = info.getPlayerIndex();
                rewardsCounts[+playerIndex] += rewards.getFaintedReward();
            });

            const player = (game.players ?? [])[0];
            if (player !== undefined) {
                const numMajorEdges = player.eventHandler.edgeBuffer.numEdges;

                maxMajorEdges = Math.max(maxMajorEdges, numMajorEdges);

                console.log(maxMajorEdges);
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
