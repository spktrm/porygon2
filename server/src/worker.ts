import { MessagePort, parentPort, workerData } from "worker_threads";
import { Game } from "./game";
import { Action } from "../protos/action_pb";

const game = new Game({
    gameId: workerData.workerCount,
    port: parentPort as unknown as MessagePort,
});

async function main() {
    while (true) {
        await game.run();
        game.reset();
    }
}

main();

parentPort?.on("message", (data: Buffer) => {
    const action = Action.deserializeBinary(new Uint8Array(data));
    const playerId = action.getPlayerindex() ? "p2" : "p1";
    game.queues[playerId].enqueue(action);
});
