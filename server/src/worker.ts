import { MessagePort, parentPort, workerData } from "worker_threads";

import { Game } from "./game";
import { Action } from "../protos/action_pb";

const isTraining = workerData.socketType === "training";

const game = new Game({
    gameId: workerData.workerCount,
    isTraining,
    port: parentPort as unknown as MessagePort,
});

async function main() {
    while (true) {
        await game.run();
        game.reset();
    }
}

main();

// Allocate a rolling buffer to store incoming messages
let buffer = Buffer.alloc(0);

parentPort?.on("message", (data: Buffer) => {
    // Update rolling buffer with incoming information
    buffer = Buffer.concat([buffer, data]);

    while (buffer.length >= 4) {
        // Read the message length
        const messageLength = buffer.readUInt32BE(0);

        // Check if the full message has been received
        if (buffer.length < 4 + messageLength) {
            break;
        }

        // Extract the full message
        const message = buffer.subarray(4, 4 + messageLength);
        buffer = buffer.subarray(4 + messageLength); // Remove the processed message from the buffer

        const action = Action.deserializeBinary(new Uint8Array(message));
        const playerId = action.getPlayerindex() ? "p2" : "p1";
        game.queues[playerId].enqueue(action);
    }
});
