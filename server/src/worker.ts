import { MessagePort, parentPort, workerData } from "worker_threads";
import { Game } from "./game";
import { Action } from "../protos/action_pb";
import { writeFileSync } from "fs";

const game = new Game({
    gameId: workerData.workerCount,
    port: parentPort as unknown as MessagePort,
});

async function main() {
    while (true) {
        await game.run();

        // try {
        //     await runGameWithTimeout(game, 1000); // Wait at most 1 second
        // } catch (error) {
        //     const world = game.world;
        //     if (world) {
        //         writeFileSync(
        //             `./logs/${game.gameId}.json`,
        //             JSON.stringify({
        //                 log: world.log,
        //                 inputLog: world.inputLog,
        //             })
        //         );
        //     }
        //     break; // Exit the loop if a timeout occurs
        // }

        game.reset();
    }
}

main();

let buffer = Buffer.alloc(0);

parentPort?.on("message", (data: Buffer) => {
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
