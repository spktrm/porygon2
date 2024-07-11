import path from "path";

import { Socket, createServer } from "net";
import { Worker } from "worker_threads";
import { existsSync, unlinkSync } from "fs";
import { socketPath } from "./data";

// Clean up any previous socket file
if (existsSync(socketPath)) {
    unlinkSync(socketPath);
}

let workerCount = 0;
const workers: Worker[] = [];

function allocateWorker(ws: Socket) {
    const worker = new Worker(path.resolve(__dirname, "../dist/worker.js"), {
        workerData: {
            workerCount,
        },
    });

    worker.on("message", (state: Uint8Array) => {
        // Calculate the size of the state array
        const stateSize = state.length;

        // Create a buffer for the size
        const sizeBuffer = Buffer.alloc(4); // 4 bytes for a 32-bit integer
        sizeBuffer.writeUInt32LE(stateSize, 0); // Write the size as a big-endian 32-bit integer

        ws.write(sizeBuffer);
        ws.write(state);
    });

    workers.push(worker);
    const currWorkerIndex = workerCount;
    workerCount++;
    return currWorkerIndex;
}

const server = createServer((ws) => {
    // if (workerCount >= maxNumWorkers) {
    //     throw new Error("Maximum number of workers reached");
    // }
    const workerIndex = allocateWorker(ws);

    ws.on("data", (data: Buffer) => {
        const worker = workers[workerIndex];
        worker.postMessage(data, [data.buffer]);
    });

    ws.on("error", (error) => {
        console.error("Socket error:", error);
    });
});

server.listen(socketPath, () => {
    console.log(`WebSocket server is running on ${socketPath}`);
});
