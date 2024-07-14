import path from "path";

import { Socket, createServer } from "net";
import { Worker } from "worker_threads";
import { existsSync, unlinkSync } from "fs";
import { Command } from "commander";

const program = new Command();

program.option("-t, --type <string>", "server type");
program.parse(process.argv);

interface SocketServerArgs {
    type: "trainig" | "evaluation";
}

const options = program.opts<SocketServerArgs>();
const socketType = options.type;
const socketPath = `/tmp/pokemon-${socketType}.sock`;

// Clean up any previous socket file
if (existsSync(socketPath)) {
    unlinkSync(socketPath);
}

let workerCount = 0;
const workers: Worker[] = [];
const workerClients: Map<number, Socket> = new Map();

function allocateWorker(ws: Socket) {
    const workerPath = path.resolve(__dirname, "../server/worker.js");
    const worker = new Worker(workerPath, {
        workerData: {
            workerCount,
            socketType,
        },
    });

    const currWorkerIndex = workerCount;
    workerClients.set(workerCount, ws);
    workerCount++;

    worker.on("message", (state: Uint8Array) => {
        // Calculate the size of the state array
        const stateSize = state.length;

        // Create a buffer for the size
        const sizeBuffer = Buffer.alloc(4); // 4 bytes for a 32-bit integer
        sizeBuffer.writeUInt32LE(stateSize, 0); // Write the size as a little-endian 32-bit integer

        if (ws.writable) {
            ws.write(sizeBuffer);
            ws.write(state);
        }
    });

    worker.on("error", (error) => {
        console.error(`${socketType} worker ${currWorkerIndex} error:`, error);
        // Handle worker error (e.g., clean up resources, restart worker, notify client, etc.)
        if (workerClients.has(workerCount)) {
            const client = workerClients.get(workerCount);
            if (client) {
                client.end(
                    `${socketType} worker ${currWorkerIndex} encountered an error.`,
                );
                workerClients.delete(workerCount);
            }
        }
        worker.terminate();
    });

    worker.on("exit", (code) => {
        if (code !== 0) {
            console.error(
                `${socketType} worker ${currWorkerIndex} stopped with exit code ${code}`,
            );
        }
        // Cleanup and potentially restart the worker
        if (workerClients.has(workerCount)) {
            const client = workerClients.get(workerCount);
            if (client) {
                client.end(
                    `${socketType} worker ${currWorkerIndex} has exited.`,
                );
                workerClients.delete(workerCount);
            }
        }
    });

    workers.push(worker);

    return currWorkerIndex;
}

const server = createServer((ws) => {
    const workerIndex = allocateWorker(ws);
    console.log(`${socketType} client ${workerIndex} has connected`);

    ws.on("data", (data: Buffer) => {
        const worker = workers[workerIndex];
        worker.postMessage(data, [data.buffer]);
    });

    ws.on("error", (error) => {
        // console.error("Socket error:", error);
    });

    ws.on("close", () => {
        console.log(`${socketType} client ${workerIndex} disconnected`);
        // Cleanup worker-client association
        workerClients.delete(workerIndex);
        workers[workerIndex].terminate();
    });
});

server.listen(socketPath, () => {
    console.log(`WebSocket server is running on ${socketPath}`);
});
