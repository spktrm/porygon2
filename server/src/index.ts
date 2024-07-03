import path from "path";
import { Socket, createServer } from "net";
import { Worker } from "worker_threads";
import { existsSync, unlinkSync } from "fs";
import { AsyncQueue } from "./utils";
import { cpus } from "os";

const socketPath = "/tmp/pokemon.sock";

// Clean up any previous socket file
if (existsSync(socketPath)) {
    unlinkSync(socketPath);
}

class WorkerContainer {
    worker: Worker;
    taskQueue: AsyncQueue<{ size: Buffer; msg: Uint8Array }>;
    conn: Socket;

    constructor(worker: Worker, conn: Socket) {
        this.worker = worker;
        this.taskQueue = new AsyncQueue();
        this.conn = conn;
    }

    async run() {
        while (true) {
            const el = await this.taskQueue.dequeue();
            const { size, msg } = el;
            try {
                this.conn.write(size);
                this.conn.write(msg);
            } catch (error) {
                console.error("Error writing to connection:", error);
                // Handle error as appropriate, possibly break the loop or re-queue the task
            }
        }
    }

    // Method to add tasks to the queue
    addTask(task: { size: Buffer; msg: Uint8Array }) {
        this.taskQueue.enqueue(task);
    }
}

let workerCount = 0;
const maxNumWorkers = 12;
const containers: WorkerContainer[] = [];
const workerToClient: Map<WorkerContainer, Socket> = new Map();
const clientToWorker: Map<Socket, WorkerContainer> = new Map();

function allocateWorker(ws: Socket) {
    const worker = new Worker(path.resolve(__dirname, "../dist/worker.js"), {
        workerData: {
            workerCount,
        },
    });
    const container = new WorkerContainer(worker, ws);
    container.run();
    worker.on("message", (state: Uint8Array) => {
        // Calculate the size of the state array
        const stateSize = state.length;

        // Create a buffer for the size
        const sizeBuffer = Buffer.alloc(4); // 4 bytes for a 32-bit integer
        sizeBuffer.writeUInt32LE(stateSize, 0); // Write the size as a big-endian 32-bit integer

        container.taskQueue.enqueue({ size: sizeBuffer, msg: state });
    });

    containers.push(container);
    workerToClient.set(container, ws);
    clientToWorker.set(ws, container);
    const currWorkerIndex = workerCount;
    workerCount++;
    return currWorkerIndex;
}

function deallocateWorker(ws: Socket) {
    const container = clientToWorker.get(ws);
    if (container) {
        container.worker.terminate();
        containers.splice(containers.indexOf(container), 1);
        workerToClient.delete(container);
        clientToWorker.delete(ws);
        workerCount--;
    }
}

const server = createServer((ws) => {
    if (workerCount >= maxNumWorkers) {
        throw new Error("Maximum number of workers reached");
    }
    const workerIndex = allocateWorker(ws);
    let buffer = Buffer.alloc(0);

    ws.on("data", (data: Buffer) => {
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

            // Find the appropriate container
            const container = containers[workerIndex];

            // Post the message to the worker
            container.worker.postMessage(message);
        }
    });
    ws.on("close", () => {
        deallocateWorker(ws);
    });
    ws.on("error", (error) => {
        console.error("Socket error:", error);
    });
});

server.listen(socketPath, () => {
    console.log(`WebSocket server is running on ${socketPath}`);
});
