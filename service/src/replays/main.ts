import * as fs from "fs/promises";
import * as path from "path";
import { StreamHandler } from "../logic/handler";

function processGame(data: any) {
    const states: Uint8Array[] = [];

    const handlers = [0, 1].map((playerIndex) => {
        return new StreamHandler({
            gameId: 0,
            sendFn: async (state) => {
                states.push(state.serializeBinary());
                return "";
            },
            recvFn: async () => {
                return undefined;
            },
            playerIndex: playerIndex as 0 | 1,
        });
    });

    const chunks = data.log.split("\n|\n");
    for (const chunk of chunks) {
        for (const handler of handlers) {
            handler.ingestChunk(chunk);
        }
    }

    return states;
}

async function readJsonFilesInFolder(folderPath: string): Promise<any[]> {
    try {
        const files = await fs.readdir(folderPath);
        const jsonFiles = files.filter(
            (file) => path.extname(file) === ".json",
        );

        const readPromises = jsonFiles.map(async (file) => {
            const filePath = path.join(folderPath, file);
            const fileContent = await fs.readFile(filePath, "utf-8");
            return JSON.parse(fileContent);
        });

        return await Promise.all(readPromises);
    } catch (error) {
        console.error("Error reading JSON files:", error);
        throw error;
    }
}
async function saveStatesInChunks(
    states: Uint8Array[],
    chunkSize: number,
    folderPath: string,
) {
    let currentChunkSize = 0;
    let currentChunk: Uint8Array[] = [];
    let chunkIndex = 0;

    for (const state of states) {
        if (currentChunkSize + state.length > chunkSize) {
            // Save current chunk to a file
            await saveChunk(currentChunk, chunkIndex, folderPath);
            chunkIndex++;
            currentChunk = [];
            currentChunkSize = 0;
        }

        currentChunk.push(state);
        currentChunkSize += state.length;
    }

    // Save the last chunk if it has any states
    if (currentChunk.length > 0) {
        await saveChunk(currentChunk, chunkIndex, folderPath);
    }
}

async function saveChunk(
    chunk: Uint8Array[],
    index: number,
    folderPath: string,
) {
    const chunkBuffer = Buffer.concat(chunk);
    const filePath = path.join(folderPath, `states_chunk_${index}.bin`);
    await fs.writeFile(filePath, chunkBuffer);
}

const CHUNK_SIZE = 100 * 1024 * 1024; // 100 MB in bytes

async function main() {
    const folderPath = "../replays/data/gen3randombattle";
    const replays = await readJsonFilesInFolder(folderPath);

    const states: Uint8Array[] = [];
    for (const log of replays) {
        states.push(...processGame(log));
    }

    await saveStatesInChunks(states, CHUNK_SIZE, folderPath);
}

main();
