import * as fs from "fs/promises";
import * as path from "path";
import { StreamHandler } from "../logic/handler";

function getReward(data: {
    log: string;
    players: [string, string];
    [k: string]: any;
}) {
    const { log, players } = data;
    for (const line of [...log.split("\n")].reverse()) {
        if (line.startsWith("|win|")) {
            const winnerName = line.split("|")[2];
            if (winnerName === players[0]) {
                return [1, -1];
            } else if (winnerName === players[1]) {
                return [-1, 1];
            }
        } else if (line.startsWith("|tie|")) {
            return [0, 0];
        }
    }
    return [0, 0];
}

function processGame(data: any) {
    const states: Uint8Array[] = [];

    const [p1Reward, p2Reward] = getReward(data);

    const handlers = [0, 1].map((playerIndex) => {
        return new StreamHandler({
            gameId: 0,
            sendFn: async (state) => {
                const info = state.getInfo();
                if (info) {
                    info.setPlayeronereward(p1Reward);
                    info.setPlayertworeward(p2Reward);
                    state.setInfo(info);
                }
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
        const format = folderPath.split("/").at(-1) ?? "";
        const jsonFiles = files.filter(
            (file) => path.extname(file) === ".json" && file.startsWith(format),
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
    let stateLengths: number[] = [];

    for (const state of states) {
        if (currentChunkSize + state.length > chunkSize) {
            // Save current chunk to a file with metadata
            await saveChunk(currentChunk, stateLengths, chunkIndex, folderPath);
            chunkIndex++;
            currentChunk = [];
            stateLengths = [];
            currentChunkSize = 0;
        }

        currentChunk.push(state);
        stateLengths.push(state.length);
        currentChunkSize += state.length;
    }

    // Save the last chunk if it has any states
    if (currentChunk.length > 0) {
        await saveChunk(currentChunk, stateLengths, chunkIndex, folderPath);
    }
}

async function saveChunk(
    chunk: Uint8Array[],
    stateLengths: number[],
    index: number,
    folderPath: string,
) {
    const chunkBuffer = Buffer.concat(chunk);
    const metadataBuffer = Buffer.from(JSON.stringify(stateLengths));
    const filePath = path.join(folderPath, `states_chunk_${index}.bin`);
    const metadataPath = path.join(folderPath, `states_chunk_${index}.json`);

    await fs.writeFile(filePath, chunkBuffer);
    await fs.writeFile(metadataPath, metadataBuffer);
}

const CHUNK_SIZE = 100 * 1024 * 1024; // 100 MB in bytes

async function main() {
    const folderPath = "../replays/data/gen3randombattle";
    const replays = await readJsonFilesInFolder(folderPath);

    const states: Uint8Array[] = [];
    for (const log of replays) {
        console.log(log.id);
        states.push(...processGame(log));
    }

    await saveStatesInChunks(states, CHUNK_SIZE, folderPath);
}

main();
