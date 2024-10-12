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
                return 1;
            } else if (winnerName === players[1]) {
                return -1;
            }
        } else if (line.startsWith("|tie|")) {
            return 0;
        }
    }
    return 0;
}

function stringToHash(str: string): number {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
        const char = str.charCodeAt(i);
        hash = (hash << 5) - hash + char;
        hash |= 0; // Convert to a 32-bit integer
    }
    return hash;
}

async function processGame(data: any) {
    const states = [];

    const reward = getReward(data);

    const handlers = [0, 1].map((playerIndex) => {
        return new StreamHandler({
            gameId: 0,
            sendFn: async (state) => {
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

    const finalStates = handlers.map(async (handler) => {
        const state = await handler.getState(512);
        const info = state.getInfo();
        if (info) {
            info.setWinreward(reward);
            info.setGameid(stringToHash(data["id"]));
            state.setInfo(info);
        }
        states.push(state);
        return state.serializeBinary();
    });

    return finalStates;
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

function flattenUint8ArrayArray(arrays: Uint8Array[]): Uint8Array {
    const lengths = arrays.map((arr) => arr.length); // Get the lengths of each Uint8Array
    const totalLength =
        lengths.reduce((acc, len) => acc + len, 0) + arrays.length * 4; // For lengths metadata

    // Create a buffer large enough to store all the arrays and their lengths
    const result = new Uint8Array(totalLength);
    const dataView = new DataView(result.buffer); // Use DataView to store lengths as 32-bit integers

    let offset = 0;
    arrays.forEach((arr, i) => {
        // Write the length of each Uint8Array (4 bytes)
        dataView.setUint32(offset, arr.length);
        offset += 4;

        // Write the Uint8Array data
        result.set(arr, offset);
        offset += arr.length;
    });

    return result;
}

async function main() {
    const folderPath = "../replays/data/gen3randombattle";
    const replays = await readJsonFilesInFolder(folderPath);

    const statesPromises: Promise<Uint8Array>[] = [];
    for (const log of replays) {
        console.log(log.id);
        const promises = await processGame(log);
        statesPromises.push(...promises);
    }

    const states = await Promise.all(statesPromises);
    const binary = flattenUint8ArrayArray(states);

    await fs.writeFile(path.join(folderPath, "states.bin"), binary);
}

main();
