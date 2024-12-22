import { ObjectReadWriteStream } from "@pkmn/streams";
import { Player } from "../server/player";
import { readFileSync, readdirSync, writeFileSync } from "fs";
import { PokemonIdent } from "@pkmn/protocol";
import { IndexValueFromEnum } from "../server/state";
import { Dataset, Trajectory } from "../../protos/state_pb";
import { join } from "path";

class OfflineStream extends ObjectReadWriteStream<string> {
    constructor() {
        super();
    }

    _write(message: string) {
        this.push(message);
    }
}

function getNextSwitch(chunk: string, playerIndex: number) {
    for (const line of chunk.split("\n")) {
        if (line.startsWith(`|switch|p${playerIndex + 1}`)) {
            const ident = line
                .split("|")[2]
                .split(`p${playerIndex + 1}a: `)
                .at(-1) as PokemonIdent;
            return ident;
        }
    }
    throw new Error("Nothing Found");
}

async function processReplay(filePath: string) {
    const replay = JSON.parse(readFileSync(filePath).toString());
    console.log(`Processing file: ${filePath}`);

    const inputLog: string[] = replay.inputlog.split("\n");

    if (replay.inputlog === null) {
        throw new Error("no input log");
    }

    const batches = [];

    for (const i of [0, 1]) {
        const chunks: string[] = [];

        let currentChunk = "";
        for (const line of replay.log.split("\n")) {
            currentChunk += `${line}\n`;
            if (
                line.startsWith("|turn|") ||
                line.startsWith(`|faint|p${i + 1}`) ||
                (line.startsWith(`|move|p${i + 1}`) &&
                    line.split("|")[3] === "Baton Pass")
            ) {
                chunks.push(currentChunk);
                currentChunk = "";
            }
        }
        chunks.push(currentChunk);

        const playerInputLog = inputLog.filter((x) =>
            x.startsWith(`>p${i + 1}`),
        );

        const stream = new OfflineStream();
        for (const chunk of chunks) {
            stream.write(chunk);
        }
        stream.pushEnd();

        let count = 1;

        const batch = new Trajectory();

        async function send(player: Player) {
            const nextInput = playerInputLog[count - 1];
            if (nextInput !== undefined) {
                const nextChunk = chunks[count];
                const state = player.createState();
                if (nextInput.split(" ")[1] === "switch") {
                    try {
                        const nextSwitchIdent = getNextSwitch(nextChunk, i);
                        // console.log(nextSwitchIdent);
                        const label = IndexValueFromEnum(
                            "Actions",
                            `switch_${nextSwitchIdent}`,
                        );
                        // console.log(label);
                        batch.addActions(label, count);
                        batch.addStates(state, count);
                        // eslint-disable-next-line @typescript-eslint/no-unused-vars
                    } catch (err) {
                        /* empty */
                    }
                } else {
                    const label = IndexValueFromEnum(
                        "Actions",
                        `move_${nextInput.split(" ").at(-1)}`,
                    );
                    // console.log(label);
                    batch.addActions(label, count);
                    batch.addStates(state, count);
                }
                count += 1;
            }
            return -1;
        }

        async function recv() {
            return undefined;
        }

        const player = new Player(
            0,
            0,
            stream,
            0,
            send,
            recv,
            null,
            undefined,
            true,
            i,
        );
        await player.start();

        const reward = replay.log.includes(
            `|win|${player.privateBattle.sides[i].name}`,
        )
            ? 1
            : replay.log.includes(
                  `|win|${player.privateBattle.sides[1 - i].name}`,
              )
            ? -1
            : 0;
        for (const [index] of batch.getStatesList().entries()) {
            batch.addRewards(reward, index);
        }

        batches.push(batch);
    }
    return batches;
}
async function main() {
    const directoryPath = "../replays/data/gen3randombattle";
    const files = readdirSync(directoryPath);
    let dataset = new Dataset();
    let currSampleCount = 0;
    let sampleCount = 0;
    let datasetCount = 0;

    for (const [index, file] of files.entries()) {
        if (file.endsWith(".json")) {
            const filePath = join(directoryPath, file);
            try {
                const batches = await processReplay(filePath);
                for (const batch of batches) {
                    dataset.addTrajectories(batch, sampleCount);
                    const numActions = batch.getActionsList().length;
                    sampleCount += numActions;
                    currSampleCount += numActions;

                    // Save every 50_000 items and then clear the dataset
                    if (currSampleCount > 50_000) {
                        const datasetBinary = dataset.serializeBinary();
                        const outputFilePath = join(
                            directoryPath,
                            `dataset_binary_part_${datasetCount}.dat`,
                        );
                        writeFileSync(outputFilePath, datasetBinary);
                        console.log(
                            `Binary data saved at ${outputFilePath} (count: ${sampleCount})`,
                        );

                        // Clear the dataset for next batch of data
                        dataset = new Dataset();
                        datasetCount += 1;
                        currSampleCount = 0;
                    }
                }
            } catch (err) {
                console.log(err);
            }
        }
        console.log(sampleCount);
        console.log(index / files.length);
    }

    // After finishing all files, if there's remaining data that wasn't saved
    // (less than a multiple of 1000), save it and clear the dataset.
    const finalDatasetBinary = dataset.serializeBinary();
    const finalOutputFilePath = join(
        directoryPath,
        `dataset_binary_part_${datasetCount}.dat`,
    );
    writeFileSync(finalOutputFilePath, finalDatasetBinary);
    console.log(`Final binary data has been saved to ${finalOutputFilePath}`);

    // Clear the dataset after saving
    dataset = new Dataset();
}

main().catch((error) => {
    console.error("An error occurred:", error);
});
