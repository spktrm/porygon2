/**
 * Single-replay exporter — encodes ONE replay JSON through the same state
 * encoder as the offline shard exporter and writes one shard-format record:
 *
 *     [uint32-LE length][EnvironmentBatch proto bytes]
 *
 * so rl/offline/dataset.py's record parser consumes it unchanged. Used by
 * the potential-function visualizer (rl/offline/visualize.py) and handy for
 * debugging individual replays.
 *
 * Usage (from service/, after tsc):
 *   node dist/scripts/exportReplay.js <replay.json> <out.bin>
 *
 * Must run with CWD=service/ (data.ts loads ../constants and ../data
 * relative to the working directory). Prints a JSON stats line to stdout on
 * success, e.g. {"perspectives":[0,1],"states":[42,42]}.
 */

import * as fs from "fs";

import { EnvironmentBatch } from "../../protos/service_pb";
import { encodePerspective, ReplayFile } from "./offlineWorker";

function main() {
    const [inPath, outPath] = process.argv.slice(2);
    if (!inPath || !outPath) {
        console.error(
            "usage: node dist/scripts/exportReplay.js <replay.json> <out.bin>",
        );
        process.exit(1);
    }

    const replay: ReplayFile = JSON.parse(fs.readFileSync(inPath, "utf-8"));
    const lines = replay.log.split("\n");

    const batch = new EnvironmentBatch();
    const perspectives: number[] = [];
    const stateCounts: number[] = [];
    let maxLength = 0;
    for (const playerIndex of [0, 1] as const) {
        const encoded = encodePerspective(replay, lines, playerIndex);
        if (encoded !== null) {
            batch.addTrajectories(encoded.trajectory);
            perspectives.push(playerIndex);
            stateCounts.push(encoded.states);
            maxLength = Math.max(maxLength, encoded.states);
        }
    }
    if (perspectives.length === 0) {
        console.error(
            "replay has no decided outcome (no |win|/|tie| line) — nothing to encode",
        );
        process.exit(2);
    }
    batch.setMaxTrajectoryLength(maxLength);

    const record = batch.serializeBinary();
    const prefix = Buffer.alloc(4);
    prefix.writeUInt32LE(record.length, 0);
    fs.writeFileSync(outPath, Buffer.concat([prefix, Buffer.from(record)]));
    console.log(JSON.stringify({ perspectives, states: stateCounts }));
}

main();