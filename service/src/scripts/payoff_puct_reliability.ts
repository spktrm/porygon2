/** Replay fixed public roots through the actual PUCT estimator; no actor or live state. */
import fs from "fs";
import path from "path";
import { createHash } from "crypto";
import { PRNG, PRNGSeed } from "@pkmn/sim";
import { ObservedBattle } from "../server/baselines/mcts_observation";
import { makeRootSampler } from "../server/baselines/mcts_simulator";
import { searchMcts } from "../server/baselines/mcts";
import { getSearchPotential } from "../server/baselines/search_potentials";
import { DEFAULT_MCTS_OPTIONS } from "../server/baselines/potential_mcts";

const roots = fs
    .readFileSync(path.resolve(process.argv[2]), "utf8")
    .trim()
    .split("\n")
    .map(
        (line) =>
            JSON.parse(line) as {
                rootIndex: number;
                observation: ObservedBattle;
                actions: string[];
                pair: number;
                game: number;
                turn: number;
                actualPotential: number;
                rootPotential: number;
            },
    );
const captured = fs
    .readFileSync(path.resolve(process.argv[3]), "utf8")
    .trim()
    .split("\n")
    .map(
        (line) =>
            JSON.parse(line) as {
                pair: number;
                game: number;
                turn: number;
                legal: string[];
            },
    );
const output = path.resolve(process.argv[4]);
if (fs.existsSync(output)) throw new Error("Refusing to overwrite PUCT audit");
for (const root of roots) {
    const source = captured.find(
        (record) =>
            record.pair === root.pair &&
            record.game === root.game &&
            record.turn === root.turn,
    )!;
    const results: unknown[] = [];
    for (let repeat = 0; repeat < 4; repeat++) {
        let material = JSON.stringify(root.observation);
        if (repeat > 0) material += `:audit-repeat:${repeat}`;
        const digest = createHash("sha256").update(material).digest();
        const seed: PRNGSeed = `${digest.readUInt16LE(0)},${digest.readUInt16LE(2)},${digest.readUInt16LE(4)},${digest.readUInt16LE(6)}`;
        const random = new PRNG(seed);
        try {
            const result = searchMcts(
                makeRootSampler(
                    root.observation,
                    source.legal,
                    random,
                    3,
                    getSearchPotential("original"),
                    "tactical",
                ),
                { ...DEFAULT_MCTS_OPTIONS, random: () => random.random() },
            );
            results.push({ ...result, seed });
        } catch (error) {
            results.push({ error: String(error), seed });
        }
    }
    fs.appendFileSync(
        output,
        JSON.stringify({
            rootIndex: root.rootIndex,
            originalAction: root.actions[0],
            actual: root.actualPotential,
            initial: root.rootPotential,
            results,
        }) + "\n",
    );
    console.log(root.rootIndex);
}
