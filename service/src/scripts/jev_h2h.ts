import { execSync } from "child_process";
import * as dotenv from "dotenv";
import fs from "fs";
import path from "path";
import { BattleStreams, PRNG, PRNGSeed, Teams } from "@pkmn/sim";
import { TrainablePlayerAI } from "../server/runner";
import {
    CheckpointMeta,
    checkpointSource,
    DecisionSource,
    playDecisions,
} from "../agents/decision";
import { JevClient } from "../jev/client";
import { JevDecode, JevMeta, jevSource } from "../jev/source";
import {
    GameOutcome,
    mean,
    pairedBreakdown,
    percentile,
    totalScore,
    wilsonInterval,
} from "../stats";

dotenv.config({ path: path.resolve(__dirname, "../../../.env") });

const FORMAT = "gen9randombattle";
const MAX_TURNS = 300;
const GAMES_PER_PAIR = 2;
// Names route players: a `baseline` or `eval-heuristic` prefix would send a
// side into the service's scripted baselines instead of its decision queue.
const CHECKPOINT_NAME = "ckpt";
const JEV_NAME = "jev";

interface Arguments {
    pairs: number;
    out: string;
    seedOffset: number;
    rlServer: string;
    checkpoint: string;
    temperature: string;
    jevModel: string;
    jevDecode: JevDecode;
    jevDeadlineMs: number;
    jevContextBudget: number;
    concurrency: number;
    maxSpend: number;
}

interface GameRecord {
    pair: number;
    game: number;
    jevSide: "p1" | "p2";
    outcome: GameOutcome | "invalid";
    invalidReason?: string;
    turns: number;
    seconds: number;
    jevDecisions: number;
    jevLatencyP50: number;
    jevLatencyP95: number;
    jevMeanConfidence: number;
    jevInputTokens: number;
    jevPayloadChars: number;
    truncatedDecisions: number;
    checkpointDecisions: number;
    checkpointMeanVWin: number;
    checkpointMeanEntropy: number;
}

function parseArguments(argv: string[]): Arguments {
    const values = new Map<string, string>();
    for (let index = 0; index < argv.length; index += 2) {
        values.set(argv[index].replace(/^--/, ""), argv[index + 1]);
    }
    const read = (name: string, fallback: string) =>
        values.get(name) ?? fallback;
    const out = values.get("out");
    if (out === undefined) {
        throw new Error("--out <directory> is required");
    }
    return {
        pairs: Number(read("pairs", "200")),
        out,
        seedOffset: Number(read("seed-offset", "0")),
        rlServer: read("rl-server", "http://localhost:8001"),
        checkpoint: read("checkpoint", "undeclared"),
        temperature: read("temperature", "undeclared"),
        jevModel: read("jev-model", "typesafe/jev-1.13"),
        jevDecode: read("jev-decode", "argmax") as JevDecode,
        jevDeadlineMs: Number(read("jev-deadline-ms", "20000")),
        jevContextBudget: Number(read("jev-context-budget", "28000")),
        concurrency: Number(read("concurrency", "4")),
        maxSpend: Number(read("max-spend", "10")),
    };
}

async function playGame(
    pair: number,
    game: number,
    seedOffset: number,
    decideCheckpoint: DecisionSource<CheckpointMeta>,
    decideJev: DecisionSource<JevMeta>,
): Promise<GameRecord> {
    const seedIndex = pair + seedOffset + 1;
    const battleSeed: PRNGSeed = `9714,${seedIndex},831,719`;
    const teams = [
        Teams.pack(
            Teams.generate(FORMAT, { seed: `9714,${seedIndex},101,201` }),
        ),
        Teams.pack(
            Teams.generate(FORMAT, { seed: `9714,${seedIndex},102,202` }),
        ),
    ];
    const streams = BattleStreams.getPlayerStreams(
        new BattleStreams.BattleStream(),
    );

    // Same battle seed and rosters in both games of a pair; the agents trade
    // sides, so each plays each roster once.
    const jevName = `${JEV_NAME}-${pair}-${game}`;
    const checkpointName = `${CHECKPOINT_NAME}-${pair}-${game}`;
    let jevSide: "p1" | "p2" = "p1";
    let names = { p1: jevName, p2: checkpointName };
    if (game === 1) {
        jevSide = "p2";
        names = { p1: checkpointName, p2: jevName };
    }
    const first = new TrainablePlayerAI(names.p1, streams.p1);
    const second = new TrainablePlayerAI(names.p2, streams.p2);
    first.opponent = second;
    second.opponent = first;
    let jevPlayer = first;
    let checkpointPlayer = second;
    if (game === 1) {
        jevPlayer = second;
        checkpointPlayer = first;
    }

    let invalidReason: string | undefined;
    const voidGame = async (reason: string) => {
        if (invalidReason === undefined) {
            invalidReason = reason;
            await streams.omniscient.write(">forcetie");
        }
    };
    // A failed decision still has to answer its request, or the player's
    // loop waits on it forever; the game is already void by then.
    const lowestLegalCell = (player: TrainablePlayerAI) => {
        if (player.legalChoiceByCell.size === 0) {
            return 0;
        }
        return Math.min(...player.legalChoiceByCell.keys());
    };

    const jevMetas: JevMeta[] = [];
    const checkpointMetas: CheckpointMeta[] = [];
    const guardedJev: DecisionSource<JevMeta> = async (player, state) => {
        try {
            return await decideJev(player, state);
        } catch (error) {
            await voidGame(`jev: ${String(error)}`);
            return {
                cell: lowestLegalCell(player),
                meta: { asked: false, truncatedTurns: 0, droppedChatLines: 0 },
            };
        }
    };
    const guardedCheckpoint: DecisionSource<CheckpointMeta> = async (
        player,
        state,
    ) => {
        try {
            return await decideCheckpoint(player, state);
        } catch (error) {
            await voidGame(`checkpoint: ${String(error)}`);
            return {
                cell: lowestLegalCell(player),
                meta: { vWin: NaN, logProb: NaN, entropy: NaN },
            };
        }
    };

    const started = Date.now();
    const loops = [
        first.start(),
        second.start(),
        playDecisions(jevPlayer, guardedJev, (decision) => {
            if (decision.meta.asked) {
                jevMetas.push(decision.meta);
            }
        }),
        playDecisions(checkpointPlayer, guardedCheckpoint, (decision) =>
            checkpointMetas.push(decision.meta),
        ),
    ].map((loop) => loop.catch((error) => voidGame(`loop: ${String(error)}`)));

    await streams.omniscient.write(
        `>start ${JSON.stringify({ formatid: FORMAT, seed: battleSeed })}\n` +
            `>player p1 ${JSON.stringify({ name: names.p1, team: teams[0] })}\n` +
            `>player p2 ${JSON.stringify({ name: names.p2, team: teams[1] })}`,
    );

    let turns = 0;
    let winner: string | undefined;
    let tied = false;
    let capped = false;
    for await (const chunk of streams.omniscient) {
        for (const line of chunk.split("\n")) {
            if (line.startsWith("|turn|")) {
                turns = Number(line.split("|")[2]);
                if (turns >= MAX_TURNS && !capped) {
                    capped = true;
                    await streams.omniscient.write(">forcetie");
                }
            }
            if (line.startsWith("|win|")) {
                winner = line.slice("|win|".length);
            }
            if (line === "|tie" || line === "|tie|") {
                tied = true;
            }
        }
    }
    await Promise.all(loops);

    let outcome: GameOutcome | "invalid" = "invalid";
    if (invalidReason === undefined) {
        if (winner !== undefined && winner.startsWith(CHECKPOINT_NAME)) {
            outcome = "win";
        } else if (winner !== undefined) {
            outcome = "loss";
        } else if (tied) {
            outcome = "tie";
        } else {
            invalidReason = "battle ended with neither a winner nor a tie";
        }
    }

    const latencies = jevMetas.map((meta) => meta.latencyMs ?? NaN);
    return {
        pair,
        game,
        jevSide,
        outcome,
        invalidReason,
        turns,
        seconds: (Date.now() - started) / 1000,
        jevDecisions: jevMetas.length,
        jevLatencyP50: percentile(latencies, 0.5),
        jevLatencyP95: percentile(latencies, 0.95),
        jevMeanConfidence: mean(jevMetas.map((meta) => meta.confidence ?? NaN)),
        jevInputTokens: jevMetas.reduce(
            (sum, meta) => sum + (meta.inputTokens ?? 0),
            0,
        ),
        jevPayloadChars: jevMetas.reduce(
            (sum, meta) => sum + (meta.payloadChars ?? 0),
            0,
        ),
        truncatedDecisions: jevMetas.filter((meta) => meta.truncatedTurns > 0)
            .length,
        checkpointDecisions: checkpointMetas.length,
        checkpointMeanVWin: mean(checkpointMetas.map((meta) => meta.vWin)),
        checkpointMeanEntropy: mean(
            checkpointMetas.map((meta) => meta.entropy),
        ),
    };
}

function summarise(records: GameRecord[], client: JevClient, stopped: string) {
    const valid = records.filter((record) => record.outcome !== "invalid");
    const outcomes = valid.map((record) => record.outcome as GameOutcome);
    const count = (outcome: string) =>
        records.filter((record) => record.outcome === outcome).length;
    return {
        stopped,
        games: records.length,
        valid: valid.length,
        invalid: count("invalid"),
        checkpointWins: count("win"),
        checkpointLosses: count("loss"),
        ties: count("tie"),
        checkpointWinRate: wilsonInterval(totalScore(outcomes), valid.length),
        paired: pairedBreakdown(
            valid.map((record) => ({
                pair: record.pair,
                outcome: record.outcome as GameOutcome,
            })),
        ),
        gamesWithTruncation: records.filter(
            (record) => record.truncatedDecisions > 0,
        ).length,
        meanTurns: mean(valid.map((record) => record.turns)),
        jevModelEcho: client.modelEcho,
        jevCalls: client.calls,
        jevInputTokens: client.inputTokens,
        jevCostUsd: client.costUsd,
        jevCharsPerToken:
            records.reduce((sum, record) => sum + record.jevPayloadChars, 0) /
            client.inputTokens,
        jevLatencyP50: percentile(
            valid.map((record) => record.jevLatencyP50),
            0.5,
        ),
        jevLatencyP95: percentile(
            valid.map((record) => record.jevLatencyP95),
            0.95,
        ),
    };
}

async function main() {
    const args = parseArguments(process.argv.slice(2));
    const apiKey = process.env.OPENROUTER_API_KEY;
    if (apiKey === undefined || apiKey === "") {
        throw new Error("OPENROUTER_API_KEY is not set in the repo-root .env");
    }
    if (fs.existsSync(args.out)) {
        throw new Error(`${args.out} exists; results are never overwritten`);
    }
    const ping = await fetch(`${args.rlServer}/ping`);
    if (!ping.ok) {
        throw new Error(`RL server /ping returned ${ping.status}`);
    }
    fs.mkdirSync(args.out, { recursive: true });
    fs.writeFileSync(
        path.join(args.out, "manifest.json"),
        JSON.stringify(
            {
                started: new Date().toISOString(),
                gitSha: execSync("git rev-parse HEAD").toString().trim(),
                format: FORMAT,
                maxTurns: MAX_TURNS,
                paired: "Fixed rosters and battle seed per pair; the agents exchange sides and rosters in the second game",
                checkpointNote:
                    "checkpoint and temperature are DECLARED by the launcher, not read back from the inference server",
                args,
            },
            null,
            2,
        ),
    );

    const client = new JevClient({
        apiKey,
        model: args.jevModel,
        deadlineMs: args.jevDeadlineMs,
    });
    const sampleRandom = new PRNG(`9715,${args.seedOffset + 1},17,23`);
    const decideJev = jevSource(client, {
        decode: args.jevDecode,
        contextBudgetTokens: args.jevContextBudget,
        random: () => sampleRandom.random(),
    });
    const decideCheckpoint = checkpointSource(args.rlServer);

    const records: GameRecord[] = [];
    let nextPair = 0;
    let stopped = "completed";
    const worker = async () => {
        while (nextPair < args.pairs) {
            if (client.costUsd >= args.maxSpend) {
                stopped = `max spend of ${args.maxSpend} USD reached`;
                return;
            }
            const pair = nextPair;
            nextPair += 1;
            for (let game = 0; game < GAMES_PER_PAIR; game++) {
                const record = await playGame(
                    pair,
                    game,
                    args.seedOffset,
                    decideCheckpoint,
                    decideJev,
                );
                records.push(record);
                fs.appendFileSync(
                    path.join(args.out, "games.jsonl"),
                    JSON.stringify(record) + "\n",
                );
                console.log(
                    `pair ${pair} game ${game} jev=${record.jevSide} -> ${record.outcome} ` +
                        `turns=${record.turns} ${record.seconds.toFixed(0)}s ` +
                        `spend=$${client.costUsd.toFixed(4)}`,
                );
            }
        }
    };
    await Promise.all(Array.from({ length: args.concurrency }, () => worker()));

    const summary = summarise(records, client, stopped);
    fs.writeFileSync(
        path.join(args.out, "summary.json"),
        JSON.stringify(summary, null, 2),
    );
    console.log(JSON.stringify(summary, null, 2));
}

main().then(
    () => process.exit(0),
    (error) => {
        console.error(error);
        process.exit(1);
    },
);
