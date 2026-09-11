/** Paired offline evaluation; omniscient output is used only for scoring. */
import fs from "fs";
import path from "path";
import { BattleStreams, PRNG, PRNGSeed, Teams } from "@pkmn/sim";
import { TrainablePlayerAI } from "../server/runner";
import { evalActionMapping } from "../server/eval";
import {
    DEFAULT_MCTS_OPTIONS,
    createPotentialMctsAction,
    mctsDiagnostics,
} from "../server/baselines/potential_mcts";

import { PayoffRecorder } from "../server/baselines/payoff_audit";
import { MATRIX_OPTIONS } from "../server/baselines/matrix_search";
import { SearchPotentialName } from "../server/baselines/search_potentials";
const experiment = {
    potential: (process.argv[4] ??
        DEFAULT_MCTS_OPTIONS.potential) as SearchPotentialName,
    iterations: Number(process.argv[6] ?? DEFAULT_MCTS_OPTIONS.iterations),
    maxMillis: Number(process.argv[7] ?? DEFAULT_MCTS_OPTIONS.maxMillis),
    maxDepth: Number(process.argv[8] ?? DEFAULT_MCTS_OPTIONS.maxDepth),
    exploration: Number(process.argv[9] ?? Math.SQRT2),
    selection: (process.argv[10] ?? DEFAULT_MCTS_OPTIONS.selection) as
        | "uct"
        | "puct"
        | "matrix",
    prior: (process.argv[11] ?? DEFAULT_MCTS_OPTIONS.prior) as
        | "uniform"
        | "tactical",
    cpuct: Number(process.argv[12] ?? DEFAULT_MCTS_OPTIONS.cpuct),
};
const experimentAction = createPotentialMctsAction(experiment);
const seedOffset = Number(process.argv[5] ?? 0);

class EvaluationPlayer extends TrainablePlayerAI {
    searchActor = experimentAction;
    payoffAudit?: PayoffRecorder;
    measurements: {
        matrix?: NonNullable<typeof mctsDiagnostics.last>["matrix"];
        millis: number;
        searched: boolean;
        iterations: number;
        voluntary: boolean;
        switched: boolean;
    }[] = [];
    override async getChoice(): Promise<string> {
        this.payoffAudit?.resolve(this);
        const request = this.getRequest();
        const before = mctsDiagnostics.searches;
        const started = performance.now();
        let actor = evalActionMapping[this.baselineIndex];
        if (this.baselineIndex === 3) actor = this.searchActor;
        const action = actor({ player: this });
        const choice = this.choiceFromAction(action);
        this.payoffAudit?.begin(this, choice);
        const searched = mctsDiagnostics.searches > before;
        let iterations = 0;
        if (searched) iterations = mctsDiagnostics.last!.iterations;
        let matrix: NonNullable<typeof mctsDiagnostics.last>["matrix"];
        if (searched) matrix = mctsDiagnostics.last?.matrix;
        this.measurements.push({
            matrix,
            millis: performance.now() - started,
            searched,
            iterations,
            voluntary:
                !!request.active &&
                !request.forceSwitch &&
                !request.teamPreview,
            switched: choice.startsWith("switch "),
        });
        return choice;
    }
}

async function main() {
    const pairs = Number(process.argv[2] ?? 50);
    const output = path.resolve(
        process.argv[3] ?? "../runtime/potential-mcts-h2h",
    );
    fs.mkdirSync(output, { recursive: true });
    const resultsPath = path.join(output, "games.jsonl");
    if (fs.existsSync(resultsPath))
        throw new Error("Refusing to overwrite existing evaluation");
    fs.writeFileSync(
        path.join(output, "manifest.json"),
        JSON.stringify(
            {
                pairs,
                budget: experiment,
                matrixOptions: MATRIX_OPTIONS,
                payoffAudit: process.argv[13] === "audit",
                seedOffset,
                actionPolicySeed:
                    "9712,pair+seedOffset+1,game+1,side+1 (independent actor stream)",
                format: "gen9randombattle",
                maxTurns: 300,
                paired: "Fixed rosters and battle seed; algorithms exchange rosters/sides in second game",
                note: "100 ms soft deadline makes visit counts hardware dependent; no training connection",
            },
            null,
            2,
        ),
    );
    for (let pair = 0; pair < pairs; pair++) {
        const battleSeed: PRNGSeed = `9711,${pair + seedOffset + 1},831,719`;
        const firstTeam = Teams.pack(
            Teams.generate("gen9randombattle", {
                seed: `9711,${pair + seedOffset + 1},101,201`,
            }),
        );
        const secondTeam = Teams.pack(
            Teams.generate("gen9randombattle", {
                seed: `9711,${pair + seedOffset + 1},102,202`,
            }),
        );
        for (let game = 0; game < 2; game++) {
            const streams = BattleStreams.getPlayerStreams(
                new BattleStreams.BattleStream(),
            );
            let firstIndex = 3;
            let secondIndex = 2;
            if (game === 1) {
                firstIndex = 2;
                secondIndex = 3;
            }
            const first = new EvaluationPlayer(
                `baseline-first:${firstIndex}`,
                streams.p1,
            );
            const second = new EvaluationPlayer(
                `baseline-second:${secondIndex}`,
                streams.p2,
            );
            for (const [side, player] of [first, second].entries()) {
                const actionRandom = new PRNG(
                    `9712,${pair + seedOffset + 1},${game + 1},${side + 1}`,
                );
                player.searchActor = createPotentialMctsAction({
                    ...experiment,
                    actionRandom: () => actionRandom.random(),
                });
            }
            if (process.argv[13] === "audit") {
                for (const player of [first, second]) {
                    if (player.baselineIndex === 3)
                        player.payoffAudit = new PayoffRecorder(
                            (record) =>
                                fs.appendFileSync(
                                    path.join(output, "payoffs.jsonl"),
                                    JSON.stringify(record) + "\n",
                                ),
                            { pair, game },
                        );
                }
            }
            first.opponent = second;
            second.opponent = first;
            const beforeFallbacks = { ...mctsDiagnostics.fallbacks };
            const started = Date.now();
            let turns = 0;
            let outcome = "missing";
            let truncated = false;
            let failure: string | undefined;
            const loops = [first.start(), second.start()].map((promise) =>
                promise.catch(async (error) => {
                    failure = String(error);
                    await streams.omniscient.write(">forcetie");
                }),
            );
            await streams.omniscient.write(
                `>start ${JSON.stringify({ formatid: "gen9randombattle", seed: battleSeed })}\n>player p1 ${JSON.stringify({ name: first.userName, team: firstTeam })}\n>player p2 ${JSON.stringify({ name: second.userName, team: secondTeam })}`,
            );
            for await (const chunk of streams.omniscient) {
                for (const line of chunk.split("\n")) {
                    if (line.startsWith("|turn|")) {
                        turns = Number(line.split("|")[2]);
                        if (turns >= 300 && !truncated) {
                            truncated = true;
                            await streams.omniscient.write(">forcetie");
                        }
                    }
                    if (line.startsWith("|win|")) {
                        outcome = "simple";
                        if (line.slice(5).endsWith(":3")) outcome = "mcts";
                    }
                    if (line === "|tie" || line === "|tie|") outcome = "tie";
                }
            }
            await Promise.all(loops);
            const fallbacks: Record<string, number> = {};
            for (const [reason, count] of Object.entries(
                mctsDiagnostics.fallbacks,
            )) {
                const delta = count - (beforeFallbacks[reason] ?? 0);
                if (delta) fallbacks[reason] = delta;
            }
            const record = {
                pair,
                game,
                battleSeed,
                firstTeam,
                secondTeam,
                firstIndex,
                outcome,
                truncated,
                failure,
                turns,
                millis: Date.now() - started,
                fallbacks,
                cumulativeSampleChoiceFailures: {
                    ...mctsDiagnostics.choiceFailures,
                },
                invalidChoices:
                    first.invalidChoiceCount + second.invalidChoiceCount,
                first: first.measurements,
                second: second.measurements,
            };
            fs.appendFileSync(resultsPath, JSON.stringify(record) + "\n");
            first.payoffAudit?.finish();
            second.payoffAudit?.finish();
            first.destroy();
            second.destroy();
            await streams.omniscient.writeEnd().catch(() => {});
            console.log(
                JSON.stringify({
                    pair,
                    game,
                    outcome,
                    turns,
                    millis: record.millis,
                    failure,
                    fallbacks,
                    cumulativeSampleChoiceFailures: {
                        ...mctsDiagnostics.choiceFailures,
                    },
                }),
            );
            if (failure || outcome === "missing" || record.invalidChoices)
                throw new Error("Invalid evaluation game; inspect results");
        }
    }
}
main().catch((error) => {
    console.error(error);
    process.exitCode = 1;
});
