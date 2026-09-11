import { expect, test } from "vitest";
import { runBattle } from "./harness";
import { mctsDiagnostics } from "../server/baselines/potential_mcts";

test(
    "potential MCTS completes a singles battle through the real service action decoder",
    { retry: 2 },
    async () => {
        const before = mctsDiagnostics.searches;
        const results = await runBattle({
            smogonFormat: "gen9randombattle",
            baselineIndex: 3,
        });
        expect(results).toHaveLength(1);
        expect(results[0].stateCount).toBeGreaterThan(0);
        expect(mctsDiagnostics.searches).toBeGreaterThan(before);
        console.log(
            "MCTS service counters",
            JSON.stringify({
                searches: mctsDiagnostics.searches - before,
                fallbacks: mctsDiagnostics.fallbacks,
                lastIterations: mctsDiagnostics.last?.iterations,
                lastMillis: mctsDiagnostics.last?.elapsedMillis,
            }),
        );
    },
);
