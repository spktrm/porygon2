// Endless random-action soak over random formats — the long-running fuzz
// counterpart to the bounded vitest suite (battle.test.ts). Both share the
// invariants in harness.ts; here a violating battle is logged and the loop
// keeps going, so one bad seed doesn't end an overnight soak.
import { runBattle } from "./harness";

async function main() {
    let historyLength = 0;
    let packedHistoryLength = 0;
    while (true) {
        try {
            const results = await runBattle();
            historyLength = Math.max(
                historyLength,
                ...results.map((r) => r.historyLength),
            );
            packedHistoryLength = Math.max(
                packedHistoryLength,
                ...results.map((r) => r.packedHistoryLength),
            );
            console.log(historyLength, packedHistoryLength);
        } catch (error) {
            console.error("An error occurred during the battle:", error);
        }
    }
}

main().catch((error) => {
    console.error("An error occurred in the main execution:", error);
});
