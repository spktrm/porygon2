import { defineConfig } from "vitest/config";

export default defineConfig({
    test: {
        include: ["src/**/*.test.ts"],
        // Real battles with random actions: generous per-test budget, and
        // battles are stateful singletons of the sim — run serially.
        testTimeout: 180_000,
        hookTimeout: 60_000,
        pool: "forks",
        poolOptions: { forks: { singleFork: true } },
    },
});
