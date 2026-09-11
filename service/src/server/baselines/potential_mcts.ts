/** Fair-information service baseline: sampled-world MCTS with potential leaves.
 * Ordinary Gen 9 singles requests only; unsupported reconstructions explicitly
 * fall back to SimpleHeuristic. Never receives a live sim or opponent object.
 */
import { searchMatrix, MATRIX_OPTIONS } from "./matrix_search";
import { SearchPriorName } from "./search_priors";
import { createHash, randomInt } from "crypto";
import { PRNG, PRNGSeed } from "@pkmn/sim";
import { getSearchPotential, SearchPotentialName } from "./search_potentials";
import { Action } from "../../../protos/service_pb";
import type { EvalActionFnType } from "../eval";
import { StateHandler } from "../state";
import { captureObservation } from "./mcts_observation";
import { makeRootSampler, normaliseChoice } from "./mcts_simulator";
import { searchMcts, SearchResult } from "./mcts";
import { GetSimpleHeuristicAction } from "./simple_heuristic";

export const MCTS_BUDGET = Object.freeze({
    iterations: 256,
    maxDepth: 1,
    maxMillis: 100,
});
export const mctsDiagnostics = {
    searches: 0,
    choiceFailures: {} as Record<string, number>,
    fallbacks: {} as Record<string, number>,
    last: undefined as (SearchResult & { elapsedMillis: number }) | undefined,
};

export interface PotentialMctsOptions {
    potential: SearchPotentialName;
    iterations: number;
    maxDepth: number;
    maxMillis: number;
    exploration?: number;
    selection?: "uct" | "puct" | "matrix";
    cpuct?: number;
    prior?: SearchPriorName;
    actionRandom?: () => number;
}
export function createPotentialMctsAction(
    options: PotentialMctsOptions,
): EvalActionFnType {
    const potential = getSearchPotential(options.potential);
    if (
        !Number.isInteger(options.iterations) ||
        options.iterations < 1 ||
        !Number.isInteger(options.maxDepth) ||
        options.maxDepth < 1 ||
        !Number.isFinite(options.maxMillis) ||
        options.maxMillis <= 0
    )
        throw new Error("Invalid potential MCTS budget");
    if (
        options.selection !== undefined &&
        options.selection !== "uct" &&
        options.selection !== "puct" &&
        options.selection !== "matrix"
    )
        throw new Error("Unknown search selection");
    if (
        options.prior !== undefined &&
        options.prior !== "uniform" &&
        options.prior !== "tactical"
    )
        throw new Error("Unknown search prior");
    if (
        options.cpuct !== undefined &&
        (!Number.isFinite(options.cpuct) || options.cpuct <= 0)
    )
        throw new Error("Invalid PUCT coefficient");
    if (options.selection === "matrix" && options.maxDepth !== 1)
        throw new Error("Matrix search requires depth one");
    return ({ player }) => {
        const start = Date.now();
        try {
            const observation = captureObservation(player);
            const playerIndex = player.getPlayerIndex()!;
            new StateHandler(player).getActionMask({
                request: observation.request,
                allyActive: player.publicBattle.sides[playerIndex].active,
                enemyActive: player.publicBattle.sides[1 - playerIndex].active,
            });
            const cells = new Map<string, number>();
            for (const [cell, command] of player.legalChoiceByCell)
                cells.set(normaliseChoice(command), cell);
            if (!cells.size) throw new Error("empty_root_actions");
            const digest = createHash("sha256")
                .update(JSON.stringify(observation))
                .digest();
            const seed: PRNGSeed = `${digest.readUInt16LE(0)},${digest.readUInt16LE(2)},${digest.readUInt16LE(4)},${digest.readUInt16LE(6)}`;
            const random = new PRNG(seed);
            const makeSampler = (generator: PRNG, worlds: number) =>
                makeRootSampler(
                    observation,
                    [...cells.keys()],
                    generator,
                    worlds,
                    potential,
                    options.prior ?? "uniform",
                );
            let result: SearchResult;
            if (options.selection === "matrix") {
                const samplers = Array.from(
                    { length: MATRIX_OPTIONS.worlds },
                    () => {
                        const worldSeed: PRNGSeed = `${random.random(65536)},${random.random(65536)},${random.random(65536)},${random.random(65536)}`;
                        return makeSampler(new PRNG(worldSeed), 1);
                    },
                );
                result = searchMatrix(samplers, {
                    ...options,
                    actionRandom:
                        options.actionRandom ??
                        (() => randomInt(0x100000000) / 0x100000000),
                    selection: undefined,
                    random: () => random.random(),
                });
            } else {
                result = searchMcts(makeSampler(random, 3), {
                    ...options,
                    selection: options.selection,
                    random: () => random.random(),
                });
            }
            const selected = cells.get(result.action);
            if (selected === undefined)
                throw new Error("search_returned_illegal_action");
            mctsDiagnostics.searches++;
            mctsDiagnostics.last = {
                ...result,
                elapsedMillis: Date.now() - start,
            };
            const action = new Action();
            action.setCell(selected);
            return action;
        } catch (error) {
            if (error instanceof Error && "choiceDetail" in error) {
                const detail = String(error.choiceDetail);
                mctsDiagnostics.choiceFailures[detail] =
                    (mctsDiagnostics.choiceFailures[detail] ?? 0) + 1;
            }
            let reason = "unknown_search_error";
            if (error instanceof Error) reason = error.message;
            mctsDiagnostics.fallbacks[reason] =
                (mctsDiagnostics.fallbacks[reason] ?? 0) + 1;
            return GetSimpleHeuristicAction({ player });
        }
    };
}
export const DEFAULT_MCTS_OPTIONS = Object.freeze({
    ...MCTS_BUDGET,
    potential: "original" as const,
    selection: "puct" as const,
    prior: "tactical" as const,
    cpuct: 1,
});
export const GetPotentialMctsAction =
    createPotentialMctsAction(DEFAULT_MCTS_OPTIONS);
