/** Offline-only conditional payoff audit. Reads saved public observations and
 * resolved public replies, never a live simulator or opposing private request.
 */
import fs from "fs";
import path from "path";
import { AnyObject, Battle, Dex, PRNG, PRNGSeed } from "@pkmn/sim";
import { PayoffRecord, PublicReply } from "../server/baselines/payoff_audit";
import {
    buildSampledBattle,
    SimulatorSearchState,
    simulatorActions,
} from "../server/baselines/mcts_simulator";
import { getSearchPotential } from "../server/baselines/search_potentials";
import { sampleProbability } from "../server/baselines/puct";
import { searchPriors } from "../server/baselines/search_priors";

function commandForReply(
    battle: Battle,
    reply: PublicReply,
): string | undefined {
    const side = battle.sides[1];
    if (reply.kind === "switch") {
        const slot = side.pokemon.findIndex(
            (mon) =>
                Dex.toID(mon.species.name) === reply.identity &&
                !mon.isActive &&
                !mon.fainted,
        );
        if (slot < 0) return undefined;
        return `switch ${slot + 1}`;
    }
    const request = side.activeRequest as AnyObject;
    const slot = request.active[0].moves.findIndex(
        (move: AnyObject) =>
            move.id === reply.identity && !move.disabled && move.pp !== 0,
    );
    if (slot < 0) return undefined;
    if (reply.tera) {
        if (!request.active[0].canTerastallize) return undefined;
        return `move ${slot + 1} terastallize`;
    }
    return `move ${slot + 1}`;
}

function simulate(
    snapshot: string,
    ownAction: string,
    reply: string,
    seed: PRNGSeed,
): { value: number | null; error?: string } {
    const battle = Battle.fromJSON(snapshot);
    battle.resetRNG(seed);
    const state = new SimulatorSearchState(battle);
    try {
        state.advance(ownAction, reply);
        return { value: state.evaluate() };
    } catch (error) {
        let reason = String(error);
        if (error instanceof Error && "choiceDetail" in error)
            reason += ": " + String(error.choiceDetail);
        return { value: null, error: reason };
    } finally {
        state.dispose();
    }
}

function main() {
    const input = path.resolve(process.argv[2]);
    const output = path.resolve(process.argv[3]);
    const worldCount = Number(process.argv[4] ?? 16);
    const chanceCount = Number(process.argv[5] ?? 16);
    const mode = process.argv[6] ?? "conditional";
    if (mode !== "conditional" && mode !== "prior")
        throw new Error("Unknown payoff audit mode");
    fs.mkdirSync(output, { recursive: true });
    if (fs.existsSync(path.join(output, "roots.jsonl")))
        throw new Error("Refusing to overwrite payoff audit");
    const records = fs
        .readFileSync(input, "utf8")
        .trim()
        .split("\n")
        .map((line) => JSON.parse(line) as PayoffRecord);
    const selected: PayoffRecord[] = [];
    const perGame = new Map<string, number>();
    const exclusions: Record<string, number> = {};
    for (const record of records) {
        if (
            record.exclusion ||
            record.actualPotential === undefined ||
            !record.reply
        ) {
            const reason = record.exclusion ?? "missing_actual";
            exclusions[reason] = (exclusions[reason] ?? 0) + 1;
            continue;
        }
        const identity = `${record.pair}:${record.game}`;
        const count = perGame.get(identity) ?? 0;
        if (count >= 2) continue;
        perGame.set(identity, count + 1);
        selected.push(record);
    }
    fs.writeFileSync(
        path.join(output, "manifest.json"),
        JSON.stringify(
            {
                input,
                recorded: records.length,
                selected: selected.length,
                exclusions,
                worldCount,
                chanceCount,
                mode,
                selection:
                    "First two eligible complete public transitions per game",
                conditioning:
                    "conditional mode fixes the public resolved reply; prior mode samples replies without hindsight",
                note: "Same chance seed across alternative own actions within each world; no live state truth",
            },
            null,
            2,
        ),
    );
    const potential = getSearchPotential("original");
    for (const [rootIndex, record] of selected.entries()) {
        const worlds: {
            snapshot?: string;
            reply?: string;
            error?: string;
            rootValue?: number;
            details?: AnyObject;
            priors?: number[];
            opponentActions?: string[];
            opponentPriors?: number[];
            priorValues?: (number | null)[];
            values?: (number | null)[][];
            errors?: Record<string, number>;
        }[] = [];
        const priorTotals = record.legal.map(() => 0);
        for (let world = 0; world < worldCount; world++) {
            let battle: Battle | undefined;
            try {
                const seed: PRNGSeed = `5171,${rootIndex + 1},${world + 1},101`;
                battle = buildSampledBattle(record.observation, seed);
                const active = battle.sides[1].active[0];
                const ownActive = battle.sides[0].active[0];
                const priors = searchPriors(
                    battle,
                    0,
                    record.legal,
                    "tactical",
                );
                priors.forEach((prior, index) => (priorTotals[index] += prior));
                const reply = commandForReply(battle, record.reply!);
                const opponentActions = simulatorActions(battle, 1);
                const opponentPriors = searchPriors(
                    battle,
                    1,
                    opponentActions,
                    "tactical",
                );
                let replyMass = 0;
                let familyMass = 0;
                const rankedReplies = opponentActions
                    .map((action, index) => ({
                        action,
                        prior: opponentPriors[index],
                    }))
                    .sort((left, right) => right.prior - left.prior);
                for (const [index, action] of opponentActions.entries()) {
                    if (action === reply) replyMass += opponentPriors[index];
                    if (
                        reply &&
                        action.replace(" terastallize", "") ===
                            reply.replace(" terastallize", "")
                    )
                        familyMass += opponentPriors[index];
                }
                worlds.push({
                    opponentActions,
                    opponentPriors,
                    snapshot: JSON.stringify(battle.toJSON()),
                    reply,
                    rootValue: potential(battle),
                    priors,
                    details: {
                        replyMass,
                        familyMass,
                        replyInTop3: rankedReplies
                            .slice(0, 3)
                            .some((entry) => entry.action === reply),
                        species: active.species.name,
                        moves: active.moveSlots.map((move) => move.id),
                        ability: active.ability,
                        item: active.item,
                        maxhp: active.maxhp,
                        stats: active.storedStats,
                        teraType: active.teraType,
                        ownStats: ownActive.storedStats,
                        ownHp: ownActive.hp,
                        ownMaxhp: ownActive.maxhp,
                        ownRequest: battle.sides[0].activeRequest,
                        seed,
                    },
                });
            } catch (error) {
                worlds.push({ error: String(error) });
            } finally {
                battle?.destroy();
            }
        }
        const ranked = record.legal
            .map((action, index) => ({ action, prior: priorTotals[index] }))
            .sort((left, right) => right.prior - left.prior);
        const actions = [
            ...new Set([
                record.ownAction,
                ...ranked.slice(0, 3).map((entry) => entry.action),
            ]),
        ];
        const bestSwitch = ranked.find((entry) =>
            entry.action.startsWith("switch "),
        );
        if (bestSwitch && !actions.includes(bestSwitch.action))
            actions.push(bestSwitch.action);
        for (const [worldIndex, world] of worlds.entries()) {
            if (!world.snapshot) continue;
            world.errors = {};
            const retainError = (reason: string | undefined) => {
                if (reason)
                    world.errors![reason] = (world.errors![reason] ?? 0) + 1;
            };
            if (mode === "prior") {
                world.priorValues = [];
                for (let chance = 0; chance < chanceCount; chance++) {
                    const replyRandom = new PRNG(
                        `5173,${rootIndex + 1},${worldIndex * chanceCount + chance + 1},11`,
                    );
                    const reply =
                        world.opponentActions![
                            sampleProbability(world.opponentPriors!, () =>
                                replyRandom.random(),
                            )
                        ];
                    const result = simulate(
                        world.snapshot,
                        record.ownAction,
                        reply,
                        `5172,${rootIndex + 1},${worldIndex * chanceCount + chance + 1},7`,
                    );
                    world.priorValues.push(result.value);
                    retainError(result.error);
                }
            } else {
                if (!world.reply) continue;
                world.values = actions.map(() => []);
                for (const [actionIndex, action] of actions.entries()) {
                    for (let chance = 0; chance < chanceCount; chance++) {
                        const result = simulate(
                            world.snapshot,
                            action,
                            world.reply,
                            `5172,${rootIndex + 1},${worldIndex * chanceCount + chance + 1},7`,
                        );
                        world.values[actionIndex].push(result.value);
                        retainError(result.error);
                    }
                }
            }
        }

        const serialised = worlds.map((world) => ({
            reply: world.reply,
            error: world.error,
            rootValue: world.rootValue,
            details: world.details,
            values: world.values,
            priorValues: world.priorValues,
            errors: world.errors,
        }));
        fs.appendFileSync(
            path.join(output, "roots.jsonl"),
            JSON.stringify({
                rootIndex,
                pair: record.pair,
                game: record.game,
                turn: record.turn,
                observation: record.observation,
                next: record.next,
                trace: record.trace,
                rootPotential: record.rootPotential,
                actualPotential: record.actualPotential,
                reply: record.reply,
                actions,
                worlds: serialised,
            }) + "\n",
        );
        console.log(
            JSON.stringify({
                rootIndex,
                worlds: worlds.filter((world) => world.snapshot).length,
                replySupported: worlds.filter((world) => world.reply).length,
                actions: actions.length,
            }),
        );
    }
}
main();
