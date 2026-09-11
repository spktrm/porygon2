# Service baselines

Baseline index **3**, `potentialmcts`, runs fair-information sampled-world MCTS
inside the TypeScript service. It uses no Python/JAX inference. Select it with
`eval_baseline=3` when explicitly configuring a future learner run, or the usual
baseline username suffix `:0003` in a standalone service game. The default
baseline remains index2. The Python change only registers the evaluation name.
No training run is restarted by adding this baseline.

## What it searches

- Both sides sample actions using positive advantage plus a shrinking prior
  term before the simulator resolves their simultaneous actions. Opponent Q
  and node value reverse sign. This is a Jaxcalibur-style PUCT adaptation;
  the original decoupled UCT selector remains a matched experimental control.
  Tree nodes represent action histories.
- Each rollout clones one of up to three sampled worlds reconstructed from
  the baseline's own request and public observations. Hidden opponent sets
  come from the Gen9 random-battle generator, retaining revealed moves,
  abilities and items. Unseen roster members are sampled without duplicate
  base species. This is a coarse prior, not exact posterior inference.
- `@pkmn/sim` resolves hypothetical moves and switches, chance outcomes and
  subsequent replacement requests. The live simulator, its RNG, opponent
  object, private opponent request and submitted opponent choice are never
  inputs to the search.
- A completed simulated game returns win+1/loss-1/tie0. An unfinished leaf
  uses the bounded human-outcome position estimate at unit scale. The PBRS
  coefficient eta and episode-centred shaping differences are **not** search
  rewards: this baseline uses the potential as a leaf-value heuristic.
- Defaults: PUCT, tactical priors, original potential, cpuct=1, at most256
  simulations, depth1 decision transition, and a100ms soft deadline. At least one iteration completes;
  a synchronous simulation cannot be interrupted midway, so this is not a
  hard100ms latency guarantee. Root action is selected by visits, then mean.

This is an open-loop, determinised search baseline, not a full information-set
solver or a guarantee of Nash-equilibrium play. The original leaf coefficients
are shared with the unchanged Python PBRS research reference through
`rl/offline/position_potential_fit.json`. Search-only alternatives live in
`search_potentials.ts`; they do not change learner rewards.
The service runtime must have that repository-relative JSON file available.

## Explicit support and fallbacks

MCTS currently accepts ordinary **Gen9 singles move requests**. It reconstructs
own known stats/HP/moves/PP, revealed opposing HP/sets, public boosts, ordinary
tera and permanent entry hazards. Opponent stats/moves/items not revealed are
hypotheses. Hypothetical simulator leaves evaluate the sampled worlds; they are
not access to actual hidden game state.

Team preview, root forced/pivot replacement requests, doubles/other generations,
weather/terrain, pseudo-weather, timer-dependent side conditions, sleep/toxic
counters, complex volatiles/Illusion and unsupported species priors use
SimpleHeuristic. Forced replacements reached **within a rollout** are simulated.
Known residual limitations include inferred stat spreads, opponent damage-roll
conditioning, unrevealed interactions and exact historical state such as
last-used turn/fake-out eligibility. This is approximate root reconstruction,
not a claim that an observation uniquely determines the simulator state.

Fallbacks and search errors are counted by reason in exported `mctsDiagnostics`,
along with the last search's visits, means, completed iterations and latency.
The service integration test prints these counters. They are module-local to
each worker, not currently a W&B metric or a persisted per-game audit. Report
fallback coverage when evaluating strength: games can mix MCTS and heuristic
actions. The existing doubles alignment defect remains unresolved; the
multi-active potential alone does not make the battle adapter doubles-ready.

## Files and tests

- `mcts.ts`: shared UCT/PUCT search, backups and rollout ownership.
- `puct.ts`: sampled positive-advantage/prior probability formula.
- `search_priors.ts`: uniform or tactical heuristic action priors.
- `search_potentials.ts`: original, material, strategic, tactical and roster leaves.
- `mcts_observation.ts`: explicit public/own-request information boundary.
- `mcts_simulator.ts`: sampled worlds, simulator action enumeration and clones.
- `position_potential.ts`: shared-coefficient position evaluation.
- `potential_mcts.ts`: legal action-cell adapter, budgets and fallback counters.

From `service/`:

```
npx vitest run src/server/baselines/mcts.test.ts src/server/baselines/puct.test.ts src/server/baselines/search_potentials.test.ts src/tests/potential_mcts.test.ts
npx tsc --noEmit
```

The focused tests cover terminal wins, adversarial responses, potential leaves,
fixed-seed reproducibility, iteration/deadline accounting, potential symmetry,
public-only capture with forbidden hidden getters and a live public-data control,
root reconstruction, clone isolation across80 rollouts, legal simulator choices,
forced replacements and a complete service game through the actual decoder.
Fixed-iteration tests are reproducible; the wall-clock deployment budget can
change the completed iteration count with machine load.

The clone regression is important: the simulator's `toJSON()` aliases its log
array. Passing the same object repeatedly to `fromJSON()` shared rollout logs
and eventually tripped the simulator's1000-unsent-line guard. Cached roots are
now JSON strings, so each clone owns its arrays and cannot contaminate another.

Correctness tests cover PUCT probability arithmetic, prior override by wins,
adversarial mixed play, seeded reproducibility and pure positive heuristic
priors. They do not themselves establish strength; use paired battle evaluation.
No evaluation against the learned agent has been performed.

## Standalone paired head-to-head

From `service/`, run without a learner or WebSocket server:

```
npx ts-node src/scripts/potential_mcts_h2h.ts 50 ../runtime/potential-mcts-h2h-new
```

This runs 100 Gen9 random battles: each pair keeps the two generated teams and
battle seed fixed while the algorithms exchange teams/sides. The output
folder must be new. `manifest.json` records the budget and protocol;
`games.jsonl` records packed teams/seeds, outcomes, truncations, live invalid
choices, per-decision timing/search/switch counters and fallback reasons.
A 300-turn cap is recorded separately from natural draws. Omniscient output
is used only to score completed games; player decisions use the normal
service adapters. Compare uncertainty across pairs, since the two games in
a pair are correlated. The wall-clock budget prevents exact reproducibility
of visit counts even with saved teams and seeds.

## PUCT rule and limits

[Reference description](https://jaxcalibur.github.io/#search): this adaptation
samples actions with weights `max(Q - V, 0) + cpuct / sqrt(max(1, N)) * P`.
Here Q and V are mean sampled returns in [-1,1], N is node visits, and P is
normalised over legal actions. Unvisited Q starts at V. Tactical P mixes 5%
uniform probability so unusual legal actions retain exploration mass.

This adopts the published selection rule, with important differences: heuristic
priors/potential leaves, an action-history tree, three coarse sampled worlds,
and no learned surprise-revelation cutoff. In particular simulated opponent
planning can overestimate knowledge of our own hidden set. The actual baseline
still cannot inspect the live opposing private team, submitted choice or RNG.
This is not a reproduction of Jaxcalibur's neural search system or its strength.

`createPotentialMctsAction(options)` exposes potential, prior, selection,
iteration/depth/time budgets and cpuct. `DEFAULT_MCTS_OPTIONS` owns deployed
baseline-3 defaults; the overall learner eval baseline remains index2.
The H2H script uses these defaults unless positional overrides are supplied:
`pairs output potential seedOffset iterations millis depth exploration selection prior cpuct`.
For example, the original UCT control is:

```
npx ts-node src/scripts/potential_mcts_h2h.ts 50 ../runtime/uct-control original 1000 64 100 3 1.4142135623730951 uct uniform 1
```

Every run writes its resolved options to the manifest. The new search code takes
effect when the service is rebuilt; adding it does not restart a running service.

## Measured screen (2026-09-11)

Twelve candidates/configurations were screened on40 tuning games each. The
selected PUCT/original-leaf/tactical-prior/depth1 configuration won30/40 there,
then **120/200 (60%)** on fresh games against SimpleHeuristic; pair-bootstrap
95% interval54.5–65.5%. The requested70% target was not achieved.
On the first100 fresh games, the old UCT/original/depth3 control won43 while
PUCT won59 on exactly matched teams/seeds/sides: +16 percentage points,
pair-bootstrap95% interval+7 to+25 points. This supports the combined change,
not an isolated selector-only improvement.

PUCT searched52.5% of ordinary decisions, averaged100.63ms/57.78 iterations per
successful search, and had59 sampled trapping-choice failures with legal
heuristic fallback. There were no live invalid choices or truncated games.
The sampled-opponent and fallback limitations remain. All four replacement
leaf evaluators lost the tuning selection; the original PBRS files are unchanged.
Local detailed artefacts are under `runtime/search-potentials-20260911/`.

## Experimental one-turn regret matching

`createPotentialMctsAction({ ...DEFAULT_MCTS_OPTIONS, selection: "matrix" })`
selects the service-only local matrix solver. PUCT remains baseline3's default.
Use `maxDepth: 1`. This is not full-game CFR: it retains three highest-prior own
actions and three replies in each of three sampled worlds, caches their
simulated payoffs, then runs1,024 signed-regret matching updates. Our policy is
shared across worlds; opposing policies condition on the sampled opposing set.
It samples an action from the averaged own policy, rather than taking an argmax.

Files: `regret_matching.ts` (pure solver), `matrix_search.ts` (sample/cache/solve),
`regret_matching.test.ts` (equilibrium/information-set/budget/RNG tests).
The 256-call cap includes three inspection clones; the100ms soft deadline
reserves5ms for solving. An incompletely sampled matrix triggers an explicit
`matrix_budget_incomplete` fallback; missing cells are not treated as zero.
Diagnostics include cell/evaluation counts, restricted-game equilibrium gap,
retained/legal action counts and root policy probabilities. A small gap is
not a certificate for the full battle or for omitted actions.

Live deployment draws use private cryptographic randomness independent of the
observation-derived simulator seed. Tests and the benchmark inject separate
seeded action streams for reproducibility. The benchmark manifest records that
seed rule and `MATRIX_OPTIONS`. No actual opposing private state or live
simulator RNG is accessed. The existing opponent-information approximation
and unsupported-state fallbacks remain.

From `service/`, compare on matching seed offsets and new output directories:

```
npx ts-node src/scripts/potential_mcts_h2h.ts 50 ../runtime/matrix-new original 3000 256 100 1 1.4142135623730951 matrix tactical 1
npx ts-node src/scripts/potential_mcts_h2h.ts 50 ../runtime/puct-new original 3000 256 100 1 1.4142135623730951 puct tactical 1
npx vitest run src/server/baselines/regret_matching.test.ts
```

The original PBRS files and learner rewards are unchanged.

The frozen matrix experiment won51/100 versus PUCT56/100 on matched fresh
Gen9 random battles (seed indices3001–3050). Paired difference−5pp,95% bootstrap
interval−17 to+7pp: no demonstrated improvement. PUCT remains the default.
Mean matrix gap0.000353 refers only to the restricted estimated game; its
roughly27 cells averaged2.4 payoff samples each. Two matrix-coverage fallbacks,
zero live invalid choices/failed/truncated games.31 focused tests passed.
Local records: `runtime/matrix-search-20260911/`.

## Payoff reliability audit

The optional final H2H argument `audit` records public root observations and
subsequently resolved public replies in `payoffs.jsonl`. It does not expose
opposing private information to the policy. `payoff_reliability.ts` replays
saved roots with independent world/chance seeds in `conditional` or `prior`
mode; `payoff_puct_reliability.ts` repeats the actual adaptive search.

On 46 matched supported nonterminal roots from 29 games, next-original-potential
RMSE was 0.034 conditional on the actual reply, 0.066 under the initial tactical
reply prior, 0.070 for adaptive PUCT, and 0.086 copying the current potential.
This diagnoses reply-distribution mismatch; it does not validate potential-to-win
alignment or establish the best search depth. The actual reply appeared among
three highest-prior opposing commands in only 58.8% of sampled worlds.
Conditional hidden-world variance was small, but conditioning itself removes
hidden-action uncertainty. Do not generalise that result to all battle states.

The audit also found cosmetic species with no random-set table entry blocking
whole roots, even when fainted. Cosmetic set-prior canonicalisation now recovers
all 96 such failed world constructions in the saved sample, while preserving
observed identity and keeping mechanically different formes distinct. No
post-fix win-rate improvement has been established. Forecast figures above use
the pre-fix matched population. Local artefacts: `runtime/payoff-reliability-20260911/`.
