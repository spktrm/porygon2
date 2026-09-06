# Porygon2 — agent instructions

## Scope and sources

These instructions apply throughout this repository. They are based on
`CLAUDE.md`, `LESSONS.md`, and the implementation inspected on 2026-09-07.
Keep durable project guidance here; record experiment results and removal or
restoration history in `LESSONS.md` rather than expanding the startup context.

- Read the relevant code before changing it. Some READMEs and historical code
  comments describe retired architectures or commands. Executable definitions,
  configuration, and contract tests establish current behaviour.
- Consult `CLAUDE.md` for the companion agent guidance. Keep shared project
  rules consistent when updating either file; Claude-specific hooks and tools
  are not automatically available to other agents.
- Search `LESSONS.md` before proposing a mechanism, choosing a coefficient,
  deleting or restoring functionality, or writing a plan's declined options.
  It records measurements, failed approaches, and revert handles.
- `docs/` contains local design documents and `docs/plan-template.md`. It is
  gitignored: do not commit it, cite it as public, or assume a fresh clone has it.
- Check `git status --short` before editing. Preserve unrelated working-tree
  changes, including changes inside the `data/ps` submodule.

## What this project does

Porygon2 trains Pokémon Showdown agents through asynchronous self-play,
primarily for generation 9. A TypeScript service runs battles through `@pkmn`
packages and exchanges protobuf messages over WebSockets with Python actors.
Actors produce trajectory chunks; a JAX/Flax learner trains from replay and
publishes parameter snapshots to the actor population and league.

The main path is:

```text
@pkmn battle simulation
  -> service/src/server/state.ts: EnvironmentState protobuf
  -> rl/environment/: decode observations, actions, and history
  -> rl/online/player_actor.py: play games and emit overlapping chunks
  -> rl/online/buffer.py: replay
  -> rl/online/training/: batch, compute targets, update parameters
  -> actor/league snapshots and sharded checkpoints
```

The learner uses the GPU. Actors currently default to CPU inference in f32
through `player_actor_device="cpu"`; a GPU batching path also exists. This is
distinct from the GPU requirement for real-model tests. Do not impose a global
CPU JAX backend to configure actors.

Self-play supplies the training and exploration signals. Do not introduce
scripted heuristics or human-derived reward shaping. The standalone replay
critic remains a research tool; its former RL shaping consumption path is retired.

## Environment and commands

Run commands from the repository root unless a working directory is specified.
Python is the repository virtual environment: **`env/bin/python`**, not system
Python and not the obsolete `venv/` path in the root README. Python dependencies
are in `requirements.txt`; `pyproject.toml` configures formatting and linting.
Node dependencies and scripts are in `service/package.json` and `data/package.json`.

| Task | Command | Notes |
| --- | --- | --- |
| Fast Python tests | `env/bin/python -m pytest tests/ -m "not slow"` | Prefer a relevant file or selection for a narrow change. |
| Real-model tests | `env/bin/python -m pytest tests/ -m slow` | GPU only, when no training run is live; see testing rules below. |
| Service tests | `cd service && npm test` | Bounded Vitest battle-invariant suite. |
| Service typecheck | `cd service && npx tsc --noEmit` | Checks TypeScript without a build. |
| Service ESLint | `cd service && npm run lint` | Separate from the formatting script. |
| Build service and example fixtures | `cd service && npm run compile-base` | Removes/rebuilds `dist`, then runs `src/tests/ex.ts`. |
| Regenerate protobuf bindings | `bash scripts/compile_protos.sh` | Regenerates both language bindings and the enums source. |
| Format and lint | `source env/bin/activate` then `bash scripts/lint.sh` | Rewrites multiple trees; run before staging a commit. |
| Focused static checks | `env/bin/ruff check rl/ tests/ scripts/` | Uses the repository exclusions and load-order exceptions. |
| Launch/restart training | `bash start.sh [arguments]` | Operational action: restarts the `train` tmux session. |
| Attach to training | `tmux attach -t train` | Pane 0 is service; pane 1 is learner. |
| W&B inspection | `env/bin/python scripts/wb.py ...` | Compact query commands below. |

`npm run test-soak` in `service/` is an endless fuzz loop, not a normal test
command. The Makefile's `kill` target kills the tmux server and Python/Node
processes broadly; do not use it for routine task cleanup.

`scripts/lint.sh` runs Prettier on service/data TypeScript, autoflake/isort/Black
on several Python trees, and Ruff on `rl/`, `tests/`, and `scripts/`. It is
broader than a single-file formatter. Account for existing dirty files and
inspect the resulting diff before staging. Do not run this mutating script
merely to validate a Markdown-only edit.

## Repository map

### Protocol, service, and environment

- `proto/service.proto`, `proto/features.proto`: shared wire schemas.
- `proto/enums.proto`: **generated**, not a hand-editable schema. Its source is
  `data/data/data.json`, processed by `proto/scripts/make_enums.py` at the start
  of `scripts/compile_protos.sh`.
- `service/protos/`, `rl/environment/protos/`: generated bindings. Regenerate
  both sides together; do not patch the output or remove generated imports.
- `service/src/server/index.ts`, `worker.ts`: service coordination and workers.
- `service/src/server/runner.ts`: `TrainablePlayerAI`, `createBattle`, and
  opponent cross-references needed for wedged-battle teardown.
- `service/src/server/state.ts`: `StateHandler.build`, state encoding, and
  history windowing/rebasing in `getHistory`.
- `service/src/tests/harness.ts`: invariants shared by the bounded
  `battle.test.ts` suite and the `main.ts` soak runner.
- `rl/environment/interfaces.py`: pytree dataclasses crossing environment,
  model, actor, and learner boundaries.
- `rl/environment/data.py`: environment feature counts, action layout, masks.
- `rl/environment/utils.py`: protobuf-to-NumPy decoding, history preparation,
  inference buckets, and `clip_history_windows_tail`.
- `rl/environment/env.py`: Python environment/service interaction.

Put a constant in `proto/` only when **both the service and Python read it**.
Model-only layout belongs in `rl/model/constants.py`; environment-side layout
belongs in `rl/environment/data.py`. Model token types do not belong in the
wire enums: that experiment was reverted and is documented in `LESSONS.md`.

### Model

- `rl/model/config.py`: model configuration; `rl/online/config.py` is the
  separate learner/actor/run configuration.
- `rl/model/constants.py`: `SEQUENCE_LAYOUT`, derived offsets and named slices,
  action-bank mappings, policy-readable rows, and `SEQUENCE_READ_MASK`.
  Current layout: **80 learner rows, 73 policy-readable actor rows**. Older
  references to an unmasked 61-row trunk describe an earlier architecture.
  Derive counts and offsets from the layout; never copy literals into heads.
- `rl/model/encoder.py`: feature embeddings, entity-local pooling, and sequence
  assembly. `_assemble_sequence` centralises additive row identities and is
  separately testable from the batched forward.
- `rl/model/history_encoder.py`: per-slot recurrent history encoding and
  request alignment. Actor-side recurrent carry avoids replaying all history.
- `rl/model/trunk.py`: unshared pre-RMSNorm attention blocks. Preserve the
  policy/privileged read partition in every block.
- `rl/model/heads.py`: `FlatActionReadout`, policy metrics, and value heads.
  Switching reads private sheet rows, moves use a move/target bilinear, and
  standalone actions read target rows. The deployable critic reads `CLS`;
  the privileged critic reads `VALUE_CLS`.
- `rl/model/player_model.py`: encoder/head wiring, actor vs learner paths,
  privileged outputs, and shared-head transition evaluation.
- `rl/model/transition.py`: stochastic latent transition over policy-readable
  post-trunk rows, chance-code prior/posterior, grounding, next-action-mask,
  and kind/done prediction. Imagined rows use the shared policy/value heads.
- `rl/model/search.py`: static-shape, depth-1 expectimax for an evaluation
  actor. Samples chance codes from the prior and adds an action-value bonus
  to legal policy logits. Its control is the same checkpoint and temperature.
- `rl/model/modules.py`: generic primitives. Keep architecture beside its wiring.

### Actors, learner, persistence, and research

- `rl/online/main.py`: process setup, configuration/resume, actor and eval wiring.
- `rl/online/player_actor.py`: whole-game play, statistics, and `chunk_spans`.
- `rl/online/agent.py`: actor network/device handling.
- `rl/online/inference.py`: zero-wait batched GPU inference and parameter LRU;
  CPU actors follow their own inference path.
- `rl/online/buffer.py`: replay stores; capacity counts **chunks**, not games.
- `rl/online/guards.py`: explicit evaluation-to-training leak gate.
- `rl/online/league.py`: league/PFSP bookkeeping and snapshots.
- `rl/online/training/train_step.py`: jitted update, losses, EMA target updates,
  and non-finite gate. Its static config and donated states are load-bearing.
- `rl/online/training/batching.py`: static shape lattice and `stack_batch`.
- `rl/online/training/run_state.py`: mutable `RunState` bundle.
- `rl/online/training/workers.py`: background workers and replay-reuse control.
- `rl/online/training/league_ops.py`: checkpoint pacing, snapshots, payoff reads.
- `rl/online/training/targets.py`: v-trace/Retrace and f32 recursions.
- `rl/online/training/loss.py`, `controllers.py`, `telemetry.py`,
  `diagnostics.py`: objectives, controllers, metrics, and RAM attribution.
- `rl/online/training/learner.py`: construction, loop, and periodic scheduling.
  Prefer free functions over `RunState` where practical so tests can use stubs.
- `rl/checkpoint.py`: sharded checkpoint persistence and loading.
- `rl/offline/harness.py`: play games with plain parameters and re-run training
  heads on chunks for joint per-row diagnostics. Other offline probes and the
  replay critic trainer are described in `rl/offline/README.md`.
- `service/src/scripts/offline.ts`: replay-to-protobuf shard export using the
  same live state encoder; both perspectives of a game stay in one record.
- `data/`, `embeddings/`, `replays/`, `scrape/`: data preparation and collection.
  `data/ps` is a Git submodule, not ordinary project-owned source.
- `inference/`, `serve/`, `heuristics/`: ancillary inference, deployment, and
  baseline tooling. Inspect their callers before assuming they are the live path.
- `runtime/`, `ckpts/`, `jax_cache/`, `wandb/`, replay shards, and generated
  data are local artefacts. Do not commit them or expose `.env` credentials.

## Contracts that must survive changes

### Shapes, chunks, and history

- Learner shapes come from the fixed, enumerated `player_shape_lattice`, a
  chain ascending in both dimensions and ending at the full stored shape.
  All variants are precompiled at the first batch so failures happen early.
  Never introduce data-derived learner shape families: surprise late compiles
  have caused repeated OOM failures. Geometric actor inference buckets are a
  separate mechanism.
- Batch trimming is lossless. Time trimming removes only trailing terminal
  copies; history trimming requires every chunk's valid field and packed rows
  to fit. Do not silently discard real history to fit a smaller shape.
- Chunks overlap by one row. The last row is bootstrap-only unless it is the
  game's actual done row. Outcome statistics and builder losses must gate on
  terminal chunks; `win_reward[-1]` is not a real outcome in other chunks.
- Field history references packed-cache rows with absolute indices. Window
  and rebase both together through `clip_history_windows_tail`, mirroring
  service `getHistory`. Independent slicing silently corrupts entity reads.
- Align history to requests by **request-count value**, not window position.
  Preserve carry reset/rewrite behaviour when changing recurrent inference.

### Information boundaries

- Policy inputs must equal what deployment can observe. Learner-only opponent
  truth and `VALUE_CLS` may feed the privileged critic but must never influence
  policy-readable rows. `SEQUENCE_READ_MASK` enforces this transitively at
  every trunk block; conventions or a final-output mask are insufficient.
- Actor mode omits the learner-only rows. Preserve named head indices and
  actor/learner policy equivalence within the expected numerical tolerances.
- `player_privileged_targets=False` selects the deployable-head estimator.
  Keep that meaningful control. The documented privileged-target gate is
  `player_priv_value_head_r2 >= player_value_head_r2` from 20k onward; consult
  the experiment record before changing the premise or interpreting it.
- Evaluation trajectories never enter training. Gate on explicit `is_eval`,
  not usernames or actor naming conventions.
- Replay critic train/eval splits are per game, retaining both perspectives
  together. Spectator exports are public-view observations, not live private
  requests; do not treat their placeholder masks as recorded player actions.

### Precision and initialisation

- Learner parameters are stored in f32; the forward generally computes bf16
  through `cfg.dtype`. CPU actor f32 is an intentional separate configuration.
- Pass `dtype=` to layers. A missing dtype can promote against f32 parameters
  and upcast all downstream activations. Fix the layer rather than scattering
  `.astype()` repairs. Losses and value recursions intentionally use f32.
- `tests/test_dtype_policy.py` owns the allowed f32 activation exceptions.
- Zero initialisation needs a live gradient path. In the move/target bilinear,
  query is zero-initialised and key is live. Making both factors zero stalls
  learning; adding normalisation on the zero query changes its gradient scale.
- Respect the identity-at-initialisation behaviour of the RMSNorm scale and
  any measured zero-init paths. Test gradients as well as initial outputs.

### Persistence and service lifecycle

- Checkpoint components use atomic write-then-rename with writer-unique
  temporary names. Concurrent periodic/emergency writers must not collide.
  Readers must skip temporary files.
- Failed checkpoint restoration must fail visibly, never become a silent
  scratch run. A step > 0 checkpoint without league state refuses to load.
- Parameters and EMA target parameters must not alias donated buffers.
  Parameter-only restoration must seed targets too. Publish host snapshot
  copies, not live buffers the next train step will donate.
- Checkpoint metadata records provenance; current configuration handles
  schema drift without requiring retired controller fields.
- Route games to workers by game hash, never assumed globally serial reset
  pairs. Concurrent actors invalidate the pairing assumption.
- Keep `player.opponent` cross-references used in wedged-battle teardown.
  Avoid double destruction: `BattleStream._writeEnd` can call destroy again.
- Worker protobuf payloads arrive as `Uint8Array`, not necessarily `Buffer`.
  Each worker is a separate V8 isolate; coordinator heap usage is not total
  service heap usage.
- Illusion reveals may require remapping history events from a disguise slot
  to the true entity. Preserve rewrite signalling through the wire and carry.

## Testing and validation

- Choose checks for the affected contract. Do not add tests for a trivial
  documentation edit or rerun expensive suites without a reason.
- Fast suite: `env/bin/python -m pytest tests/ -m "not slow"`. For a narrow
  seam, select its test file or test names first.
- Real-model `gpu`/`slow` tests run on the GPU, **never with
  `JAX_PLATFORMS=cpu`**. The CPU backend pays a separate compile cache and
  introduces bf16 tolerance artefacts. Do not run expensive model forwards or
  train-step tests while a learner is live: host RAM pressure can trip its
  OOM guard as well as consume GPU memory. Check process/resource state first.
- `tests/conftest.py` sets environment variables before JAX imports, disables
  W&B syncing, avoids GPU preallocation, and controls test compile-cache noise.
  Do not move these imports ahead of their setup.
- Share `real_model_and_trajectory` and use `real_model_apply`. Never call
  `network.apply` eagerly in a test: use the shared jitted fixture or `jax.jit`.
  Do not initialise a full model separately per module.
- Masking and zero-init invariance tests need positive controls proving they
  could fail. A closed gate or inert initial head can make a broken test pass.
- Match tests to the seam: `test_chunking.py` and `test_history_suffix.py` for
  windowing; `test_history_carry.py` and `test_actor_carry_loop.py` for carry;
  `test_privileged_partition.py` for leaks; `test_dtype_policy.py` for precision;
  checkpoint/resume tests for persistence; `test_search.py` and transition
  tests for imagined-policy behaviour.
- Service tests run serially in a single Vitest fork because battles use
  stateful simulator singletons. Doubles/VGC alignment tests are explicitly
  skipped for a known defect; singles tests have bounded retries for the
  recorded Illusion/forme-change false-positive class. Do not remove these
  labels or claim doubles are supported based on a passing suite.
- Run `bash scripts/lint.sh` before committing code, then stage. Formatting
  after staging can leave the committed version unformatted. Report actual
  validation and any checks that could not run.

## Design and coding conventions

- Use Australian English in new code, comments, and filenames (`-ise`,
  `-isation`), while preserving external API names.
- No conditional expressions/ternaries. Use plain `if`/`else` and bind in
  the branches rather than assigning a default and immediately overwriting it.
- No single-letter names in new code. If two quantities exist only to form a
  product, name the product the implementation actually uses.
- Express one operation once, parameterised by real variation. Duplicated
  call sequences and comments saying “must mirror X” signal a missing shared
  implementation. Put identities and layout arithmetic beside their owner.
- Keep generic primitives generic; keep architecture beside its wiring.
  Do not add a configuration flag without a meaningful off mode.
- Structure-only changes must be bit-identical. A numerical change belongs
  in a separate commit with its reasoning and evidence.
- Measure a mechanism's worth before removing it; record the number and
  revert handle in `LESSONS.md`. Preserve matched controls that distinguish
  architectural defects from loss/gradient-path defects.
- Preserve the intentional E402 exceptions in `pyproject.toml`. Environment
  setup and `load_dotenv()` sometimes must execute before heavy imports.
- Commit style: terse `topic: dense one-liner — mechanism/consequence`, with
  no AI attribution. Do not commit or push unless the task authorises it.

## Research and telemetry discipline

- Before changing an update rule with a published reference (for example
  R-NaD, NashPG, or AlphaStar), enumerate discrepancies from that implementation
  first. Optimiser momentum, correction/clip ordering, and centred logits can
  be the stability mechanism. Invented additions come after the reference diff.
- Before introducing a new force on logits, answer: is its total per-cell
  force bounded after all analytic shifts; what opposes motion along the
  softmax-invariant mean direction; what does optimiser momentum do near its
  equilibrium; and which shared high-gain parameter routes carry the gradient?
  Instrument drift on those routes before launching the experiment.
- If coefficient cuts only delay the same pathological attractor, stop
  retuning scale and investigate the mechanism.
- Where available, use `docs/plan-template.md`: problem, evidence/diagnosis,
  principles, steps with changes/panels/acceptance/fallbacks, declined options,
  and reference numbers. Pre-register acceptance and a hold period.
- Add telemetry first only when the verdict requires a runtime signal that
  does not exist. Correctness fixes, precision fixes, and cleanup can proceed
  directly. Verify new panels against reference numbers before the run.
- Judge search against the same-checkpoint, same-temperature control, using
  win rate, uncertainty/sample counts, operator size, and runtime cost.
  Reconstruction metrics alone do not establish useful search.
- The latest recorded transition calibration (2026-09-06, checkpoint
  `ckpt_01800000`) put prior-expectation value-delta R² below copy: -0.057
  over 500 transitions with 32 prior samples. That record declined progressing
  to K=2/deeper search and identified the decode path for investigation. Read
  the latest ledger before reviving a rejected rung; this is historical
  evidence, not a claim about a currently running model.

## Operational inspection

- Use `env/bin/python scripts/wb.py latest`, `summary <keys...>`,
  `metric <name> [--run <ref>] [--last N]`, or
  `compare <run_a> <run_b> <keys...>`. Queries default to the main run;
  the project is `jtwin/pokemon-rl`. For unsupported queries, use the venv's
  `wandb.Api()` with server-side filtering and compact output.
- `scripts/wandb_views.py` saves dashboard views and prunes superseded
  same-name views. The existing project workflow calls for rerunning it after
  edits; account for that external mutation when scoping dashboard work.
- For offline games independent of the learner, use a second already-built
  service, for example
  `PORT=8081 MAX_WORKERS=2 MEMORY_STATS_PATH=/tmp/porygon2-offline-memory.json node service/dist/server/index.js`,
  and point the Python harness at `PS_SERVICE_URI=ws://localhost:8081`.
  Usernames must be unique per live game. Track and clean up only processes
  started for the task.
- `start.sh` affects existing training and attempts to stop running W&B jobs
  in the project before restarting tmux. It is not a harmless smoke test.
  When a training stop is in scope, Ctrl-C the learner pane for a full
  synchronous checkpoint; avoid a hard kill that loses in-flight state.
- A stuck learner registers faulthandler: `kill -USR1 <learner-pid>` dumps
  thread stacks. Use the exact identified process and bounded log excerpts.

## Known issues and knowledge lookup

- Doubles slot alignment remains recorded as broken, historically affecting
  about 75% of tested doubles battles. Doubles plumbing is incomplete behind
  `num_decision_slots`. Singles have a recorded roughly 1% Illusion/forme
  alignment false-positive class; these figures are historical measurements.
- `CLAUDE.md` records a deferred `packed_valid` species-sentinel issue: an
  unknown-species packed row can be undercounted and dropped. Inspect the
  current counting path before fixing; the intended exact alternatives are
  a valid-step `sum(NUM_RELEVANT)` or `max(RELEVANT_ENTITY_IDX)+1`. Treat a
  change in what the model receives as its own fix with a targeted test.
- `LESSONS.md` topic sections cover: 1 shapes/compilation/OOM; 2 precision;
  3 policy-loss lineage; 4 entropy/magnet; 5 targets/optimiser/LR;
  6 replay/staleness; 7 information sets; 8 checkpoints/resume; 9 league;
  10 retired controllers; 11 service invariants; 12 offline critic;
  13 architecture; 14 tooling. Dated ledgers above these sections hold the
  experiment-by-experiment measurements and revert handles.

Use `rg` to find the relevant mechanism, metric, date, or symbol, then read a
bounded window. Keep logs and test output filtered at the source; avoid full
W&B dumps, whole tmux histories, or repeatedly loading the large lesson archive.
Do not repeat broad validation when a focused check already establishes the
contract. Record durable findings in the appropriate document so future agents
do not have to rediscover them.
