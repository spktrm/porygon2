# Offline critic training

Trains the player model's `encoder` + `v_head` to predict final game outcome
from Pokemon Showdown replays (Monte-Carlo, public view), producing artifacts
the RL learner consumes directly. The intended use is a head start for early
self-play: warm-start the value pathway, or use the frozen critic as a
potential-based shaping signal.

## Pipeline

```
1. Download replays          python replays/main.py gen9randombattle -l 20000 --min-rating 1500
2. Export tensor shards      cd service && npm run offline -- gen9randombattle [--min-rating R] [--workers N]
3. Train the critic          python -m rl.offline.train [--num-steps 50000] [--debug]
4. Consume in RL             set init_params_ckpt_path + LOAD_STATE_MODE=params
```

## Stage 2 — exporter (service/src/scripts/offline.ts)

Replays each spectator log through the **same** state encoder as live
self-play (`TrainablePlayerAI` + `StateHandler`), from **both** players'
perspectives, on a `worker_threads` pool. Output shards live at
`replays/shards/{format_id}/shard-*.bin`; each record is
`[uint32-LE length][EnvironmentBatch proto]`, **one per replay**, holding
both perspectives' trajectories — so the trainer's per-record holdout
split is per game and can never leak a game's mirrored, label-flipped
twin across the train/eval boundary. A `manifest.json` records filters
and counts. Trajectories follow the RL
`Trajectory` convention: **only the terminal state carries the history
caches** (shared across all of that trajectory's steps), so records are
O(T) instead of O(T²) and the trainer consumes each full-history
trajectory in one history-encoder scan.

Spectator logs contain no `|request|` lines, so exported states differ from
live observations in these ways (all deterministic):

- `private_team` is all-unspecified and `my_moveset` is all-PAD rows —
  the encoding is **public-view only**.
- The action mask is all-ones; `REQUEST_TYPE` is always MOVE;
  `HAS_PREV_ACTION` is 0.
- States are emitted at `|turn|` boundaries plus one terminal state
  (live play emits per request, which additionally includes forced
  switches and team preview).

The outcome label rides in the final state's info features
(`WIN/LOSS/TIE_REWARD`), derived from the `|win|` line vs. the perspective
player's name — trajectories without a decided outcome are dropped.

## Stage 3 — trainer (rl/offline/train.py)

`Porygon2OfflineCritic` = the player model's recurrent **history pathway
only** (`Encoder.encode_history` → `PerSlotHistoryEncoder` →
`Encoder.pool_history`, shared module code and param paths with the RL
model) plus an offline-only **antisymmetric linear probe**: the pool runs
twice with side masks (shared params) — my-side slots + field, opponent
slots + field — and a single weight vector scores the flattened latent
difference, with logits `[-z, tie_bias, z]`. Mirror-antisymmetry
Φ(mirror(s)) = −Φ(s) therefore holds by construction; combined with
pair-aware batching (both perspectives of a game always share a batch,
see dataset.py) this makes game-identity memorization unable to reduce
the loss — only side-differenced structure can. The RL trunk reads the
same (unmasked) pooled latents as history-context tokens, so the
capacity trained offline is exactly what warm-starts RL. It reads nothing but the public
event stream — private team, own moveset, and action masks are
architecturally unreachable, and the history inputs are identical between
replay exports and live play, so the frozen Φ carries no train/serve bias
into RL. Loss is softmax cross-entropy (`[-1, 0, 1]` support) against the
final outcome at every pre-terminal step. Config is
`Porygon2OfflineConfig` (rl/offline/config.py) — composed from the same
`BaseTrainingConfig` as the RL learner config but fully independent of it.

Artifacts: `ckpts/offline/{format_id}/ckpt_{step:08}/` in the standard
checkpoint layout (`player/params` via cloudpickle) plus a `manifest.json`.

## Stage 4 — consumption by the RL pipeline

**Warm start:** set `Porygon2LearnerConfig.init_params_ckpt_path` to the
artifact directory and launch with `LOAD_STATE_MODE=params`. `merge_params`
overlays the trained history-pathway subtrees (`encoder/history_encoder`
plus the public entity/edge/field embedders it uses) onto the fresh init;
everything else — trunk, policy heads, `v_head`, and the offline-only
`outcome_head` (no RL counterpart) — keeps or stays at fresh init. To pin
warm-started subtrees during RL, add them to
`player_frozen_param_patterns` (e.g. `("encoder/history_encoder",)`) —
matching leaves get their optimizer updates zeroed.

**Learned potential:** set
`Porygon2LearnerConfig.offline_critic_ckpt_path`. The learner loads the
critic once at startup, keeps its params outside the train state (frozen by
construction — never donated, never in the optimizer), and computes
Φ(s) ∈ [-1, 1] per batch. Because the critic operates on the history
pathway only, no input projection is needed and none exists. Φ feeds the
potential advantage channel in `compute_player_targets`, gated by
`player_potential_advantage_coef_fn` (default 0 — set an annealed schedule
to use it). The hand-crafted `statePotential.ts` heuristic has been
removed; `INFO_FEATURE__STATE_POTENTIAL` stays zero for proto layout
compatibility.