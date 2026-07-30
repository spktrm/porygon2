# Offline critic training

Trains the player model's `encoder` + `v_head` to predict final game outcome
from Pokemon Showdown replays (Monte-Carlo, public view), producing artifacts
the RL learner consumes directly. The intended use is a head start for early
self-play: the frozen critic supplies a potential-based shaping signal
while the RL model itself trains fully from scratch.

## Pipeline

```
1. Download replays          python replays/main.py gen9randombattle -l 20000 --min-rating 1500
2. Export tensor shards      cd service && npm run offline -- gen9randombattle [--min-rating R] [--workers N]
3. Train the critic          python -m rl.offline.train [--num-steps 50000] [--debug]
4. Consume in RL             set offline_critic_ckpt_path + potential coef schedule
```

To eyeball what a trained Φ says about one game,
`python -m rl.offline.visualize <replay json | replay id | replay URL>`
writes a standalone HTML page: the rendered battle (official Showdown
replay player) next to per-turn Φ — ensemble members, mean ± std, the
uncertainty-gated potential, a mirror-antisymmetry check, and the 13-bin
margin distribution — with click-to-seek sync between chart and replay.
It encodes the log through the same exporter as the shards
(service/src/scripts/exportReplay.ts), so the states scored are exactly
the training-time states.

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
capacity stays in the critic's own history pathway. It reads nothing but the public
event stream — private team, own moveset, and action masks are
architecturally unreachable, and the history inputs are identical between
replay exports and live play, so the frozen Φ carries no train/serve bias
into RL. Loss is softmax cross-entropy over **13 margin bins** (final
alive-mon differential in [-6, +6], sign-clamped to the recorded result —
forfeits keep the true winner) at every valid step, so Φ = expected
margin ∈ [-1, 1] grades positions by decisiveness instead of only
win-probability. Forfeits get special handling (measured on 50k rated
games: ~48% played out, ~41% conceded, ~11% forfeited with the winner not
ahead on mons): games where the sign-clamp engages are **dropped**
(`drop_clamped_forfeits` — the result contradicts the position, and the
noise is perspective-consistent and side-differenced, exactly what the
antisymmetric probe would otherwise learn), and concessions are treated
as **right-censored margins** (`concession_censor_decay` — label mass
decays geometrically from the concession margin up to the winner's
alive-mon count, the hardest margin any played-out continuation could
have reached, countering the compression of conceded margins toward ±1..3
relative to played-out games). Both apply at label-construction time in dataset.py, so changing
them needs no shard re-export. For uncertainty-gated shaping, train an ensemble:
`--ensemble` trains all `num_ensemble_splits` members **simultaneously**
in one process — stacked params/optimizer with a vmapped member step
(pure parallelism, gradients never mix across members), one shard pass
routing each game to its member, and shared-holdout evals that log live
gate diagnostics (per-member metrics side by side, member std, gated
sign accuracy). Artifacts stay per-member (`{format_id}-ensk/`), so
consumption is unchanged. `--ensemble-index k` (k = 0..K-1) still trains
a single member on the identical split (same salted hash) — use it to
retrain one bad member without touching the others. Config is
`Porygon2OfflineConfig` (rl/offline/config.py) — composed from the same
`BaseTrainingConfig` as the RL learner config but fully independent of it.

Artifacts: `ckpts/offline/{format_id}/ckpt_{step:08}/` in the standard
checkpoint layout (`player/params` via cloudpickle) plus a `manifest.json`.

## Announced states (Φ_ann) — skill/luck decomposition

Each turn additionally gets an **announced state**: both players' choices
visible, chance unresolved. It is built by MASKING, not new features — the
packed edge cache keeps only `MAJOR_ARG`, `MOVE_TOKEN`, `ENTITY_IDX`
(everything else → `_UNSPECIFIED`; edges whose major arg is
`cant`/`faint`/`drag`/`replace` are outcome events and drop entirely; see
`mask_outcome_features` in rl/model/history_encoder.py), node snapshots
stay at pre-turn values, and the pre-turn recurrent state is advanced
**one** extra step with the masked messages (`announced_states_at_requests`
— no second scan). Φ_ann runs through the same antisymmetric readout
(mirror antisymmetry is automatic, zero new parameters) and trains as an
extra supervision point with the same margin label
(`announced_loss_weight`; the manifest's `announced_states` flag is the
capability marker, since the param tree can't be). Trained
Φ_ann = E[outcome | history, announced actions], which buys:

1. **Replay analysis:** per turn, decision = Φ_ann(t+1) − Φ(t) and
   dice = Φ(t+1) − Φ_ann(t+1) (damage rolls included) — the visualiser's
   momentum lane and key moments show the split, with the 🎲 log-event
   tags as labels on the quantified dice term.
2. **Dice-excised PBRS (later, learner-side):** shaped term
   γ·Φ_ann(t+1) − Φ(t) — same conditional expectation as standard PBRS
   (unbiased by the tower property), strictly lower variance: the shaping
   channel stops paying the agent for crits. Not yet wired into the
   learner — gated on a validated announced-state critic run.

Verify with `python -m rl.offline.announced_leak <replay>`: perturbs one
turn's outcome features (crit bits, damage/heal ratios, post-event hp
snapshots) and asserts Φ_ann for that turn is bit-invariant while the
realised Φ moves.

## Stage 4 — consumption by the RL pipeline

The offline critic is a **standalone model** — it shares Encoder code with
the RL model, but its trained params never enter the RL network, and the
RL model trains fully from scratch (no frozen or warm-started subtrees).
The single consumption mode is the **learned potential**: set
`Porygon2LearnerConfig.offline_critic_ckpt_path` — a single path, or a
tuple of ensemble-member paths for uncertainty gating
(Φ = mean · exp(−`potential_uncertainty_scale` · std): shaping speaks
where members agree and goes quiet off the human data distribution). The
learner loads the critic(s) once at startup, keeps params outside the
train state (frozen by construction — never donated, never in the
optimizer, stop-gradient at use), and evaluates Φ(s) ∈ [-1, 1] **once per
trajectory** as it enters the replay buffer (`Learner.enqueue_traj`),
storing the gated scalar per step on the trajectory — the frozen critic
makes Φ immutable data, so recomputing it inside the train step would
redo identical work replay_ratio × ensemble-size times, and caching
decouples learner step time from ensemble size entirely. Because the critic operates
on the history pathway only, no input projection is needed and none
exists. Φ feeds the potential advantage channel in
`compute_player_targets`, gated by `player_potential_target_adv_share_fn`
(a target-share schedule; the coefficient is solved per batch). The hand-crafted
`statePotential.ts` heuristic has been removed;
`INFO_FEATURE__STATE_POTENTIAL` stays zero for proto layout
compatibility.