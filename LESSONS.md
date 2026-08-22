# Lessons

Paid-for knowledge from this project: what was tried, what was measured, what
the verdict was. Most of it was previously recorded only in comments attached to
the code that produced it, which made it invisible the moment that code was
deleted.

Two things live here:

- **The removal ledger** — every mechanism deleted in the 2026-08-21 cleanup
  pass, with the command that brings it back.
- **The lessons themselves**, grouped by topic. Entries marked *(live)* describe
  code still in the tree; the rest describe code that is gone.

`docs/` holds the long-form design documents. It is gitignored and local to the
training box — never cite it as a public reference, and do not assume a fresh
clone has it.

---

## Removal ledger — 2026-08-21 cleanup pass

Everything below existed at tag **`pre-cleanup-2026-08-21`** (commit `e882474`).
To inspect or restore any of it:

```
git show pre-cleanup-2026-08-21:rl/online/plasticity.py        # read it
git checkout pre-cleanup-2026-08-21 -- rl/model/search.py      # bring it back
git revert <removing sha>                                      # undo one commit
```

| mechanism | paths / symbols deleted | removed in | why it went |
|---|---|---|---|
| Exploiter populations (MainExploiter / LeagueExploiter) | `learner.py` population dispatch, `_fork_population`, `_begin_exploiter_block`, `_check_exploiter_transitions`, `_check_promotion_bar`, frame-budget + `_STEP_OFFSET` tables; `player_actor.py` exploiter matchmaking; `league.py` origin split; 13 config fields; `populations/` + `scheduler` checkpoint layout | `b219d84` | `auto_exploiter_enabled=False` since 2026-08-19; three populations do not fit 12GB on this box (OOM at the first `league_exploiter` block, 2026-08-15) |
| Plasticity controller (shrink-and-perturb) | `rl/online/plasticity.py`, `learner._update_plasticity`, `_apply_plasticity_update`, controller checkpoint state, 6 config fields, `tests/test_plasticity.py` | `b219d84` | fired rarely and expensively; the Aug-2026 firing hit a consolidation phase and cost a multi-10k-step recovery |
| Single-action policy gradient | `learner.loss_pg`, `loss.py::ppo_objective`, the advantage EMA normaliser + std floor, `ema_adv_mean`/`ema_adv_std` on the TrainState, 6 config fields (`spo_objective` survives — the builder still uses it) | `4234016` | updates only the sampled action's path; replaced by all-action NeuRD |
| UPGO | `targets.py::upgo_returns`, `loss_upgo`, `RuntimeScalars.upgo_coef`, `player_upgo_coef` | `4234016` | same single-action objection; its optimistic credit had no all-action form |
| Q-boost cross-fade | `RuntimeScalars.q_boost_mix`, the PG-advantage blend, the `q_taken` / `retrace_g − v_exp` variant switch, the boost-vs-vtrace agreement diagnostics, 3 config fields | `4234016` | its only consumer was `loss_pg` |
| Stage-A root search | `rl/model/search.py` | `65a4774` | zero callers ever; the `act_search` orchestration it names was never written |
| Attention visualiser | `rl/model/viz.py` | `65a4774` | superseded by `scripts/attn_probe.py`, which is maintained |
| PriorityLock | `rl/concurrency/` | `65a4774` | referenced only inside its own file since 2025-09 |
| Offline critic analysis tools | `rl/offline/{announced_leak,baseline,causality,diagnose,visualise}.py` | `65a4774` | unimported `__main__` scripts; the offline trainer they analysed stays |
| Dead helpers + config fields | 16 never-called symbols, `artifact.save_train_state` + the cloud-upload path, `gradient_accumulation_steps`, 4 unread config fields — full list in the commit body | `65a4774` | no callers / no readers |
| p_q observer | `player_q_observer_tau`, the Boltzmann-over-Q̄_private readout and its `player_q_improve_*` metrics | `98f0873` | it was a *leading* indicator from the stage-2 era, when the Q head did not drive the policy. NeuRD now reads Q_all directly, so "what does the critic want vs what does π do" is the loss itself, not an early warning about it |
| Multi-lambda aux value heads | `aux_v_head` + `MultiLambdaValueLogitHead`, `compute_aux_value_targets`, `loss_v_aux`, `player_aux_lambdas`/`player_aux_value_coef`, the per-lambda R2 panels, and `player_bootstrap_gap` (it read the λ=1.0 MC-anchor row) | `98f0873` | representation shaping the critic stack no longer needs; **note it takes the bootstrap-bias instrument with it** — nothing now measures the main head against a Monte Carlo anchor |
| BT-rating telemetry | `rl/online/ratings.py`, `bandit_window_steps`/`bandit_min_games_per_opponent`/`bandit_min_rated_opponents`, the BT-fit auditor panels | `98f0873` | a rating needs hundreds of games per point, so it was never fast enough to act on — an auditor that outlived the controllers it audited |
| Always-true feature flags | `player_neurd_enabled`, `player_q_diagnostic_enabled` | `98f0873` | neither had a meaningful "off": one gated the sole policy gradient, the other gated loss-free logging |
| `RuntimeScalars` pytree | the class, the `scalars` arg on `train_step`, both construction sites | `1fd210c` | it carried `magnet_coef`/`neurd_coef` as traced leaves so a host controller could vary them without recompiling; nothing varied them any more. **Reintroduce it — do not widen static config — the moment a coefficient changes during a run** (LESSONS.md 1) |

---

## Removal ledger — 2026-08-22 R-NaD pass

Everything below existed at tag **`pre-rnad-2026-08-22`** (commit `6f845d2`). Same
restore recipe as above. The through-line: three exploration mechanisms (magnet
KL, epsilon ladder, critic-disagreement — as reward and as UCB selection) all
collapsed switching at ~13k, and the critic diagnosis said why — it was correctly
learning Q^π of a policy under which switching IS worse, so sampling switches more
only confirmed it. What the search-free self-play successes (DeepNash) did instead
is a reward transform against a MOVING reference policy that enters the values.

| mechanism | paths / symbols deleted | removed in | why it went |
|---|---|---|---|
| Q-ensemble UCB behaviour policy | `EnsembleGridPrior`, `ucb_tilt`, `HeadParams.ucb_c`, `q_ens_*` heads/losses/panels, InferenceServer standing head_params, `tests/test_intrinsic_reward.py::ucb`, ladder contract test | `47719b6` | σ_epi on switches tracked σ_epi on moves step-for-step through a 49%→9% switching collapse; KL(μ‖π) never left 1.5% of its cap. Shared-trunk ensembles measure head-init noise, not coverage (Kirsch 2024) |
| Critic-ensemble intrinsic reward | `EnsembleValueLogitHead`, `v_ens_head`/`v_int_head`, `compute_intrinsic_targets`, `IntrinsicTargets`, `int_rms` state leaf, 8 config fields, panels, `tests/test_intrinsic_reward.py` | `114ff5c` | int_reward switch/move pinned at 1.0 as an observer — disagreement never concentrated on post-switch states; Chen 2017 already rated disagreement-as-reward below UCB selection |
| Magnet KL | `loss_magnet_kl`, `magnet_log_policy`, `player_magnet_kl_coef` (+ its 50-line tuning history), `player_loss_magnet_kl` panel | `4254a1f` | gradient-side KL(π‖prior) carries a π prefactor (docs/entropy-gradient-pressure.md) — cannot refill a dead modality; its 0.05→0.1→0.2→0.05→0.1 ladder was compensation for a supply problem it could not fix |
| Epsilon explore ladder | `HeadParams.mix`, `behaviour_log_policy`, `Trajectory.explore`, `own_rows` gating (league cadence, builder, replay controller, the `_own` KL variant), `explore_game_prob`/`explore_eps_range`, per-game explore Agent path, `player_q_explore_*` panels, `tests/test_behaviour_mix.py` | `92b06c4` | random switches lose (explore-row post-switch return ~0.3 below post-move for the whole collapse), so behaviour-side coverage taught the critic switching is bad faster; R-NaD's penalty is the exploration mechanism and every game plays π |

## 1. Shapes, compilation, OOM

**Learner batch bucketing killed three runs.** The geometric bucket family
compiled a separate variant per data-derived shape, each with its own workspace.
The first top-bucket batch arrived ~20 minutes into a session as a *surprise*
compile: run 1786537634, the 2026-08-15 03:26 run and the 2026-08-15 23:33 run
all died there. Replaced by `player_shape_lattice`, a small enumerated ascending
chain of `(chunk_rows, history_rows)` combos, every one precompiled at the first
batch — so an OOM lands at launch, not mid-run. Trimming to the lattice must
stay lossless (T: trailing terminal-copy padding only; H: only when valid steps
and packed rows both fit).

Sizing came from a 2026-08-20 measurement at `batch_size=4`: batch-max chunk
fill mean ~42 of 64, history fill mean ~85 of 256. Retune from the
`player_shape_T/H` logs.

**Do not confuse the two bucketings.** The geometric bucket helpers in
`rl/environment/utils.py` (`geometric_bucket`, `clip_history`,
`clip_packed_history`) are the *actor/inference-path* shape reducers and are
live. The retired mechanism was the *learner batch* family, now
`_chunk_required_shape` / `_trim_to_lattice`.

**Bucket combinations multiply.** If a jitted function's batch depends on
multiple independently-bucketed fields, the number of distinct shape
combinations XLA sees is the product across fields, not the sum. *(live)*

**A scalar that VARIES during a run must never live in static config.**
`config` is a jit `static_argname`, so each distinct value is a separate
compiled executable; retaining them cost ~5GB and OOM-killed run 1326. This is
why a `RuntimeScalars` pytree carried the host-varied coefficients as traced
leaves.

That class was deleted 2026-08-21 — every mechanism that varied a coefficient
(the coef ramps, the exploiter zeroing, and every controller) had itself been
removed, so it was boxing two constants read straight off config every step.
**The rule did not go with it**: the price of the move is that retuning
`player_magnet_kl_coef` or `player_neurd_coef` now costs one `train_step`
recompile at the next launch, and the moment anything varies a coefficient
*during* a run — the magnet PI controller is the documented candidate — it needs
a traced pytree argument again. Widening the static config instead is the run-1326
failure.

**Remat policy.** The encoder is rematted with `nothing_saveable`, not the house
`checkpoint_dots` — the latter saves the very matmul outputs that blow up, and
storing them for the backward pass OOMs the train step.

**Attention width is the VRAM dial.** Player model fwd+bwd at T=64, compiled
temp size (2026-08-20): entity-local baseline 182.5MB, cross-entity pool at 2
heads 202.8MB (+11%), at 4 heads 240.9MB (+32%).

**Inference params cache.** Sizing below the actor working set causes LRU
thrash: a serial ~81MB host→device transfer per miss inside the server thread,
plus alloc/free churn in XLA's pool — the fragmentation class that killed
session 1786537634.

**Host RAM.** Session 1786537634's RSS climbed 5.9→17GB (threads 478→775) with
no way to attribute it from wandb alone; that is why the memory diagnostics and
heap census exist. The OOM guard deliberately does not try to continue in the
same process: freeing Python objects does not guarantee the OS reclaims the
memory, so a fresh process is what actually recovers. (It was added after 1361
crashed, though that crash turned out to be a websocket failure to the game
service, not RAM exhaustion.)

**Postmortem hygiene.** Use `logger.exception`, never `traceback.print_exc()` —
the latter writes raw to stderr and gets shredded line-by-line by concurrent
tqdm redraws (session 1786537634's OOM traceback was nearly unreadable). And do
not let an exception fly past an unconditional `wandb.finish()`: that made the
same crash show as three cleanly-"finished" runs and sent the postmortem down
the wrong path.

## 2. Precision

- **f32 for value recursions.** bf16 values with f32 python-scalar-promoted
  discounts made the scan carry dtype disagree and crashed the 2026-08-13
  session (fixed in 15b6a3f). The recursion must run *and return* f32.
  `tests/test_targets.py` keeps the regression.
- **bf16 log_softmax normalisation holds only to ~3e-3** — size test tolerances
  accordingly.
- **`-inf * 0` poisons a vjp.** Padded steps carry all-zero targets and zero
  weight; use a finite floor (`-1e9`) for masked logits, never `-inf`. *(live)*
- **One NaN batch poisons an EMA forever.** `mean`/`std` with `where=` go NaN on
  an all-masked batch (every row forced single-option or terminal — rare, but it
  happens), so any running statistic must be frozen on such batches rather than
  updated. The advantage-EMA normaliser this was written for is gone, but the
  constraint applies to the next running statistic anyone adds.
- **The non-finite update gate is checkpoint protection, not just numerics.** A
  poisoned update is permanent, and the next periodic save then overwrites the
  last good checkpoint with it. *(live)*

## 3. Policy loss lineage

**COMA could never have been the restorer.** The COMA loss
`-Σ_a π(a)·sg(adv(a))` has exact per-logit gradient `-π(b)·adv(b)` — the
`Σ_a π·adv` correction vanishes identically under the COMA baseline
(finite-differenced to 4e-12). That is NeuRD eq. (6): counterfactual regret
scaled by the action's own probability, so a starved switch cell gets a
restoring force proportional to how starved it already is.

**The measurement that settled it.** The 157k-step 2026-08-20 lineage measured
`absadv_ratio ~4` against `prob_ratio ~0.075`: the critic preferred switch cells
*more* than move cells, and π alone was throttling the update. Decision rule
baked into the dashboard: `grad ≈ prob` with `absadv ≈ 1` means the prefactor is
the throttle (NeuRD indicated); `absadv ≈ 0` means the critic has no switch
belief to amplify and NeuRD would amplify noise instead.

**Caveat that still applies.** `loss_q` supervises only the *taken* cell, so
untaken switch cells can be untrained rather than genuinely flat. Read
`absadv_ratio` against `player_q_switch_target_frac` before concluding the
critic "means it".

**NeuRD must be written against raw logits.** *(live)* The `log_policy` form was
tried first and failed the identity test: once the logit-gap clip zeroes cells,
the weights are no longer zero-sum and the softmax pulls in a `π(b)·Σ_a w(a)`
cross-term. Against raw logits, `d/dy_b = -w(b)` exactly, open or clipped.

**Clip and coefficient.** Advantages are not zero-mean per row, so unclipped
logits diverge; β = 2.0 is OpenSpiel's NeuRD default. The coefficient went
0.05 → 0.1 → 1.0 across 2026-08-21: 0.05 was sized as ~1% relative pressure
beside a coef-1.0 PG term; 0.1 followed from dropping π (~0.1 over ~10 legal
cells makes the raw per-cell gradient ~10x COMA's); 1.0 is what it inherits on
becoming the policy learning rate outright. Watch the scale honestly — the PG
advantages it replaced were EMA-normalised to ~unit std, while these are raw
win units (~0.1-0.2 spread), so at 1.0 a starved switch cell at adv +0.15 gets
~0.15/logit, about an order above the magnet's pull. The magnet is now the only
opposing force, so an entropy cliff or `player_neurd_clipped_switch` pinned at 1
is the signal to back off.

**Naming (2026-08-21).** Everything `player_coma_*` was renamed
`player_neurd_*` once the π-prefactor branch went — COMA proper no longer
exists in the code, only in this history. That breaks wandb metric continuity
with earlier lineages by design: the objective changed, so a chart that spans
the rename would be comparing two different losses.

**Why single-action PG was removed (2026-08-21).** Two independent reasons.
First, by then `q_boost_mix` was a hard 1.0, so `loss_pg`'s advantage was
already 100% `retrace_g − v_exp` — the v-trace advantage channel contributed
nothing, and the term was a Q-driven update wearing a PG costume. Second, and
decisive: a sampled-action objective structurally carries no information about
the action *not* taken — on a move row it says nothing about the switch that was
declined, which is exactly the axis that collapses here. NeuRD's all-action form
lands counterfactual pressure on every legal cell of every real-choice row. The
cost of the change is that the policy's only link to returns now runs through
`Q_all`, whose supervision coverage is the caveat above.

## 4. Entropy and the magnet

Four retunes, each with its own collapse:

| coef | when | evidence |
|---|---|---|
| 0.01 | baseline | lost the arm-wrestle: chocolate-silence-1307 collapsed to normalised entropy 0.27 (modality 0.17) by 190k while magnet KL climbed to 1.44, and eval strength regressed from its 56k peak |
| 0.05 | — | still insufficient |
| 0.1 | 2026-08-17 | the entropy-regularisation timeline showed the longest stable lineages (Oct–Nov 2025, 400–580k steps, 2.0–2.9 nats lifetime) ran ~2.8–5.4x today's effective pressure |
| 0.2 | 2026-08-19 | the q-boost lineage collapsed at ~3x baseline speed (modality entropy 0.87→0.27 by 18k, switch_ratio 0.45→0.02) with 0.1 holding nothing |

**The structural limit.** Reverse-KL force is π-weighted and cannot hold a floor
once a modality is starved (`docs/entropy-gradient-pressure.md` §3). No amount
of magnet coefficient fixes that; it buys time, not a floor. If a static coef
cannot hold the band, the proper fix is a PI controller on the coef with a
target-entropy schedule — not another manual bump.

**The magnet is deliberately stationary.** *(live)* A fixed anchor is what gives
regularised self-play a stable fixed point (QRE); an EMA magnet chases the
policy and degenerates into a short-horizon trust region.

**There is no automated backstop.** The AdaptivityController was removed
2026-08-13 (hard to tune, harder to predict). Modality collapse — 1330 died at
0.08 on that axis; 1328 *gained* strength at 0.18–0.26 — is watched on the
dashboard, not auto-corrected.

## 5. Targets, optimiser, learning rate

- **λ 0.99 → 0.8.** AlphaStar's own choice is TD(λ=0.8), but they could afford
  heavy bootstrapping because supervised init gave them a sane critic from step
  one. This project starts from scratch, and the 1328 five-arm sweep pointed the
  same direction (monotone lower-λ-better, confounded but directional).
- **1e-4 learning rate collapses the trunk.** Trial zany-leaf-1305 (Aug 2026):
  pre-clip grad norms 10–100x the clip, action-embedding srank at 0.27 by 13k
  steps versus 0.82 at 3e-5 — *while actor-KL sat quietly at 0.002*. **KL
  headroom is not evidence the LR can rise.** This is why the srank / dormant
  probe survives the removal of the plasticity controller that used to consume
  it.
- **Momentum.** Momentum-free Adam left all three guardrails idle (actor-KL
  0.013–0.044, grad norm 1–4 against a clip of 10).
- **Aux λ-spectrum pruning.** λ=0.2 was near-pure next-step self-distillation
  and its R² correlated 0.984 with λ=0.5's over 223k steps (run
  1786583261-main); λ=0.8 became a copy of the main head, not a horizon, once
  the main target *was* λ=0.8 v-trace at the same γ. Keep the aux coefficient
  modest — the grad-norm lesson from the integrated-critic era is that heavy aux
  gradient globally clips everything.
- **Retrace details.** *(live)* Bootstrap the last acted step on `r`, not on the
  Q readout's uncalibrated terminal estimate, so the outcome enters the
  recursion exactly once. The trace factor shifts left (`c_{t+1}`); the final
  step has no continuation to correct.
- **UPGO's asymmetry was the mechanism.** While it existed it shared the std
  divisor but was deliberately *not* mean-recentred — its positive skew (extra
  credit along better-than-expected lines) was the point, not a normalisation
  artefact.
- **Q-boost rationale, for the record.** Fan & Farina (arXiv 2605.19235): the
  boosted advantage is unbiased at λ=1 for *any* critic accuracy (Thm 3.1) and
  lower-MSE than the GAE family exactly where the policy must keep randomising
  (`Var_a[Q] > 0`) — the mixed-strategy stay/switch states the collapse forms
  in. Read its headroom on the p90 of the action-value spread, not the mean: the
  mean undersells Thm 3.1 headroom by construction when spread concentrates in
  few high-leverage states. Deleted with `loss_pg`, its only consumer.

## 6. Replay, staleness, exploration

- **Buffer capacity, not ratio, drove a strength plateau** (2048→256 chunks).
  The reuse controller is deliberately one-sided: it may cut reuse below
  nominal, never raise it. The KL target (0.045) is a pathology threshold, not
  a desirable operating point — staler data per learner step is never a win
  under a strength-per-step objective.
- **`main_player_update_steps` 10 → 50.** At 10 (~6s of training) main alone
  kept 5–10 parameter versions live at once; 50 (~30s) collapses that to ~2, and
  measured actor-KL is 0.005–0.006 against the 0.045 target.
- **`add_player_max_frames`.** At 3e6 (~11.5k steps) it filled the league with
  ~0.5-winrate near-copies of main — mirror play with extra staleness — and made
  the stagnation clock hair-trigger.
- **Mirror-only self-play does not transfer.** Mirror-only runs measured 93% vs
  Random but ~10% vs SimpleHeuristic at 163k steps: the signature of a policy
  that exploits itself and nothing else. This is what the league minimums exist
  to prevent.
- **Temperature → epsilon (2026-08-21).** A tempered collapsed policy is still
  collapsed, so the switch samples the ladder supplied shrank along with the
  collapse it was meant to counter (voluntary-switch supervision coverage
  tracked it down, 4.5% → 3.4%). An epsilon mix with the hierarchical prior has
  a floor independent of collapse depth. Supply arithmetic: `explore_game_prob ×
  E[eps] × ½ ≈ 4.7%` forced-switch rows on top of the ~3% a collapsed policy
  supplies itself. Keep eps bounded — v-trace's ρ̄ truncation bites at
  `π/μ > 1/(1-eps)` — and watch `player_isr_ess`; below ~0.9, lower the top of
  the range.
- **The importance correction is correct and still hurts.** With μ above π on
  switch rows, ISR sits below 1 there: the learner hears the collapse-
  contradicting evidence ever more faintly as the collapse deepens. Every
  individual update is properly weighted; the loop is still self-reinforcing.
- **A dedicated-actor exploration slot backfired.** Two of twelve actors
  bypassing the inference server full-time out-produced the server-queued base
  pairs ~4x, inflating an intended ~17% row share to ~44% and halving the
  effective PG/value batch.
- **Cross-population intake was removed 2026-08-15**: it conflated another
  agent's policy evidence with main's own action values, and its frozen-between-
  blocks stock went stale — foreign-row Q R² 0.27 against 0.84 for own rows.

## 7. Privileged information

*(all live — these are contracts, not history)*

- `opp_private_team` is the opponent's match-start sheet, frozen at *their*
  first request and all-zero at deploy time. It feeds only the everything-value
  readout; state, action and policy-facing streams never see it.
- The `public` value rung reads the raw pre-trunk history embeddings, not the
  state stream's history slice, because its information set must stay purely
  public-historical.
- The cross-entity attribute pool covers both sides' public rows and my own
  private sheet — exactly the player's own information set — and excludes the
  opponent sheet.
- **Two test traps, both learned the hard way.** Residual gates are zero-init,
  so a leak test at init multiplies any leaked contribution by zero and passes
  vacuously — open the gates first (`open_zero_init_paths` in `conftest.py`),
  and include a negative control proving the perturbed entity *does* respond.
  And invariance alone is one-sided: a Φ_ann that never reads the turn at all
  passes an invariance check perfectly.
- The `all == private` equality on an empty sheet no longer holds now that the
  rungs have separate query inits and residual gates; that equality test was
  removed deliberately, not lost.

## 8. Checkpoints and resume

- **Atomic writes need unique tmp names.** The periodic checkpoint worker and
  the OOM guard's emergency save can race on the same step directory (observed
  2026-08-14 at `ckpt_00020000`); with a shared `<path>.tmp` the loser's
  `os.replace` finds its tmp already consumed and crashes the save it was
  supposed to guarantee.
- **Never fail soft to a scratch start.** A bare `print()` on a failed restore
  is how run 1335's ~300k-step lineage and its league were lost between 1335 and
  1336: mode was "checkpoint", the load failed, and it silently became a fresh
  run.
- **Donation aliasing.** `params` and `target_params` must not share buffers, or
  donating the train state to the jitted step fails with a duplicate-donation
  error on the first step. Same reason the league hands out host copies:
  handing out live buffers has actors running inference on memory the donated
  train step deletes.
- **Params-mode resume must also seed `target_params`** — leaving it at fresh
  init hands v-trace a garbage reference policy for ~1/ema_rate steps.
- **Config schema drift must not fail a healthy resume.** Checkpoint meta is
  provenance only; sections written by since-removed controllers are simply
  never read.
- **A collapse probe on an old checkpoint is a contaminated reading** if the
  checkpoint predates a module the probe initialises — skip, don't report.

## 9. League and exploiters *(deleted mechanism)*

The three-population design (Main / MainExploiter / LeagueExploiter) followed
AlphaStar closely: PFSP with squared weighting, a 50% PFSP / 15% verification /
35% self-play match split, `MainExploiter.ready_to_checkpoint`'s explicit
minimum-dwell floor, and `LeagueExploiter.checkpoint()`'s probabilistic reset —
adapted here to shrink-and-perturb rather than discarding everything learned.

What it taught:

- **A 0.55 promotion bar is not a signal.** Standard error at n=20 games and
  p≈0.5 is ~0.11, so 0.55 is under half a standard error above a coin flip. 0.7
  (~1.8 SE) is a real signal. Retuned before the mechanism was ever switched on.
- **Freshly-added snapshots read near 0.5 by construction**, which looks exactly
  like a genuine exploitability hole. Run 1338 flagged two snapshots 5.5k and
  26.9k steps old whose win-rate never left 0.48–0.54 — a false positive from
  precisely this, and the reason a reliability floor exists.
- **It never ran here.** Three populations do not fit 12GB: the 2026-08-15 OOM
  was a 2.22GiB contiguous allocation failing at the first `league_exploiter`
  block against large free-but-fragmented regions.
- **Standing rule:** populations strengthen an already-good policy; they are
  never a crutch for a weak one. Re-enable on demonstrated strength (clear the
  0.29 prior peak), not on a step count.

The single-population league that remains — PFSP, snapshots, payoff table, the
15% verification slice, the checkpoint-pacing gate — is untouched and live.

## 10. Retired controllers

Every automatic controller this project has built has been removed. The pattern
is worth remembering before building another one.

- **LambdaBandit** (retired 2026-08-14) — paid an exploration tax (it must
  sometimes hold an arm it suspects is worse, to keep the uncertainty estimate
  honest) *on top of* the rating signal's own latency of hundreds of games per
  point. Slower to react than either replacement. The `bandit_` metric prefix
  survives for wandb continuity across lineages; ratings remain an auditor,
  never a control signal.
- **AdaptivityController** (removed 2026-08-13) — the commitment-covariance PI
  caused three separate bugs: unreachable-target pinning pressure at the ceiling
  in 1338/1339, a divide-by-near-zero in 1341, and an exploit_ctrl
  target-scaling bug. Its stacked event bumps held main at ~6x baseline pressure
  for a whole run.
- **ExploitabilityController** (removed 2026-08-14) — built to scale three other
  controllers' targets, it outlived all three; by the end its only action was a
  bounded nudge on the replay KL target. AlphaStar has no analogue.
- **LambdaGapController** — its one genuinely useful behaviour, forcing pure
  Monte Carlo while a freshly-perturbed critic is untrustworthy, survived as
  `upgo_coef = 0` during plasticity recovery, and died with both of them.
- **PlasticityController** (removed 2026-08-21) — shrink-and-perturb (Ash &
  Adams, arXiv 1910.08475) triggered by consecutive overdue-only league adds.
  The detection was genuinely bias-free: it read only *how* snapshots got added,
  never a hand-specified pathology. Two data points pull opposite ways, and both
  belong on the record: the Aug-2026 firing landed during a consolidation phase,
  dropped the agent below its own 50k-step snapshot (winrate 0.485) and cost a
  multi-10k-step recovery tax; but a perturbation around 179k is the one event
  that has been *observed* to revive collapsed switch mass. It was removed as
  rarely-fired machinery with a large blast radius, not as a disproven idea. The
  dormant-fraction / srank probe it consumed is kept as a pure observer.

**The through-line:** controllers here have consistently been harder to tune
than the thing they controlled, and their failures were silent and slow. Prefer
a fixed value plus a dashboard panel until a specific pathology proves a fixed
value cannot work.

## 11. Service invariants *(live)*

- **Game routing must be hash-based, not pair-based.** The "resets arrive in
  strict globally-serialised pairs" assumption that concurrent self-play threads
  never guarantee produced two sides of one game on different workers, each
  waiting forever — a silent, un-erroring hang indistinguishable from a real
  deadlock.
- **`BattleStream._writeEnd` re-runs `battle.destroy()`**, which throws on a
  second call; the rejection killed workers as an unhandled `error` (the
  2026-08-13 service crash).
- **`postMessage()` delivers protobuf payloads as plain `Uint8Array`, not Node
  `Buffer`.** `Buffer.isBuffer()` is false for those, which silently dropped
  every reset/step request — neither branch matched, no log, no error.
- **Each worker is its own V8 isolate**, so `process.memoryUsage()` from the
  coordinator only ever sees the coordinator's heap.
- **History windows are named by absolute index.** The service truncates from
  the front and *rebases* the `RELEVANT_ENTITY_IDX*` columns; a row left at the
  buffer default of 0 is silently scattered into slot 0. Never slice the field
  and packed axes independently — use `clip_history_windows_tail`, which mirrors
  `getHistory`.
- **Illusion reveals remap slots**: events since the disguised position's
  switch-in were keyed to the disguise and must be moved onto the true
  Pokémon's slot. The slot-alignment assert has a ~1% false-positive class from
  this and from forme changes (hence `retry: 2`); three independent failures
  (~1e-6 by chance) still fail the suite.
- **Doubles violate slot alignment in ~75% of battles** (622 hits over one
  ~3200-battle soak) — a pre-existing defect, skipped and labelled, not fixed.
- **Write-then-rename everywhere a Python process may read concurrently**:
  rename is atomic, so existence implies completeness.
- **Evaluation trajectories must never become training data.** The guard is
  gated on the explicit `is_eval` flag, not on actor naming, so adding or
  renaming an actor cannot silently leak eval games into training.

## 12. Offline critic *(program kept; its analysis tools deleted)*

- **Label noise is structured, not random.** Measured on 50k rated
  gen9randombattle games (July 2026): ~48% played out, ~41% conceded with the
  winner ahead, ~11% forfeited with the winner *not* ahead on mons. That last
  slice is perspective-consistent, side-differenced noise — exactly the shape
  the antisymmetric probe is built to learn.
- **Mirrored perspectives force `Φ(mirror) = −Φ`**, and the pooled unconditioned
  mean is therefore exactly zero by construction — only conditional slices test
  anything.
- **Degeneracy canary:** the masked std of the expected margin. A model
  collapsed to a constant shows ~0 there while accuracy happily tracks batch
  label composition.
- **Lower bound:** a healthy trained critic must match or beat the hand rule in
  every phase bucket. Below it late-game means a broken learned pathway, not a
  hard task.
- **Overfit-one-batch first:** 300 steps on the same batch must drive the margin
  loss to its label-entropy floor. If it does, any plateau on the full dataset is
  a capacity/data/schedule question, not a bug.
- **PBRS / potential shaping is retired** (Aug 2026) — the shaped-advantage era
  ended and `offline_critic_ckpt_path` no longer exists in the online config. The
  offline critic remains a standalone research program with its own entrypoint
  and its own wandb project.
- **Announced-state distillation was never paid for.** Measured 2026-07-30, the
  announced-movement ratio was ~0.15, so SGD never builds the
  announcement→consequence circuit on its own. The distillation KL was logged at
  weight 0 deliberately: measure the gap on a run before paying to close it.

## 13. Architecture notes worth keeping *(live)*

- **Deep, modality-separated action decoders are empirically necessary.** Make
  that depth cheaper; do not remove it.
- **Flat-at-init contract.** Every micro/macro/adapter output path is zero-init,
  so the policy starts at its hierarchical prior and Q at uniform bins — no
  lecun noise posing as action preferences for CE to unlearn or for NeuRD to
  misread. Exception: the cross-entity pool read gate starts at 1.0, because
  token content can only reach the entity vector through that read.
- **Cross-entity pooling exists because matchup reasoning is a species-token ×
  move-token comparison across two mons**, and with entity-local pooling there
  is no layer where those two tokens coexist. Cost is only the attention
  probability matrix (168² versus 12·10² + 6·8² per timestep).
- **Perspective is otherwise a whisper in these inputs** — outcome is inherently
  side-differenced, so every history message carries an explicit mine/theirs
  tag.
- **The GRU-only history readout loses the latest node**: a raw hand rule over
  snapshots beat the model on late-game states, which is why the latest-node
  path exists.
- **Zero-wait inference batching is deliberate.** There is no min-batch and no
  max-wait knob: a previous attempt foundered on tuning exactly those two, since
  any wait stalls actors at game boundaries when no further requests are coming.
- Separate adder/sampler locks would only serialise adders against adders; the
  shared RLock is what keeps notify-while-holding-the-sibling-condition legal.
- **Nov 2025 hierarchical policy head beat the current flat gram head** in
  competition and was removed in 0e23621 — a known regression, not an
  improvement.

## 14. Tooling

- **Re-run `scripts/wandb_views.py` after every edit to it.** The save/round-trip
  path materialises wandb's default "Step" x-axis onto every panel that does not
  set one, silently defeating `WorkspaceSettings(x_axis=...)`. Always key metric
  trajectories to `lifetime_step`, not `_step`.
- **`register_wandb_charts.py` creates a new preset id on every run** (names must
  be unique), so after editing the spec, bump `_CHART_NAME` and update the id
  the dashboard references.
- **Tests run on the training box.** JAX must not preallocate the GPU out from
  under a live learner, and wandb must never sync. The slow suite's host RAM can
  trip the live run's OOM guard — it killed a run on 2026-08-14 — so run it only
  when training is down. Never run pytest under `JAX_PLATFORMS=cpu`: separate
  compile cache, bf16 tolerance artefacts.
- **`ex.bin` is load-bearing**, read at import by `rl/environment/data.py` and
  backing every model-init fixture. Its content is stale (predates
  `opp_private_team`, decodes as zeros); regenerate with
  `cd service && npm run generate-ex`.
- **The JAX persistent cache works.** Startup miss spam is sub-2s compiles
  (never persisted, by design) plus model-commit HLO invalidation; a no-edit
  restart is fully warm.
