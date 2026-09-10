# Lessons — paid-for knowledge

What was tried, what was measured, what the verdict was. Most of this was once
recorded only in comments attached to the code that produced it, which made it
invisible the moment that code was deleted. Merged into ONE place 2026-08-25 —
inside CLAUDE.md, which was un-excluded from git in the same commit, since the
ledgers here carry the revert handles and are worthless without history. Split
out to this sibling file 2026-09-07: CLAUDE.md is loaded into context every
session and this archive is consulted, so the two have different jobs. Nothing
was cut in the split. CLAUDE.md carries the section index; grep here by
mechanism name, panel name or date.

Two things live here: the **removal ledgers** — every mechanism deleted in a
cleanup pass, with the command that brings it back — and the **lessons
themselves**, grouped by topic. Entries marked *(live)* describe code still in
the tree; the rest describe code that is gone.

`docs/` holds the long-form design documents. It is gitignored and local to the
training box — never cite it as a public reference, and do not assume a fresh
clone has it.

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
| `RuntimeScalars` pytree | the class, the `scalars` arg on `train_step`, both construction sites | `1fd210c` | it carried `magnet_coef`/`neurd_coef` as traced leaves so a host controller could vary them without recompiling; nothing varied them any more. **Reintroduce it — do not widen static config — the moment a coefficient changes during a run** (CLAUDE.md 1) |

---

## Removal ledger — 2026-08-22 R-NaD pass

Everything below existed at tag **`pre-rnad-2026-08-22`** (commit `6f845d2`). Same
restore recipe as above. The through-line: three exploration mechanisms (magnet
KL, epsilon ladder, critic-disagreement — as reward and as UCB selection) all
collapsed switching at ~13k, and the critic diagnosis said why — it was correctly
learning Q^π of a policy under which switching IS worse, so sampling switches more
only confirmed it. What the search-free self-play successes (DeepNash) did instead
is a penalty against a MOVING reference policy. DeepNash routes it through the
values; NashPG (2025) shows the policy objective alone is enough, and that is
what survives here — see the ledger row below.

| mechanism | paths / symbols deleted | removed in | why it went |
|---|---|---|---|
| Q-ensemble UCB behaviour policy | `EnsembleGridPrior`, `ucb_tilt`, `HeadParams.ucb_c`, `q_ens_*` heads/losses/panels, InferenceServer standing head_params, `tests/test_intrinsic_reward.py::ucb`, ladder contract test | `47719b6` | σ_epi on switches tracked σ_epi on moves step-for-step through a 49%→9% switching collapse; KL(μ‖π) never left 1.5% of its cap. Shared-trunk ensembles measure head-init noise, not coverage (Kirsch 2024) |
| Critic-ensemble intrinsic reward | `EnsembleValueLogitHead`, `v_ens_head`/`v_int_head`, `compute_intrinsic_targets`, `IntrinsicTargets`, `int_rms` state leaf, 8 config fields, panels, `tests/test_intrinsic_reward.py` | `114ff5c` | int_reward switch/move pinned at 1.0 as an observer — disagreement never concentrated on post-switch states; Chen 2017 already rated disagreement-as-reward below UCB selection |
| Magnet KL | `loss_magnet_kl`, `magnet_log_policy`, `player_magnet_kl_coef` (+ its 50-line tuning history), `player_loss_magnet_kl` panel | `4254a1f` | gradient-side KL(π‖prior) carries a π prefactor (docs/entropy-gradient-pressure.md) — cannot refill a dead modality; its 0.05→0.1→0.2→0.05→0.1 ladder was compensation for a supply problem it could not fix |
| Epsilon explore ladder | `HeadParams.mix`, `behaviour_log_policy`, `Trajectory.explore`, `own_rows` gating (league cadence, builder, replay controller, the `_own` KL variant), `explore_game_prob`/`explore_eps_range`, per-game explore Agent path, `player_q_explore_*` panels, `tests/test_behaviour_mix.py` | `92b06c4` | random switches lose (explore-row post-switch return ~0.3 below post-move for the whole collapse), so behaviour-side coverage taught the critic switching is bad faster; R-NaD's penalty is the exploration mechanism and every game plays π |
| R-NaD reward transform (reg-value stream) | `reg_v_head`, `PlayerActorOutput.reg_value`, `PlayerTargets.reg_returns`, `compute_reg_returns`, `reg_reward`/`reg_v_target`, the `V_win + V_reg` Q bootstrap on both rungs, `loss_v_reg`, `player_reg_value_coef`, panels `player_reg_value_mean`/`player_reg_reward_mean`/`player_loss_v_reg` | `27053a6` | NashPG (arXiv:2510.18183) matches R-NaD's exploitability with the own-side KL in the POLICY objective and NO reward transform, at α=0.2 and a comparable reference-reset period; V_reg shifted every cell of the Q base identically so it cancelled exactly in the NeuRD centring, and no panel could have shown an action-axis effect (reg_value_mean −0.042 against ±1). The reference's measured contribution is the immediate analytic term alone — switch tilt +0.030 = 0.55σ of the NeuRD advantage — and that is kept |

## Removal ledger — 2026-08-25 privileged-critic pass

Everything below existed at tag **`pre-qva-redesign-2026-08-25`** (commit `989e8af`).
Same restore recipe as above. The through-line: the critic was conditioned on the
opponent's team sheet, which the policy cannot see. Step-0 numbers off the live
85k-step run settled it — the privileged rung was worth 0.005 value units against
the deployable one and scored *worse* in R² on both V (0.6462 vs 0.6469) and Q
(0.7751 vs 0.7825). It was not a variance reducer; it was a nuisance variable, and
its advantage told the policy things it could not act on
(`docs/qva-redesign-step0-reference.md`).

| mechanism | paths / symbols deleted | removed in | why it went |
|---|---|---|---|
| Privileged critic (opponent team sheet) | proto `EnvironmentState.opp_private_team`, service `ensureFirstPrivateTeam`/`firstPrivateTeam`/the `getPrivateTeam` `requestOverride` arg + the frozen-sheet harness invariants, `PlayerEnvOutput.opp_private_team` and its decode, encoder `opp` stream (`priv_latent_read`, `num_priv_latents`, sheet tokens, `opp_cross_attn`, `opp_ffw`), `attend`'s `allowed` key-mask | see this pass | the sheet was worth 0.005 value units and the rung reading it scored worse than the one ignoring it; conditioning a critic on information the policy lacks makes its advantage an aliasing term, not a signal. **Keep `player.opponent` in the service** — `worker.ts` `describe()`/`abortBattle()` need it to tear down both sides of a wedged battle |
| Value ladder (all/private/public rungs) | `all_value_embeddings`/`private_value_embeddings`/`public_value_embeddings` (→ one `value_embeddings_table`), `public_v_head`, `history_to_value_public` + the raw-history side channel, per-rung read/FFW gates, `private_value_logits`/`public_value_logits`, `loss_v_private`/`loss_v_public`, `player_value_ladder_coef`, `value_ladder_logs` (incl. the `player_value_info_gap_*` panels), `tests/test_value_ladder.py` | see this pass | with no privileged rung there is nothing left to be privileged *relative to*; the public rung's only consumer was the info-gap diagnostic. Its final reading is banked in the Step-0 doc before deletion |
| Second Q rung | `private_q_adv`, `q_private_all_target`, `v_private_target`, `loss_q_private`, `q_taken_of`, `player_q_private_*` panels | see this pass | closes diagnosis #5 (Q_all/Q_private shared-parameter interference, `docs/critic-weakness-analysis.md`) structurally — the two rungs shared every head param and differed only by conditioning |

## Restoration ledger — 2026-08-25 head redesign

Not a removal pass: this one put things BACK. Tag **`pre-qva-redesign-2026-08-25`**
(commit `989e8af`) is still the revert handle. Net effect on the model is
39.11M -> 27.26M parameters (-30%): the encoder shed 12.65M with the privileged
streams, and the modality separation cost only +0.8M on the heads.

| change | what it restores / collapses | why |
|---|---|---|
| `ActionScoreHead` | the singles policy, the doubles per-stage scorer and the Q head were three hand-written copies of adapter -> src_valid -> macro/micro -> compose, and had drifted | one readout, two compositions (`reduce`); `Porygon2PlayerModel.setup` 9 module attributes -> 4 |
| `heads.compose_q` | `Q = sg(V) + A - E_sg(pi)[A]` was assembled at 4 learner call sites and undone at 2 more | the identity is written once, in the model, beside `compose_action_grid`; `targets.residual_q` deleted |
| per-slot-group micro params | the Nov-2025 per-modality decoder, flattened in `0e23621` | three groups shared one projection and differed by ONE scalar; the target group's scalar was still bitwise zero at 84.9k, so it had no readout at all |
| per-modality macro MLP | modality-level parameters | the MLP and out layer were shared across all five modalities; only the pooling query differed |
| NeuRD reads A, clip on A | — | V is a per-row constant the centring already cancelled; on Q the +-2 clip was scale-dependent (V=0.9 clipped upside 10x harder than V=-0.9) |

## Removal ledger — 2026-08-26 NashPG transition

Everything below existed at tag **`pre-nashpg-2026-08-26`** (commit `a9952f8`).
Same restore recipe as above. The through-line: NashPG's own §5.4 ablation —
swapping PPO into the reward-transform framework closes most of its gap in
larger games — says the inner update rule, not the regularisation cycle, was
the bottleneck, and the dx65cpwp runaway was a NeuRD-pathway disease the PPO
clip removes structurally (zero gradient once the ratio leaves the band in
the push direction, so nothing persists at a stiff equilibrium). The player
policy loss is now full NashPG: PPO clip 0.2 on the taken action's π/μ ratio,
batch-normalised plain-v-trace advantage from V, differentiated forward
KL(π‖π_reg) magnet at 0.2 (the same reg_params 10k snap), differentiated
entropy bonus 0.05, one `player_pg_coef` bracket, player Adam b1 back to 0.9.
`player_pg_objective` selects `ppo|spo` for A/B. The Q/advantage stack stays
trained as the matched-control observer; the policy no longer reads it.

| mechanism | paths / symbols deleted | removed in | why it went |
|---|---|---|---|
| Hierarchical NeuRD loss | `loss.hierarchical_neurd` + `HierarchicalNeurd`, the centred-logit form, the eq.-10 β=2 band, the proximal logit decay, `player_neurd_coef`/`player_neurd_logit_clip`/`player_neurd_logit_decay`, `tests/test_neurd_loss.py`, every `player_neurd_*` panel | this pass | the operator itself is replaced; the b1=0/force-clip/centred-logits triad and the log-softmax differentiation ban were all NeuRD-scoped stability machinery and retire with it — under a true PG objective the log-softmax cross-term IS the correct gradient |
| Analytic reference/entropy shifts | `targets.reference_penalty`, the ±2 total-force clip, `player_ref_eta`/`player_ref_eta_ent`, the `ref_penalty_*`/`ent_penalty_*` panels | this pass | the reference becomes NashPG's differentiated KL in the objective (`reference_kl`, now a loss term); the measured +0.030 switch tilt of the analytic form is given up KNOWINGLY — every remaining force is π-prefactored, there is no starved-cell refill any more, and switch_ratio through the 13k wire is the pre-registered acceptance gate (fallback ladder: ent 0.05→0.1, then reinstate an analytic tilt as its own commit) |
| Warm-up ramp | `loss.warmup_scale`, `player_neurd_warmup_steps`, the ramp-gated snap, `tests/test_step2_warmup.py` | this pass | it held the policy gradient off a fresh critic's noise; the PPO surrogate on a batch-normalised advantage is bounded from step one, and the snap gate is a plain step modulus |
| Distribution-space value labels | `targets.advantage_shift`, the per-atom Retrace recursion | this pass | CE labels were signed measures off the simplex; the recursion now runs in SCALAR space and projects once through `two_hot` — mean preserved exactly (v-trace is linear), labels always proper distributions, 1 channel instead of n_bins. NOTE the middle atom now means "expected value ≈ 0", not "draw" — standard MuZero/Dreamer two-hot semantics; nothing consumes the distribution shape |
| Free-level logit outputs | `PlayerPolicyHeadOutput.macro_logits`/`micro_logits` | this pass | learner-only outputs whose sole consumer was the NeuRD loss |

What came in: `loss.ppo_objective` (restored from pre-`4234016`) behind a
selector on `policy_gradient_loss` (builder keeps `"spo"`), `clip_fraction`,
`PlayerTargets.pg_advantages` (plain v-trace over V, builder-pattern, ρ-
truncated, f32), panels `player_loss_pg`/`player_ppo_clip_frac`/
`player_loss_entropy`/`player_pg_adv_{mean,std}`, and the loss-agnostic
starvation pairs renamed `player_policy_{prob,absadv}_*`. Deliberate
reference-diff residue: single grad step per batch (replay reuse ratio 8 is
the 4-epoch analogue), λ stays 0.8 (their GAE 0.95 — our monotone sweep), lr
3e-5 / grad clip 10 (theirs 3e-4/0.5 on ≤512-dim MLPs), categorical CE +
Retrace critic kept (NashPG is estimator-agnostic, §4.3).

## Addition ledger — 2026-08-26 support anchor

**Gate resolved, same day.** The NashPG switch gate FAILED (switch_ratio 0.42
→ 0.041 @33k → 0.013 @49k) while the run became the strongest lineage the
project has produced: eval wr vs SimpleHeuristic 0.167 @30k and 0.207 @49k,
against 0.030 (2wvnlsz3) and 0.039 (dx65cpwp) at matched `lifetime_step` — it
passed 2wvnlsz3's 80k mark at 30k. Critic quality is IDENTICAL across lineages
(`v_outcome_r2_{early,mid,late}` 0.06/0.25/0.60 vs −0.07/0.25/0.59), so the
~3x gain is the update rule. NashPG is vindicated; only the starvation risk
was real, and it was measured on the parameter:
`player_policy_type_scale_switch` peaked 0.0197 @13k and unwound 53% to
0.0092 @33k while the move group grew 8.5x — with `type_scale_target ≡ 0.0`
as the in-run never-trained control.

**The fallback ladder was retired UNFIRED.** Rung 1 (ent 0.05→0.1) is
falsified twice: π-prefactored (∂H/∂y_b = −π_b(log π_b + H) → 0), *and* blind
— global normalised entropy read 0.755 against modality entropy 0.22, so a
global target cannot see a modality-confined collapse (the same dilution
`train_step` documents for the μ-sampled KL). Rung 2 (analytic tilt) is
superseded by a form that is bounded where it was not.

| mechanism | what it is | why |
|---|---|---|
| `targets.support_kl` + `player_support_coef` 1.0 / `player_support_temperature` 1.2 | forward KL(p_T‖π) over legal cells against the SAME frozen `reg_params` reference raised to temperature T, p_T ∝ π_reg^(1/T). Reuses `reg_log_policy` — no extra forward pass | per-logit gradient is exactly **π(b) − p_T(b)**: no π prefactor, bounded by 1 for every π, zero-sum over legal cells — the three properties every dx65cpwp analytic shift failed. It is **all-action**, so unlike the PPO surrogate it does not need a switch to be TAKEN to act on one, and APO's `N·π < 1` absorbing bound does not apply to it. That is why it, and not a supply fix, is the mechanism at this depth |

**Direction vs temperature do different jobs.** Direction buys
mass-independence — reverse KL(π‖π_reg) has gradient π_b(log(π_b/π_reg,b) −
KL), so its force dies on a starved cell and the magnet is structurally
indifferent to a *dropped modality*; forward KL is mode-covering and can
refill one. Measured: as a cell starves 1.4e-2 → 1e-6 the magnet's restoring
force collapses 0.050 → 0.000013 while this converges on p_T(b) — 25000x
apart. T decides where the anchor SITS: π_reg is a snapshot of an
already-collapsing policy, so T=1 is a pure brake (force exactly 0 at the
reference) and T>1 re-inflates its tail through a monotone power transform
(same ranking — it never says *which* switch to make).

**SIZE THIS FAMILY ON THE EQUILIBRIUM CONDITION, NEVER ON LOSS BALANCE.** A
switch cell is only TAKEN on ~π_b of rows, so the surrogate's expected force
on it is ~π_b·(A_b − E_π[A]) while the anchor acts on EVERY row; they balance
at `|A_b| = coef·(p_T,b − π_b)/π_b`, in σ since `pg_advantages` are unit-std.
The 1/π_b is the point — the ask *diverges* as the cell starves, so this is a
floor that yields to real evidence rather than a standing tax, and it falls to
0.02σ at a healthy 0.30 mass. coef 1.0 / T 1.2 gives equilibrium switch mass
0.124 / 0.081 / 0.038 against a 0.2 / 0.3 / 0.5σ headwind. **coef 0.25 shipped
first and was wrong**: calibrated to match the entropy bonus's per-logit force
at ent 0.05, which put the fixed point at 0.014 — the collapsed status quo, a
brake and not a cure.

**T>1 is a BOUNDED dose.** The ratchet compounds *through* the reference
(π_reg re-snaps from the policy the anchor just lifted), so unopposed its only
fixed point is uniform; the PG force is what stops it. The failure mode is not
a static coefficient's — too large a coef walks the REFERENCE flatter each
snap, which reads as healthy for ~20k steps and is costly to undo. EXIT T→1.0
once `player_q_support_vol_switch_rows` ≥ 10; ABORT early if
`player_normalized_modality_entropy` climbs across 2+ snaps past ~0.85.

**Reference diff vs the NashPG implementation** (ntu-agents/nashpg @ dda50fe,
`train/core/update_agent.py`; config path `conf/default/nash_pg.yaml` ->
`conf/algorithm/nash_pg.yaml` -> `train/nash_pg.py:76` -> `update_agent`).
Confirmed faithful: `dists.kl_divergence(mag_dists)` is KL(pi || pi_mag),
matching Algorithm 4; ent_coef 0.05, mag_coef 0.2, clip_eps 0.2 all match.
Unrecorded divergences found 2026-08-26:

| theirs | ours | note |
|---|---|---|
| `optax.adamw(lr, eps=1e-5)` — explicitly overriding optax's 1e-8 | eps 1e-8 | **the one to test.** Adam is scale-invariant, so a param whose gradient has gone tiny still steps at ~full lr along a noise-dominated direction; `eps` is the ONLY damper, and 1e-5 engages 1000x sooner. Directly relevant to `type_scale_switch` drifting once switch supply dies |
| `weight_decay=1e-4` (optax.adamw default, not overridden) | 0 | ours is likely BETTER here — decay pulls unused params toward zero, which worsens forgetting — but it was undocumented |
| magnet snaps `nnx.clone(agent)`, the LIVE agent, every `num_inner_update`=1000 | snaps from `target_params` (the EMA) every 10k | ours is smoother and staler |
| advantage normalised per-MINIBATCH inside the loss fn | per-batch | minor |
| `mag_divergence_type: "kl" | "l2"` | kl only | the l2 variant is MORE pi-prefactored, not less — measured force at a starved cell 0.0048 vs kl's 0.0386 at pi=1e-2, and exactly 0 by pi=1e-6 where kl still has 1.3e-5. **The reference offers no mass-independent magnet in either mode**, which is why the support anchor had to be built rather than adopted |

Also theirs: PPO value clipping (`0.5*max(sq, sq_clipped)`) — declined, we keep
categorical CE + Retrace (§4.3 estimator-agnostic); 4 epochs x 4 minibatches =
16 grad steps per collected batch vs our 1 with replay reuse 8;
`only_use_player0_experience=False` is hardcoded at their call site, not config.

**Not bundled, deliberately:** DAPO's clip-higher. The symmetric PPO clip is
itself implicated as a *cause* — DAPO measure that clipped actions are
predominantly π < 0.2 and ours sit at 0.0144, while CE-GPPO name the discarded
quadrant (positive-advantage, low-probability) as the exploration-preserving
one; our `adv_rms_switch` 0.127 vs `adv_rms_move` 0.029 says switch cells
carry 4.4x the advantage magnitude on the grid. But widening ε_high loosens
what removed the dx65cpwp runaway, so it wants its own decision and its own
commit.

## Addition ledger — 2026-08-26 anchor phase 3 (advantage tilt)

**The through-line of the whole collapse lineage, stated once.** Every
mechanism ever tried supplied exactly one of the two things a
conditional-value action needs — probability MASS or DISCRIMINATION (which
switch) — and in this game mass without discrimination is *negative*
evidence (random switches lose ~0.3): entropy/reverse-KL magnet neither
(π-prefactored); ε-ladder mass only → taught the critic the collapse faster;
PPO discrimination only (taken cells, expected force ~π_b); anchor T>1 mass
only → `type_scale_switch` → 0 (phase 1). The papers are not wrong — their
convergence stories assume support never numerically dies, which sampled PPO
+ function approximation + ISR fade violates, and the reference diff already
established no reference impl carries a mass-independent restorer. Meanwhile
the discrimination signal existed in-run and was discarded beside the anchor:
`absadv_ratio` 3.47 vs `prob_ratio` 0.093 @101.5k — the same signature that
motivated NeuRD at 157k. Under the phase-2 T=1.0 brake, `vol_switch_rows`
decayed 9.3 → 3: T=1 exerts no force at a reference that re-snaps from a
collapsing policy.

| mechanism | what it is | why |
|---|---|---|
| `player_support_adv_temperature` (τ) + tilt in `support_target` | p* ∝ π_reg^(1/T) · exp(sg(A_target)/τ) over legal cells, same forward-KL loss — MPO's E-step target (arXiv:1806.06920) with π_reg as prior under the anchor's existing mode-covering projection. Tilt sg'd, exponent clipped ±3; A_target is the TARGET net's advantage, so no gradient path opens into the observer stack. τ=0.0 bitwise off; launch 0.125 (~adv_rms_switch: +1σ = one e-fold of reference mass) once the T=1 telemetry panels (`player_support_{switch,move}_mass`, `player_support_force_{switch,move}`) validate against `prob_switch` at a snap | gradient stays π−p*: bounded by 1 for ANY A, zero-sum, prefactor-free — and now carries WHICH cell. Within a modality p* ranks cells by the critic, so the anchor TRAINS `type_scale_switch` instead of flattening it (phase 1's failure), and the snap ratchet compounds toward absadv_ratio → 1 (policy-critic consistency), not uniform: anchor lifts a cell → it gets taken → Retrace labels it → A corrects → anchor relaxes. The chicken-and-egg closed through play |

**The gate is re-registered on discrimination, not frequency.** The
strongest-ever lineage (eval wr 0.19–0.29 @101k, at the 0.29 all-time peak)
has collapsed switching, and `vol_switch_rows` rises under any
mass-restoring force whether or not it helps — Goodhartable alone. Judge
this family on `type_scale_switch` training + `absadv_ratio` declining
toward 1 + eval-wr trajectory, with raw frequency as an observer. That also
makes the objective question empirical: wr climbing as discriminate mass
returns = the collapse was costing strength; absadv_ratio → 1 with mass
still low = low switching was locally correct against this league and the
anchor's job is re-learnability. Acceptance/abort/fallback pre-registered in
the config block; fallback is DAPO clip-higher (own commit), never a τ
retune.

**PHASE 3 RESULT (110.4k–223k) and PHASE 4 (2026-08-27).** Each half of the
tilt proved out on the axis it owned; the composition failed on the axis
neither owned. `type_scale_switch` trained for the first time ever (0.0008
→ 0.0039) and wr set new all-time highs (0.35 @223k) — but raw A's
MODALITY-LEVEL sign is negative on switch cells (the −0.11..−0.18 mean Q^π
gap; absadv_ratio is a MAGNITUDE, not a sign), so the tilt subtracted
switch mass from p* and the snap cycle compounded it: ask 0.065 → 0.0024,
`force_switch` NEGATIVE by 223k — the support anchor inverted into an
anti-switch force, `vol_switch_rows` hit 0, and `absadv_switch` then
DECAYED unsupervised (0.17 → 0.10). **Two lessons with teeth:** (1) the
"absadv_ratio → 1" acceptance read cannot fire by arbitrage once supply is
0 — it starves instead, which is unfalsifiable, not vindication; (2)
"yields to evidence" is wrong at the modality level, because that sign IS
the self-confirming Q^π view the anchor exists to break. Phase 4 =
`targets.centre_within_modality` (the critic says WHICH switch, never
WHETHER — a whole-modality shift of A leaves p* unchanged, test-pinned
with the within-group ranking as positive control) + T back to 1.2 for the
modality axis (its phase-1 within-modality flattening now opposed by the
tilt), recovery mode from full absorption. Abort instruments are now the
ones that actually saw each failure: `support_switch_mass` falling across
2 consecutive snaps / `force_switch` persistently negative (the phase-3
shape); `type_scale_switch` falling while switch mass rises (the phase-1
shape). Also on the record twice now: pick the abort metric from the
mechanism you fear — the entropy>0.85 guard watched the wrong direction
while entropy fell to 0.025.

## Removal + addition ledger — 2026-08-27 factorised objective, anchor retired

**REMOVED: the support-anchor family** (`targets.support_kl`/`support_target`/
`centre_within_modality`, `player_support_{coef,temperature,adv_temperature}`,
`loss_support`, all `player_support_*` panels). Four phases, one through-line:
every mass-restoring variant either ERASED within-modality discrimination
(T>1's mode-covering target + the snap ratchet: type_scale_switch 0.0122 →
0.0003 in phase 1) or pointed the anchor DOWN (raw tilt, phase 3: the
modality-level Q^π sign inverted force_switch by 223k) or ratcheted down
against the PG (centred tilt + T 1.2, phase 4: ask 0.0131 → 0.0031 across
the 230k snap) — and behaviour-side injection teaches the MEAN switch's
losing value, so the modality-level lesson each era wrote was "switching is
bad", correct for the policy the injection created. Revert handles: the
phase commits and this file's 2026-08-26/27 ledgers.

**ADDED: the factorised policy objective** — the Oct–Nov 2025
training-signal structure rebuilt natively on the composed head, plus the
same split pushed into the PG itself:

| piece | what | why |
|---|---|---|
| `loss.factorised_log_probs` + actor-stored `PlayerPolicyHeadOutput.macro_log_prob` | exact level split via the composition identity (modality logsumexp of log_policy == composed_macro; micro = joint − macro); μ_macro stored at act time, μ_micro derived | per-level PPO: two ratios, two clips, SAME normalised v-trace advantage (level score functions sum to the joint PG — only trust-region geometry changes). The composed single band let macro drift consume trust region the micro level needed on exactly the rare-modality rows; `player_ppo_clip_frac_{macro,micro}` makes that observable |
| `loss.factorised_entropies` | unit-weight H(macro) + H(micro\|taken modality), each masked-averaged over its OWN row set (macro: ≥2 live modalities; micro: taken modality ≥2 legal cells), each NORMALISED by its own log k (raw H caps at log k — an 8-cell move row outweighed a 2-switch row ~3x) | the joint H decomposes as H(macro) + Σ_m π_m·H(micro\|m): its which-axis pressure died with modality mass and one budget was payable wherever cheapest. Unit weights = per-axis budgets; masked average = inverse-frequency amplification; normalisation = k-independent temperature coef/log k. A temperature evidence beats (eq. π ∝ exp(A/coef) per axis), never a target — the injection post-mortem's law |

Magnet stays joint (factorised magnet is the recorded follow-up). Fresh
lineage from step 0 (6ta9hmp6 archived at 243.5k,
`ckpts/archive/gen9_phase4_6ta9hmp6_final_20260827`): the which-axis terms
act on taken-switch rows, so the clean test is organic early switching
feeding them — the Oct–Nov replication read. Pre-registered acceptance
(judged against 6ta9hmp6's own early curve): type_scale_switch ≥0.01
through the 13k wire and NOT unwinding ≥50% by 33k (6ta9hmp6 lost 53%);
wr vs SimpleHeuristic ≥0.167 @30k; offline within-switch spread read at
~20k. Abort: head param rms >0.07; clip_frac_micro pinned ~1; entropy
panels cliffing. Fallback ladder: per-level clip epsilons, factorised
magnet, then the engine-fork counterfactual-labels workstream (scoped
independently — it is the evidence fix regardless).

## Removal + addition ledger — 2026-08-28 entropy floor; per-level PPO retired

**The BR probe delivered the project's first ground-truth exploitability
readings** (mechanism: 2026-08-27 BR child runs, `start_br.sh`). Against the
factorised-objective lineage @77638: flat 0.509 over ~2000 games at a
20k-step budget, then — budget extended — a climb from ~26k to a **0.57–0.58
plateau held 60k→118k** (stopped there; published `p_100077638` at br_steps
120670). The exploit is substantially the SWITCH AXIS: the BR's voluntary
switches against the frozen target return +0.8..+0.95 while its own
prob_switch recovered 0.006 → ~0.02 and its inherited anti-switch
`type_scale_switch` unwound −0.038 → −0.012. Three verdicts with teeth:
(1) low switching was NOT locally correct — the collapse costs measurable
strength (the phase-3 open question, closed); (2) the machinery CAN learn
switching the moment labels say switches win (stationary opponent) —
supervision coverage, not capacity, again; (3) a BR "not exploitable"
certificate is budget-relative — the 20k read expired at ~33k (the BR must
first climb out of the collapsed prior it inits from).

**REMOVED: the per-level PPO split** (`factorised_log_probs`, actor-stored
`macro_log_prob`, per-level ratios/clips, `player_{loss_pg,ppo_clip_frac}_{macro,micro}`
panels; revert commit 4317fb4, landed b1c767f/923ebb2). The factorised-PPO
lineage answered its own gate — `type_scale_switch` trained NEGATIVE
(−0.026 @13k, −0.039 @77k), prob_switch 0.131 → 0.02 — and the BR probe
located the deficit in evidence supply, not trust-region geometry. Back to
ONE pi/mu ratio, one clip. The per-level ENTROPY terms and their row masks
STAY.

**ADDED: entropy-floor dual controllers** (commit 5285d9e). Each per-axis
entropy coefficient is now a SAC-style dual temperature: log-space leaves
on the player TrainState (traced — varying them costs NO recompile, the
run-1326 rule satisfied structurally), dual ascent `log α += lr·(target −
H_norm)` per axis after each grad step, clipped [0.005, 0.5], frozen on
empty-axis batches (`average()` reads 0.0 there — a spurious full deficit),
updated before the finite gate so poisoned batches revert alphas too. Init
= `player_ent_coef` (0.05); targets `player_ent_target_{macro,micro}` = 0.5
normalised; target 0.0 = that axis's controller OFF (bit-identical static
behaviour). This is CLAUDE.md 4's own prescription firing after the
four-retune static-coef falsification (0.01/0.05/0.1/0.2 all collapsed).
**Calibration that reframed the ask**: the Nov-2025 stable lineage
(generous-sky-444, 412k steps) held raw action entropy 0.5–1.4 nats — the
SAME band today's global H occupies; the remembered "3 nats" was the
per-head-sum loss metric. What the old model held and today loses is
entropy DISTRIBUTED ACROSS BOTH AXES; a raw-nats global floor is both
infeasible (avg max legal H ≈ 1.9 nats) and blind to modality collapse
(the old global-target dilution lesson). Temperature-not-target law
respected: the controller picks only the temperature (equilibrium still
π ∝ exp(A/α) per axis); evidence keeps deciding WHICH cells hold the mass.

**Pre-registered acceptance** (fresh lineage from step 0; judged against
the factorised lineage's and 6ta9hmp6's early curves): `entropy_macro`
holds ≥ ~0.45 through 33k with `player_ent_alpha_macro` OFF the 0.5
ceiling; prob_switch not collapsing to ≤0.02 by 33k (factorised lineage
hit 0.019 @32k); wr vs SimpleHeuristic ≥ 0.167 @30k; and the new
instrument — a BR probe against the ~77k ckpt reading ≤ the 0.57–0.58 the
77638 probe read. **Abort**: alphas pinned at max with eval-wr falling
(the floor is a pure tax); entropy AT target while `type_scale_switch` →
0 (mass without discrimination — the phase-1 shape, watched on the
mechanism actually feared, per the twice-paid abort-instrument lesson).

## Removal + addition ledger — 2026-08-29 flat trunk, flat readout, Q retired

Everything below existed at tag **`pre-flat-trunk-2026-08-29`** (on `main`).
Same restore recipe as the other passes. **28.72M -> 13.47M parameters (-53%),
`encoder.py` 1998 -> ~1230 lines, 189 input tokens -> a 61-row sequence.**

The through-line: almost none of the structure was load-bearing for the
INFORMATION the model needs. The block masks, the residual-stream gates, the
Perceiver bottleneck and the macro/micro composition each solved a routing
problem that plain self-attention over one short sequence solves by
construction — and at 61 rows an all-pairs attention is 3.7k cells against the
~24k the masked routing plus its two feeding cross-attention reads paid. The
masks were buying their own complexity.

| mechanism | paths / symbols deleted | why it went |
|---|---|---|
| Round trunk + three residual streams | `encoder.RoundBlock`, `GroupNorm`, `action_input_norm`/`action_out_norm`, `attend`, every `*_gate` param, `LatentInputRead` + `cfg.encoder.num_latents`/`latent_read`, `ActionSlotRead` + `_action_slot_queries` + `slot_position`, `InputTokenSet`, `tag_with_species`, `_current_entity_tokens`, `value_embeddings_table`, `cfg.encoder.num_rounds`/`round.*`, `rl/offline/gate_contribution.py` | 14.76M params (51% of the model) for five block-masked gated attentions per round over 3 streams, fed by a 48-latent bottleneck. Replaced by `rl/model/trunk.py`: 6 ungated pre-RMSNorm blocks over ONE sequence, 6.32M. The 2026-08-24 gate-contribution finding (all six FFWs contributing <= 5e-4 of their stream) is retired STRUCTURALLY — there are no gates — rather than tuned around |
| Attribute-token input layout | 189 raw tokens (10-11 per entity) -> 61 rows, one per THING | see the caveat below: this re-adopts entity-local pooling, deliberately |
| Hierarchical action head | `heads.{calculate_hierarchical_prior, MicroHead, ActionAdapter, MacroHead, MacroMicroHead, ActionScores, ActionScoreHead, compose_action_grid}`, `cfg.{policy_head, advantage_head}` | 2.65M params over two instantiations became 0.13M over three small readouts (`FlatActionReadout`): a scalar per sheet row for switching, ONE bilinear for moves x targets, a scalar per target row for pass/default |
| Q/advantage observer | `advantage_head`, `heads.compose_q`, `PlayerActorOutput.{advantage,q}`, `loss_q`, `player_q_coef`, the Retrace baseline in `targets.compute_player_targets` (`adv_taken`), `telemetry.{q_fit_telemetry, modality_means}` + the Q/policy head-leaf tables, `rl/offline/overfit_probe.py`, `tests/test_{q_identity,train_step_q}.py`, and every `player_{loss_q, q_r2*, q_mse, q_action_var*, adv_rms*, adv_type_scale*, q_pivotal_*, q_switch_move_gap, q_saturation_frac, q_calibration_r2_*, q_label_var_*, retrace_baseline_*, policy_absadv_*, q_loss_share_*, mv_*_gap_critic, policy_micro_local_*, policy_adapter_rms, policy_type_scale_*, q_grad_norm_*}` panel | the policy stopped reading it at the NashPG switch (2026-08-26), which left it a matched control for an architecture that no longer exists. **Last readings, banked before deletion** (qb8b82oi @ lifetime_step 289235): `q_r2` 0.8817, `q_r2_switch_voluntary` 0.9162, `adv_rms_move` 0.0339, `adv_rms_switch` 0.1016, `adv_rms_target` 0 (never trained — singles has no target modality), `policy_absadv_ratio` 4.509, `q_action_var_uniform_p90` 0.00705, `retrace_baseline_abs` 0.0197, `q_switch_move_gap` -0.1386, `q_support_vol_switch_rows` 9. So the critic fit well and *preferred* switch cells 3.0x on |A| against a 0.032 switch mass — the same signature that has stood since the 157k 2026-08-20 lineage — and the value targets it shifted moved by 0.0197. The v-trace baseline reverts to V, which `targets.py`'s own note already promised was the exact revert |
| Packed 1681-bit action mask | `EnvironmentState.bytes action_mask` on the write path (the field survives, renamed `packed_action_mask`, ONLY so the 8.8GB of `replays/shards` still parse), `state.ts`'s four hand-copied enum tables, `runner.ts::getShowdownTargetFormat` + `CHOOSABLE_TARGETS` + the whole parallel decoder, `isStruggling`, `getActionMask`'s unused `format` arg | 211 bytes carried a ~10-element choice set over a grid of which ~84% is unreachable in ANY format. See the addition row |

**Two §13 lessons were deliberately set aside. They are decisions, not
oversights, and both are recorded so a future reader sees the choice:**

1. *"Deep, modality-separated action decoders are empirically necessary"* (the
   Nov-2025 hierarchical head beat the flat gram head). **Judged confounded.**
   The flat readout goes in and the pre-registered gate below judges it.
2. *"Cross-entity pooling exists because matchup reasoning is a species-token x
   move-token comparison across two mons, and with entity-local pooling there
   is no layer where those two tokens coexist"* — and the 2026-08-28 audit,
   which moved AWAY from pooled entity vectors toward raw token reads.
   One-token-per-entity re-adopts the pooling. **Partial mitigation, by
   construction:** my 16 candidate moves stay their own rows and the four
   entity-derived target rows carry the opposing actives, so *my move x their
   mon* — the direction a decision actually turns on — keeps both operands as
   separate rows. What is given up is THEIR individual revealed moves. If
   matchup reasoning proves to be the deficit, the fix is explicit matchup rows
   (each of my mons x each of theirs, the 24 tokens the reference design uses),
   NOT re-unpacking attributes.

**Additions.**

| mechanism | what it is | why |
|---|---|---|
| `rl/model/trunk.py` (`Trunk`, `TrunkBlock`) | 6 unshared pre-RMSNorm blocks, self-attention + one SHARED SwiGLU MLP, plain residual, `nothing_saveable` remat, over the 61-row sequence | no gates and no masks: `RMSNorm` is `normed * (1 + scale)` with `scale` zeros-init, i.e. exactly identity at init, and the residual adds are ungated, so the trunk is live at step 0 by construction and an "is it wired" test needs no gate opening |
| `heads.FlatActionReadout` | three readouts over named rows; the bilinear survives ONLY for moves x targets | **init contract: every logit is exactly 0, so the policy starts UNIFORM over legal cells** and `compute_policy_metrics(prior=None)` is the anchor. Reaching that without re-creating the two-factor stall is the subtlety: `query` is zero-init and `key` is not, so the zero factor's gradient is a rank-1 outer product of LIVE rows (it moves at step 1, key unfreezes at step 2) — one zero factor over a live input, not a scalar times a random grid. **No qk layer-norm on this head**: its input is identically zero at init and its Jacobian goes as 1/sqrt(eps) straight into the zero-init kernel |
| `loss.uniform_kl_rows` + `player_uniform_kl_coef` 0.05 | forward KL from UNIFORM over legal cells, inside the pg bracket | per-logit gradient is exactly **pi_b - 1/k**: bounded by 1 for any pi, zero-sum, **no pi prefactor** — the three properties every mass-restoring mechanism this project tried has lacked. Note the DIRECTION: reverse KL(pi ‖ u) is, up to a constant, the entropy bonus already in the objective. And unlike the four support-anchor phases the reference is a CONSTANT, so it cannot collapse, cannot be ratcheted flat by re-snapping from an already-collapsing policy, and cannot invert on the modality-level sign of Q^pi. 0.0 is a bit-identical off |
| `proto.ActionMask` + `ActionRequestKind` | a structured mask shaped like the DECISION: 6 switch bits, 16 move slots x 17 target bits, a standalone-action mask, the request kind and the ally half | the 41x41 grid was a model-side artefact that leaked onto the wire. The service now builds the mask and the Showdown choice string TOGETHER, as one `cell -> choice` map, and `choiceFromAction` is a lookup in it |
| `player_pointer_*` / `player_trunk_*` panels | rms of each readout leaf against its known init, plus per-subtree grad norms | required, not decoration: the dx65cpwp runaway lived entirely in head params and was invisible on wandb. The flat readout's failure mode is the SAME shape (a two-factor product with one zero-init factor), so `query` must leave 0 within ~200 steps and `key` must leave lecun 0.0625 shortly after. src/tgt stay split — the 7.5x asymmetry was itself the diagnostic |

**THREE MASK BUGS the one-table refactor made unrepresentable**, all live for
months and all invisible because nothing asserted the mask and the decoder
agreed and the sim's rejections were logged rather than counted:

1. **The wildcard suffix was hardcoded to `" terastallize"`** while the mask
   legalised a wildcard for mega evolution, Ultra Burst and Z-moves too, so
   those cells decoded to a choice Showdown rejects.
2. **Move targets were resolved through a different move list than the mask
   indexed**: the mask read `active.maxMoves.maxMoves` under Dynamax, the
   decoder `request.active[i].moves[j]`.
3. **Team preview lit one cell per REMAINING POSITION — up to 7 — while the
   decoder ignored the target entirely.** Up to 42 grid cells stood for 6 real
   choices, so the policy spread its mass over exact duplicates and the
   micro-entropy cell count was inflated to match. One cell per candidate now.

Also fixed: `switch_slots` alone cannot say WHICH active a switch replaces, so
the message carries `active_slot`; without it a singles battle legalises
`ALLY_2_SWITCH` and the model can pick a cell the service cannot decode. Caught
by the new invariant, not by reasoning.

**Instruments.** `service/src/tests/harness.ts` now asserts that the wire mask
and the decoder name exactly the same cells (with an independent second
implementation of the derivation, because the thing under test is that the two
LANGUAGES agree), that a request carrying a choice never legalises nothing, and
that the sim rejected no choice at all — `invalidChoiceCount`, counted rather
than logged. **3810 soak battles, zero violations.** Python side:
`tests/test_action_mask.py` round-trips the message through
`_grid_from_structured_mask`, with the positive control that `kind` genuinely
selects which HALF of the grid names the incoming mon.

**Pre-registered acceptance** (fresh lineage from step 0; baselines banked from
qb8b82oi before deletion, since the gate as written is one the CURRENT lineage
FAILS — it dips before the dual controller recovers it):

| lifetime_step | H_macro | alpha_macro | prob_switch |
|---|---|---|---|
| 12952 | 0.4888 | 0.0050 (at the MIN) | 0.0368 |
| 32865 | 0.3123 | 0.0431 | 0.0196 |
| 76998 | 0.4229 | 0.0755 | 0.0249 |

So the honest bar is **beat 0.3123 / 0.0196 at 33k**, with 0.45 / 0.02 as the
aspiration rather than a pass/fail line. Plus: eval wr vs SimpleHeuristic **at
temp = 1.0** (the new arm — see below) at 30k; a BR probe against the ~77k
checkpoint reading <= the 0.57-0.58 the 77638 probe read; `separation_probe
--probe c` at a matched step, `RESERVE_j`/alive/hp moving off ~0.00 toward the
0.35-0.44 controls (READ THE Y-STD COLUMN FIRST); and steps/sec above the
current lineage, which is the point of the change.

**Abort:** `player_pointer_query_rms` or `..._key_rms` still at init by 2k (the
two-factor stall, live); alphas pinned at max with eval wr falling; entropy at
target while `prob_switch` -> 0 (mass without discrimination — the phase-1
shape, watched on the mechanism actually feared).

**EVAL TEMPERATURE IS NOT COMPARABLE ACROSS THIS BOUNDARY.** `ActionScoreHead`
divided BOTH levels by `temp`, so eval at 0.5 sharpened the modality marginal
as well as the within-modality choice; the flat head's single division does
not. The two differ by a per-modality reweighting that relatively down-weights
the concentrated switch modality, so the SAME policy reads as switching more
under the new head — which would look like a free improvement and is pure
parameterisation. `rl/online/main.py` now runs the last eval slot at temp = 1.0
(identical under both), named `EvalActor-simpleheuristic-t1-*`, and that is the
cross-lineage number. temp = 0.5 stays as the within-lineage trend.

**Scoped out deliberately, recorded as the follow-up:** collapsing the action
space itself to the three blocks end to end. Modality would become block
membership and `FLAT_MODALITY_MASK` would go, but it moves the stored
trajectory shapes and the chunk contract, so it wants its own commit. Also
still open: `EntityPrivateNodeFeature` has no hp/status/fainted/boosts, so the
switch readout depends on the trunk routing a candidate's condition from its
PUBLIC row — a two-hop path over 61 rows now rather than a maze, which is the
reason to EXPECT probe C to move, but it is an expectation and probe C is the
instrument either way.

## Removal ledger — 2026-08-30 NashPG-verbatim bracket

One commit (the revert handle). The player policy bracket is the reference
actor loss verbatim (ntu-agents/nashpg `update_agent.py`, mag_divergence
"kl"): `surrogate + ent_coef*(-H_joint) + mag_coef*KL(pi || pi_reg)`, with
`player_pg_objective` defaulting to "spo" (the smooth quadratic; "ppo" stays
the A/B) and `player_ent_coef` 0.01 STATIC on the plain JOINT entropy — which
is, up to a constant, the reverse KL to uniform, i.e. NashPG's own form.

| mechanism | symbols | why |
|---|---|---|
| Entropy-floor dual controllers + per-level entropy loss | `log_ent_alpha_{macro,micro}` TrainState leaves + their ckpt scalar plumbing (old ckpts still load; `apply_player_scalars` ignores their alpha keys), `loss.entropy_floor_step`, the per-axis loss terms, `player_ent_target_{macro,micro}` / `player_ent_alpha_{lr,min,max}`, `tests/test_entropy_floor.py` | on wy8m50ic BOTH alphas sat at the 0.005 MINIMUM for the whole 185k-step run while H_macro held 0.45–0.70 — the flat trunk holds its entropy for free, so the controller was pure machinery (the §10 rule firing again). `factorised_entropies` and the `player_entropy_{macro,micro_taken}` panels survive as OBSERVERS — macro dying while joint H holds is the collapse shape the global panel cannot see, and nothing defends the floor any more |
| Uniform-KL zero-avoider | `loss.uniform_kl_rows`, `player_uniform_kl_coef`, its gradient test | reference alignment (NashPG carries no mass-independent restorer). Given up KNOWINGLY — this was the only pi-prefactor-free force in the bracket; `prob_switch` / `player_vol_switch_rows` are the watch, and the term is one commit away if the collapse returns. **RESTORED 2026-08-31 — the watch fired, one day later** (see the addition ledger below) |
| Last Q machinery | `targets.compute_q_onestep_targets`, `q_mask` (→ `acted_mask`), the one-step-label panels (`player_q_target_*`, `player_v_onestep_r2`, `player_q_target_edge_frac`), dead `pi_target` compute and two bare no-op statements | the labels had no consumer since the Q head retired 2026-08-29. Supply counters KEPT, renamed off the q prefix: `player_{vol,forced}_switch_rows`, `player_chunk_vol_switch_frac`, `player_taken_{,voluntary_}switch_frac` — fresh wandb continuity by design (the coma→neurd precedent) |

Also in: `masked_policy` (exp→mask→renormalise written once in train_step),
wandb views tidied and re-saved. Fresh lineage from step 0 (the loss moved);
wy8m50ic archived at ~185k. Same acceptance table as the flat-trunk gate,
plus the entropy observers now watched UNDEFENDED — a macro-entropy cliff
under the static 0.01 is the abort, answered by re-instating the floor or
the uniform-KL as their own commits, never by a silent coef bump.

## Addition ledger — 2026-08-31 zero-avoider restored, and WHY no coefficient works

**The abort fired within a day and the watch was the right one.** wy8m50ic's
alpha panels stop at ~187k (the dual controllers' removal); everything after
that is the undefended static-0.01 era, and it is the only era in the run's
history below 0.35 macro entropy:

| lifetime_step | 114k | 180k | 217k | 230k | 246k | 266k |
|---|---|---|---|---|---|---|
| `player_entropy_macro` | 0.41 | 0.46 | 0.43 | 0.36 | 0.31 | **0.19** |

with `player_policy_prob_switch` 0.026 -> 0.012 and `player_vol_switch_rows`
15 -> 2 across the same window. Confounded with depth (114k also dipped to
0.41 with the floor live), so the reading is directional, not causal.

**The argument that retires "just raise `player_ent_coef`" permanently.** The
two terms have the SAME minimiser — entropy IS reverse KL to uniform, up to a
constant — so as objectives they are the same thing pointed opposite ways.
What differs is what happens against an opposing force, and the decisive fact
is that the surrogate's expected force on a cell is ALSO pi-prefactored (the
cell is taken on ~pi_b of rows, so ~pi_b * A_b). Against the entropy bonus the
pi_b cancels on BOTH sides, leaving a pure temperature; against forward KL it
cancels on one:

```
entropy   d/dy_b = -pi_b (log pi_b + H)   equilibrium  pi_b ~ exp(-|A|/alpha)   EXPONENTIAL decay
unif-KL   d/dy_b =  pi_b - 1/k            equilibrium  pi_b ~ coef/(k |A|)      LINEAR decay
```

Measured at k=10 (pg_advantages are unit-std, so |A| is in sigma):

| headwind | ent 0.01 | ent 0.05 | ent 0.2 | unif-KL 0.05 |
|---|---|---|---|---|
| 0.1 sigma | 4.5e-05 | 1.4e-01 | 6.1e-01 | 3.3e-02 |
| 0.2 sigma | 2.1e-09 | 1.8e-02 | 3.7e-01 | 2.0e-02 |
| 0.5 sigma | 1.9e-22 | 4.5e-05 | 8.2e-02 | 9.1e-03 |
| 1.0 sigma | 3.7e-44 | 2.1e-09 | 6.7e-03 | 4.8e-03 |

Read across the entropy columns: there is no alpha that holds a floor without
setting the temperature of every OTHER cell to the same value — mass without
discrimination, the phase-1 support-anchor shape. That is what the four
retunes (0.01/0.05/0.1/0.2) each measured one at a time. Two further
properties fall out of the same algebra and are worth stating once: forward
KL contains `-(1/k) log pi_b`, a log BARRIER that diverges as the cell dies,
where reverse KL contains `-pi_b log pi_b`, which vanishes there — entropy
scores a dead cell as costing nothing; and `H` appears inside entropy's own
per-cell force, so entropy earned on a wide move row is directly SUBTRACTED
from the pressure on a starved switch row. That is "one fungible budget,
payable wherever it is cheapest" written in the gradient, and it is why the
live split (`entropy_micro_taken` 0.61 against `entropy_macro` 0.25) is a
stable state rather than a transient.

| mechanism | what | why |
|---|---|---|
| `loss.uniform_kl_rows` + `player_uniform_kl_coef` 0.05 | forward KL from the CONSTANT uniform over each row's legal cells, inside the pg bracket | restored verbatim from `daa4228^`. Sized on the equilibrium above, never on loss balance: 0.05 holds ~0.02 mass against a 0.2 sigma headwind and ~0.009 against 0.5 sigma, and the ask RELAXES as evidence arrives. The one deliberate divergence from the reference actor loss, which carries no mass-independent restorer in either `mag_divergence` mode |

**Judged on a matched BR pair, not on the main lineage.** `sp75c-ckpt_00254992`
launched 2026-08-31 against target `ckpt_00254992` with the same
`--br-init shrink-perturb --br-perturb-frac 0.75` as `sp75b` (vm9b7p07,
stopped at 65k and published into the parent's `players/`), so the control is
identical in init, target and budget and differs by this one term. Read
`prob_switch` / `entropy_macro` against vm9b7p07 at matched steps;
`player_loss_uniform_kl` should FALL as mass returns (the term relaxing, not a
standing tax); the exploitability read is
`league_main_v_254992_winrate` against the 0.57-0.58 the 77638 probe set.
**Abort:** the phase-1 shape again — mass rising while the readout stops
discriminating (`player_pointer_*` drift flat, entropy at ceiling), or
`loss_uniform_kl` pinned with switch mass unmoved, which would say the term is
paying and buying nothing.

**PHASE 1 RESULT (coef 0.05, read at 69k on the matched pair). The abort
fired on its first clause, and the mechanism is not the scale.** Everything
the term promised on its own axis arrived, and the BR was half as strong for
it:

| @ lifetime_step 69k | sp75b control | sp75c uniKL 0.05 |
|---|---|---|
| `entropy_macro` | 0.264 | **0.727** |
| `player_policy_prob_switch` | 0.0151 | **0.0650** |
| `player_policy_prob_ratio` | 0.060 | **0.330** |
| `player_vol_switch_rows` | 3 | **23** |
| `league_main_v_254992_winrate` | **0.343** | 0.186 |

The control collapsed exactly on the recorded schedule (macro 0.94 -> 0.26,
switch 0.115 -> 0.015, supply 35 -> 3) and sp75c held all of it — so this is
not "the term failed to fire", it is the term firing hard and costing
strength. No slower-but-higher story either: control slope 50k->69k
~0.005/1k, sp75c 50k->82k ~0.0022/1k, and sp75c @82k (0.215) is still under
the control @50k (0.248). `mag_coef` 0.2 was nowhere near binding in either
arm (`ref_kl` 0.01-0.03 both), and `ent_coef` 0.01 was identical in both, so
neither is implicated. One seed, one control — the 2x gap sustained over 50k
steps is well outside seed noise at this magnitude, but it is a single pair.

**The design error, stated plainly: the term is a ROW flattener, not a
MODALITY one, and the ledger row above claimed otherwise.** `KL(u || pi)` runs
over every legal cell of the row, so its `pi_b - 1/k` pull separates moves
from each other with exactly the force it uses to restore switch mass — it
buys WHETHER-to-switch and pays in WHICH-move-to-pick. `entropy_micro_taken`
sat at 0.93 normalised (near-uniform over the taken modality's cells) and was
NOT trending down, against the control's 0.84 and falling. That is the phase-1
support-anchor shape, mass without discrimination, which this form was
asserted to avoid: the constant reference does fix the RATCHET (it cannot be
walked flat by a re-snapping reference) but says nothing about the
FLATTENING, and those were conflated when the term was written.

**Next reads, in order.** (1) Cheap: coef 0.025, same form, `sp75d` — mass is
linear in coef at fixed |A|, so equilibrium ~0.035 against control 0.015 and
sp75c 0.055. Exploit recovering to near-control at ~2x the control's mass
means the trade-off curve has a usable middle; exploit still depressed
falsifies the FORM and not the scale. (2) The structural fix if (1) fails:
move the uniform reference to the MODALITY MARGINAL — forward KL from uniform
over modalities, gradient `pi_m - 1/M` with M = 2-3, silent within a modality
— restoring mass on the axis that dies and leaving every within-modality
choice to the critic. That is the phase-4 law reached from the other
direction: the regulariser says WHETHER, never WHICH.

## Addition ledger — 2026-09-01 centralised value, discrete belief code

The user-decided amendment of the no-privileged-info invariant (see the
Invariants section for the amended rule and its gate). One fresh lineage
carries this pass plus the TGN history restructure; plan and baselines in
`docs/central-value-baselines.md` (d7zdz8hw @33k/50k banked there).

| mechanism | what | why |
|---|---|---|
| `opp_private_team = 17` + `REQUEST_LAG = 22` | the opponent's live request serialised through the SAME EntityPrivateNodeFeature rows by ONE parameterised `getPrivateTeam(sourcePlayer)`; staleness = clip(observer turn − source turn, 0, 8), identically 0 on the own channel (rqid deltas drift on force-switch turns — not comparable across players); every degrade (deploy, un-ingested request, spectator) is the all-zero buffer. Field 16 was RE-USED by the action mask after the old deletion — 17, never reuse | the deleted 2026-08-25 sheet was STATIC team truth; the request is live HP/status/PP — the state a decision actually turns on. Harness truth invariant extended to the channel, race-free via `lastSerialisedOppRequest` (the exact object build() serialised; the live request moves between build and check). 5014-battle soak, zero violations |
| leak partition: `OPP_PRIVATE_ENTITY` (6) + `VALUE_CLS` (1), 61 → 68 rows, `SEQUENCE_READ_MASK` | policy-readable rows keep the complete 61×61 attention among themselves and gain ZERO in-edges from the partition; secret rows may read anything but VALUE_CLS; VALUE_CLS reads all, read by nothing (out-degree 0) — leak-freedom transitive across the 6 blocks by induction, ANDed into the trunk mask every block | asymmetric actor-critic (Lambrechts et al. 2025: removes the value-aliasing penalty) with the leak objection answered structurally rather than by convention; V is learner-only so deployment is untouched. `tests/test_privileged_partition.py` pins bit-identity of every policy output under opp-team perturbation WITH both positive controls |
| privileged V + `player_privileged_targets` | second `CategoricalValueLogitHead` on VALUE_CLS, train-gated; CE on the SAME win_returns; True routes v-trace bootstraps → pg_advantages through it, False is bit-for-bit the old estimator (the live fallback) | the deployable head stays trained unchanged — the matched control the 2026-08-25 diagnosis never had. Panels: `player_loss_v_win_priv`, `player_priv_value_head_r2` (THE discriminator, beside the deployable R²), `player_priv_value_gap` (the 0.005 number re-measured) |
| opponent discrete code (`_opp_code_rows`) | per mon: same private embedder → (16 groups × 16 classes) multi-softmax, 1% unimix floor, straight-through argmax → concat of code-table vectors IS the secret-row content; `entity_index_tag` + `opp_private_side_bias` give it the shared additive identity | Dreamer's discrete-latent reading of "the opponent's private state is a KNOWN discrete state" with none of the machinery (no learned posterior, no KL balancing — the privileged value loss grounds the code through the straight-through estimator). Collapse instrument: `player_code_perplexity_{mean,min}` (pinned at 1 = dead group). The belief head (next commit) predicts the code from public rows |

| TGN history: directed messages + HISTORY_ENTITY rows | messages gain the step's SOURCE half — masked-mean node+edge latents of the rows carrying a real major arg, plus `is_src`, so mover and target coexist in ONE vector (the relation the per-slot scatter destroyed); `_embed_edge` reads the never-read `FROM_TYPE_TOKEN*`; the additive history-into-public-row merge (the 2026-08-28 "11th attribute token") is DELETED and history becomes its own 12-row policy-readable HISTORY_ENTITY group (68 → 80 rows) carrying GRU state + the latest-node snapshot the RL path used to discard, joined to its board row by the shared entity_index_tag | the current encoder was a degenerate Temporal Graph Network (Rossi et al. 2020): destination-only messages, memory-only readout. Reference diff recorded: TGN's source MEMORY in messages is carry-dependent (serial, against the ~26us/step bound) so the source's raw cache embedding stands in; sum aggregation kept (Souza et al. 2022); the latest-node-beside-memory readout is TGN's staleness fix, independently rediscovered by the ledger. ZERO added serial work — everything lands in the batched precompute. Panel: `player_history_src_frac` (expect >> 0.5) |

Also in this pass, on the CURRENT lineage: history reads all EIGHT
`RELEVANT_ENTITY_IDX` columns (was 0..3 against the service's 8 — spread
rows silently dropped; params-compatible), and `ex.bin` regenerated with
the opponent channel live (57/58 steps populated; step 0 is the documented
un-ingested-request race).

## Removal + addition ledger — 2026-08-31 grid retired, modality-marginal KL

One fresh lineage (new param tree accepted). Revert handle: the two commits on
`flat-trunk` following `4c3948d`.

| mechanism | what | why |
|---|---|---|
| 41x41 action grid, everywhere | `Action{src,tgt}` -> `Action{cell}` (an index into the 295-cell BLOCK SPACE: ActionMask's own fields flattened — 6 switch, 16x17 move x target, 17 standalone; layout documented beside the proto message, offsets derived from the slot-list lengths on BOTH sides); `FlatActionReadout` emits the three blocks it always computed and the scatter dies; the service builds `structuredMask` + `legalChoiceByCell` directly from block cells and its internal `OneDBoolean` grid dies; `FLAT_MODALITY_MASK` -> `CELL_MODALITY_MASK` (length 295, IDENTICAL per-cell values, so `entropy_macro` and every modality consumer keeps its meaning); `ally_switch_bias` (2,1) folds to one `switch_bias` scalar (row 1 never trained in singles; at preview a uniform shift over an all-switch set is softmax-invariant); `src_index`/`tgt_index` leave the stored pytrees; `kind`/`active_slot` become purely decoder-side (test-pinned: the mask cells are kind-invariant, harness.ts asserts the decode half) | ~82% of the grid was unreachable in any format; the readout, the wire mask and the service's choice map were all already block-shaped, and every grid artefact was adapter code between them. Stored mask leaf shrinks 5.7x. Shards survive via the packed-grid fold shim (`_cells_from_packed_grid`); prev-action info features keep their ActionEnum vocabulary via decode-time conversion (`cellToEnumPair`). 2049-battle soak, zero mask/decoder violations; probe C's row addressing ported off the dead 41-slot action stream onto named sequence rows in the same pass (it silently indexed sequence rows by ActionEnum VALUE — reads were garbage on the flat trunk) |
| `loss.uniform_kl_rows` -> `loss.uniform_kl_modalities` | forward KL from uniform over LIVE MODALITIES; modality-level gradient exactly pi_m − 1/M; the loss reads the marginals ALONE, so within-modality redistribution is identically invariant and the per-cell force follows the policy's own conditional. Metric renamed `player_loss_modality_kl` (the loss changed meaning — the coma->neurd precedent). Coef stays `player_uniform_kl_coef` = 0.025: with M = 2–3 against the row form's k ~ 10 the same coef buys ~5x the modality mass — equilibrium switch mass ~0.06 at 0.2 sigma headwind, ~0.025 at 0.5 sigma | the pre-registered structural fix from the sp75c falsification: the row form's pi_b − 1/k separated moves from each other with the same force it restored switch mass with — every mass metric bought (macro-H 2.8x, prob_switch 4.3x) and the exploit HALVED (0.186 vs 0.343 @69k), `entropy_micro_taken` pinned 0.93. The phase-4 law — the regulariser says WHETHER, never WHICH — is now an algebraic identity of the term rather than an aspiration. Tests: pi_m − 1/M gradient, full pull at a starved marginal, within-modality swap invariance WITH the cross-modality positive control, forced rows silent by construction |
| private TRUTH CHANNEL + alignment key | `EntityPrivateNodeFeature` += HP_RATIO/STATUS/HAS_STATUS/TOXIC_TURNS/SLEEP_TURNS/FAINTED (encodings identical to the public fields) + ENTITY_IDX (1 + stable entity index, 0 = never fielded); hp/status/fainted parsed from the REQUEST's own condition string; the encoder widens `private_state_linear` with the condition block and applies a shared `entity_index_tag` table to public rows (via PUBLIC_ORDER) and private rows (via ENTITY_IDX) so a sheet row and its public row carry the SAME additive identity | the switch-action slots were condition-blind: probe C measured the trunk's public-row workaround at the floor (r ~ 0.00 vs 0.35-0.44 controls) and the private rows carried only static set descriptors. A JOIN cannot fix it: under a my-side Illusion the public row BLENDS two mons' histories until `|replace|`, so only the request has the truth — and the harness truth invariant caught a second sourcing trap ON ITS FIRST RUN: the privateBattle member's hp is log-event-driven and reads 0/0 before the log's first reading (request said 252/342), which is why the CONDITION STRING and not the client object is the source. The two condition blocks are not duplicates — they coincide only when no deception is active and diverge into truth-vs-opponent-belief exactly when it is. Decode right-pads short `private_team` buffers (shards store all-zero blocks; appending safe, renumbering never). Known ~0.1% class: a disguised my-side mon's index attaches to no public row until reveal — the wire is CORRECT there (no public identity yet), vitest retry absorbs it |


Also in this pass: player Adam eps 1e-8 -> 1e-5 (the reference diff's flagged
"one to test"; builder unchanged). sp75c stopped at ~283k (winrate vs target
0.48 and climbing, prob_switch ~0.026 held; published `p_100254992`) — the
proof that mass + strength coexist under a zero-avoider, and the measurement
that priced the row form's WHICH-tax.

## Addition ledger — 2026-09-05 hidden-token belief label (B3 fired)

The revealed-row control (2026-09-04) caught the belief label: the code is
trained only through the privileged value CE, so it encoded whatever the
sheet carries, public tokens included, and a control reading the matched
mon's OWN pre-trunk public row converged to the head's accuracy
(irqeetfg: belief 0.89, revealed 0.87, species 0.59;
`player_belief_context_margin` 0.20 @634k → 0.017 @1.15M — the launch
margin was the fresh control lagging, not inference). B3's pre-registered
verdict: hidden-token code.

| mechanism | what | why |
|---|---|---|
| `encoder.OppCodeLabels` + `Encoder._hidden_code` | the SAME trained pool + `opp_code_logits` applied a second time per opp mon with every token the matched public row already shows masked out (id-equality on species/ability/item; moves by SET against MOVEID0-3; the state token never hidden — hp/status are what the revealed row reads; an unmatched mon is all hidden), hard argmax, no unimix, all under stop_gradient; `hidden_any` False = fully-revealed mon, skipped by the loss. `belief_alignment` moved into the encoder (computed once, re-exported from player_model). NO new params — the critic's secret rows and its full code are untouched (user constraint: the critic's INPUT stays normal, only the belief TARGET is hidden-only); checkpoint-mode resume | the label can encode nothing the public row carries, so predicting it from that row is inference about unseen tokens by construction. Panels: `player_belief_hidden_frac` (of matched mons, the share with a hidden token), `player_hidden_code_perplexity_{mean,min}` (the label's own usage over the scored rows; min 1 = dead label). Every `player_belief_*` panel breaks meaning at irqeetfg ~1.15M — kept the names so the positive control reads ACROSS the boundary: `player_revealed_belief_accuracy_above_marginal` must FALL to the species control's, `player_belief_accuracy_above_marginal` must stay above both; read on above-marginal, never raw accuracy (perplexity floors it) |

In randbats a mon's hidden set is nearly independent of context given
its own revealed tokens, so the achievable margin is small by nature —
the gate is `above_marginal` > species AND > revealed after a 20k hold,
else the head is a hidden-token hash lookup and goes. Tests:
`tests/test_hidden_code_label.py` (label blind to hp WITH the hidden-move
positive control; full reveal empties `hidden_any`, unmatching refills
it), `test_belief_telemetry` row_mask narrowing; the control tests score
the hidden label. Owed at the next stop: `tests/test_train_step.py` (2GB
VRAM free beside the live learner — not compiled here).

## Probe ledger — 2026-09-05 stochastic transition, Step 1 (the mean head priced)

Step 1 of the stochastic-transition plan (local `docs/stochastic-transition-plan.md`):
instruments only, no behaviour change. The question was whether the
`dynamics_delta_head` — a conditional MEAN over the transition between two
of MY requests, which folds the opponent's unobserved choice, the engine's
dice and the reveals into one average — is a usable one-step model, or
whether search needs a sampleable latent.

| piece | what |
|---|---|
| `rl/offline/transition_probe.py` | plays self-play games on a checkpoint through the second service and reads two things per t→t+1 pair: the residual `delta − pred` on hp-moved public rows projected onto `dynamics_hp_basis` (GMM 1-vs-2 BIC, Ashman's D, with the fainted/survived split as the positive control), and the VALUE GAP — the predicted pre-trunk rows substituted into the REAL t+1 sequence (row biases re-added, the substitution propagated onto the four entity-derived target rows), the frozen target network's `V(sub)` against `V(real t+1)`. Two variants: history rows live, and history rows MASKED from the trunk's read — the entity diaries and field memory at t+1 already encode the real outcome, so with them live even a COPY predictor (rows unchanged from t) reads a small gap; the masked read is the honest one, and the INFO row's request kind still leaks a force-switch either way |
| `transition_edges` / `transition_reveals` + panels | learner-side: spanned window steps per transition (`FIELD_FEATURE__REQUEST_COUNT == t+1`), `player_transition_edges_{mean,p90}`, `player_transition_reveal_frac` (a matched opponent row whose `REVEALED_ID_COLUMNS` changed), `player_dynamics_gain_public_{short,long,reveal,no_reveal}` (≤2 vs ≥4 edges). Read on the next restart — irqeetfg predates them |

**Numbers (ckpt_01220000, 60 self-play games, 3340 transitions; hp_moved
0.762 of transitions, faint 0.334, reveal 0.375, edges mean 2.61 / p90 4).**
Value gap is `|V(sub) − V(real t+1)|` in CAT_VF_SUPPORT units (lower = the
predicted state is worth what the real one is worth):

| | mean head | copy predictor | \|V(t+1) − V(t)\| |
|---|---|---|---|
| all, history live | 0.031 (p90 0.078) | 0.043 | 0.118 |
| hp_moved, history live | 0.034 (p90 0.082) | 0.049 | 0.125 |
| faint, history live | 0.034 | 0.052 | 0.152 |
| all, history masked | 0.045 | 0.062 | — |
| hp_moved, history masked | **0.051** (p90 0.111) | 0.074 | — |
| faint, history masked | 0.051 | 0.083 | — |
| short / long, masked | 0.045 / 0.038 | 0.057 / 0.069 | — |
| reveal / no_reveal, masked | 0.053 / 0.040 | 0.072 / 0.056 | — |

Signed bias `V(sub) − V(real)` +0.007. Residual on hp-moved rows (n=3862,
0.295 fainted): hp-subspace gain 0.611; far_frac (|residual| ≥ half the
true delta) 0.694 — 0.372 on fainted rows, 0.829 on survived; GMM BIC
1-comp 9360 vs 2-comp 8825 (two favoured), component means −0.157 / +0.498
at weights 0.76 / 0.24, Ashman D 0.68 (overlapping, not separated);
faint control fainted +0.754 vs survived −0.316, D 1.55.

**Verdict: neither pre-registered branch fired cleanly.** The ≥0.10
value-gap bar FAILED (0.034 live / 0.051 masked on hp-moved rows) — the mean
head's state is worth within ~0.05 of the real one, roughly a third of the
step-to-step value movement; the residual IS two-component by BIC but the
components overlap (D 0.68 against the ≥2 that "bimodal" means), and a
faint is largely predictable (far_frac 0.37 on fainted rows — the mean is
NOT sitting between the branches on most faints). The re-plan branch
(unimodal AND gap < 0.03) did not fire either: masked gap 0.051, BIC says
two. What the read does establish: the mean closes only ~30% of the copy
predictor's gap (0.074 → 0.051 masked), `long` transitions cost the mean
nothing over `short` (0.038 vs 0.045 — the semi-Markov diagnosis is NOT
supported on this instrument), and reveals cost ~0.013. The go/shrink
decision is the user's; the learner-side `gain_long < gain_short` clause
can only be read after the next restart.

## Removal + addition ledger — 2026-09-05 stochastic transition, Step 2 (the latent model)

Step 2 of the plan (local `docs/stochastic-transition-plan.md`; maths in
`docs/stochastic-transition-notes.md`). The user's call after Step 1's
mixed read: proceed in full with the chance code — in this metagame chance
is less prevalent, in other formats a sampleable transition is necessary.
Checkpoint-mode by-path resume of irqeetfg (the policy's inputs and the
value/readout params are unchanged; the delta head's leaves drop, the
transition leaves init fresh).

| mechanism | what | why |
|---|---|---|
| `dynamics_delta_head` (REMOVED) + `player_dynamics_gain_*` / `player_loss_dynamics` / `player_dynamics_head_*` panels | the conditional-MEAN predictor of the 21 pre-trunk target rows, live 362.8k–1.22M. Last readings banked in the Step-1 ledger (gain public 0.528 / hp_moved 0.588 / loss 0.453 @1.168M; value gap 0.034 live / 0.051 masked on hp-moved rows) | a mean over discrete unobserved branches cannot be rolled out; Step 1 priced it. Its LABEL survives as the grounding head |
| `rl/model/transition.py::TransitionModel` | g(h_t, a, z) → ĥ_{t+1} over the 73 post-trunk policy-readable rows: 2 `TrunkBlock`s of the trunk's own shape under the policy-readable sub-block of `SEQUENCE_READ_MASK`, conditioned by ONE vector added to every row (`action_proj` of the taken cell's src/tgt readout rows + `code_proj` of the code-table embedding); ĥ = rows + `out_proj`(blocks). Chance code z: `transition.code_groups` 2 × `code_classes` 16, prior MLP over [masked-mean(h_t); src; tgt], posterior the same features + masked-mean(sg(rows at t+1)), f32 logits, 1% unimix, straight-through argmax; the training decode uses the posterior sample, a no-gradient decode from the prior MODE feeds the `_prior` panels. `code_groups = 0` is the static mean latent model (drops exactly `{code_table, code_proj, prior_net, posterior_net}`) | **`out_proj` is THE single zero factor**: g is exactly the copy predictor at init (consistency gain 0, the old head's own contract), its gradient is the live block output ⊗ residual, so it moves at step 1 and `code_proj` / `action_proj` / the blocks / `code_table` unfreeze at step 2 — the readout's query/key rule, pinned by `test_out_proj_is_the_single_zero_factor` (fresh: only out_proj live; opened via `open_zero_init_paths(..., ["out_proj"])`: the rest live). Only policy-readable rows exist here, so a rollout carries nothing privileged (`test_privileged_partition` pins every transition leaf bit-identical under `opp_private_team` perturbation, the posterior included — it reads leak-free rows) |
| heads on ĥ (`player_model._forward_transition`) | consistency: per-row normalised MSE vs sg(next rows), per-`SequenceGroup` scale with the 1e-2 floor, learner-only groups skipped; grounding: `ground_head` MLP per imagined target row → the 21 DYNAMICS_TARGET_ROWS' pre-trunk content at t+1, in the NEXT step's layout; value: the SHARED `v_head` on ĥ's CLS row, CE to the t+1 `win_returns`; policy: the SHARED `action_head` on ĥ, forward KL to sg π(t+1) over the real next mask; `mask_head` (a second `FlatActionReadout`): 295-cell BCE on the next `action_mask`; `cls_head` on ĥ's CLS: next `ActionRequestKind` CE + done BCE | every label is OBSERVED (the opponent's choice is never one). `v_head` and `action_head` TRAIN through imagined rows — MuZero's value/policy targets on the shared heads, which is what makes `player_transition_value_r2` a calibration read of the same head search will call |
| positional t→t+1 pairing | `next_rows = sg(concat(rows[1:], rows[-1:]))`; the last chunk row self-pairs and is masked by `valid_step = acted_mask[:-1] & value_mask[1:]`; grounding gathers the prediction into the next step's row order through `dynamics_alignment`'s `next_index` and normalises by `\|target_{t+1} − target_t\|²`, so an ALIGNED copy predictor scores exactly gain 0 across a resort and a scatter-negated one 4 (test-pinned); `player_transition_rows_frac` = valid transitions / (T−1) | the old head predicted the delta in the CURRENT layout and the target rows re-sort between requests (actives first) |
| loss | `player_dynamics_coef` 0.5 × [cons + ground + value + policy + mask + kind + done + `player_transition_dyn_coef` 0.5 · max(F, KL(sg q‖p)) + `player_transition_rep_coef` 0.1 · max(F, KL(q‖sg p))], `player_transition_free_nats` F = 1.0 per transition summed over groups; every KL f32 | DreamerV3's balancing and free bits verbatim; test pins that doubling dyn_coef doubles the PRIOR's gradient only and that identical prior/posterior under F=1 gives zero code gradients with `player_transition_kl_free_frac` 1 |
| panels `player_transition_*` | `kl` (+`_short/_long`, `_reveal/_no_reveal`), `kl_free_frac`, `{prior,post}_perplexity_{mean,min}`, `prior_post_agree`, `gain_{public,private,field}` (posterior decode), `gain_public_prior` / `gain_hp_moved_prior` (**expected BELOW the mean head's 0.528 / 0.588** — a sample from a two-branch law is further from the truth in MSE than the mean), `cons_gain_<group>`, `value_r2` beside `player_value_head_r2`, `value_gap`, `mask_acc` / `mask_recall` / `mask_exact_frac`, `kind_acc`, `done_acc`, `hp_share`, `edges_{mean,p90}`, `reveal_frac`, `{out_proj,action_proj,code_proj,code_table}_rms`, `{transition,blocks,prior,posterior}_grad_norm`; wandb section "3b · Transition model" | the coma→neurd precedent: the loss changed meaning, so the `player_dynamics_*` names retire and the new ones start fresh at the restart step |

**Recorded divergences from the plan text.** (1) The consistency target
and the posterior read sg'd rows from the SAME online forward, not the EMA
`player_target_pred` forward — the EMA rows would have cost a second
73-row trunk pass per learner step for a target whose only virtue is
smoothness, and the trunk already sees these rows under stop-gradient.
(2) `done_auc` shipped as done BCE + `done_acc` (an AUC is not computable
inside the jitted step without a sort; accuracy against `done_frac` reads
the same thing). (3) `transition.unroll` was written and deleted unread —
the K-step variant is Step 5's rung and gets its knob with its consumer.
(4) `rl/offline/transition_probe.py` now decodes from the prior MODE
(`ground_prior`); the sampled-decode calibration read
(`|E_z V(g) − V(real)|`) is scoped with Step 3's search module, which is
where the batched prior sampling lives.

**Pre-registered acceptance (20k hold after the resume)** — in the plan and
unchanged: `kl` 0.5–3.0 nats, `post_perplexity_min` ≥ 1.5, `prior_perplexity_min`
≥ 1.3, `gain_public` ≥ 0.68, `gain_hp_moved` ≥ 0.75, `value_r2` ≥ 0.85,
`mask_acc` ≥ 0.98 with `kind_acc` ≥ 0.95, `kl_long > kl_short` and
`kl_reveal > kl_no_reveal` by ≥ 0.2, `player_value_head_r2` /
`player_learner_actor_forward_kl` / temp-1 eval wr inside their pre-change
band, `learner_steps_per_sec` ≥ 6.5 (from 7.19). Abort ladder in the plan
(posterior collapse → rep 0.05 + F 2 once; copying → K 8; dead groups → FSQ;
strength cost → coef 0.25 then sg on g's input; `value_r2` < 0.7 → search
does not launch).

**LAUNCH CHECK, 1266k–1292k (2026-09-05): the grounding head shipped
predicting CONTENT and was rewritten to the DELTA form, own commit.** Every
other panel opened inside its band (`kl` 0.65–0.76, `post_perplexity_min`
~4.2, `prior_perplexity_min` ~3.6, `mask_acc` 0.995, `kind_acc` 0.99,
`value_r2` tracking `player_value_head_r2` batch for batch, `out_proj_rms`
0.0073, forward KL ~0.003) — but `player_loss_transition_ground` started at
17.8 (→ 4.8 by 1292k), `gain_public` −12 → −5.4, `gain_private` −36 → −5.9,
and `player_transition_grad_norm` sat at 22 → 18.3 against the global clip
of 10 (`blocks_grad_norm` 1.3 → 3.2). The head predicted the 21 target
rows' full pre-trunk content at t+1 through a lecun-init output, and the
loss normalised that by the DELTA's mean squared size — so the static
tokens' reconstruction error (species, item, moveset: unchanged across a
transition, large in norm) was scored against a normaliser sized for what
changes, and one head's gradient alone was 2x the clip, spending every
other loss's step (the eed695e shape again). Fix: `ground_delta_head`
predicts the t → t+1 CHANGE of each target row in the next step's layout
(the old delta head's label) through a ZERO-init output, so the head starts
exactly at the copy predictor (gain 0, loss 1) and every number it emits is
on the delta's own scale. RENAMED, not edited in place: the by-path merge
loads any same-name same-shape leaf, so a content-trained kernel would have
resumed under the delta loss at ~18 again — a changed-meaning module with
unchanged shapes must change its NAME to init fresh (Adam moments included).
Rule with teeth: **a grounding head predicts the delta, zero-init, or its
static-token reconstruction error dominates the gradient.** Tests pin the
all-zero prediction at exactly loss 1 / gain 0 across a resort, the exact
delta at 0, its negation at 4, and content-constant invariance.

**LAUNCH CHECK 2, 1294k–1310k (2026-09-05): the relaunch fixed the
gradient's MAGNITUDE and not its DIRECTION — the switch axis collapsed
under the transition term, and the model is now a learner-side observer.**
Windowed 4k means, 1250k → 1294k (delta-head relaunch) → 1310k:
`player_policy_prob_switch` 0.036–0.041 → 0.018–0.029 → 0.013–0.015,
`player_entropy_macro` 0.52 → 0.29, `player_vol_switch_rows` 12 → 3.5,
`player_taken_voluntary_switch_frac` 0.107 → 0.035, `player_loss_modality_kl`
1.0 → 1.33 (the zero-avoider losing), `player_trunk_grad_norm` 3.7–5.3 →
8–11.5 with `player_gradient_norm` 10.4–13.7 against the clip of 10, while
`player_transition_grad_norm` sat at 3.2–3.8 — the transition SUBTREE was
quiet and the trunk was not, because the transition losses reached the
trunk and the shared heads through paths no transition panel counted.
Mechanism: with `out_proj_rms` 0.0056 and shrinking, ĥ_t ≈ h_t, so the
next-policy loss through the LIVE shared readout was KL(π_{t+1} ‖ π_t) on
the real trunk — an unregularised temporal-smoothing force (entropy, the
modality KL and the magnet see real rows only) that reads anti-switch on
every voluntary-switch row, with the consistency loss pulling h_t toward
h_{t+1} in the same shape. The strength-cost abort ladder's first rung
(coef 0.5 → 0.25) was SKIPPED under the standing law — a coefficient cut
that only delays onset is falsified — and its second rung landed directly,
widened to both ends: g's INPUT rows and the consistency target are both
under `stop_gradient`, and the shared `action_head` / `v_head` are applied
to imagined rows through FROZEN copies of their params
(`clone().apply(stop_gradient(<head>.variables), ...)` — same output, no
param gradient, live input gradient, no duplicate tree). g learns to
write rows the real heads already read, which is what search needs; the
heads and the trunk never learn from imagined rows. `player_transition_value_r2`
keeps its meaning (a calibration read of the head search will call) —
what changes is that it no longer trains that head. Rule with teeth: **a
learned model's losses reach the model and nothing the model reads — the
representation it predicts and the heads it is scored by are frozen from
its side.** The reach test is inverted to pin exactly `{transition}` on
opened params, with the real value CE + log-policy as the control that
the encoder and the shared heads are reachable. Checkpoint-mode resume
(no param moved); the Step 2 20k hold restarts from this relaunch, and
the strength clause is judged from it.

**LAUNCH CHECK 3, 1266k → 1312k (2026-09-05, the observer relaunch
resumed from `ckpt_01266269` after the collapse era was archived to
`ckpts/archive/gen9_transition_collapse_irqeetfg_20260905/`): the
policy side is fixed and the CHANCE CODE is not learning.** Trunk grad
norm back to 3.5–5.5, `prob_switch` held ~0.04, every observed head in
band (`mask_acc` 0.994, `kind_acc` 0.998, `value_r2` 0.89) — but
`player_transition_kl_free_frac` sat at 0.93–0.94: 93% of transitions
had a total KL (0.64 nats, summed over the 2 groups) UNDER the 1-nat
free-bits floor, so both KL halves gave zero gradient on them.
`prior_grad_norm` 0.12 → 0.08, `prior_post_agree` 0.58 → 0.30,
`gain_public_prior` 0.19 → −0.01 (a decode from the prior's mode is the
copy predictor — fatal to search, which samples z from the prior), while
the posterior trained through the straight-through decode alone
(`posterior_grad_norm` 0.23 → 0.37, `gain_public` 0.26 → 0.43) with its
usage collapsing (`post_perplexity_mean` 2.6 → 2.35 of 16). Also
`kl_long` 0.52 < `kl_short` 0.66 and falling, reveal margin +0.04. Two
defects, two commits (Step 2b in the plan):

| change | what | why |
|---|---|---|
| `player_transition_free_nats` 1.0 → 0.0625 | DreamerV3's 1 nat is over a 32-group code — 1/32 nat per group; ours has 2. The clip stays on both halves (reference form); at KL 0.64 it no longer binds, so the prior trains on every transition and the rep half pulls the posterior toward it | the floor was sized for the reference's code width and trapped ours. The old comment's "F 1 → 2" collapse fallback is RETIRED — that rung would re-trap the prior; the floor never goes up. Stochastic MuZero's form (prior CE to the sg'd code, no floor) is the F = 0 end of the same knob; 0.0625 keeps a floor against over-compression once the KL does fall |
| `transition.RowRead` (`cfg.transition.row_read_width` 16) + `prior_net` / `posterior_net` → `prior_read_net` / `posterior_read_net` | ONE shared Dense(D → 16) per row, masked by validity, flattened in row order (73 × 16): prior reads `[RowRead(h_t); src; tgt]`, posterior `[prior features; RowRead(h_{t+1} − h_t)]` under `row_valid & next_valid`; the 73-row mean pool is deleted. RENAMED so the by-path merge inits the whole nets and their Adam moments fresh (the grounding-head rule: only the first kernel's shape moves, the later layers would resume weights trained on features that no longer exist); `code_table` / `code_proj` / the blocks / `out_proj` carry over | the mean pool cancelled row identity — a row's additive bias is present at t and t+1 alike, so "my active lost 40%" and "theirs did" pooled to the same vector and a one-row change was diluted 73×; the posterior could not see the branch it was meant to code (`kl_long` 0.52 < `kl_short` 0.66, reveal margin +0.04). Every reference posterior reads the FULL next embedding. Tests: the same delta on row 2 vs row 9 moves the posterior logits (a mean pool is invariant to it by construction); invalid rows contribute exactly zero WITH the live-row control |

**Pre-registered acceptance (20k hold from this relaunch), the Step 2
gate plus the mechanism reads:** `kl_free_frac` ≤ 0.2; `prior_grad_norm`
not falling toward 0; `prior_post_agree` ≥ 0.5 and rising (from 0.30);
`gain_public_prior` > 0.2 and within 0.2 of `gain_public`;
`post_perplexity_mean` ≥ 3 (from 2.35); `kl_long > kl_short` and
`kl_reveal > kl_no_reveal` by ≥ 0.2; `kl` in 0.5–3.0; `mask_acc` /
`kind_acc` / `value_r2` / `player_value_head_r2` / prob_switch / trunk
grad norm inside their bands (the observer form guarantees the last
three — a move there is a leak). **Abort ladder:** posterior collapse
(`kl` < 0.1 for 5k at the copy predictor's gains) → `rep_coef` 0.1 → 0.05
once, never F upward; prior not tracking (`agree` flat < 0.4 with
`kl_free_frac` ≤ 0.2) → the prior's READ is the deficit, a learned-query
attention read, own commit; `kl_long` still ≤ `kl_short` with the row
read live → the code is chance-dominated on this clock, drop the clause.
Declined: K 16 → 8 (usage is low because nothing trained it up), β_dyn
up (zero gradient, not a small one), FSQ (dead-group rung unfired,
`post_perplexity_min` 2.16), resuming from 1266k again (the policy side
is fine, g's blocks are 46k ahead).

## Addition ledger — 2026-09-06 stochastic transition, Step 3 (search on an eval actor)

**Step 2b read at ~1545k (the 20k hold, banked before Step 3 launched):
the code is informative but low-entropy, and the gate did not pass
cleanly.** Passing / moving the right way: `prior_grad_norm` flat at
0.15–0.18 (not falling), `gain_public_prior` 0.29–0.31 against
`gain_public` 0.47–0.51 (the prior's mode decodes a real branch, within
0.2 of the posterior's), `kl_long` / `kl_short` 0.21 / 0.14 and
`kl_reveal` / `kl_no_reveal` 0.21 / 0.16 (both signs right; the margins
are short of 0.2 nats because the TOTAL KL is 0.18). Failing: `kl`
0.12 → 0.18 (band 0.5–3.0), `kl_free_frac` 0.46 → 0.37 (gate ≤ 0.2 —
the 0.0625 floor still binds on a third of transitions),
`post_perplexity_mean` 2.61 (gate ≥ 3), `prior_post_agree` peaked 0.58
at 1370k and slid to 0.46. No abort clause fired; the observers held
(`prob_switch` 0.037–0.043, `value_r2` 0.84–0.92, `mask_acc` 0.995,
~6.2 steps/s). The plan's verdict for a code that carries SOMETHING is
to price it by play rather than retune it, so Step 3 launches on this
model as-is; the KL band is re-judged on the search read.

| mechanism | what | why |
|---|---|---|
| `rl/model/search.py::depth_one_expectimax` + `cfg.search` (`enabled` False, `num_samples` 8, `temp` 0.1, `max_cells` 16) | rung 1 of the plan: for every legal root cell (the first `max_cells` by index; `legal_truncated` counts the rest) sample `num_samples` codes from the transition PRIOR, decode each with `imagine`, read the SHARED `v_head` on the imagined CLS row; `Q(a) = E_z[V(g(h, a, z))]`, scattered back onto the 295 cells with a `segment_sum` (padding lands on cell 0 with value 0 — a set-scatter would corrupt a legal cell 0), and `Q / temp` added to the readout's logits as `search_bonus` before the sampler. `_legal_logits` writes the −1e9 masking once for the readout and the search | the model is judged by PLAY, not by MSE: the same params through a network with `search.enabled` (a static config branch — the search slot builds its own network object and shares the params container) at temp 1.0 against the `-t1` slot is the matched pair. `search_bonus=None` is bit-identical to today's forward; the searched arm's sampling stream differs by one extra `make_rng("sampling")` split, by construction |
| `SearchOutput` on `PlayerActorOutput.search` (`root_kl`, `search_value`, `root_value_gap`, `num_legal`, `legal_truncated`; every leaf `()` off the search arm) | per-decision diagnostics computed in the forward: `root_kl` = KL(π_search ‖ π) over legal cells, `search_value` = E_{π_search}[Q], `root_value_gap` = search value − V at the root | the training path stores no search leaves (`()`), so the buffer, the chunk contract and the shape lattice are untouched; the eval actor reads them off the last chunk |
| `eval_search_slots` (1) + `EvalActor-simpleheuristic-search-*` | one more eval thread against the LAST baseline, `is_eval=True` (the leak guard is flag-gated), same EMA params; per-game logs from `main.eval_game_logs`: `switch-frac-*` (voluntary switches per decision that offered both a switch and a non-switch cell — read beside the `-t1` slot's, which now logs it too), `ms-per-step-*` (unroll wall time per real step, the search's price on the CPU actor path), `search-root-kl-*`, `search-value-gap-*`, `search-legal-truncated-*`; wandb section "3c · Search eval" | the headline is `wr(search) − wr(t1)` on the same checkpoint — the model's worth in play. 0 = the search network is not even built |

**Pre-registered acceptance (read over ≥ 300 eval games per arm):**
`wr(search) − wr(t1)` ≥ +0.03 (a third of Jaxcalibur's 100–150 Elo from
ENGINE search is the bar for a learned one-step model), `root_kl` in
0.05–0.5 (an operator that moves nothing is inert, one that replaces π is
reading noise — the smoke read on fresh-opened params was 0.01),
`root_value_gap` sign-consistent with the wr delta. Rung 2 (pUCT over
decision nodes, chance nodes sampling z) launches only on a pass and is
judged on a further +0.02. **Fallback, in the plan's words:** wr delta
≤ 0 with `root_kl` > 0.05 → the model is confidently wrong, back to the
calibration reads (`value_r2`, the offline prior-sample gap) before any
search change; wr delta ~0 with `root_kl` ~0 → search agrees with π and
rung 2 is the next READ, not a fix. No `temp` / `num_samples` ladder —
that is the coefficient-retune trap. Tests: `tests/test_search.py` —
root Q ranks the legal cells by a hand-set imagined value (the positive
control), illegal cells exactly 0, cell 0 legal under padding equals its
own value, truncation counted; diagnostics 0 at zero bonus and
`search_value = Σ π_search·q` under a tilt; `eval_game_logs` on a
synthetic trajectory (acted rows only, `()` leaves skipped); gpu/slow:
the real network with `action_head` + `out_proj` OPENED (fresh params
make every logit 0 and g the copy predictor, so the read would pass
vacuously) — the searched arm's entropy differs from the plain arm's
and the plain arm carries no search leaves.

## Addition ledger — 2026-09-06 stochastic transition, Step 3b (the model priced by search; B and D)

**Step 3 launch read fired the "search agrees with π" branch** (`root_kl`
0.015–0.04, 3.4x ms/decision, search removing switch mass 0.131 →
0.089), and the "Do 1" probe priced it: `rl/offline/search_samples_probe.py`
on ckpt_01560000, 500 self-play roots, the EXACT `Q(a) = E_z[V(g(h,a,z))]`
over all 256 codes (expectation units over CAT_VF_SUPPORT, range 2 —
halve for win probability): `spread` (max − min Q over legal cells) mean
0.039 / median 0.029, `sigma_z` (std of V over z at fixed a) 0.060,
`oracle_gain` (best Q − E_π[Q], the most a perfect operator could buy)
0.014 / p90 0.026, `root_kl` at 1/8/32/64 draws 0.079/0.029/0.022/0.020
with `snr` 0.70/1.98/3.97/5.61 — ~0.02 is the depth-1 operator's ceiling
and sampling is NOT the lever; `q_best_switch_minus_best_move` −0.090
(the model's Q^π ranks a move above every switch at 90%+ of roots;
diagnosis (f), open). **Corrections banked so they are not re-made:**
`value_r2` 0.84 was VACUOUS — V barely moves between requests (mean
|ΔV| 0.12), so V(h_t) scores high on the t+1 label by itself; the honest
number is an R² on the CHANGE in value, where the copy predictor is
exactly 0. Search removing switch mass is faithfulness to Q^π (the
entropy and modality terms hold mass above greedy), not a broken model.

| mechanism | what | why |
|---|---|---|
| A: `player_transition_value_delta_r2{,_prior,_switch,_move}`, `player_transition_value_gain{,_prior}`, `value_gap_{prior,switch,move}`, `policy_kl_copy`, `pred_rms`, `player_value_head_grad_norm`; `TransitionOutput.pred_prior` | delta-R² = R² of (V(ĥ) − V(h_t)) on (V(h_{t+1}) − V(h_t)) over valid transitions, copy = 0; value gain = (ce_copy − ce_imagined)/max(ce_copy − ce_real, 1e-3) on the t+1 two-hot label, copy 0 / real 1; every `_prior` panel reads a no-gradient decode from the prior's MODE — the rollout-side number search samples from | pre-fix offline (ckpt_01560000, `--calibration`): delta-R² posterior 0.446 / prior **−0.242** (move-taken −0.283 drives it), MSE gain 0.296 / −0.106 — the posterior decode moves value the right way, the prior's mode is WORSE than not rolling out. The CE-form offline read (1.006 / 0.659) is outlier-driven against a one-hot MC label and is not to be read; the learner panel uses the two-hot v-trace label |
| B: `player_transition_value_trains_v_head` True + `player_transition_cons_coef` 1.0 → 0.0 | the shared `v_head` trains through the imagined CLS row (MuZero's value target on the real t+1 win_returns — the same labels on a wider input distribution); the trunk stays unreachable (g's input sg'd) and the readout FROZEN on imagined rows (launch check 2's collapse lived there); False = the frozen clone bit for bit. Raw-row consistency leaves the gradient, still logged as a read (`cons_gain_<group>`, `loss_transition_cons`) | the per-row MSE is minimised by the conditional MEAN of h_{t+1}, exactly what a sampleable model must not produce, and it was the largest term the blocks saw — `out_proj_rms` 0.0156 (a quarter of lecun) and prior-mode grounding 0.336 vs the mean head's 0.528 read as its signature. **The observer rule is AMENDED by user decision: transition losses reach `{transition, v_head}` and nothing else** (reach test pins the pair on / `{transition}` off) |

**B RESULT (1654k → 1674k, the 20k hold, 2026-09-06): the matched
control held and NOTHING on the decode side moved — diagnosis (a)
falsified as written.** Windowed means first half / second half /
last quarter: `out_proj_rms` 0.0168 / 0.0174 / 0.0178 (bar > 0.03);
`gain_public` 0.513 / 0.509 / 0.515 and `gain_hp_moved` 0.565 / 0.559 /
0.562 (bars 0.528 / 0.588 — still below the deleted mean head);
`value_delta_r2` 0.302 / 0.291 / 0.294 (bar ≥ gain_hp_moved) with
`_prior` −0.118 / −0.108 / −0.092 (below copy); `value_gain` 0.216 /
0.212 / 0.224 (positive, NOT rising) with `_prior` −0.42 / −0.39 / −0.31;
`kl` 0.173 flat (predicted 0.3–0.6 — the consistency term was not
starving the code); `pred_rms` 0.88 flat. Consistency reads drifted as
the clause allowed (`cons_gain_cls` −0.06 → −0.38, `cons_gain_field`
0.70 → 0.52) but the bars they were conditional on did not pass. Matched
control: `player_value_head_r2` 0.921 → 0.926 (band 0.90 ± 0.02),
`player_loss_v_win` 0.489–0.494 (0.485 ± 0.01), priv 0.922; leak reads
`prob_switch` 0.040 → 0.043, `entropy_macro` 0.53 → 0.54, trunk grad
norm 3.99; 5.33 steps/s against the pre-stop 5.83 (the ≥ 6.5 gate was
already failing before B; read as no new cost). Abort rung (2) fired —
grounding bars failed with the control fine and `out_proj_rms` < 0.03 —
so D lands as pre-registered. Rule with teeth: **removing the force
that looked like the cause did not unpin the single zero factor; two
gradient regimes (with and without consistency, frozen and live head)
left `out_proj` at a quarter of lecun scale, so the bottleneck is not
a loss term** — the next unfired rung after D is the prior's READ.

| mechanism | what | why |
|---|---|---|
| D: `player_transition_rep_coef` 0.1 → 0.0 | Stochastic MuZero's posterior form — the posterior is pulled toward nothing, the prior chases the sg'd posterior at `dyn_coef` 0.5 under F 0.0625; the KL-side posterior gradient is exactly 0 (test-pinned WITH the rep 0.1 control, which the default no longer supplies) | trigger as pre-registered: B's grounding bars failed AND `kl` < 0.5 at the hold's end. Acceptance (20k): `kl` into 0.5–3.0, `post_perplexity_mean` ≥ 3 (from 2.62), `prior_post_agree` ≥ 0.4 (from 0.45), prior-mode grounding ≥ 0.62 × posterior. Abort: `kl` > 4 or `agree` → 1/16 for 5k → restore 0.1 (a reference-form toggle, never a retune); D falsified → the prior's read (learned-query attention over rows), own commit. The old "posterior collapse → rep 0.05" rung is retired |

## Addition ledger — 2026-09-06 stochastic transition, Step 3b D result, posterior sampling, Step 4 event probe

**D RESULT (1682k → 1754k, read on 10k windows, 2026-09-06): kept.**
No abort clause fired and three of four acceptance clauses passed —
`kl` 0.17 → 0.64 (band 0.5–3.0; the rep half was the brake, as D
predicted), `kl_free_frac` 0.41 → 0.06 (the floor no longer binds),
`prior_post_agree` 0.45 → 0.47 (≥ 0.4), prior-mode grounding
`gain_public_prior` 0.29 → 0.33 against `gain_public` 0.51 → 0.53
(0.63× — at the 0.62 bar), `gain_hp_moved_prior` 0.34 → 0.42 (0.73×).
The clause that FAILED: `post_perplexity_mean` 2.62 → 2.60, flat at
~2.6 of 16 per group under F 0.0625, RowRead, rep 0.1 and rep 0.0
alike. Rollout-side value decode unmoved: `value_delta_r2_prior` −0.12
→ −0.05 (still below the copy predictor's 0), `value_gain_prior`
−0.66 → −0.32, posterior `value_delta_r2` 0.29 flat; `out_proj_rms`
0.0176 → 0.0208 (bar > 0.03 unmet in a THIRD gradient regime). Matched
control fine throughout (`player_value_head_r2` 0.92, `loss_v_win`
0.49, `prob_switch` 0.040–0.044, trunk grad norm 3.6–4.0, ~5.0 steps/s).

| mechanism | what | why |
|---|---|---|
| `rl/offline/event_probe.py` (Step 4b widened; ckpt_01682407, 60 games / 120 sides / 3639 transitions) | the opponent's first EXECUTED decision event on the spanned edges (NONE / MOVE / SWITCH / CANT; MOVE_TOKEN on MOVE rows) predicted from FROZEN inputs by a stop-gradient linear + MLP readout, held out by side; reported as accuracy above the majority marginal (marginals NONE 0.153 / MOVE 0.624 / SWITCH 0.186 / CANT 0.037; edges mean 4.0, p90 6). **A measurement of what each input carries, never a label the model sees** | above-marginal linear / MLP: posterior z **+0.158 / +0.204**, prior z +0.118 / +0.134, prior features +0.071 / +0.134, `RowRead(h_{t+1} − h_t)` +0.232 / +0.261, pooled rows +0.079 / +0.130, rows PCA +0.114 / +0.105; MY-event control +0.20–0.27, so the label's ceiling is ~0.25 and the plan's 0.3 bar was above what the label allows; the move token reads ~0 from every input (n-limited — needs ~300 games and a coarser label). Verdict: prior z ≈ the best MLP on the prior's own input, so the prior's READ is NOT the deficit (the plan's attention-read rung does not fire); `h_t` holds ~half the ceiling of opponent intent; the posterior carries ¾ of what the next state reveals; posterior − prior ≈ 0.07 is the opponent-choice content of the chance node. Readout caveats: C fixed at 0.1, the mirrored side is in-sample |
| `transition.straight_through_sample` (commit `b88463f`, relaunched from `ckpt_01759998` at 1760k) | the training decode draws z from the 1%-unimix'd posterior categorical (`jax.random.categorical`, `hard + probs − sg(probs)`) instead of its argmax; rng = `make_rng("sampling")` inside the learner apply only (per-step keys over T, per-batch keys fold `step_count` into a fixed root — no checkpoint leaf); without an rng the module is the argmax bit for bit, so the prior-MODE decode, every `_prior` panel, `prior_post_agree` and search are untouched. Liveness panel `player_transition_post_sample_is_mode` (fraction of valid transitions whose drawn class is the mode in every group; exactly 1.0 = the rng never reached the learner) | the diagnosis for flat usage was rich-get-richer through the argmax: only the mode's class ever received the decode gradient. User's stated worst case: the other codes encode redundancy rather than being ignored. **20k HOLD (1760k → 1793k, 8k windows): the mechanism is live and the diagnosis is FALSIFIED as the cause of flat usage.** `post_sample_is_mode` 0.82 → 0.84 (18% of draws off the mode — live — and sharpening), `post_perplexity_mean` 2.54 → 2.50 (bar ≥ 3; FELL), `post_perplexity_min` 2.04 → 1.99. What did move: `kl` 0.64 → 0.92 → 0.91 (in band), `prior_post_agree` 0.47 → 0.53 → 0.55 (rising), `prior_grad_norm` 0.38 → 0.57, `kl_long` 1.14 vs `kl_short` 0.74 (the ≥ 0.2 margin clause PASSES for the first time), `kl_reveal` 1.01 vs `kl_no_reveal` 0.85 (+0.16, just under). Unmoved: `gain_public` 0.53 / `_prior` 0.35, `gain_hp_moved` 0.58 / `_prior` 0.44, `value_delta_r2` 0.27 / `_prior` −0.09..−0.12 (below copy), `out_proj_rms` 0.0215 → 0.0222, `pred_rms` 0.89. Controls: `player_value_head_r2` 0.92–0.93, `loss_v_win` 0.487, `prob_switch` 0.040, `entropy_macro` 0.52, trunk grad norm 3.9–4.0, 5.5–5.6 steps/s. Mechanism as now understood: with rep 0.0 nothing opposes sharpening, and a sampled decode PAYS for its off-mode draws in the decode losses (a rarely-used class's `code_table` row decodes worse), so the STE gradient pushes mass back onto the mode — rich-get-richer through the decode loss, which sampling cannot fix. KEPT (nothing regressed, the KL and its splits improved); usage widening is not a lever this family has, and a usage-entropy force would be a new force answering the four questions first |

**Search read, 1/4-subsampled wandb rows (2026-09-06):** since launch
t1 0.481 (n 295) vs search 0.556 (n 270), **+0.074 ± 0.042, z 1.77**;
the D era alone +0.063 ± 0.061 (0.460 vs 0.522); `root_kl` mean 0.027
(below the 0.05 band floor). Point estimate above the +0.03 bar,
confidence not yet — read again at ≥ 300 games per arm on the full
rows. Note the t1 arm itself drifted 0.50 → 0.46 over the D era while
the search arm held 0.52 — inside the stalled 0.40–0.48 band, not a
new regression.

**Where this leaves the ladder.** The plan's next rung on a D failure
was the prior's attention read; the event probe retired it (the prior
reads its input as well as an MLP can). Three gradient regimes have
now left `out_proj` at 0.02 and the prior-mode value decode below
copy, so the bottleneck is on the DECODE side. Owed before any search
or structural change: the expectation-form calibration read —
`value_delta_r2` of `E_z[V(g(h, a, z))] − V(h_t)` over prior SAMPLES
on the real change — because every `_prior` panel decodes the prior's
MODE, and a mode decode of a 55%-agree prior is expected to read below
copy on the 45% where the mode is the wrong branch even when the
mixture is calibrated. That is the number search uses; if it clears
copy the mode panels were the wrong instrument and C (K = 2) / rung 2
proceed on it; if it does not, the decode path (`code_proj` →
`out_proj`) is the falsified piece and gets its own commit.

**EXPECTATION-FORM READ (ckpt_01800000, 2026-09-06, 500 transitions
off self-play games through the second service, 32 prior draws per
transition; `search_samples_probe.py --calibration --calibration-samples`):
the expectation does NOT clear copy — the decode path is the falsified
piece.** `value_delta_r2` = R² of the decoded CHANGE in value
(V(decode) − V(h_t)) on the real change (V(h_{t+1}) − V(h_t)); the copy
predictor scores exactly 0, higher is better. All / switch taken (118) /
move taken (382): posterior decode **0.210** / 0.243 / 0.192; prior
MODE −0.104 / −0.101 / −0.120; prior EXPECTATION over 32 samples
**−0.057** / −0.013 / −0.081; a SINGLE prior sample −0.270 / −0.208 /
−0.302. `expect_sigma_z` (std of V over the 32 draws at the taken
action) 0.050 (switch 0.027, move 0.057), `abs_real_delta_v` 0.115,
sign_acc 0.63. The MSE-gain form on the MC outcome (copy 0, real 1)
reads negative everywhere except the posterior on switch rows (+0.43)
and is the noisier instrument. The ordering sample < mode < expectation
< 0 is what averaging out a ~0.05-wide branch spread buys: the
expectation removes the sampling variance and still lands below copy,
so the prior's mixture is NOT calibrated on value — the mode panels
were the RIGHT instrument, only slightly pessimistic (−0.10 vs −0.06).
Also on the record: the posterior decode reads 0.21 here against 0.446
on ckpt_01560000 (pre-B, cons_coef 1.0) — different games, n=500, but
the learner panel's 0.27–0.29 across the B/D holds says the same:
dropping raw-row consistency cost the posterior decode's value
alignment and bought nothing on the prior side. Pre-registered
consequence: C (K = 2) and rung 2 do NOT proceed; the decode path
(`code_proj` → `out_proj`, the single zero factor pinned at 0.02 rms
through four gradient regimes) gets its own structural commit. The go
decision is the user's.

## Diagnostic audit — 2026-09-07 latent transition at 1.835M

Read-only model audit of `irqeetfg`, final step 1,835,215, resumed at
1,759,999 on commit `b88463f`. Full local W&B history supplied 75,217
learner rows. No model change or restart was performed. The final window
1,815,000–1,835,215 contains 20,216 rows: mean posterior/prior-mode
`value_delta_r2` 0.264 / -0.081; public grounding gain 0.532 / 0.352;
KL 0.909, free fraction 0.031, posterior usage perplexity 2.508 (minimum
1.975), prior/posterior joint-mode agreement 0.552. Real critic R² 0.923,
switch mass 0.0418. Posterior mask accuracy 0.9953 hides recall 0.8265
and exact-set accuracy 0.5322. CLS consistency gain deteriorated from
-1.318 at 1.760–1.780M to -2.915 in the final window; its loss is disabled.
These are means of logged batch metrics, not pooled transition-level scores.

Full resumed-segment EMA evaluation: temperature-1 control 685/1410 wins
(0.4858), search 725/1333 (0.5439), difference +0.0581; nominal independent
binomial 95% interval [+0.0207, +0.0954]. Evolving checkpoints and unpaired
games limit this comparison. Search root KL 0.02194; mean game-step time
218 ms versus 59 ms across EMA and main evaluation games. No legal-cell
truncation was logged. This supports a depth-1 benefit against this evaluator,
not a deeper-search verdict or a general opponent-strength claim.

**Corrections to earlier mathematical interpretations:**

- Learner `delta_gain` uses `1 - SSE / sum(delta²)`, with copy exactly 0.
  Offline `search_samples_probe._r2` uses centred SST. Copy there is
  `-n * mean(delta)² / SST`, not necessarily 0. The historical -0.057
  expectation score alone therefore does not prove worse-than-copy MSE.
  Recompute both on the same transitions before using that gate. Average
  batch ratios are also unstable: five switch-score rows in this segment
  have magnitude >100, with a minimum near -1.206e6. Pool numerator and
  denominator and bootstrap by game. The final-window prior CE is 0.51726
  versus copy 0.51526 and real 0.48944: ratio of window means gives -0.0775,
  while mean logged CE-gain ratios gives -0.447.
- Posterior-conditioned MSE has optimum `E[h_next | h, a, z]`; it does not
  inherently erase branches distinguished by z. Removing consistency on
  the unconditional-mean argument was not mathematically compelled.
  EfficientZero (arxiv.org/abs/2111.00210) supplies precedent for consistency,
  but its end-to-end deterministic model is not this observer architecture.
- A small `out_proj` parameter RMS does not determine its functional gain
  or the code-to-value Jacobian. The prior-mixture error does not isolate
  that projection as the cause. Code usage perplexity is marginal entropy,
  not mutual information or evidence that all available classes are needed.

**Concrete structural limitation:** `TransitionModel.imagine` applies the
current `row_valid` to its blocks and final output. Encoder move/target
validity depends on the current legal action mask. A currently invalid row
is thus identically zero after imagination even when valid at the next
request; its output gradient is zero too. In particular force-switch to
move requests cannot create the new move rows. Bias-free readouts cannot
recover those rows' missing state dependence. Fix by separating observed
input validity from future output queries/validity, retaining policy-only
inputs and fixed shapes. Measure newly-valid-row and request-kind splits;
do not feed the real next validity to the deployed decoder.

**Unmeasured distributional hypothesis:** the 2x16 prior is a product of
categoricals. Averaging the posterior over possible next observations need
not preserve that independence; correlated posterior code pairs can produce
unsupported cross-pairs under prior sampling. Fit/evaluate a conditional
joint or autoregressive prior with encoder and decoder frozen, splitting by
whole game. A 256-pair enumeration can remove sampling noise from offline
value calibration. This hypothesis is not established by current marginal
perplexity/agreement panels. DreamerV3's representation KL encourages
predictability under its factorised prior (arxiv.org/abs/2301.04104); this
run has rep coefficient 0 and is not the reference objective. Stochastic
MuZero also differs through its afterstate model and VQ chance encoder.

## Probe ledger — 2026-09-07 Step 1: the calibration rescored on the gate's own instrument

The 2026-09-06 expectation-form read (−0.057, "the decode path is the
falsified piece") was scored with `search_samples_probe._r2`, the CENTRED
R² (copy = −n·mean(Δ)²/SST), per-transition means, no interval, 32 prior
draws. Rescored with the probe rewritten to the learner's UNCENTRED
`delta_gain` (copy = 0 exactly), sums pooled over transitions before
dividing, the prior expectation taken EXACTLY over all 256 joint codes
weighted by the product prior, a 1000-replicate bootstrap over whole
GAMES (both self-play sides together), and splits by taken modality,
newly-valid rows at t+1 and the (t, t+1) request-kind pair. 40 self-play
games / 80 sides / 2343 decisions / 1199 transitions, the SAME games for
both checkpoints (`--games-pkl`, played by ckpt_01800000, seed 1).

| read (`value_delta_r2`, copy 0 / real 1) | ckpt_01800000 | ckpt_01835216 |
|---|---|---|
| all: posterior | +0.333 [+0.215, +0.447] | +0.341 [+0.271, +0.410] |
| all: prior MODE | −0.208 [−0.420, −0.007] | −0.196 [−0.380, −0.054] |
| all: prior EXPECTATION (exact 256) | **−0.073 [−0.186, +0.040]** | **−0.056 [−0.151, +0.020]** |
| all: one prior sample | −0.242 [−0.419, −0.085] | — |
| move taken (n 924): expectation | −0.106 [−0.220, +0.005] | −0.114 [−0.217, −0.031] |
| switch taken (n 275): expectation | +0.124 [−0.150, +0.395] | +0.178 [+0.072, +0.242] |
| newly-valid rows at t+1 (n 417): posterior / expectation | +0.153 / −0.067 | +0.200 / −0.017 |
| no newly-valid row (n 782): posterior / expectation | +0.403 / −0.075 | — |
| request switch → move (n 139): posterior / expectation | +0.263 / −0.014 | +0.164 / −0.034 |
| copy under the centred R² (the 2026-09-06 offset) | −0.001 all, −0.274 on move → switch | — |

`expect_sigma_z` (std of V over the exact code mixture) 0.061;
`prior_mode_mass` (the joint mode's prior probability) 0.48;
`abs_real_delta_v` 0.125.

**Verdict, on the pre-registered clause (docs/latent-world-model-plan.md
§5 Step 1).** The all-rows interval STRADDLES 0 on both checkpoints: the
block on C (the two-step unroll) and rung 2 (the pUCT tree) stays, and
Steps 3–4 (the clean consistency control, the prior-family control)
decide. The centred-vs-uncentred offset the audit raised is real but
numerically negligible on all rows (−0.001) — the 2026-09-06 number was
not wrong for that reason; it was unpooled, sampled and interval-free.
What the splits add: on MOVE rows (77% of transitions) the prior
expectation is below copy with the interval clear of 0 at 1.835M; on
SWITCH rows it is above copy, clear of 0. The prior's mixture is
mis-calibrated where the opponent's move choice is the chance content,
and useful where the branch is which Pokémon comes in. The posterior
decode reads 0.15–0.20 where a row becomes valid at t+1 against 0.40
where none does — the validity defect (Step 2) reaches the CLS read
through the blocks even though CLS itself is never masked. The mode
decode is worse than the expectation everywhere (−0.20 vs −0.06), so the
learner's `_prior` panels remain the pessimistic instrument; the
expectation is the number search consumes.

| mechanism | what | why |
|---|---|---|
| `search_samples_probe.py`: `delta_gain` (uncentred, primary) + `r2_centred` (+ `copy_delta_r2_centred`), `code_grid` + `--calibration-enumerate`, `resample_games` + `--bootstrap N`, `calibration_splits`, `--calibration-only`, `Transition.{game,kind,next_kind,newly_valid}`, `prior_mode_mass`; `tests/test_search_samples_probe.py` (copy 0 / real 1 on the uncentred read, the centred copy offset as the positive control, the code grid + product weights against the exact mixture, sides kept together by the bootstrap) | the accounting fix; `value_delta_r2_*` now means the SAME formula on the learner and the probe | `train_step.delta_gain_terms` + `player_transition_value_delta_{sse,energy}{,_switch,_move}` and `_sse_prior` panels (logs only, bit-identical): the switch split's per-batch ratio is dominated by tiny denominators, the window read is 1 − mean(sse)/mean(energy) — wandb view "Transition delta sums" |

Reproduce: `PS_SERVICE_URI=ws://localhost:8081 env/bin/python -m
rl.offline.search_samples_probe --ckpt ckpts/gen9/ckpt_01800000 --games 40
--pairs 4 --roots 1200 --calibration-only --calibration-enumerate
--bootstrap 1000 --games-pkl <pkl> --seed 1` (second service from
`service/`: `PORT=8081 MAX_WORKERS=2 node dist/server/index.js`).

## Addition + removal ledger — 2026-09-07 latent actions: the world model in MuZero/Dreamer form, nothing masked past the root

**The change.** `rl/model/transition.py` rewritten (plan
`~/.claude/plans/modular-knitting-petal.md`, approved 2026-09-07 after three
review passes). g(h_t, u, z) over the 73 policy-readable rows with the
action and the chance code entering as TOKENS beside the rows (plus a learned
slot embedding on the rows), through `dynamics_blocks` under an all-True
mask: `imagine(rows, action_one_hot, code_one_hot)` takes no validity, no
legal mask and no future input. u is ONE 64-class LATENT ACTION: the action
encoder q(u | h, a) (one cross-attention read of the sequence by the taken
cell's rows) at observed states, the candidate generator rho(u | h) (two
decoder blocks, causal over the candidate tokens, cross-attention over the
rows) at imagined nodes, drawing J = 8 distinct codes WITHOUT replacement
inside the smallest rho prefix holding 0.99 mass. The learner unrolls K = 2
transitions from every start step along the recorded actions, recomputing
the encoder (the alignment loss) and the chance posterior at the imagined
state, with every reader (value, kind, done, the conditional terminal
outcome, the generator) trained at every predicted state; 0.5 gradient scale
into the unrolled state (MuZero's heuristic). Search (`rl/model/search.py`)
is the explicit decision / chance recursion B_d(h) = (1 − c) T + c Σ mu Q_d,
mu the KL-regularised softmax over the occupied candidates, chance averaged;
depth 1 at the root over the exact legal cells is the operational control,
depth 2 a second baseline search eval slot (`eval_search_depth`); search runs
NOWHERE else. Config: `action_classes` 64, `num_candidates` 8,
`max_cells` 16 (shared with search), `mass_threshold` 0.99, `unroll_steps` 2,
`player_transition_decode_coef` 1.0, `player_transition_align_coef` 1.0,
`cons_coef` stays 0.0 (a read), `eval_search_depth` 2.

| mechanism | added / removed | why | revert handle |
|---|---|---|---|
| `imagine` masked by the CURRENT `row_valid` (blocks + output zeroing) and the posterior's `row_valid & next_valid` delta read | REMOVED | a row that appears at t+1 was identically 0 with no gradient (LESSONS 2026-09-07 audit; posterior value read 0.40 without vs 0.15–0.20 with an appearing row). Validity is content the rows carry: the real target rows are zero where the trunk zeroed them, so the losses teach absent → 0 and appearing → content | `git show e6a50d7:rl/model/transition.py` |
| `mask_head` (a second `FlatActionReadout` predicting the next legal set) + its BCE | REMOVED | legality at an imagined node is the generator's mass, and a rollout never enumerates concrete cells past the root; last readings irqeetfg 1.815–1.835M: `mask_acc` 0.9953 / `mask_recall` 0.8265 / `mask_exact_frac` 0.5322 (high per-cell accuracy is not a solved legal set: half the masks were inexact) | same |
| the frozen action readout on imagined rows under the REAL next legal set (`transition_log_policy`, `player_loss_transition_policy`, `policy_kl_copy`) | REMOVED | the imagined node's policy is the generator over the latent alphabet; last readings `loss_transition_policy` 0.30–0.37 vs `policy_kl_copy` 0.64–0.87 (the imagined rows beat the copy on the real-cell policy, banked) | same |
| `action_proj` (broadcast add of the cell's two rows), `code_proj`, `blocks` / `out_proj`, `prior_read_net` / `posterior_read_net`, `row_read` bias | REMOVED / RENAMED | tokens route the condition per row and per head instead of a constant offset every row must subtract; the renamed leaves init fresh on the by-path merge (input width or meaning changed); `RowRead` bias-free so a zero row reads exactly 0 | same |
| `slot_embedding`, `action_table`, `condition_type_embedding`, `chance_token_proj`, `dynamics_blocks`, `dynamics_out_proj`, `action_encoder`, `candidate_generator`, `prior_latent_net`, `posterior_latent_net`, `terminal_outcome_head` | ADDED | the plan's module table (Appendix A.1); the transition subtree ≈ 8M params at D = 256 | this commit |
| exact decode objective H_w(A \| U, h) summed over every code (uniform reference weights over the legal cells) | ADDED (`player_transition_decode_coef`) | a sampled straight-through CE drops the derivative of the sampling distribution; the symmetric collapsed encoding is a STATIONARY point of this objective (exactly zero gradient, pinned by test), not a repelled one — collapse is diagnosed, never escalated by coefficient | coef 0 = the control |
| conditional terminal-outcome head (3 logits, trained only on actual terminal successors inside the joint termination NLL) | ADDED | V is unconditional: at a node with continuation c the blend (1 − c)V + c E[Q] counts the continuation branch twice (a 50% terminal win aliased with a 50% continuation at −1 reads −0.5 instead of 0); the backup needs E[outcome \| terminal] | `git show` this commit |
| the K = 2 unroll with the alignment loss | ADDED | depth-2 search reads the generator, the prior, V and done on IMAGINED rows; nothing trained those readers on imagined input (cons is 0). The alignment loss is a hypothesis with its own panels (`align_kl_k1/k2`) | `unroll_steps` 1 = the single-step model; `align_coef` 0 |

**Resume.** Checkpoint-mode from `ckpts/gen9/ckpt_01835216` (irqeetfg,
1,835,216). The manifest is silent on the transition architecture by design
(the by-path merge handles it). Merge audit (2026-09-07, `merge_params` of
the checkpoint onto the new init, 17.42M loaded → 20.46M fresh, transition
subtree 8.49M): KEPT FRESH `transition/{action_encoder, action_table,
candidate_generator, chance_token_proj, condition_type_embedding,
dynamics_blocks, dynamics_out_proj, posterior_latent_net, prior_latent_net,
slot_embedding, terminal_outcome_head}` and NOTHING outside the transition;
DROPPED `transition/{row_read/read/bias, action_proj, blocks, code_proj,
mask_head, out_proj, posterior_read_net, prior_read_net}`; RESUMED
`transition/{cls_head, code_table, ground_delta_head, row_read (kernel)}` —
exactly the plan's Appendix A.1 table.

**Probe (docs/pre-mcts-validation-2026-09-07.md, fixed the same day).** The
calibration read had fixed the action code at its argmax and integrated
chance under it — a read conditional on one code, not the deployed
expectation E_u E_z V(g(h, u, z)). `search_samples_probe._calibration` now
integrates the taken action's `--calibration-actions` (8) most probable
codes with renormalised weights (`action_retained_mass` reported) and the
chance prior under each (`expect`), keeps the conditional read under its
own label (`expect_mode`), draws `sample` from the joint, and reports
`expect_sigma_u` beside `expect_sigma_z`. The hold's calibration study
reads `expect`.

**Launch (2026-09-07).** Attempt 1 (16:23) failed at the first lattice
compile: the offset-leading transition leaves came out of the learner's
batch vmap as (K, B, T) — `PlayerActorOutput.batch_out_axes()` (59c3b16); the
full-lattice train_step smoke, which the GPU-held pre-launch check could not
run, would have caught it. Attempt 2 (16:38, wandb run irqeetfg resumed)
tripped the host-RAM guard at 1,836,000, 784 steps in (available 0.135 <
0.15; learner RSS 12.06 GB at the last diag against the 9.3–9.7 GB this
era's checkpoint launched at, node 1.45 GB): the depth-2 eval arm cost 3930
ms per step on its CPU thread against 362 (depth 1) and 141 (plain) — 11x
depth 1 — so `eval_search_depth` 2 → 0 and attempt 3 resumes from
ckpt_01836000. Readings at 784 steps: `out_proj_rms` 0.0034 (left 0),
`decode_acc` 0.74, `action_mi` 1.25 nats, `action_perplexity` 4.2 of 64,
`generator_kl_first` 0.23 against a target entropy of 0.93, imagined
`value_r2` 0.971 vs real 0.985, `value_delta_r2` −0.21 (below copy this
early), trunk grad norm 2.3, no skipped update.

**Pre-registered (plan §8).** 2k launch check: `out_proj_rms` leaving 0,
`decode_acc` above 1/num_legal and rising, `action_mi` > 0 and rising,
`generator_kl_first` falling, `transition_value_r2` tracking
`player_value_head_r2`, `prob_switch` 0.040 ± 0.004, trunk grad norm
3.5–5.5, steps/s within 20% of 5.5. The 20k hold's acceptance table and the
non-inferiority margins (critic R² drop ≤ 0.02, base win-rate drop ≤ 2 pts,
throughput loss ≤ 20%) are in the plan; depth-2 reads stay diagnostic until
the calibration checks pass.

## Actor compilation correction — 2026-09-07 irqeetfg

The 17:06 restart's actor cache misses named historical unused parameter
branches (`transition` with different children, retired `dynamics_delta_head`)
and static `Agent` identities. At the diagnostic snapshot there were 423
nested/outer tracing warnings but only 21 at the outer actor call; three
learner executable writes matched the fixed lattice. Do not count nested
Flax tracing warnings as independent XLA builds or infer learner recompilation
from this actor-side evidence.

`player_model.actor_params_view` now supplies actor-required variables before
`DeviceParamsCache` transfer: encoder, action head, deployable value head,
optional doubles conditioning, and transition for search. Stored snapshots
are untouched; required branches fail visibly when absent. `Agent` dispatches
to module-level JITs keyed by the apply callable, with parameters, RNG and
head scalars dynamic, so equivalent Agent instances share traces. The GPU
inference server receives the same projection. First-trace logs record a
parameter shape/tree fingerprint and history/carry signatures. Revert handles:
`actor_params_view`, its main/inference wiring and the module-level
`_step_player`/`_step_builder` move (parent revision `aaed1a3`).

Focused actor/device, search and carry-loop fast checks passed. Two GPU tests
compare full versus projected trees with live action/dynamics paths: plain
and depth-one searched output pytrees are bit-identical. A stub trace-count
test checks reuse across Agent identities, temperatures, parameter values and
retired branches, with a required-shape change as the positive retrace control.
Ruff and diff whitespace checks passed. This removes demonstrated sources of
specialisation; it does not yet establish the steady-state RAM/throughput gain.

User authorised stop/fix/restart. The first interrupt landed inside a JAX GC
callback and was ignored; a second graceful interrupt saved checkpoint
`01846215` and the learner exited cleanly at 17:49:59. Relaunched at 17:53:29
with explicit checkpoint mode/path in the existing train pane, retaining the
service and learning/search configuration. W&B confirmed `irqeetfg` resumed;
verified progress through 1,846,386, no skipped update and a recent 5.34
steps/sec. Initial plain actor traces share one parameter fingerprint across
32/64/128 history buckets; the longer performance hold is not yet measured.
Local plan and measurements:
`docs/irqeetfg-restart-diagnosis-2026-09-07.md`.

## Removal ledger — 2026-09-02 entity_index_tag: measured dead, deleted

The 2026-08-31 alignment key — one (13, 256) table added to a sheet row by
the wire's ENTITY_IDX and to its public row (and history row, and opp
secret row) by PUBLIC_ORDER, so the two rows describing one mon carried
the same additive tag for attention to match on. Three reads on
ckpt_00182000, all one way (`docs/lineage-instrumentation-plan.md` §6a):

| read | number |
|---|---|
| tag rms, init → 182k | 0.0634 → 0.0661 (4%; `sequence_row_bias` moved 40% in the same window) — never trained |
| tag / other addends, history rows | 0.028 (alarm < 0.05) — drowned from step 0 |
| public-ONLY content (boosts, active) read from the sheet row, post- vs pre-trunk | 0.129/0.067/0.682 vs 0.143/0.092/0.771 (ceiling on the public row 0.310/0.247/0.843) — post ≤ pre on every label |

The third read is key-agnostic: it asks whether ANY public-only content
crossed onto the sheet row, and it did not — not via the tag, and not via
the far stronger shared content (one species/ability/item/move embedder
feeds both rows; species is unique per side in randbats). So the join was
dead, not the key drowned, and the tag was neither helping nor hindering
(3% of a row's norm, unchanged from init). The explicit gather fix (hits
matrix @ public rows through a zero-init projection, §9 of the plan) was
scoped and DECLINED by the user in favour of removal; the design is on
record there if the switch axis ever demands it.

| mechanism | symbols | note |
|---|---|---|
| `entity_index_tag` | the param, its four add sites in `_assemble_sequence`, `player_entity_index_tag_{rms,grad_norm}`, `rl/offline/history_addends.py`, `test_entity_index_tag_links_private_to_public` (→ `test_entity_idx_is_not_row_content`, the inverse pin) | the wire column `ENTITY_PRIVATE_NODE_FEATURE__ENTITY_IDX` STAYS — `belief_alignment` matches sheet row to public row through it, structurally, not through any learned tag. History rows keep their positional pairing (row bias i ↔ public row i). Resumed in checkpoint mode via the new by-path merge (233f707): leaf dropped from all four trees, league and Adam intact |

## Removal ledger — 2026-09-02 history encoder restructure

One deletion per commit; the through-line and the design are in the
2026-09-02 plan (parallel-scan backbone, gestalt out, step GAT in; fresh
lineage). Each row's commit is its revert handle.

| mechanism | symbols | why |
|---|---|---|
| Announced-state path (Φ_ann) | `history_encoder.{mask_outcome_features, _ANNOUNCEMENT_EDGE_COLUMNS, _OUTCOME_MAJOR_ARGS, SplitGRUCell.__call__, PerSlotHistoryEncoder._advance, announced_states_at_requests}`, `encoder.encode_history_with_announced`, offline `Porygon2OfflineCritic.{_history_tokens_with_announced, announced, with_aux_and_announced}`, `train.{_announced_metrics, _announced_enabled, _train_method, _unpack_outputs}` + the `announced_*` eval keys and manifest flag, `artifact.has_announced_states` + the `announced=` potential arg, `announced_loss_weight`/`announced_distill_weight`, the offline "Announced head" wandb section | broken since the 2026-09-01 GRU hoist (`announced_states_at_requests` called the pre-hoist `project_inputs` signature) and ON BY DEFAULT in the offline trainer (`announced_loss_weight` 1.0) — the one-step masked advance was the only caller of the per-step `_advance`, i.e. the serial GRU cell the parallel-scan backbone deletes. The skill/luck decomposition (decision = Φ_ann(t+1) − Φ(t), dice = Φ(t+1) − Φ_ann(t+1)) and dice-excised PBRS it existed for never shipped a validated critic (announced-movement ratio ~0.15, 2026-07-30 — SGD never built the circuit unpaid). Structure-only: the RL forward is untouched |
| Slot gestalt (`ctx = h_slots.mean(-2)` on the slot input) | `_scan_step`'s ctx concat; `slot_cell` input 6D → 5D (`[messages ; field_vec ; flat_field]`) | redundant twice over — `flat_field` is fed the SUM of every message, and the trunk attends over the HISTORY_ENTITY rows at read time — diluted by every untouched slot sitting at `initial_slot_state`, and on the serial carry tail. Fresh params (kernel shape moved). `tests/test_history_encoder.py` pins it at the scan-step level: slot k's update never reads slot j's state, with the field carry (moves every slot) and the slot's own carry as the two positive controls |
| Masked SOURCE MEAN in the history message (the 2026-09-01 TGN source half) | `src_node`/`src_edge`/`src_weights` in `__call__`; `message_projection` input 5D+3 → 2D+3 (`[node ; edge ; side ; is_src ; field]`); the `is_src` predicate and the relevant-edge gather written once (`source_rows`, `relevant_edges`) for the encoder and the wire-side `player_history_src_frac` | replaced by `StepAttention` — ONE GAT layer over the live rows of a step, self included, `q`/`k`/`v` lecun over `[node ; edge ; side ; is_src]`, −1e9 floor AND re-masked probs on padded keys (a 1-row step is exactly its own value), `attn_out` zeros-init (one zero factor over live inputs — messages are `message_projection` alone at step 0, the projection moves at step 1). The mean was invertible in singles (2-source steps are 2-row steps: other = 2·mean − self) and lossy in doubles (a spread move: 2 movers, 3 targets, one average for all); mover/affected are row FEATURES, never a key partition (self-targeting moves are one row on both ends). `cfg.encoder.history_step` (2 heads × qk 32); probs/masks ride `PerSlotHistoryOutput` for the Step-5 panels. Tests: silent-at-init WITH the live-kernel control, non-zero `attn_out` grad at init, padded row places and receives no mass WITH the live-row control, 1-row step probs exactly one-hot |
| GRU backbone (`SplitGRUCell` + `nn.scan`) | `SplitGRUCell`, `_GateParams`, `_scan_step`, the `nn.scan` call, `SCAN_UNROLL`, the hoisted `project_inputs` calls; per cell the three orthogonal D×D recurrent Denses (`hr/hz/hn`) → `GatedLinearCell` (minGRU, Feng et al. 2024: `z = σ(W_z x)`, `h̃ = W_h x`, `h = (1−z)h_prev + z h̃`; the candidate no longer reads the carry) + `gated_linear_scan` (f32 `jax.lax.associative_scan` over the affine coefficients `(1 − write·z, write·z·h̃)`; write = 0 is the identity, so untouched slots hold `initial_slot_state` bit-exactly); two parallel scans — field first, its shifted states become an `xs` column of the slot gates — ZERO serial work | **the bench gate CLEARED** (real module, full GPU, B=16 broadcast of the ex.bin trajectory, median of 50; actor forward GRU → linear): H=256 B=1 7.72 → 2.15 ms (−72%), B=4 8.11 → 4.16 (−49%), B=8 11.36 → 7.08 (−38%), B=12 14.43 → 9.85 (−32%), B=16 17.44 → 12.76 (−27%); H=512 B=1 13.82 → 2.88 (−79%), B=4 14.32 → 6.70 (−53%), B=8 20.87 → 12.34 (−41%), B=12 25.56 → 17.05 (−33%), B=16 31.95 → 22.92 (−28%). The GRU is LATENCY-bound (near-flat in B — the ~26µs/step floor the 2026-09-01 hoist could not move) and the scan is throughput-bound, so the win shrinks with batch; the inference server has 12 actors, so B=16 is never realised and the ≥30% bar clears at every live batch. Recorded honestly: it is an actor-cost read of THIS lineage only — after the stored-state pass (§3b) a per-request scan is ~5 steps and the parallel form's payoff moves to the learner. `test_chunking` window invariance on the new backbone: value log-prob max diff 0.0337 against the 0.05 tolerance (the bf16 GEMM shape-dependence survives an f32 recursion; not tightened, not loosened). The recursion lives in a free function, so no dtype-allowlist entry. Tests: slot isolation WITH the moved-slot control, field-into-every-slot, field-never-reads-slots, scan == serial `lax.scan` to 1e-5, unwritten units hold init exactly |

## Addition ledger — 2026-09-02 actor-side history carry (incremental inference)

Follows the parallel-scan restructure directly. Every actor request used
to re-run the whole history pathway over the full window — packed-cache
embedding (≤1024 rows), message projection + step GAT (≤512 steps), both
scans, the cummax — bucketed to ≥64 steps, when a request adds ~3 steps
/ ~5 packed rows (ex.bin: 168 steps, 268 rows over 58 requests). The
minGRU scan is `h_t = A_t⊙h_0 + B_t` with `h_0` already an argument, so
resuming from a carried post-window state over the new suffix is the SAME
function, exact up to the bf16 GEMM leading-dim class (0.05 on log-probs,
`tests/test_chunking.py`). Three user decisions shaped it: **actor-side
only** (learner path, chunk contract, buffer, shape lattice untouched —
`player_chunk_history_underrun` 0.0000 over 3000 samples says the 256-row
window always holds whole games, so an actor carrying from h0 over the
game computes what the learner computes; that panel is the watch that
would say otherwise), **the carry is optional** (`valid=False` or `()`
leaves is today's function bit for bit; the inference server is
stateless), and **the wire is unchanged plus one int**
(`history_rewrite_count`; the actor slices the suffix itself).

| piece | what | why |
|---|---|---|
| `ActorStats` + `actor_time_*` / `actor_infer_*` panels (§9b) | lock-guarded mean sink drained by the learner every `actor_stats_log_steps`; per-step timers (service wait, decode, clip, inference) and the server's phases (queue wait, stack, lock wait, forward, device_get, batch size, history level) | the actors logged nothing; this is the baseline the pass is judged against. `service_wait` includes the OPPONENT actor's whole step in self-play — "waiting on the game server", not service CPU |
| `EnvironmentState.history_rewrite_count = 18` | incremented INSIDE `EdgeBuffer.remapEntitySlot` — the one place past rows are rewritten (an Illusion `\|replace\|`) — so a second caller can never forget it; harness asserts monotone, vitest positive control on the Illusion team class | the carry's only invalidation the window itself cannot show. Field 17 is the opponent request; never reuse |
| `HistoryCarry` (slot f32 (12,D), field f32 (3,D), node snapshots (12,D), `valid`) on `PlayerActorInput`/`PlayerActorOutput`, `resolve_initial`, `history_carry_from`, `invalid_history_carry` | every leaf defaults to `()`; `isinstance(valid, tuple)` is the STATIC no-carry branch (no `where` in the learner trace — bit-identical by construction, verified on 100 leaves), else `jnp.where(valid, carried, h0)` per leaf. The returned carry is the post-window f32 state taken BEFORE the cfg.dtype cast; request-count stamping makes the gather select the last valid step, so no alignment leaf is needed. The actor strips it before the row is stored (`without_history_carry`) | the f32 leaves are the scan's own recursion state — the value-recursion rule, not a dtype-policy breach |
| `_cut_history_windows` + `clip_history_suffix` + `ACTOR_HISTORY_MIN_LENGTH` | the joint cut and IDX rebase written ONCE (the tail clip calls it); the suffix = steps with `FIELD_FEATURE__INDEX` after the carried step and exactly the packed rows they reference, each axis on its own bucket; `None` when the window no longer continues from the carried step; zero new steps is a valid all-zero window (the encoder returns the carry itself). The "must match" constant between actor and server is one symbol | token-for-token reconstruction and rebase pinned in `tests/test_history_suffix.py` |
| `PlayerActor(history_carry_width)` ← `player_actor_history_carry` (default True) | per game: carry, last consumed index, rewrite count; recompute with an INVALID carry (leaves present so a server batch always stacks, `valid` False) at game start / rewrite / gap, each counted (`actor_history_recompute_{frac,game_start,rewrite,gap}`, `actor_history_suffix_{steps,rows}`); `Agent._step_player` forwards the carry, `_run_group` stacks carry leaves on axis 0 and fills a no-carry request in a mixed group with an invalid one | False = today's full-window request on every step — the control arm and the abort switch |
| `ACTOR_HISTORY_MIN_LENGTH` 64 → 32 | levels 32/64/128/256/512: one extra forward-only compile per batch bucket on the ACTOR family only | the suffix runs at the smallest bucket; the scan is O(log H) so 16 buys little over 32 |

**The plan's params-version guard is OMITTED as vacuous**: `ParamsContainer`
is an immutable NamedTuple and `unroll` runs a whole game on ONE
container, so a mid-game version change is structurally impossible; a
future refactor that hands actors a mutable container must add the guard
(recompute on `_version_key` change), not inherit its absence.

**Tests** (`tests/test_history_carry.py`, gpu/slow; `tests/test_actor_carry_loop.py`
plain python): (a) the 58 ex.bin requests served from the previous carry
over the suffix alone match the full-window forward within 0.05 on
log_policy and value log-probs, carried f32 states within 0.05 of the
full scan's at the end, shifted-carry control; (b) garbage leaves under
`valid=False` == the from-scratch forward bit for bit, same garbage under
`valid=True` differs; (c) a zero-step window returns the carry itself /
h0; (d) suffix reconstruction; (e) reason dispatch + stats; (f) a mixed
server group vs single forwards. The carry tests open the zero-init
readout (`open_zero_init_paths(..., ["action_head"])`) — on fresh params
every logit is exactly 0, so a "the carry moves the policy" control would
pass or fail vacuously. **(a)/(b)/(c)/(f) are OWED a run**: landed
against a live learner, fast suite green, slow suite at the next stop.

**Pre-registered acceptance (hold 2k learner steps):** `actor_steps_per_sec`
up and `actor_time_inference` down against the Step-0 baseline below;
`actor_infer_forward` down by the history share at the live batch
(34-56% at levels 128-512); system steps/sec above irqeetfg's 4.17-4.41 at
matched lifetime_step; `actor_history_recompute_frac` ≤ 0.05 (> 0.1 = a
continuity bug); `player_learner_actor_forward_kl{,_switch,_move}` and
`player_replay_realised_ratio` inside their pre-change band. **Abort** →
`player_actor_history_carry=False` (no revert). **Declined, recorded:**
delta wire (gated on the service_wait / process_state share), server-side
carry table (identity + eviction for no numeric gain), stored-state chunks
(underrun 0 — nothing to fix), gating the returned carry on `cfg.train`
(two paths for a leaf XLA drops anyway). The parallel scan's payoff after
this pass is the recompute fallback and the learner's chunk-length scans.

**Reference numbers.** irqeetfg (launched on f478015, BEFORE the Step-0
telemetry commit — so it carries NO `actor_*` panels; the plan's "lands
first" was not honoured by a restart) system rate 4.17 / 4.25 / 4.41
steps/sec at 1k-4k / 4k-8k / 8k-13.8k, vs d7zdz8hw 3.66 and yt3qp960
3.53 (steady 4.12); learner alone 12.3. The per-phase baseline therefore
comes from the relaunch itself: a bounded `player_actor_history_carry=
False` window on the SAME code (the timers live, the carry off), then
the carry on — the matched pair the speed comparison is read from.
**Matched pair, read 2026-09-02** (same code, same box, irqeetfg resumed
from ckpt_00042422; OFF = lifetime_step 43.0k-47.66k, ON = 47.9k-49.4k;
window means over the 10-step drains, ms):

| | carry OFF | carry ON | |
|---|---|---|---|
| `learner_steps_per_sec` (system rate) | 5.18 | 5.60 | **+8%** |
| `actor_steps_per_sec` (pool) | 77.7 | 86.7 | **+11%** |
| `actor_time_step_total` | 199 | 176 | -11% |
| `actor_time_inference` | 170 | 147 | -13% |
| `actor_infer_queue_wait` | 118 | 76 | **-36%** |
| `actor_infer_forward` | 33.1 | 47.9 | +45% (see below) |
| `actor_infer_batch_size` | 2.94 | 4.74 | |
| `actor_infer_forward` PER REQUEST | 11.2 | 10.1 | -10% |
| `actor_infer_lock_wait` | 12.1 | 17.4 | |
| `actor_infer_history_level` | 1.42 | 0.004 | as designed |
| `actor_time_history_clip` | 0.27 | 1.17 | the suffix cut's python |
| `actor_history_recompute_frac` | - | 0.035 (all game starts) | gate <= 0.05 PASS |
| `actor_history_suffix_{steps,rows}` | - | 2.6 / 4.1 | ex.bin predicted ~3 / ~5 |
| `player_learner_actor_forward_kl` | 0.008 | 0.007 | in band |
| `player_replay_realised_ratio` | 8.00 | 8.00 | in band |
| `player_policy_prob_switch` | 0.033 | 0.034 | unchanged |

**Verdict: a real but modest win, and the "forward down by the history
share" clause of the acceptance FAILED on its own instrument, for a
reason the panel makes legible.** Every request now lands at level 0
(32 rows), so the grouping key no longer scatters the queue across
levels 1-2 and the server takes bigger groups (2.9 -> 4.7); the forward
grew in absolute terms with the batch and shrank only 10% per request.
The 34-56% history share was measured on an UNCONTENDED GPU; under a
live learner the server's forward is dominated by stream contention
(lock wait and forward both inflate with the train step), so removing
the history compute moved the per-request forward by a tenth, not a
third. The win that did arrive is in the QUEUE: -36% wait, which is
what the actors actually spend a step on (queue 76 of 147 ms inference,
of 176 ms step). Both windows also sit above irqeetfg's pre-stop
4.17-4.41 (5.2 / 5.6) — the extra level-0 bucket and the restart are
confounded there; only the OFF-vs-ON pair is the clean read. Next lever
by measurement, not this pass: the server's serial loop under GPU
contention (queue + lock + forward = 141 of 147 ms), i.e. the learner's
train step and the actor forward sharing one device stream.

## Removal + addition ledger — 2026-09-03 gpu_lock retired, actors on the CPU

Follows the history-carry pass, which left the actor step at 147 ms of
inference — queue 76 + lock 17 + forward 48 — none of it compute: the
server's forward queued behind train-step kernels on ONE device stream,
while the host sat at load 1.1 on 20 cores. Two commits, one number moved.

| mechanism | what | why |
|---|---|---|
| `gpu_lock` (REMOVED, `19d804b`, structure-only) | the learner held it for `device_put(batch)` + the ASYNC dispatch of `train_step` — released before the GPU finished, so it never serialised execution nor bounded VRAM; the server and `Agent.step_player` forwarded on their own device copies (no donation hazard). Its ONE real job — `main.py`'s eval thread reading the learner's LIVE device buffers that the next step donates — became learner-thread publication: `publish_live_params` → `EvalSnapshot(step_count, main, ema)` on `RunState.eval_snapshot`, read lock-free (`create_params_container(run_state, target=...)` gains the EMA side). `actor_infer_lock_wait` panel gone | no lock is simpler than a smaller one: with the reads moved to the thread that owns the buffers, nothing remained to lock |
| `player_actor_device = "cpu" \| "gpu"` (ADDED) | "cpu": every PlayerActor and BuilderActor runs its own batch-1 forward on the host through `Agent.step_player` — no server, no queue — on an f32 actor network (`get_player_model_config(..., dtype=)`; XLA:CPU only emulates bf16; params stored f32 either way). "gpu": today's batched bf16 InferenceServer, the control arm and the abort switch. Bench 2026-09-03 beside the live learner, ex.bin at the live shapes: suffix 32/32 B=1 17.8 ms; 12 concurrent B=1 threads 60.5 ms/call, 196 fwd/s aggregate (a CPU *server* at B=8 serial: 128/s — declined) | the actors leave the learner's stream entirely; expected ~2x actor throughput and the learner alone at its 12.3 steps/s ceiling |
| `DeviceParamsCache(device, field)` | the server's `_get_device_params` LRU lifted into ONE object: host `ParamsContainer` → that field committed to `device`, keyed by container IDENTITY (the league hands out one object per version; the eval thread's main and EMA containers share `(step_count, frame_count)` and would alias under `_version_key`, kept for GROUPING only). `Agent` owns two (player, builder) and takes the host container on both paths — the per-game `device_put` in `unroll_and_push` and the builder loop die | 12 actors share one device copy per version instead of 12 x 54 MB per game (the host-RAM lesson) |
| `joint_history_level` + `pad_history_to_level` | the server's joint bucket pad written once in `rl/environment/utils.py` and applied on the direct path too | bucket combinations multiply: the CPU path compiles one variant per level (5), not per (history level x packed level) pair. `tests/test_actor_device.py` pins the two call sites to one shape |
| rng keys committed to `agent.device` | `PlayerActor`/`BuilderActor` seed keys `device_put` once; every split stays there | under "cpu" no GPU kernel runs for an actor at all — verified: 2 offline games on ckpt_00216496 through the second service, process GPU footprint 292 MiB before and after (the CUDA context alone) |

`Agent.step_player` records `actor_infer_forward` + `actor_infer_history_level`
into the shared `ActorStats` (nested inside `actor_time_inference`, NOT a
`STEP_PARTS` entry); the server-only timers are simply absent under "cpu".
Three Agent instances each trace `_step_player` (`self` static): 3 x 5
levels of CPU compiles, once.

**Pre-registered acceptance** (checkpoint-mode resume from the 2026-09-03
stop; 2k warm-up, then a 2k hold; against the carry-ON window above):
`actor_time_inference` <= 70 ms (from 147); `actor_steps_per_sec` >= 130
(from 86.7); `learner_steps_per_sec` >= 8 (from 5.6; solo ceiling 12.3);
`player_learner_actor_forward_kl` <= 0.012 (from 0.007 — f32-vs-bf16 on
log-probs is ~3e-3, so inside 2x, else the dtype gap is real);
`player_replay_realised_ratio` 8.00 (the PI may CUT reuse under fresher
data, never raise it); `actor_history_recompute_frac` <= 0.05 unchanged.
**Abort** → `player_actor_device="gpu"` (no revert): forward KL out of
band; `learner_steps_per_sec` FALLING while actors climb (the CPU pool
starving the learner's host thread — cap XLA:CPU intra-op threads or drop
to 8 actors); `service_wait` rising (host load pinning the node service).
**Declined:** a batched CPU server (above); a narrower lock (nothing left
to lock); `jit(device=)` (deprecated in jax 0.10.2 — committed inputs carry
the computation); a second CUDA context / MPS (time-slices the same GPU).
Follow-up ONLY after a full lineage on "cpu": the server and its level-0
grouping become dead code and go, taking the flag's off with them.

## Removal ledger — 2026-09-03 entity attention pool → masked sum

One structural commit, fresh lineage (param tree moves: 13.75M → 11.64M,
−2.11M, the pool was 15% of the model). Revert handle: the commit itself.

**The measurement.** Probe E (`rl/offline/type_probe.py`, ef08214) read the
policy's mass on immune targets at 0.372 against 0.403 under uniform — the
model barely computes type matchups — while the types ARE on the wire as
explicit multi-hot columns (species.npy 1342/1344/1346, moves.npy
711/713/715, one column per type). A supervised ceiling on 27.4k records
(held out by chunk, majority floor 0.549, the readout's own bilinear form,
type one-hots as the 0.987 control) located the loss:

| operands of the move x target bilinear | held-out acc |
|---|---|
| attention-pooled rows, pre-trunk (the assembled input) | 0.600 (train 0.937) |
| attention-pooled rows, post-trunk (TODAY'S READOUT INPUT — `player_model.py` scores the trunk's output rows) | 0.503 (train 0.992) |
| type-supervised bottleneck on the same pre-trunk rows | 0.813 |
| raw attribute multi-hots, rank 64 / 256 | 0.801 / 0.804 |
| **SUMMED rows** (learned linear per attribute, no attention) | **0.793** |
| shuffled labels | 0.390 |

Information and form both suffice (0.81 through a 19-d type bottleneck);
what fails is that a 256-d bilinear over pooled rows prefers the identity
shortcut (species x species pair memorisation — pairs never recur across
randbats games) to the type subspace, and the attention pool makes the type
subspace LESS legible than the linear sum (post ≤ pre on every read). The
sum recovers 0.60 → 0.79 for free; the residual gap to 0.80 is the same
shortcut inside the multi-hot itself.

| mechanism | symbols | why |
|---|---|---|
| `EntityAttentionPool` (1-layer self-attention over ~10 attribute tokens + a learned-query `TransformerDecoder` read, rematted) | → `EntitySumPool`: `sum(mask · (token + token_bias[type])) / sqrt(num_tokens)` — STATIC divisor as in `simple_sum_embeddings`, so the row is linear in its attribute multi-hots; absent tokens contribute nothing, a fully masked set is zeros; `token_bias` (the field identity) stays. `cfg.encoder.intra_entity_{encoder,pool}`, `transformer_{encoder,decoder}_kwargs`, the `decoder_*` config vars, `encoder_init_residual_scale` (0.05 — the 2026-08-24 gate-init lesson, now moot), `modules.{DecoderBlock,TransformerDecoder}` (last consumer) all deleted; `TransformerEncoder` stays for the builder | the within-entity interactions it was meant to form (species x item x moveset) never showed on any read, and the one interaction a decision measurably turns on — my move's type x their species' types — is a BILINEAR of two entities, which no intra-entity block can form and which the readout already has the form for. Also runs on every packed history-cache row, so the actor forward and the learner's history precompute both shrink |

**Kind-confounding probe (`rl/offline/kind_probe.py`, 2026-09-03, run on
ckpt_00260000 of the attention-pool lineage BEFORE any launch, with a
fresh-init trunk as the control).** The follow-up question was whether the
trunk — one attention and ONE shared MLP over ~10 row kinds told apart only
by additive biases, with `player_trunk_row_participation` falling 7.1 → 4.5
over the run — was mixing the kinds into one subspace. Three reads per
block, all held out by chunk:

| read (higher = better) | input | block 6 trained | block 6 FRESH |
|---|---|---|---|
| kind identity (ridge, kind-balanced acc) | 0.96 | 0.94 | 0.90 |
| cross-kind subspace overlap, public↔target (rank 16) | 0.54 | 0.22 | 0.41 |
| own legibility: public row → species types (argmax hit) | 0.998 | 0.880 | 0.858 |
| own legibility: move row → move type | 0.995 | **0.534** | 0.458 |
| own legibility: move row → base power (r) | 0.994 | 0.856 | 0.831 |
| own legibility: private row → species types | 0.739 | 0.630 | 0.652 |
| own legibility: history row → species types | 0.401 | 0.353 | 0.826 |

**Verdict: kinds are NOT confounded** — identity holds at every block and
training pulls the kinds' subspaces APART (every cross-kind overlap falls
from input to output, and ends well below the random trunk's). What the
trunk does do is DILUTE: a move row's own type is half as linearly legible
on the row the readout scores as on the row that entered, and the fresh
control loses the same amount — this is what six ungated residual
additions do to a linear direction, not something training built, and
training preserved slightly more of it than random. It is the mechanism
behind the ceiling's post ≤ pre. So a per-kind MLP / input projection is
NOT indicated by this data; the cheap structural answer if the sum-pool
lineage still reads its move rows dimly is a readout that ALSO sees the
assembled input row (a skip past the trunk), its own commit and gate.
Side reads: the private (sheet) row carries species type at 0.74 even at
the input, against the public row's 0.998, fresh and trained alike (the
pool, not learning — re-read under the sum pool); trained history rows
carry hp 0.95 / active 0.82 but types 0.40 (fresh 0.94: the GRU state
displaces the node snapshot); the trunk writes almost nothing into history
rows (flat through the blocks). Read the `n` column: the target-row types
read (0.64) is n-limited — the opp-active public row at the SAME steps
reads 0.644 against 0.998 at 9x the rows.

**Declined, recorded.** (1) An explicit hand-written type-interaction row
(typechart lookup of my move type x their types): rejected by the user
because move types change at run time — Tera Blast, Weather Ball, Judgment,
Ivy Cudgel, Raging Bull, Revelation Dance, the -ate abilities — and
immunities route through abilities and items (Levitate, Air Balloon), so it
would be a human-maintained rule table, against the no-human-heuristics
invariant. (2) `entity_size` → a wider readout `qk_size`: the ceiling was
flat in rank (64/256) so width is not the lever.

**Pre-registered acceptance** (fresh lineage; probe E at a matched step
against irqeetfg): post-trunk type-class accuracy of the readout's rows off
the 0.55 majority floor; the policy's immune-target mass BELOW the uniform
baseline (0.372 vs 0.403 today — lower is better, it is mass on moves that do
nothing); wr vs SimpleHeuristic at temp 1.0 at 30k ≥ irqeetfg's at the
matched `lifetime_step`. **Abort:** none needed on a structural change —
the fallback if the gate fails is a self-play damage aux head (dense
per-move label from the engine's own outcome, no rule table), its own
commit.

## Investigation ledger — 2026-09-01 flash attention: measured, declined

The old objection ("APIs too immature") is RETIRED — jax 0.10.2's
`jax.nn.dot_product_attention(implementation="cudnn")` is first-class (bool
masks, GQA, per-batch seq lengths, logsumexp residual; bf16 and head-dim 64
both fine on the 3080 Ti). What replaces it is a measured no-win at this
model's shapes. Full table in the comment beside the einsum in
`rl/model/modules.py`; the bench scripts were scratchpad-only.

| finding | number |
|---|---|
| trunk shape (61-64 rows): einsum vs best flash | einsum 2-4x FASTER (0.12ms vs 0.32-0.54ms at seq 64) — kernel overhead dominates tiny attentions |
| crossover where cudnn wins | ~256 rows with `seq_lengths`; decisive only at 2048 (3.09 vs 4.11ms, and plain xla OOMs there — the memory win is real at long seqs) |
| the mask tax | a DENSE bool mask is folded into an additive bias whose materialisation eats the flash win at every length (cudnn+mask never beats einsum past noise); `seq_lengths` avoids it but is PREFIX-only, and the trunk's validity mask is scattered — structurally inexpressible |
| the odd-length blocker | cuDNN training backward raises verbatim "Unsupported sequence length Q 61, KV 61" whenever a mask/bias is present — the trunk would need 64-padding |
| masked-query rows | dpa returns garbage (~0.4) where the einsum's double-mask returns exact 0; the trunk's block-end hard-zero would absorb it, but it is a semantic difference to re-check on any future swap |
| softcap | max pre-cap logit 7.6 on the trained ckpt_00140000 against the 50 cap — qk layer norm bounds it; INERT INSURANCE, deletable if a future swap needs it gone |

Verdict against the pre-registered bar (adopt at >=5% win at the live shape):
declined at -2x to -4x. Revisit only when a design grows the sequence past
~512 with prefix-shaped masking (matchup rows, token-level history, the
parked world model) — the crossover number above is the deliverable. Also:
even a winning kernel moves nothing end-to-end today; the 4.2 steps/sec
system rate is actor-bound (the learner alone does 12.3). The dead
commented-out dpa call from the original attempt is deleted.

## Investigation ledger — 2026-09-01 GRU scan: hoisted, unroll retuned, the big claim measured DOWN

The follow-up to the flash-attention close-out: the history GRU scan IS the
mis-shaped compute (measured 34-56% of the actor forward at buckets 128-512,
~52us/step under learner contention), and the actor side is what binds the
4.2 steps/sec system rate. The classic cuDNN-RNN restructure — hoist the
input-side gate GEMMs out of the scan as one batched GEMM, keep only the
carry-dependent tail serial — was implemented in pure JAX with an IDENTICAL
param tree (`SplitGRUCell`: children ir/iz/in/hr/hz/hn mirror flax GRUCell's
Dense layout exactly, so the live lineage's checkpoint loads unchanged).

**The pre-registered >=30% bar FAILED, and the standalone bench that promised
-38% was a strawman**: the real scan at full GPU already ran 28us/step (the
bench replica of it: 55us) — XLA had less headroom than the mock suggested.
Measured on the real module, full GPU, H=512: scan 14.46 -> 13.82ms (hoist)
-> 13.22ms (+ SCAN_UNROLL 8 -> 32) = **-8.6%**; H=256: -7.4%. Landed anyway
as a small verified win with three structural improvements riding along:
the latest-node stream left the scan carry entirely (a last-touched-value
recurrence is a parallel cummax + gather — BIT-EXACT vs the old carry), the
per-step segment_sums moved to batched vmapped precompute, and the stacked
scan outputs shrank 27 -> 15 rows/step. Equivalence gate on real params +
ex.bin at H=256: nodes exact, recurrent states max|diff| 0.035 / corr
0.999998 — the compounding bf16-reassociation class the precision ledger
predicts, from GEMM splitting; landed at a restart boundary of the live
lineage (checkpoint-mode resume, param tree unchanged).

**The diagnosis with teeth: the slim scan is DEPENDENCY-LATENCY-bound at
~26us/step** — kernel count per step barely matters (unroll 32 bought 4%).
The remaining fix classes are (a) a Pallas whole-scan fused kernel (one
launch for all H steps; ceiling = ~13ms of the ~20ms H=512 actor forward;
backward pass is the hard part) or (b) an associative-recurrence
architecture change — both recorded, neither cheap. ALSO measured while
here: the system-level actor decomposition (GPU forward vs TS service vs
python plumbing) is still unmeasured, and the inference server has no
instrumentation — that measurement outranks any further kernel work.
Ridealong fix: `rl/model/capacity.py`'s probe still unpacked the
pre-flat-trunk encoder's two return values — latently broken since
2026-08-29, exposed by the first current-arch checkpoint, ported to the
61-row sequence (action = private|move|target block, value = CLS row).

## Audit + input-read redesign — 2026-08-28 (tag `pre-read-redesign-2026-08-28`)

A full read of `rl/model/` against `service/src/server/state.ts`, treating
comments as claims to check. Everything below is one fresh lineage.

**Two my-side/opp-side tagging bugs, both live for months.** The service
writes `ENTITY_PUBLIC_NODE_FEATURE__SIDE = isMySide(n, playerIndex)`
(`state.ts:487, 1068`), so **side_bias row 1 is MINE and row 0 is theirs**.
(1) `_current_entity_tokens` tagged my private sheet `side_bias(0)` — the
opponent's row — putting 48 of the read's keys under the wrong side in a
permutation-invariant attention where that additive tag IS the identity
signal. (2) `_embed_field` used `pos_bias` rows 1/0 as the my/opp side tag,
but `pos_bias` is indexed by `ACTIVE` (= `scoreOrder`, {0,2} in singles), so
row 0 meant "benched pokemon" AND "opponent side conditions" — and it was the
only thing separating my hazards from theirs, since both share
`side_condition_linear`. Both now own dedicated params
(`private_side_bias`, `field_side_bias`), and the SIDE convention is written
once.

**`PUBLIC_ORDER` was a no-op on the RL path.** `Encoder.__call__` re-aligns
history-slot order to public-row order per request — and then the read gave
all 12 resulting rows the same `entity_bias` and the same `HISTORY_SLOT`
type, i.e. a multiset of identically-biased keys, discarding the alignment.
History row i is now public entity i's **11th attribute token**, inheriting
its entity/pos+side/group biases. Test: swapping two slots' history states
must move the latents, with the control that the same swap under the OLD
layout is bitwise inert.

**PROBE C — the switch axis is starved by construction, not just by
supervision.** `EntityPrivateNodeFeature` has NO hp/status/fainted/boosts, so
`RESERVE_j`'s warm start was a STATIC set descriptor and a candidate's
condition could only arrive through the trunk. It did not.
`separation_probe --probe c` on the entropy-floor lineage @28540:

| readout | subset | held r |
|---|---|---|
| `RESERVE_j` | alive | **-0.004** |
| `RESERVE_j` | legal switch targets | **-0.127** |
| `RESERVE_j` | shuffled (floor) | -0.007 |
| `ally_1_switch` (control) | alive | 0.351 |
| `enemy_1_target` (control) | alive | 0.440 |

The trunk routes "is this mon dead" and nothing about hp among ALIVE
candidates — exactly the set a switch decision discriminates over. **Two
instrument lessons, both nearly cost the verdict**: the unconditioned
reading is 0.447 and is ENTIRELY the alive/dead contrast (it vanishes once
conditioned on alive); and `fainted` has zero variance on the alive/legal
subsets — you cannot switch to a fainted mon — so its r there is vacuous,
not a result. Read the y-std column before reading any r.

**The board was encoded TWICE and now is not.** Once as read tokens, again as
entity-local pooled vectors warm-starting the action slots. The 12
entity-derived slots now QUERY the same token set (`ActionSlotRead`), keyed
by **species + side + role**, reading the RAW tokens rather than the latents
— no compression loss, near-exact species match, each entity's folded history
token for free, and cheaper than the twelve [10-token self-attention +
pooling decoder] runs it replaces. `LatentInputRead` split into
`InputTokenSet` + `LatentInputRead` + `ActionSlotRead`; the set is assembled
twice per forward from ONE instance (prev-action tokens are gathered FROM the
finished action sequence, so they cannot be keys of the read that builds it).
`tag_with_species` adds each entity's species embedding to all of its tokens
— without it a species query retrieves the species token only, since hp lives
on a state token with no species content and the sole shared component is
`entity_bias`, which is positional and not content-addressable.

**Two things the design pays for explicitly.** The role bias in the query is
REQUIRED (`ALLY_i_SWITCH` and `ALLY_i_TARGET` name the same mon) but is
zero-init, so at step 0 the separation is carried by `ActionSlotRead`'s
per-slot position term. Position also covers where species is NOT a stable
key: `publicBattle` carries the disguise until `|replace|` (`state.ts:2080`),
so a **my-side** Illusion shows the disguise on the public row while the
sheet shows the truth — a *wrong* match, not a missing one. Opponent-side
Illusion is self-consistent and correctly fools the model, as it fools a
human.

**Inputs that were on the wire and never read**, now wired:
`FIELD_FEATURE__TURN_ORDER_VALUE` (the within-turn edge sequence index — the
ONLY observable of relative speed, since `segment_sum` in `_observe_step`
destroys the order of a step's edges; `_embed_field` read it, returned it,
and both callers dropped it), `ENTITY_EDGE_FEATURE__HIT_COUNT`, and a request
info token carrying `REQUEST_TYPE` + `NUM_ACTIVE` (InfoFeatures, so they
cannot ride `_embed_field`, which is shared with history rows). The history
field state splits into (global, mine, theirs) with side-filtered messages —
hazards are side-differenced and one collapsed vector could only hold their
mixture. `NUM_INPUT_TOKENS` 186 -> 189, params 27.26M -> 28.71M.

**`ENTITY_EDGE_FEATURE__EFFECT_TOKEN` is never written by the service** —
always 0 on the wire. NOT deleted: the enum is contiguously numbered and the
buffers are sized by `len(keys())` but indexed by enum value, so removing 22
forces 23-33 to renumber, shifting every packed edge row and invalidating the
19 shards in `replays/shards`. Documented in the proto; delete it when those
shards are next rebuilt.

**Verified sound, do not re-litigate**: src-major grid indexing agrees across
`FLAT_MODALITY_MASK` / `MicroHead` / `src_index` / the service's `setRowCol`;
`my_moveset` row -> `MOVE_INDICES[k]` and `RESERVE_j` <-> `private_team[j]`;
public rows 0-5 mine / 6-11 theirs actives-first; unrevealed opponent mons are
`SPECIES_ENUM___UNK` not `_UNSPECIFIED`, so all 12 public rows are always live
keys and the read CAN count remaining mons (keep deliberate); masking is
NaN-safe end to end. Eval runs at `temp=0.5` and `ActionScoreHead` divides
BOTH levels by temp, so eval roughly squares the modality marginal — the
headline winrates measure a policy that switches less than the trained one.
Left as-is, recorded as a confound.

**Acceptance**: `entropy_macro` / `prob_switch` through 33k, wr vs
SimpleHeuristic at 30k, a BR probe at ~77k against the 0.57-0.58 the 77638
probe read, and `separation_probe --probe c` re-run at a matched step —
`RESERVE_j/alive/hp` should move off ~0.00 toward the controls' 0.35-0.44.

## Probe ledger — 2026-08-27 capacity falsification (separation probe)

The within-switch flatness measured on the live critic (within-row Q std
0.0196 vs move 0.0374 @ckpt_00224773) motivated restoring the Nov-2025
per-modality decoder depth. The pre-launch separation probe
(`rl/offline/separation_probe.py`) was built to gate that launch — and it
FALSIFIED the capacity hypothesis on every reading its data can support:

| reading | old arch | new arch (+3 decoders, +2.8M) |
|---|---|---|
| probe A routing separation at init | 3.76x | 3.61x |
| train fit (memorisation) | r=1.000 | r=1.000 |
| held-seen species, cross-game (3 seeds) | 0.63–0.89, mean 0.75 | 0.66–0.88, mean 0.73 |
| held-seen move ids, cross-game (122-chunk set) | 0.997 | 0.997 |
| identity, within-game rows split | 1.000 | 1.000 |
| RELATIONAL pair (species x opp active), rows split | 0.957 | 0.967 |

**Verdict: the decoders add nothing measurable; no launch on capacity
grounds** (pre-registered branch). By elimination the live deficit is
SUPERVISION COVERAGE: taken-cell-only Q loss at vol_switch_rows -> 0 asks
for within-switch ranking exactly zero times — no architecture learns from
zero labels. The supply mechanism (phase-4 anchor) is the load-bearing fix,
not capacity. The decoder commit stays in-tree unlaunched pending the
resume decision (resuming the archived lineage requires reverting it — the
manifest `action_pathway` literal correctly blocks cross-arch loads).

**Instrument lessons, paid for in one afternoon instead of a lineage:**
(1) on a FIXED batch, train fit is pure memorisation (species-slot frozen
per row; state features fit it without routing) — held-out is the only
capacity reading; (2) held-out labels keyed to identity are UNLEARNABLE for
unseen ids, and randombattle species overlap across games is ~27% (moves
~79%) — the first dramatic 0.096-vs-0.58 "architecture gap" was exactly
this artifact; (3) at train loss ~0 the interpolator's held-out behaviour
is rounding/seed sensitive — ±0.1 r across seeds at n=28 rows; single
readings are noise; (4) (species x opponent) pairs NEVER recur across
randombattle games, so cross-game relational probing is structurally
impossible here — the rows split (hold out half of each game's timesteps)
is the only relational reading and is an acknowledged lower bound; (5) a
fresh-init supervised probe can only EXCLUDE capacity explanations — it
cannot see training-dynamics pathologies, which are what remains.

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
  accordingly. That figure is for a SINGLE softmax; carried through a recurrent
  scan it compounds (see the next entry).
- **bf16 GEMM results depend on the INPUT SHAPE, so "same content, different
  padding" is not bit-identical** *(live, 2026-08-25)*. XLA autotunes a kernel
  per shape, so `A[:m] @ W` and `(A @ W)[:m]` differ by ~1 ULP — measured
  directly: for `A(2048,1026) @ W(1026,256)` in bf16, `m=708` gives maxdiff 0.5
  over 81 rows while `m=2048` is exact, and the effect is NON-MONOTONE in `m`
  (f32 same case: 1.9e-4, rel 1.3e-6). This bit
  `test_untruncated_tail_window_forward_is_identical`, which demanded
  `atol=1e-5` across a history clip that changes `message_projection`'s leading
  dim 2048 -> 708: one ULP (0.0078125) amplified by the 177-step GRU scan into
  0.023 on the value logits. **Never assert bitwise or tight equality between
  two forwards whose tensors have different shapes**, however semantically
  identical the content — assert content equality on the ARRAYS, where it is
  exactly checkable, and give the forward a precision-realistic tolerance. The
  non-monotone-in-padding signature is how you tell kernel selection from real
  data loss.
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

**NeuRD must not differentiate through the log-softmax.** *(retired with
NeuRD 2026-08-26 — under a true PG objective the log-softmax cross-term IS
the correct gradient; this ban was about NeuRD's regret-weight semantics
only.)* The `log_policy` form was tried first and failed the identity
test: once the logit-gap clip zeroes cells, the weights are no longer zero-sum
and the softmax pulls in a `π(b)·Σ_a w(a)` cross-term. But RAW free logits
overshoot the other way: the softmax-invariant mean direction gets the clip
residual unopposed — no pi, no decay, no clip sees it — and the dx65cpwp
micro runaway rode exactly that gauge freedom. The correct form is rnad.py's:
the loss reads each level's CENTRED live logits (`logit_pi - mean_logit`), so
`d/dy_b = -(w(b) - mean_legal(w))` — a pi-free linear projection (no
cross-term along π), zero-sum per level, open or clipped.

**The dx65cpwp micro runaway (2026-08-26) — diff against the reference impl
before inventing.** The entropy analytic shift (eb3bf4a) made the policy micro
level diverge twice — eta_ent 0.05 and, after f9b2481's 5x cut, again at 0.01,
both eras converging on the SAME diseased params (micro_local_tgt 25x, adapter
18x; the advantage head, same modules under the Huber Q loss, calm) — so the
coefficient was never the root. The root was three discrepancies against
rnad.py/DeepNash (Table 2, arXiv:2206.15378), each now fixed: (1) they run
Adam **b1=0** — momentum carried each push ~10 steps past the stiff
equilibrium the analytic shifts create (player-only now; the builder's PPO
surrogate keeps 0.9); (2) they **clip the total per-cell force** after every
correction — nothing in DeepNash/NashPG ever exposes an unbounded −η·log π
to the logits (reward transform / near-snap ratio / PPO clip respectively);
(3) they differentiate **centred logits** (the lesson above). The policy-head
param panels (`player_policy_micro_local_*_rms`, `player_policy_adapter_rms`)
exist because this diagnosis needed checkpoint forensics — healthy O(0.005),
diseased 0.07–0.15.

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

**NashPG closes the lineage (2026-08-26).** The sampled-action objective
returned — this time as the deliberate, reference-backed choice (NashPG §5.4:
the inner update rule was the bottleneck, PPO > NeuRD in large games), with
its eyes open about the cost: no force refills a starved cell any more, so the
all-action era's starvation instruments (`player_policy_prob_ratio` vs
`absadv_ratio`, switch_ratio through the 13k wire) are now the acceptance
gate rather than a diagnosis. See the 2026-08-26 removal ledger for the full
trade and the fallback ladder.

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

- **2026-09-07 per-chunk protection built, not activated — DELETED
  2026-09-09** (never switched on in any run; "Removal ledger — 2026-09-09
  per-chunk retention feedback" below carries the revert handle).
  `player_replay_trajectory_mode` defaults to `off`; `observe` logs detached
  per-chunk taken-action k3 mismatch with unchanged uniform sampling, and
  `protect` retires chunks above the existing KL threshold from FUTURE draws.
  This changes retention, not a corrected priority sampler or an estimate of
  learning utility. Slot/monotonic-ID/visit feedback rejects replaced occupants
  and stale visits; prefetch still counts against the global cap. Batch-level
  0.045 has not been calibrated as a noisy per-chunk decision threshold: observe
  before enabling protection. Revert handles: `training/replay.py`, buffer
  feedback/eligibility, `Trajectory.replay_{slot,id}`, batching/train-step/worker
  wiring and the config mode. Detailed maths/limits: local
  `docs/adaptive-trajectory-replay.md` §10. No live replay change or strength
  result is claimed. Validation: 36 focused buffer/chunking/replay checks
  passed; all 14 replay checks passed after the final telemetry change.
  GPU train-step activation checks await a learner-free window.

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

## 7. Information sets

*(the privileged critic was DELETED 2026-08-25 — see the removal ledger)*

- **There is no privileged input.** Every stream — state, action, value —
  sees exactly the agent's deploy-time information set. The contract is no
  longer "route the sheet to one rung"; it is "there is no sheet".
- **Two test traps, both learned the hard way.** Residual gates are zero-init,
  so a leak test at init multiplies any leaked contribution by zero and passes
  vacuously — open the gates first (`open_zero_init_paths` in `conftest.py`),
  and include a negative control proving the perturbed entity *does* respond.
  And invariance alone is one-sided: a Φ_ann that never reads the turn at all
  passes an invariance check perfectly. These outlive the ladder: any future
  "is it wired" test needs the same two halves.
- A probe that reads checkpoint params BY NAME must skip, not fail, on a
  checkpoint from a superseded architecture — otherwise it reports a
  name mismatch as the collapse it was built to detect
  (`tests/test_checkpoint_collapse.py`'s `value_embeddings_table` sentinel).

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

- **SET ASIDE 2026-08-29 as confounded** (the flat readout went in on that
  reading; the ledger above carries the gate that judges it). *Deep,
  modality-separated action decoders are empirically necessary.* Make
  that depth cheaper; do not remove it. RESTORED 2026-08-25 after a
  three-week regression: every micro parameter is now keyed by slot group
  (`SRC_GROUP_MASK`) and every macro parameter by modality, with no sharing.
  The flattened version distinguished the three groups by a single scalar
  each, and the 84.9k-step read found the target group's scalar still
  BITWISE ZERO on both the policy and the advantage head — under that
  parameterisation the scalar *was* the group's entire readout, so the group
  had never trained at all (`docs/qva-redesign-step0-reference.md`).
- **G disjoint projections are one Dense with G*qk outputs.** Attention heads
  own disjoint output coordinates, so `PointerLogits(num_heads=G)` gives
  genuinely unshared per-group projections with no vmap and no loop; the
  per-group select is a one-hot contraction over the group axis. A cell's
  group is a function of its SRC half alone, and `micro_local_tgt` is read
  under the CELL's group — the same target token scores differently
  depending on which modality is choosing it, which is the point.
- **A learned grid behind a zero-init scale is a two-factor product and it
  stalls.** Whenever per-group or per-modality parameters go behind a
  zero-init gate, pair them with a zero-init SINGLE-factor route over a live
  input (`micro_local_src`/`micro_local_tgt`), or the gate's gradient is a
  random grid's correlation with the residual and neither factor moves —
  measured: 60k steps, gate 0.03-0.06, q/k kernels still at lecun init.
- **Flat-at-init contract.** Every micro/macro/adapter output path is zero-init,
  so the policy starts at its hierarchical prior and Q at uniform bins — no
  lecun noise posing as action preferences for CE to unlearn or for the
  policy loss to misread. Exception: the cross-entity pool read gate starts at 1.0, because
  token content can only reach the entity vector through that read.
- **SET ASIDE 2026-08-29, partially and knowingly** — one token per entity
  re-adopts entity-local pooling. My 16 candidate moves and the four
  entity-derived target rows keep BOTH operands of *my move x their mon*,
  which is the direction a decision turns on; what is given up is their
  individual revealed moves. The fix if it bites is explicit matchup rows,
  not re-unpacking attributes. *Cross-entity pooling exists because matchup
  reasoning is a species-token ×
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
- **Nov 2025 hierarchical policy head beat the flat gram head** in competition
  and was removed in 0e23621 — a known regression. Addressed 2026-08-25 by
  the per-group/per-modality restoration above; judge it on
  `player_adv_rms_{move,switch,target}` all moving separately and on the
  three `type_scale` entries leaving zero.
- **RETIRED 2026-08-29** with the advantage head — there is one readout and
  no second composition. *One readout, two compositions.* The policy and the advantage score the
  SAME src x tgt grid and differ only in `reduce` — `ActionScoreHead`. Before
  2026-08-25 the sequence (adapter -> src_valid -> macro/micro -> compose)
  was open-coded three times (singles policy, doubles stage, Q head) and had
  drifted between the copies. If a fourth consumer of the action axis
  appears, it is a third `reduce`, not a fourth copy.

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

## MCTS evaluation pilot parked — 2026-09-07

At user request, irqeetfg stopped gracefully at checkpoint `ckpt_01861967`
(step 1,861,967; 26,751 updates after latent-model rewrite). Training remains
stopped. Local report: `docs/mcts-ablation-2026-09-07.md` (gitignored).
Optional MCTS now works through `sh eval.sh --search mcts`; plain stays default.
Five independent 100-game arms completed against SimpleHeuristic: plain 45 wins,
MCTS depth 1 48, MCTS depth 2 50. These small differences do not establish a
search gain. Fixed-checkpoint 500-transition prior-expectation delta gain was
-0.0817 (95% game bootstrap [-0.1579,-0.0251]); doubling chance draws gave -0.0818.

Performance audit found a concrete issue: the model's outer head-output vmap
turns conditional MCTS expansion into masked evaluation. A two-expansion toy
executed imagine twice without vmap and 64 times under vmap. The current
`mcts_model_calls` field counts logical expansions, not physical calls.
Warmed single-root GPU medians: plain 1.65 ms, expectimax depth 1 3.67 ms,
expectimax depth 2 31.84 ms, MCTS depth 1 118.43 ms, MCTS depth 2 175.83 ms.

mctx commit b53073fd5035618228a717e29254e36ceb6f0645 separates cheap traversal
from one batched expansion per simulation. The user parked work before applying
that layout fix. Next: separate traversal/expansion, preserve scalar guards in
single-root search, compare bit-identical outputs and remeasure. Original source,
full outputs, timing logs and a GPU trace are preserved in runtime/; exact paths
and remaining steps are in the local report. No performance fix or promotion
should be inferred from the functional pilot. No commit was made.

## MCTS GPU layout correction — 2026-09-07 resumed

Implemented mctx-style read-only traversal followed by expansion and backup in
`rl/model/mcts.py`; statically omit candidate generation at depth 1. Preserve
chance-bank keys, PUCT, terminal backup and visit policy. On the parked benchmark
(checkpoint 01861967, EMA, 13 legal actions, history bucket 4, 30 warmed keys),
median GPU latency falls 118.43 → 27.21 ms at depth 1 (4.35×) and 175.83 →
124.06 ms at depth 2 (1.42×). All arrays in 60 full actor outputs match the saved
originals bit-for-bit. This changes runtime cost, not evidence of search strength.

Declined scalar head lax.map: 18.90/74.35 ms, but changes bf16 computations and
some visits (274 differing returned leaves across 30 shallow outputs). Reverted
that trial and retained the vectorised heads to honour structural equivalence.
Original code/outputs remain in runtime/; local report has exact paths.

vmap still executes masked expansions, so mcts_model_calls means accepted cache
entries rather than physical evaluations. Expansion is now outside traversal,
bounding executed dynamics calls to one per simulation per root. Six new callback
counter tests cover scalar/map/vmap at depths 1/2 and prove the physical bound;
depth-one candidate generation never runs. The GPU measurements and numerical
comparisons used no live learner. Training remains stopped, no commit made.

## MCTS singleton batching guards — 2026-09-07 second performance pass

Retained vectorised heads and added inference-only `_guarded_cond` in mcts.py:
choose the branch with a scalar predicate for singleton vmap, but execute the
chosen branch using the original batched arithmetic. This avoids the bf16 changes
of the rejected scalar-head trial. All captured arrays must be explicit: ordinary
jax.closure_convert omits some constant-valued, batched support arrays; the helper
uses make_jaxpr + jax.extend.core and custom_vmap. No reverse-mode rule is supplied.
Mixed multi-root predicates keep the ordinary masked rule.

On the same 30-key GPU benchmark, depth 1 improves 27.21 → 19.77 ms and depth 2
124.06 → 77.51 ms (5.99× / 2.27× versus the original prototype). All returned
arrays in all 60 actor outputs are still bit-identical to the original. A profiled
depth-two decision drops 65,058 → 40,163 GPU events. Sixteen MCTS tests pass,
including singleton physical-call counts, multi-root positive controls, captured
integer/float values, and nested batching. Original and first-fix artefacts remain
in runtime/; local report records exact paths. Training remains stopped; no commit.

## Replay comparison at 01861967 — 2026-09-07

Read-only model/training audit; no human matches. Frozen EMA, plain GPU actor,
Gen9 random battles: 400 games vs SimpleHeuristic at T=1 (198 wins, 49.5%,
Wilson 95% 44.6–54.4), 400 at T=.5 (247 wins, 152 losses, one zero result;
61.75% wins, 56.9–66.4), plus 100 self-play games. Simulator seeds unpaired.
Compared protocol logs with 2,000 seeded-sample human replays, both ratings
>=1900; 1,196 uploads from 2026. No human-relative Elo inference is possible.

Tera by turn 3: 81.8% T=1, 83.0% T=.5, 83.0% self-play, 3.1% humans.
Median use turn: 2/1/1 vs human 18 (conditional on use). Voluntary switches
per observed move-or-switch: 12.8%/4.5%/14.3% vs human 20.2%; 18.9% in
974 human games with one side actually losing six. These are not mask-conditioned
choice rates. SimpleHeuristic itself switched 1.5% and never used Tera.
One T=.5 protocol log ended without an outcome marker (turn 75, zero reward);
exclude it from completed-game behaviour (399), retain in 400 outcomes.

Manual cases distinguish demonstrated setup/finishing and Leech Seed/Protect
wins from repeated ineffective attacks with demonstrated alternatives, and a
failure to leave an immune matchup. Encore and passive winning lines make raw
repeated-action counts unsafe as blunder labels. Lower temperature improves
baseline results but preserves both early Tera and some tactical failures.

New objective hypothesis, NOT tested or implemented: `CELL_MODALITY_MASK`
separates MOVE/WILDCARD, so `uniform_kl_modalities` pressures the one-use Tera
category towards equal marginal mass on every eligible decision. Investigate
regulariser grouping while retaining stay/switch support before widening the
architecture or reducing all exploration. Timing differences do not prove early
Tera loses games. Current-state policy/attribute probes must discriminate rare
samples from confidently wrong rankings before proposing an input-row skip or
self-play action-effect auxiliary. Human frequencies are diagnostic, never rewards.

Local report: `docs/replay-audit-01861967-2026-09-07.md` (gitignored); raw logs,
results, selected human paths, analysis scripts and aggregates:
`runtime/replay-audit-01861967/`. No model/configuration changes or commit;
training remained stopped, task service stopped. Built service must run with
`service/` cwd here (`../constants/data.json` is cwd-relative); initial root-cwd
launch failed before any game and was replaced. No removal/restoration involved.

## Priority investigation at 01861967 — 2026-09-08

Follow-up to the replay audit; no model/configuration change, training launch or
human match. Local report `docs/priority-investigation-01861967-2026-09-08.md`;
reproducible scripts/raw data in `runtime/priority-audit-01861967/` (gitignored).

Four fresh EMA/T=.5/GPU arms of 400 SimpleHeuristic games: unchanged 255 wins,
hold Tera until turn5 236, until turn10 232, never Tera 236. Independent simulator
seeds; all differences' 95% intervals cross zero. **No evidence that forcing later
Tera helps.** This weakens the first audit's priority, not proof that a differently
trained resource policy cannot improve. On 816 visited eligible requests, mean
P(Tera|move)=.496. The current modality loss's conditional derivative is
`.025/M*(2q-1)`, pulling towards .5; its average is only −.0000687. The derivative
shifting all Tera logits with others fixed averages +.00272 (less Tera); distinguish
conditional resource choice from move/switch allocation. Finite difference error
<2.4e−12. No regulariser change or scripted delay justified yet.

New 400-game full trajectories confirm **confident tactical misranking**: Hypno
into revealed Soundproof at turns 14–16 has stored actor Psychic Noise probabilities
.99783/.99550/.99222, with Focus Blast legal and Soundproof present in observations.
The side ultimately wins; terminal success can coexist with locally ineffective
play. Basic type awareness is nevertheless real: 876 requests with an immune and
nonimmune regular damaging choice carry immune mass .166 T1/.129 T.5 versus .410
uniform (game bootstrap T1−uniform [−.279,−.211]). These labels are diagnostic and
exclude selected dynamic type/ability cases; not a complete mechanic oracle.

Three held-GAME representation splits on 200 games: move type pre 100%, post 81–83%;
opp type on target row pre 84%, post 54–59%; matchup class on move row pre 52–54%,
post 65% (majority 54–57%). The trunk computes useful relations while weakening raw
attribute accessibility. A projected bilinear probe is not an architecture ceiling.
Prioritise a matched self-play action-effect auxiliary versus small input-row
readout connection experiment; do not combine them or infer width is the lever.
No auxiliary or architecture experiment implemented in this investigation.

Posterior-expectation gap closed on the same 500 transitions/24 self-play games:
exact 256 chance codes, top 8 action mass .9912. Posterior mode delta gain +.0441;
posterior expectation +.0429, 95%[−.0100,+.0891]; prior expectation −.0818,
[−.1570,−.0248]. Original reference reads reproduced. Posterior mode selection was
not hiding a strong expectation. Moves drive weakness; switch posterior expectation
+.1156[+.0015,+.1943], prior −.0022[−.0912,+.0529] (exploratory splits).
Initial batched diagnostic OOM resolved by sequential actions/chance batches 16;
production code unchanged. This metric tests taken-action prediction, not action
ranking. Existing transition stop-grad boundary does not directly train policy rows.

Existing depth 1 expectimax, 16 seeds per case, leaves Psychic Noise first at .99786
and immune Spirit Shackle first at .95764 in all seeds. Focus Blast's mean predicted
Q advantage .00282 versus .773 required to reverse the policy with Q/.1; U-turn
.00119 versus .333 in the second case. No root truncation. Do not amplify this
uncalibrated signal or promote deeper search; fix tactical discrimination first.
No full tests or commit; diagnostic forwards/syntax, game bootstrap, stored actor
probabilities and raw per-seed checks performed. Task service stopped.

## Addition ledger — 2026-09-08 consistency restoration preflight, absent-row amplification

**Offline gradient read at `ckpt_01861967`; no training updates applied.**
The proposed consistency 0 → 1 continuation was preflighted using the exact
`train_step` loss prefix, current full stored shape (T=64, B=4, H=256), saved
learner parameters, target/reg parameters, and restored Adam moments. Both
coefficients use identical batches and step-derived sampling keys; shared
value-head training remains True. Eight batches from the saved 24 self-play
games give total gradient norms **11.8–21.1 off versus 2476.8–8256.0 on**.
Global clipping at 10 reduces the other objectives' incoming gradients to
**0.143–0.547%** of their off-arm scale. This is NOT the parameter-step ratio:
restored Adam's encoder update norm is 0.791–0.845 times control; action-head
update norm is 0.846–0.968 times control. Optimiser history matters.

**99.29–99.65% of the consistency loss comes from PREV_ACTION.** All 3072
stored rows in these self-play chunks have HAS_PREV_ACTION=0 (includes stored
padding). The real trunk hard-zeros these absent rows. Their copy-movement
normaliser is exactly zero in all eight batches; imagined-versus-real squared
error is 19.92–43.07, divided by the 0.01 floor, giving group losses
1991.9–4306.5. The final objective averages ten policy-readable groups, then
multiplies dynamics by 0.5. The service clears `actionEnumPairs` after assembling
each request's choice; its previous-action features concern earlier choices
within a request, so absence in singles is consistent with the current producer,
not evidence of an outdated replay exporter. `imagine` deliberately produces
all rows without a future-validity oracle, while consistency scores every group
without a row-validity mask. Thus dormant coordinates can dominate restoration.

An independent eight-batch stress read from 400 saved heuristic-opponent games
also finds amplification (off norms 16.3–963.5, on 4249.8–8454.0); this is an
off-policy stress sample, not a substitute for the self-play read. The structural
stop-gradients still exclude the real encoder/trunk and policy from this loss;
global clipping and saved optimiser moments are the indirect coupling. Tiny
non-transition differences in subtraction of two bf16 total gradients are not
by themselves evidence that the explicit stop-gradient boundary is broken.

**Verdict:** supports a large restoration mismatch, not a causal claim that a
late-trained trunk is trapped in bad parameter space. Drift while consistency
is off and absent-row normalisation are concrete alternatives. Do not discard
the lineage or treat a coefficient-1 hold as a clean test of useful latent
prediction before addressing this distinction. Isolated full-state arm launchers
were prepared but the long self-play forks were NOT launched after this
preflight finding. No production configuration or gradient boundary changed.
Local reproducibility: `runtime/consistency-ablation-01861967/gradient_audit.py`,
`selfplay-gradient-audit.json`, `gradient-audit.json`; launcher preparation lives
beside them. The script adds scalar numerator/normaliser reads to the extracted
loss; forwards/gradients are jitted on GPU. These local artefacts are gitignored.

## Addition ledger — 2026-09-08 latent matching, validity versus normalisation

**Matched offline ablation, same eight self-play batches and full optimiser
state as the restoration preflight.** Task-local `matching_audit.py` wraps the
model only to expose detached target-label validity and real-state energy;
the transition's inputs, attention and predictions are unchanged. The mask
includes a row when valid at EITHER endpoint, retaining appearances and
 disappearances. Group losses average only groups with eligible rows.

| Consistency at coefficient 1 (outer dynamics coefficient still 0.5) | Raw consistency loss | Total gradient norm | Added-gradient norm |
|---|---:|---:|---:|
| Existing all-row, movement-normalised | 200.62–432.16 | 2476.84–8255.97 | 2476.99–8255.99 |
| Endpoint-union validity, original movement normaliser | 1.4395–1.5591 | 11.8845–21.1643 | 0.9567–1.2175 |
| Same validity, real-state energy normaliser | 0.5126–0.6674 | 11.8235–21.1444 | 0.2800–0.4980 |

The consistency-off total gradient range is 11.8125–21.1406. Off losses for the
mask-only variant equal the original exactly. With masking alone, global clip
scales retain 99.394–99.888% of control; restored Adam encoder update norms
retain 99.824–99.960%, action-head updates 99.828–99.993%. Added consistency
versus other transition gradients has cosine -0.048 to +0.025, not a strong
aligned or opposing force on these batches. All eligible group mean movement
normalisers exceed 313, far above the 0.01 floor.

The energy alternative divides by the detached group mean of
(||h_current||² + ||h_next||²)/2. Its smaller gradient does not establish better
learning: changing denominator also changes the effective coefficient. The
mask-only result already resolves the measured spike. Prefer validity-aware
matching as the first controlled continuation; retain copy-relative gains for
diagnostics and do not conflate an alternative normaliser with a masking fix.
A fixed movement floor remains susceptible to genuinely static VALID groups
and is width-dependent because coordinate errors are summed; that is a residual
possibility, not observed dominance in these eight batches. Pure cosine matching
would discard magnitude information read by the shared heads, so it is not an
equivalent replacement.

The dominant excluded coordinates are the two PREV_ACTION rows, zero-based
indices 58–59 in both the full and selected policy layout. `imagine` keeps all
rows live deliberately so future-appearing rows can be predicted without an
oracle. The missing mask is in the comparison loss, not a reason to restore
current-validity masking inside the transition. Future validity is used only as
a training label. Ignoring absent coordinates does not prove that their imagined
activations are harmless in deeper unrolls; held-checkpoint prior-expectation
calibration and play remain required. No production code/config or training run
changed. Results: local `valid-movement.json`, `valid-energy.json`,
`matching-summary.json` alongside the diagnostic script. GPU jitted forwards and
gradients completed; script syntax checked. No new matches were run.

## Addition ledger — 2026-09-08 production consistency validity fix

Applied the measured mask-only change to production. `PlayerActorOutput` now
carries `transition_cons_valid` (T, rows; T, B, rows after batching), produced as
current OR next validity in `_forward_transition`. It is a loss-side label only.
`transition_losses` intersects it with eligible first-step transitions, averages
only nonempty groups, and uses the same mask on the newly-valid diagnostic.
Empty groups/splits report gain 0 rather than an artificial perfect gain 1.
The movement normaliser, its floor, coefficient 0, parameter tree and transition
attention remain unchanged. No training restart, commit or push.

The mask has no generation, format or PREV_ACTION special case. Present previous
choices in doubles remain supervised, including appearance and disappearance
between decisions. Tests prove a large error in absent previous-action slots
has zero gradient, the same slots have a live gradient when valid (even with
zero movement), empty groups do not dilute the average, all-absent labels give
zero consistency gradient, and the bootstrap tail remains excluded. A real-model
GPU test varies HAS_PREV_ACTION through absent/appearing/present/disappearing
states and verifies both previous-action row masks through the actual encoder
and output wiring. This validates the doubles previous-choice seam, not the
separate known end-to-end doubles slot-alignment defect.

Validation: all 34 fast tests in `tests/test_transition_model.py` passed; the
new real-model mask test passed on GPU. Its initial test-only NumPy/JAX fixture
conversion error was corrected before the passing rerun. Focused Ruff, Black
check and `git diff --check` passed. The unrelated dirty `data/ps` submodule was
preserved. Earlier offline results above remain the pre-fix experiment record.

## Addition ledger — 2026-09-08 main-run consistency restoration

User authorised resuming the main lineage with the corrected consistency loss.
`player_transition_cons_coef` restored to 1.0; shared value-head training remains
True, outer `player_dynamics_coef` remains 0.5, and other coefficients are
unchanged. Replaced the stale config comment claiming posterior-conditioned MSE
necessarily destroys stochastic branches; history and its falsification remain
in the earlier ledgers. The two focused consistency coefficient/masking tests
passed with the new default; focused Ruff/Black and diff checks passed.

Launch: `bash start.sh --load-mode checkpoint --init-ckpt
/home/joseph/Documents/porygon2/ckpts/gen9/ckpt_01861967`, tmux `train`, learner
PID 168034, log `runtime/learner_20260908_100541.log`. Explicit checkpoint mode
requests full optimiser/target/reference/league restoration at step 1,861,967;
this is the main continuation, not the earlier prepared isolated forks.
Initial startup config confirms consistency 1.0 and live shared value head.

Review after approximately 20k additional updates (around step 1,882,000):
compare consistency and posterior/prior value calibration against the recorded
preflight checkpoint; rerun the saved-game prior-expectation probe at a held
checkpoint before claiming better search. Check deployable value quality and
same-temperature baseline play alongside gradient/clipping and update skips.
The older architecture's posterior R² 0.40 threshold is not a measured baseline
for this checkpoint. Initial startup success alone does not establish benefit.

Startup verification: full checkpoint and 15-opponent league restored, W&B
`irqeetfg` resumed, fixed shape precompilation completed and updates advanced.
At lifetime step 1,862,046 the live summary reads total gradient norm 11.9266,
transition gradient norm 3.1152, consistency loss 1.20953 and update-skipped 0.
This confirms the restored coefficient is operating without the preflight's
thousands-scale gradient in this observed update; it is not a strength verdict.
A requested bounded W&B history scan failed with the API's "Step column '_step'
not found in schema" error, so these are summary observations, not a verified
zero-skip count over every startup update. No learner errors were found in the
bounded startup log check. Run: https://wandb.ai/jtwin/pokemon-rl/runs/irqeetfg.

## Addition ledger — 2026-09-08 consistency continuation, first 5.8k updates

Read completed records from the current local W&B event file (avoids the API
history scan failure above), through lifetime step 1,867,813: 5,846 learner
records beginning at 1,861,968, all update-skipped values zero. First versus
latest 1,000-update windows: consistency loss 1.0992 → 0.9746; CLS consistency
gain -0.3181 → -0.06285 (substantially closer, still worse than copy on CLS);
total gradient norm mean 7.701 → 7.780; deployable value-head R² mean
0.9152 → 0.9278; throughput 4.775 → 4.734 steps/s. Across the first 5,627
records, total gradient norm ranged 3.87–65.53 and transition norm 1.90–10.82:
no recurrence of the thousands-scale restoration pathology.

Pooled value-delta diagnostics (1 - sum residual / sum target energy), first
versus latest 1,000: posterior 0.07593 → 0.08910, prior MODE 0.03758 → 0.04436.
These are modest changes, not the held-checkpoint prior-expectation test.
Grounding gains slightly declined (public posterior 0.2518 → 0.2406, prior
0.1966 → 0.1881); policy switch-cell probability remained 0.04075 → 0.04149.
No blanket downstream improvement claim is warranted.

Completed live EMA evaluation games since restart: plain temp 0.5 131/218
(60.1%); plain temp 1.0 38/109 (34.9%); depth-1 search temp 1.0 54/108 (50.0%).
Search MUST be compared with temp 1.0, its configured control, not temp 0.5.
The apparent +15.1pp search margin is encouraging but these are unpaired games
across changing checkpoints, not a frozen-checkpoint causal ablation of the
consistency change. Small main-parameter eval samples were excluded from these
EMA totals. Continue the planned roughly 20k hold; alignment has recovered more
clearly than value calibration so far. Local artefacts: `read_live.py`,
`live-history.json`, `live-review.json` under
`runtime/consistency-ablation-01861967/`. No extra matches, tests or training
mutations were needed for this inspection.

## Addition ledger — 2026-09-08 trunk homogeneity at 10.6k after restart

Streamed the local W&B records retaining only three scalar fields (no model
forward while training): 10,621 records through step 1,872,588. First 1,000
versus latest 1,000 (1,871,589–1,872,588) mean trunk row cosine 0.11176 →
0.10787; centred participation ratio 6.8939 → 6.9042. No progressive global
row-homogeneity collapse is evident during this continuation. `row_homogeneity`
reads the full valid learner sequence, including privileged rows, so this global
mean does not isolate policy-readable rows or within-move/target similarity.
Participation is spectral concentration, not an exact rank or count of encoded
features. High-energy variation in about seven effective directions does not
prove the other directions are absent or unusable.

The pre-restart held-game representation probe remains evidence for selective
loss of easy feature access (move type 100% → 81–83%; target-row opponent type
84–85% → 54–59%), while matchup-class decoding from move rows improved to ~65%.
This is not proof of erased information or of global row collapse. Do not add a
row-diversity penalty or reset the trunk on these aggregate metrics alone;
causal feature-access/readout tests would distinguish that concern. Local scalar
report: `runtime/consistency-ablation-01861967/homogeneity-review.json`.

## Addition ledger — 2026-09-08 belief review, last 10k at step 1,873,286

Streamed completed local W&B records, retaining only belief/code scalars:
10,000 updates spanning 1,863,287–1,873,286. Window means on the same masked
hidden-code population: belief accuracy 0.84729, per-batch marginal-majority
baseline 0.40482, species-only 0.55093, own-revealed-row control 0.80337.
Above-marginal accuracies: belief 0.44247, species 0.14611, revealed 0.39855.
Context margin 0.04393; species margin 0.29637. CE: belief 0.40438, revealed
0.52006, species 1.31520 (belief versus revealed ~22.2% lower).

First versus last 2,000 updates within that window: belief accuracy
0.84629 → 0.84637; context margin 0.04377 → 0.04354; CE 0.40822 → 0.40584.
Healthy, stable prediction above controls, not a fresh improvement trend.
Hidden-label perplexity averages 4.73996; average weakest-group perplexity
2.87527, minimum over recorded batches 1.30451: no group pinned at 1 evident
on this instrument. Hidden fraction among matched mons 0.98574; matching
coverage 0.63991, so the metrics do not score every opponent mon.

The own-revealed-row control remains strong: most predictability is available
locally, and the incremental context contribution is ~4.4pp. This does not
fulfil the older B3 expectation that the revealed baseline would converge to
the species baseline; beating both controls alone is not evidence for that
stronger claim. These are replay-training accuracies/CE for a learned hidden
code, not exact hidden-item/moveset accuracy, a held-out calibration assessment,
or proof that the auxiliary improves wins. Local report and stream reader:
`runtime/consistency-ablation-01861967/belief-review.json`, `belief_review.py`.
No model forwards or live-run mutations were performed.

## Addition ledger — 2026-09-08 saved dashboard eval-panel mismatch

Read the live saved `Signal health` workspace (`nw-yqg7vvun701-v`) through
Workspace.from_url, without saving it. Its definitions match the local script:
At a glance's winrate/margin/payoff/count panels request simpleheuristic actor
suffixes `-0`, `-1`, `-2`; live `-0`/`-1` use temp 0.5, while the third live
actor is now `-t1-2` at temp 1.0. The requested `-2` series is stale and is not
its replacement. Section 3c correctly compares `-t1-2` with `-search-3`, both
temp 1.0, and includes the currently disabled historical `-search-d2-4` key.
Thus these sections do not plot equivalent actor populations.

Smoothing also differs: 3c's first panel applies UI time-weighted smoothing
0.98 to raw per-game `ema-wr-*` observations; its second plots the learner's
`smoothed-wr-*` values with no extra UI smoothing. At a glance's smoothed-winrate
panel uses the latter estimator but different actors. Learner smoothing has a
200-game half-life per actor and resets at process startup. At a glance's
UI-smoothed payoff panel uses payoff (-1/0/+1), not binary wins, and UI factor
0.95. Neither section's x-axis differs: both explicitly use lifetime_step.
No dashboard or script was changed during this diagnostic. Correcting labels
and replacing retired actor keys is warranted; published-view updates should
stay scoped to the relevant saved training view.

## Addition ledger — 2026-09-08 current-run dashboard refresh

Updated and published `Signal health` in place, retaining view ID
`VmlldzoxNzg4Mjk2OQ==` and URL
`https://wandb.ai/jtwin/pokemon-rl?nw=yqg7vvun701`. Overview and section 3c
now share one T=1 plain-versus-depth-1-expectimax win-rate panel definition:
learner 200-game-half-life smoothing, no additional UI smoothing, lifetime
steps and a 0–1 scale. T=0.5 controls are explicitly separate. Removed stale
actor `-2` and inactive depth-2 search series, replaced retired policy-head
and own-split forward-KL keys, and replaced GPU batching panels with the
currently emitted CPU inference timing/history metrics. Separated trunk
cosine/participation and parameter/gradient magnitudes onto distinct panels.
Consistency comments now reflect the restored coefficient and masked loss.

Read back the published view: 116 panels, 338 explicit metric references,
all present in current run `irqeetfg` history; the two T=1 comparisons have
identical metric, axis, smoothing and range settings. Black, focused Ruff,
and `git diff --check` passed. Added `--project rl|offline|both` so publishing
can remain scoped; this refresh did not touch the offline or personal view
or the learner. Verification is recorded locally in
`runtime/consistency-ablation-01861967/dashboard-published-verification.json`.

SDK trap: passing an internal `nw-…-v` name directly as the URL's `nw` token
to `Workspace.from_url` reads successfully but wraps the internal name again;
saving then makes the view disappear from the ordinary saved-view listing.
Restored the original name on the same ID, normalised update URLs before SDK
loading, and added saved-URL/ID read-back before stale-view pruning. Final
server listing contains the original Signal health and personal workspace.

## Addition ledger — 2026-09-08 restart health at step 1,888,347

Read completed local W&B records for the current irqeetfg restart, steps
1,861,968–1,888,347 (26,380 updates). Compare first/latest 2,000 updates:
mean global gradient 7.807→7.908, transition gradient 4.135→4.502; zero skipped
updates throughout, full-window max gradient 119.812, latest-window max 24.468.
Consistency loss 1.0624→0.86175; CLS gain -0.2298→-0.01486, private entity
-0.08463→0.10897, move slot -0.00403→0.12923. PREV_ACTION remains absent/unscored
in singles; its zero metric does not validate doubles. Decode CE 0.25964→0.20172,
generator loss 0.98514→1.00371. Pooled logged SSE/energy delta gain rises
0.08064→0.10295 for posterior and 0.04478→0.05494 for prior; switch gain barely
moves 0.05533→0.05561. These replay-training diagnostics are not the historical
frozen multi-sample prior calibration and must not be equated with it.

Belief accuracy 84.343→85.252%, revealed 79.932→80.933%, context margin
4.411→4.319pp. Hidden-code mean perplexity 4.754→4.643 and minimum-code
perplexity mean 2.851→2.584 merit monitoring, not a collapse verdict. Trunk
cosine 0.11034→0.11266, participation 6.8696→6.8809; normalised action entropy
0.51061→0.51337. Value target-fit R² 0.9191→0.9238, but actual outcome R²
0.1587→0.1524 and early-game outcome R² -0.0841→-0.1471 remain weaknesses.

EMA eval since restart: T=.5 pooled 616/990 wins (62.22%); T=1 plain 235/495
(47.47%), depth-one expectimax 240/486 (49.38%). Latest 200-game-half-life
smoothed rates: T=.5 actors 63.77%/60.28%, T=1 plain 50.39%, search 47.32%.
Latest 2k updates contain only 38 plain-T1 and 37 search games. No established
search advantage or causal win-rate benefit from consistency restoration.
Read-only review; no matches launched or learner changes. Local report/script:
`runtime/consistency-ablation-01861967/latest-health.json`, `latest_health.py`.

## MCTS rerun after consistency restoration — 2026-09-08

At user request, gracefully stopped irqeetfg at full checkpoint `01889162`,
27,195 updates after previous pilot `01861967`. Repeated all five 100-game
SimpleHeuristic arms, frozen EMA, GPU, T=1, seed 123, unpaired simulator seeds.
MCTS uses 64 simulations / four chance slots with depth caps 1 and 2; this is
not a simulation-budget sweep. New W/D/L: plain 55/0/45, MCTS1 47/0/53,
MCTS2 42/1/57, expectimax1 45/0/55, expectimax2 51/0/49. Old win counts were
45/48/50/41/50 respectively. MCTS does not improve in this pilot; margins
against plain change +3→-8pp and +5→-13pp. Small independent samples do not
establish a causal regression or benefit from restored consistency.

All 500 games completed, no failures/abandonments, finite saved diagnostics,
zero legal overflow. MCTS root KL 0.02668/0.02910; expectimax 0.001256/0.001820.
Depth-two candidate mass ~95.2%/95.4%; MCTS mean logical expansions 13.57/41.76.
Elapsed incl. compilation: plain 26.9s, MCTS1 97.0s, MCTS2 225.4s, expectimax1
38.8s, expectimax2 126.0s. Historical timing comparison is confounded by the
intervening bit-identical MCTS optimisations and compilation cache state.
Learner remains stopped; isolated service cleaned up. No production changes.
Local report `docs/mcts-ablation-2026-09-08.md`; raw arm outputs/logs under
`runtime/ablation-01889162-*`; comparison includes Wilson intervals and old/new
search diagnostics in `runtime/ablation-01889162-comparison.json`.

## World-model diagnosis at 01889162 — 2026-09-08

After the MCTS rerun, recalibrated latest EMA on the SAME saved 500 transitions
from 24 old-checkpoint self-play games (no new games). Existing
`runtime/priority-audit-01861967/posterior_calibration.py`, latest checkpoint,
top eight latent action codes (99.1246% mean retained mass), 64 prior draws per
code, exact 256-code posterior expectation, 2,000 whole-game bootstraps.
Prior-expectation value-delta gain -0.06393, 95% [-0.10488,-0.01227], compared
with old -0.08184. Posterior expectation +0.04939 [-0.04293,+0.15730], old
+0.04289. Move subset (394) prior -0.07557 [-0.12871,-0.01171]; switches (106)
+0.00012 [-0.09465,+0.06498]. Latest prior still loses to copy; favourable old/new
point movement is not a significant improvement claim, and real-state value
labels change with each checkpoint. This evaluates taken actions on a fixed old
state distribution, not counterfactual action ranking or current-policy rollout
calibration. Raw results/log/pickle: `runtime/posterior-calibration-01889162.*`.

Code confirms two gradient boundaries relevant to interpretation: real trunk
rows and consistency targets stop-gradient; restoring consistency trains the
transition decoder, not a predictability objective into the trunk. Main imagined
losses decode posterior samples with next-state information; prior is matched by
balanced KL, while the explicit prior-mode value path is stop-gradient telemetry.
Thus good posterior reconstruction need not imply useful deployed expectations.
This is an objective mismatch to investigate, not proof a prior-sample realised-
next-state MSE would fix it (stochastic futures need distributional treatment).

MCTS confound remains separate: forced one-visit coverage of every legal root
cell and raw visit-count sampling with 64 simulations puts at least 1/64 mass
on each legal action. Ten legal actions imply at least 9/64 probability away
from any single preferred action, even before other visits. This may weaken a
confident policy independently of model accuracy; no value-blind matched control
was run, so do not attribute the complete gameplay deficit to the world model.

## Research interpretation — hidden-information world models, 2026-09-08

Research review after 01889162 ablation: Stochastic MuZero supports learned
chance/afterstates (Go, 2048, backgammon), but does not establish sound hidden-
information adversarial search. ReBeL and Student of Games reason over public
beliefs and information-set-aware values/strategies using game models. DeepNash
shows strong hidden-information play without search. LAMIR (arXiv 2510.05048,
reviewed v1) is especially relevant to simulator-independent inference: learns
both players' information-state abstractions, joint-action dynamics, legal
masks, rewards/termination and uses CFR+ resolving. Its stated limitation is
absence of explicit chance nodes; not a ready stochastic-Pokémon replacement.

Inference for this project: reconsider the transition/search interface before
width or token count. Current own-action dynamics marginalise opponent choices
and environmental chance together; useful against a fixed opponent distribution,
but not a model that supports separately changing the opponent strategy. Joint
choices, persistent hidden-state uncertainty, public/private observation updates,
and values appropriate to a changing strategy profile are candidate requirements
for an adversarial-search prototype. This is a research direction, not an approved
architecture replacement or proof these omissions caused the 100-game deficits.
Keep policy-gradient baseline; test representation/action-effect calibration and
value-blind search control before promoting search. Existing frozen-state value
calibration measures agreement with the checkpoint critic, not true counterfactual
outcomes. Stop-gradient alone is not a diagnosed bug; LAMIR also decouples parts
of abstraction and dynamics learning.

Sources: https://openreview.net/pdf?id=X6D9bAHhBQ1 ;
https://arxiv.org/abs/2007.13544 ; https://arxiv.org/abs/2112.03178 ;
https://arxiv.org/abs/2206.15378 ; https://arxiv.org/html/2510.05048v1 .
No implementation, coefficients or learner state changed during this review.

## Scoped joint latent/public-belief prototype — 2026-09-08

User requested concrete scope after ruling out exact private-state enumeration.
Wrote local `docs/public-belief-world-model-scope-2026-09-08.md` against current
code and plan template. First deliverable is aligned joint-decision self-play data
plus a capacity-matched joint-action prediction experiment with frozen policy.
Later gates cover learned public/private abstraction, filtering, information-set
solver/value contracts, then powered play. Existing per-mon belief codes are not
joint public ranges; policy-readable rows are not strictly public. Explicitly
record submitted actions (not merely executed events), asynchronous requests,
visibility and doubles previous-choice sequencing. Paired actor choices are a
new learner-only self-supervised target, replacing the former no-opponent-label
convention only inside the proposed prototype. No exact team enumeration, simulator
at search time, policy rewrite, coefficient change or training restart authorised
by this scoping document. Learner remains stopped at 01889162.

## Scope revision accepted — unpaired interval model, 2026-09-08

User preferred avoiding joint training records because own requests need not align
with opponent submissions. Revised the existing public-belief scope document in
place: first audit current unilateral intervals, then compare inferred opponent-
behaviour plus residual-chance latents against a capacity-matched combined-chance
model. No paired action dataset, wire change or cross-player join prerequisite.
Intervals can span zero/one/multiple opponent decisions; absent executed events
are not absent submitted actions. Behaviour/chance disentanglement is unidentifiable
without sufficient constraints/evidence: a predictive latent is not automatically
an opponent-controlled action. Added explicit strategic interpretation gate before
public-belief solving, including simultaneous-choice secrecy and feasible interval
semantics. Paired collection is deferred only; if strategic semantics cannot be
established from unpaired data, retain predictive planning rather than claiming an
adversarial solver. This supersedes the preceding scope's A+B data requirement.
No production changes or restart; learner remains stopped at 01889162.

## Unilateral interval Stage A audit — 2026-09-08

Added NumPy-only `rl/offline/interval_data.py`: adjacent own source/action/successor
extraction preserves terminal successors, excludes padding and bootstrap-only
sources, retains same-request previous-choice microsteps, and reports original
history-window indices. Stable game splits require a caller-supplied identity
shared by both perspectives; the trajectory schema itself has no game ID or
training/evaluation provenance. No cross-player join or private-truth input.

Historical `runtime/mcts-readiness-01861967-games.pkl`: 48 sides / 48 chunks,
1,449 real intervals including 48 terminal successors, 1,234 move and 215 forced-
switch requests, 3,798 observed history steps, two zero-new-history intervals,
zero missing retained prefixes. All actions legal and per-side interval identities
unique. No preview, doubles, previous-choice or same-request coverage. Keep this
calibration corpus held out; new standalone research-training collection requires
an explicit checkpoint/format/purpose/game-grouping manifest. Adjacent opposite
player indices are compatible with harness grouping, not sufficient provenance.

Service `getChoices()` only increments requestCount after collecting per-slot
choices: doubles/preview microsteps can share a counter. HISTORY_STEP_COUNT is
retained window length, not an absolute event count. Missing visible events are
not absent opponent submissions, and post-trunk history deltas are not pure
opponent behaviour. Ten focused extraction/visibility/split tests pass; Black and
Ruff pass. No production changes, learner restart or real-model forward in this
audit. Fixture coverage does not repair known doubles service alignment defects.
Details are in local gitignored `docs/interval-data-audit-2026-09-08.md`.

## Interval-model prototype and hold launched — 2026-09-08

Implemented accepted A+B prototype in `rl/model/interval_transition.py`,
`rl/offline/interval_data.py`, `interval_features.py`, `train_interval.py` and
focused tests. No production transition/search/actor/replay protocol change.
Historical calibration archives remain held out. Fresh explicitly research-training
self-play at frozen EMA 01889162: 128 games, 256 sides, 7,413 intervals; 101 games /
5,750 intervals train and 27 games /1,663 intervals held out by stable game hash.
Collection provenance and features: `runtime/interval-01889162/`.

Concrete first factorisation: two conditional categorical codes using the existing
2x16 decoder alphabet. Combined posterior's first code sees full successor-row
movement; history posterior's first code sees only HISTORY_ENTITY/HISTORY_FIELD
movement. Residual posterior sees full movement and the first code. Prior samples
ancestrally; the conditional residual prior in KL has stop-gradient on the posterior
first-code input. Both conditional arms have exactly 6,975,808 parameters and
identical initial values; trained legacy control has 4,628,384. Warm start strictly
copies compatible decoder leaves; new networks receive identical seeded init.
This is a testable predictive restriction, not identified opponent-action semantics:
post-trunk history carries contextual changes and can include chance/own effects.

All arms freeze policy, own-action encoder and critic. New isolated Adam at existing
3e-5 LR; objective is one-step validity-aware consistency (coef1), frozen-next-critic
categorical distillation (coef1), existing KL dyn .5 / rep0 / free .0625. This differs
from production multi-step/grounding/generator/outcome training and is documented
in each manifest. It evaluates critic agreement, not counterfactual ground truth.
No representations or production coefficients are retuned in this experiment.

200-update preflight with eight prior samples: prior delta gain legacy
-0.06455→-0.05216, combined -0.07605→-0.03547, history -0.07605→-0.02739.
Final 95% whole-game intervals respectively [-.0991,-.0090], [-.0877,.0054],
[-.0705,.0090]. Final gradient norms 4.77/1.37/1.29; finite updates/evaluations,
no established improvement over matched control. Logs, isolated optimizer/parameter
checkpoints and raw held-out reads: `runtime/interval-preflight-01889162/`.
18 focused tests passed (intervals, same-request/previous choices, terminal/bootstrap,
posterior visibility with positive controls, live prior independence, KL gradient
boundary, exact matched initialisation, strict warm start, masked loss, bootstrap).
Black/focused Ruff/diff checks passed. Existing doubles service defect and missing
preview/doubles/other-format corpus coverage remain explicit limitations.

Started standalone `interval-model` tmux session: 25,000 updates per arm, legacy /
combined / history sequentially, 32 prior samples, held-out evaluations every 5k
including 15k/20k/25k. Run directory `runtime/interval-hold-01889162/`, progress
`runtime/interval-hold.log`. Results pending; no promotion or semantic claim.
Production learner remains STOPPED at 01889162. Task collection service cleaned up.
No commit made. Exact commands and objective differences in rl/offline/README.md.

## Interval hold completed — gate failed, 2026-09-08

All three isolated arms completed 25k updates. Held-out prior delta gains
(start→25k): legacy -.06323→-.34507, combined -.07473→-.29279,
history -.07473→-.58858. Final 95% whole-game bootstrap intervals respectively
[-.57781,-.16364], [-.47573,-.14073], [-.95633,-.31948]. Every arm remained
below copy at 15k/20k/25k; none passed the pre-registered gate. Final history-minus-
combined gain -.29579, paired 10k whole-game bootstrap [-.51785,-.09521] over
27 held-out games, with identical interval IDs and copy-energy targets verified.
This is conditional on this training seed/corpus, not multi-seed architecture proof.

Training consistency decreases to ~.49–.50 while held-out value-change prediction
worsens, including posterior reads (~-.16 to -.17). Final training KL .0026 legacy,
.0125 combined, .0068 history is below the .0625 free-nats threshold; this alone
does not prove code collapse. All reported metrics finite, every update checked
for finite loss/gradient norm. No evidence of gradient explosion. Shared deterioration
also implicates limited-data generalisation/objective alignment; do not attribute
all degradation to the split. 25k*32/5750 = ~139 sampled passes over train intervals.

Verdict: do not promote the history restriction, add public-belief solver stages,
or change production based on this experiment. Next diagnostic should compare
train/held-out prediction under the same evaluator and inspect latent usage before
retuning or collecting more data. Results/checkpoints persist under
runtime/interval-hold-01889162; paired summary comparison.json. tmux interval-model
has exited normally; production learner remains stopped at 01889162.

## Interval hold diagnosis — generalisation and coarse codes, 2026-09-08

Saved-checkpoint evaluation on BOTH train and held-out games with the same frozen
critic and 32-draw prior/posterior expectations completed. At 25k, train/held-out
prior gain: legacy +.98830/-.34506, combined +.98939/-.29279, history
+.98906/-.58858. Posterior expectation train/held-out: +.98897/-.16352,
+.98983/-.16268, +.98974/-.18010. Legacy already +.94355/-.33526 at 5k.
Thus the offline objective fits training value targets extremely well but fails
to generalise; this is not evidence that the objective cannot fit value targets.
Held-out reconstruction still improves (legacy .85903→.70931), while value KL
worsens (.06846→.09522; train .06712→.00127). Prior/posterior KL is .00525
train versus 1.398 held out in final legacy. No claim about production online
training follows automatically from this frozen-feature, 101-game experiment.

Legacy posterior BEFORE offline training selects only modal pairs (3,14) and
(15,5), exactly separating successor forced-switch/terminal (1,392 intervals)
from nonterminal normal-move requests (6,021). At 25k legacy retains two pairs
and 100% held-out phase purity; combined/history purity 99.88%/98.56%. Group
marginal perplexity ~1.7, conditional ~1.08–1.12. Two groups mostly repeat the
same coarse phase distinction; soft probabilities may still contain finer detail.
Code interventions change decoded rows and values, so the route is live rather
than entirely ignored. Uniform interventions include improbable combinations;
nonzero usage/intervention response does not identify opponent-behaviour semantics.

Next controlled experiment: fix legacy architecture/objective and compare fixed-
corpus reuse with fresh training data at matched update counts; track unique games,
reuse and early train/held-out value gains. Keep a new untouched final split since
existing held-out games now inform experiment selection. Data volume/reuse is a
hypothesis to test, not a demonstrated causal fix. Do not promote history-only
factorisation or prescribe code-entropy forcing from these results. No new games,
training, or production change in this diagnostic; learner remains stopped.

Artefacts: runtime/interval-diagnosis-01889162/{diagnose.py,report.json}, per-row
reads/interventions and trajectory-derived successor labels. Prior held-out gain
reproduction within 1.4e-5, single-posterior squared errors exact; minor prior
rounding changes with fused diagnostics were measured. Diagnostic exited normally.
Full local report: docs/interval-diagnosis-2026-09-08.md (gitignored).

## Interval data/reuse control registered — 2026-09-08

User authorised the recommended follow-up. Collecting 512 fresh self-play games
at frozen EMA 01889162, T1, seeds 918–921 (four 128-game collections). Original
101-game training pool is the fixed control; fresh arm adds new games except the
stable-hash 20% reserved as a new final test. Both arms share old 27-game validation
and new final test. Legacy architecture, one-step offline objectives, Adam 3e-5,
clip10, batch32, seed42 and 25k update budget stay fixed. This tests increased
corpus size/reduced reuse together, not streaming or opponent-policy diversity.

Registered reads 0/200/1k/5k/15k/20k/25k with train diagnostics, unique sampled
games/intervals and sampled passes. Final test only at fixed 25k; no checkpoint
selection. Primary relative gate: paired whole-game 95% final-test interval for
fresh-minus-fixed prior gain excludes zero positively; usefulness also requires
fresh prior gain above copy with 95% interval excluding zero. No automatic
production promotion. Local plan docs/interval-reuse-control-2026-09-08.md;
commands, provenance and outputs runtime/interval-reuse-01889162/.

Extended shared offline evaluator with optional training diagnostics, early
checkpoints and explicit final-test/training-eligibility masks. Whole-game split
validation rejects row leakage and shared-game leakage. Twelve focused NumPy
interval tests pass; focused Black/Ruff and diff checks pass. No production
training restarted. Results pending.

## Interval data/reuse control completed — 2026-09-08

Collected all 512 fresh self-play games without failures/abandonment. New final
test: 100 games /5,856 intervals. Larger training pool: original101 + new412 =
513 games /30,420 intervals, compared with fixed101 /5,750. Both original legacy
models completed 25k updates with matched architecture/loss/optimiser/seed/batch;
all finite. Fixed final validation reproduced historical -.345067 exactly.

At 5k, deterministic train-subset/validation gains: fixed +.94609/-.33527;
larger +.32869/-.04477. At 25k: fixed +.98758/-.34507; larger +.88890/-.25648.
Reuse 139.13 versus 26.30 sampled passes. Larger pool reduces the development
train/validation gap, but both deteriorate after early improvements. Best scheduled
validation is around1k: fixed -.02489, larger -.01452, still below copy.

Preselected 25k NEW final test, 10k whole-game bootstrap: fixed prior gain
-.45713,95%[-.62146,-.32033]; larger -.37195,95%[-.51606,-.24735]. Paired
larger-minus-fixed +.08518,95%[-.02546,+.20132], identical game/interval IDs and
copy energies asserted. Relative improvement gate INCONCLUSIVE; above-copy gate
FAILED. Do not call reduced reuse a proven fix, or infer data is irrelevant.
This single-seed fivefold-corpus experiment still repeats data26times and uses
one frozen policy. No automatic promotion, solver expansion, or further collection.
Early stopping/reuse-limited training is a possible next controlled test, requiring
a registered selection rule and new untouched test games; do not reuse this final
test to claim selected-checkpoint performance. Offline findings do not directly
prescribe production replay reuse because objectives/representation training differ.

Artefacts: runtime/interval-reuse-01889162/{comparison.json,composition.json},
{fixed,fresh}/legacy checkpoints and reads, exact partitions, collect.py/run.py.
Full local result docs/interval-reuse-control-2026-09-08.md. Task service cleaned
up, experiment exited normally, production learner still stopped at01889162.
Twelve focused interval/partition tests and focused Black/Ruff/diff checks passed.
No commit made.
Final state verification: all control parameters, optimiser state leaves and step
are bit-identical to the original hold; diagnostic additions preserved updates.

## Direct interval-value probe registered — 2026-09-08

User authorised the next bounded diagnostic. Direct current-state + latent-action
predictor versus identical state-only control, with the existing legacy transition
as reference. Frozen EMA01889162 actor/action encoder/critic and the same513-game
training pool. Existing27 + previously examined100 held-out games are development
validation only (127 games); no evaluation games become training. Stop at5k,
read0/200/500/1k/2k/5k, select each arm by minimum development SSE, earliest tie.

Direct probe copies width16 RowRead and action embedding; same hidden widths as
prior, predicting centred categorical-logit residual from the frozen current critic
log probabilities. Zero final kernel starts at copy; train with next-critic CE only.
State-only zeroes action embedding input, retaining identical seeded parameters.
Actor/action encoder/critic frozen; exact64-code own-action marginal at evaluation.
Fewer parameters/no consistency or chance KL make this a path/objective diagnostic,
not an architecture-only ablation. No simulator labels or opponent joins.

Collect one new128-game final set only if selected action development gain and
its advantage over selected state-only both exceed1e-6 (exclude numerical copy
roundoff). Freeze selections first; final success needs above-copy and paired
above-state-only whole-game95% intervals. No post-test checkpoint selection.
Plan docs/direct-interval-probe-2026-09-08.md; runtime/direct-interval-01889162/.
Fourteen focused tests pass: partition boundaries, copy-init/live output gradient,
identical controls, positive action-dependence check and validation-only selection.
Focused Black/Ruff/diff pass. Production remains stopped; results pending.

## Direct interval-value probe completed — 2026-09-08

All three arms completed5k finite updates on513 training games, with127 previously
examined games for development. Both886,563-parameter direct controls select
step0/copy; all nonzero scheduled checkpoints are below copy. At5k train-subset /
development gain: action+.24827/-.08121, state-only+.24280/-.08264. Legacy selects
1k with development gain+.002274,95%[-.014657,+.016714], then falls to-.05550 at5k.
Selected-development intervals do not correct checkpoint selection and are not
untouched-test evidence. Final-test collection gate FAILED; no new games collected.

Conclusion: simplifying to direct next-critic distribution prediction did not fix
generalisation; decoder-only blame is not established. Nor does this bounded small
probe prove the trunk uninformative. Frozen action encoding, RowRead/input access
and critic targets remain shared possible bottlenecks. No production replacement,
extra hold, retuning or after-the-fact checkpoint selection. This is observational
critic agreement, not counterfactual action advantage or outcome prediction.

Full direct controls have bit-identical initial params/optimiser state. Legacy5k
params/optimiser/step are bit-identical to prior larger-pool run. Direct output
gradient paths live; final norm .622/.604, max centred residual1.450/1.417.
Action encoder not universally collapsed:9 modal codes, marginal perplexity
4.70train/4.41development. This does not verify within-state legal-action separation.
Across36,657 multiple-legal-action intervals, recorded policy entropy mean.91069
nats; only1.90% below.1, and1.17% taken probability>.99. Pervasive deterministic
play is not supported as the explanation for this corpus.

Artefacts: runtime/direct-interval-01889162/{selection.json,composition.json},
{action,state,transition/legacy}/ reads/checkpoints, behaviour-variation.json,
action-code-usage.json, code-sha256.json. Local report
 docs/direct-interval-probe-2026-09-08.md. Fourteen focused tests and focused
Black/Ruff/diff pass. Experiment exited normally; production remains stopped.

## Latest action-code audit — 2026-09-08

Read-only EMA01889162 audit,2,560 saved multi-legal-action roots across640 games,
16,834 legal actions, no truncation. Exact uniform-reference action reconstruction
87.213% vs chance18.295%, MI1.5872nats. Existing decode objective is implemented
and configured coefficient1.0; do not propose adding/re-enabling it as missing.

Strong pair alias (TV<.1; optimal binary decode<55%): ordinary versus Tera version
of SAME move35.62% of817 pairs; ordinary/ordinary7.34% of12,009; switch/switch6.42%
of11,354; ordinary/switch.31% of24,162. Same-move ordinary/Tera modal match39.17%.
None of the strongly aliased pairs have identical source/target input rows. For
ordinary/Tera aliases input RMS difference median.7842, relative RMS.4943. Specific
action distinctions are lost despite distinct encoder inputs; not literal trunk-row
identity. f32 rerun gives87.209% accuracy and35.86% same-move aliases, robust aggregate.

Downstream128-root/128-game audit, all64 action codes exact, two32-draw chance groups
with common randomness. Well-separated q pairs (TV>.9,n2,266): median embedding RMS
.17656, fixed-chance decoded-row RMS.15165, prior expected-value gap.00200, predicted
value-distribution TV.00190. Replicate action-difference correlation.9552/sign agreement
94.53%. Per-root legal-action value spread median.00531/mean.00756. Decoder responds;
value predictions remain tightly clustered. Strong aliases(n125) have median value
gap8.8e-6 and distribution TV1.9e-5. No true unplayed-action outcomes were measured.

Numerical caveats: cached taken-code guard initially failed (max difference.2739),
median2.4e-7,p99 .03863,4/2,560 modal changes. Metadata/order verified; bf16/f32
aggregates stable, but full fused-encoder rounding source not isolated. Do not claim
bit identity. Initial Gram row-distance formula lost small differences; explicit
pair subtraction recomputation in stable-row-rms.npz is authoritative. Final summary
uses stable distances. NumPy confusion agrees with executable exact_decode_loss;
finite outputs and mixture-TV contraction checks pass. No production tests needed
for runtime-only audit; no games/training/production changes.

Next supported diagnostic: root-only direct source/target action rows versus the
categorical bottleneck, with other inputs/targets fixed. This does not establish
that bypassing fixes value prediction or justify eliminating latent imagined actions,
code-entropy forcing or a full architecture reset. Scope remains singles gen9random.
Artefacts runtime/action-audit-01889162/, local report
 docs/action-code-audit-2026-09-08.md. Production remains stopped at01889162.

## Observed-root action bypass registered — 2026-09-08

User authorised a bounded root-only bypass test after action-code collisions were
measured. Direct-value probe now accepts recorded concrete cells, gathers existing
source/target rows through chosen_bank_rows and projects them with a trainable copy
of action_encoder/query_proj. This replaces categorical sampling/action-table input;
state RowRead, shared MLP width/init, frozen critic targets, centred residual and
CE loss remain fixed. This is an input-path comparison, not equal parameter count
or equal feature-scale attribution. No imagined-node or production model change.

Same513 train/127 development games, seed42, Adam3e-5, clip10, batch32,5k ceiling,
reads0/200/500/1k/2k/5k, earliest maximum validation gain selection. Recover all own
recorded cells in exact export order, checking game/request/terminal and legality.
Rerun categorical control and require bit-identical5k parameters/optimiser state
against previous direct probe. State-only reference is unchanged and reused.
Final128-game collection only if selected bypass exceeds copy and both references
by>1e-6; selected checkpoints frozen first. No extended hold or coefficient sweep.

Fifteen focused tests pass, including shared initialisation, exact copy, live output
and action-projection gradients, and action-specific row sensitivity. Focused Ruff
passes; formatting applied. Plan docs/root-action-bypass-2026-09-08.md, runtime
runtime/root-action-bypass-01889162/. Production remains stopped; results pending.

## Observed-root action bypass completed — 2026-09-08

Both direct-value arms completed5k finite updates on the same513 training/127
development games. All37,939 concrete cells aligned with saved game/request/terminal
records and passed legality checks. Shared RowRead/prediction-MLP initial parameters
bit-identical;886,563 categorical versus1,001,507 bypass parameters. Categorical
final params/optimiser/step bit-identical to prior direct probe.

Bypass development gain at200/500/1k/2k/5k: -.01912/-.01838/-.02729/-.04840/-.07064.
Categorical -.02051/-.02344/-.03315/-.06124/-.08121. BOTH select step0/copy under
registered selection; state-only reference also copy. Final-test gate FAILED;
no new games or after-the-fact extra training. At5k train-subset gains+.24827
categorical/+.29496 bypass. Paired development bypass-minus-categorical+.010573,
95%[-.007006,+.028677] over127 games (10k whole-game bootstrap), inconclusive.
Bypass absolute gain-.070642,95%[-.112796,-.036794], clearly below copy.

Conclusion: bypassing categorical action input has not demonstrated useful
prediction; do not change production conditioning on this evidence. Action-code
collisions are real but not established as a sufficient explanation. Shared frozen
state inputs, critic targets, objective/metric alignment and generalisation remain
unresolved. Bypass still uses a learned finite-width query projection; different
parameter count/input scale preclude categorical-alias-only attribution.

Fifteen focused tests and focused Black/Ruff/diff checks passed. Final bypass grad
norm.5603, max centred residual1.7096; finite throughout. Runtime
runtime/root-action-bypass-01889162/{selection.json,comparison.json,composition.json}
plus reads/checkpoints/scripts. Local report docs/root-action-bypass-2026-09-08.md.
Experiment exited normally, production remains stopped at01889162, no new service.

## Interpretation reset after bypass controls — 2026-09-08

The failed offline variants share frozen features, one behaviour-policy corpus,
next-frozen-critic targets and mostly distribution CE. They are not independent
falsifications of all world-model architectures. Train/held-out overfitting is
measured; a specific shared root cause is not. Treat next-critic copy-relative gain
as a diagnostic, not the sole acceptance test for a complete planning model.
A Bellman-consistent value already predicts policy-averaged future return; under
correct payoff/terminal conventions the policy-averaged TD residual has zero mean.
An action-conditioned gain must exploit predictable action advantages/critic errors
against stochastic residual variance. This does not prove the achievable gain is
zero here or validate the current world model.

Checked a possible terminal-accounting explanation using existing bypass5k reads:
254 of7,519 development intervals end terminally and contribute24.77% of copy
energy. Categorical terminal/nonterminal gains+.00706/-.11028; bypass+.04616/
-.10909. Failure remains on nonterminal transitions. Production value targets
include real terminal rows/payoffs; search's deeper backup separately uses predicted
continuation and conditional terminal outcome. Do not claim a discovered terminal
bug or that the offline critic-only objectives reproduce the full training/search
backup. Next priority is shared-target/calibration/coverage audit against recorded
returns and production conventions before further architectural variants. This
entry is interpretation and read-only analysis, not authorisation to restart training.

## Shared target/loss audit completed — 2026-09-08

Read-only saved direct checkpoints at0/200/1k/5k, all127 development games/7,519
intervals plus fixed1,663 training read; all640 games/37,939 intervals aligned with
recorded final outcomes and opposite-perspective checks. Prediction reproduction
within3e-7, finite. No new games/training/restart.

At5k teacher-distribution KL improves from copy.075158 to categorical.070812,
bypass.070417,state.070956; all CE improvements positive under10k whole-game
bootstrap. Yet scalar successor MSE worsens8.12%/7.06%/8.26%. Bypass outcome-MSE
improvement+.203%,95%[-.777%,+1.183%], inconclusive; other arms likewise inconclusive
at5k. At1k all three significantly worsen outcome MSE about.86–.87%. Sampled-code
versus mixture CE gap only~3e-6, and state-only reproduces mismatch.

Production uses two-hot scalar V-trace/TD(lambda) return targets, including shifted
win_returns for imagined values; offline probes distil frozen successor critic
probabilities. These are different targets/training paths. Categorical outputs are
not automatically calibrated outcome probabilities. Finite-sample CE improvement
need not improve scalar mean SSE; do not diagnose all failed probes as inability
to learn or as independent falsifications of the architecture. Previous measured
overfitting and action collisions remain; this audit does not establish useful search.

On-policy whole-game deployable lambda=.8 diagnostic is not exact privileged,
importance-corrected, chunked production target. Bypass5k gain+.194%,95%[-2.512%,
+2.822%], no established benefit. Next proposed bounded control separates scalar
loss alignment (same successor teacher) from return-target choice, retaining
state-only/action-conditioned controls before any architecture change. Not launched.
Local report docs/shared-target-audit-2026-09-08.md; reproducible reads/scripts in
runtime/target-audit-01889162/, including report.json and uncertainty.json.

## Base-policy sample accounting audit — 2026-09-08

User prioritised base-policy learning efficiency relative to Jaxcalibur's reported
almost 100M self-play games. Read-only W&B and current-code audit: irqeetfg is
finished; checkpoint 01889162 records 1,889,162 updates and 385,205,336 frames.
Current recorded batch size is 4, reuse cap 8. A 1,000-point sampled history has
median realised replay ratio 8 and median chunk length 29.75; it is not a census.
Assuming batch size 4 throughout, 7,556,648 chunk draws / eight uses gives about
944,581 fresh chunks. This is NOT a unique-battle count: long games make multiple
chunks; mirror self-play contributes both sides, historical opponents only main.
Do not present the estimate as measured games or infer comparable ladder strength.
Architecture changes and partial parameter migrations also prevent treating the
lifetime update count as one fixed-architecture training experiment.

New accounting finding: train_step increments frame_count by (~done).sum(),
whereas actor terminal padding has done=False. Lattice trimming retains some such
padding. A 48-row chunk with 29 decisions and terminal at index 29 counts 47 frames,
not 29. Replay repeats these counts, and nonterminal bootstrap rows count too.
The actual policy/value objectives use separate cumsum-done/chunk masks; this is
not evidence those objectives train on padding. frame_count also paces league
snapshots, so correcting it requires considering pacing and checkpoint continuity,
not silently relabelling historical totals. No counter or scheduling change made.
Saved accounting: runtime/base-policy-audit-2026-09-08/accounting.json.

Architecture priorities remain hypotheses: the 09-08 held-game type-accessibility
read supports testing a small input-row readout connection separately from an
observed self-play action-effect auxiliary. The 09-02 private/public join finding
is historical and needs current remeasurement; its gather was previously declined.
Current transition inputs and base-policy targets stop-gradient; its shared value
head can train, but transition losses do not directly shape the policy trunk.
No learner restart, new games, model forward, architecture edit or experiment launch.

## Input-to-readout connection implemented — 2026-09-08

User authorised the connection after the base-policy audit. Encoder now returns
only the three assembled policy banks beside its unchanged contextual sequence.
FlatActionReadout.connect_inputs adds input_scale[bank] * original_row to each
contextual bank: private sheet, move, target. The diagonal scale is f32, cast to
the activation dtype, zero-initialised, 3 * 256 = 768 parameters under action_head.
This preserves the existing head's feature coordinates without a new projection
or loss. At a restored live head, all three banks receive scale gradients
immediately; with a fresh all-zero head the readout must first leave zero, as
with its existing bilinear key path. The input connection is policy-only:
critics and transition dynamics still receive the original trunk output. Search
root priors/diagnostics use the same connected policy rows as the plain actor.

Config action_head.input_connection=False disables the route and its gradients.
Old league parameter trees without input_scale retain the original policy;
ordinary fresh-init/by-path restoration seeds the new leaf at zero. No checkpoint
rewritten and no training restart. This is a sample-efficiency hypothesis based
on the recorded held-game attribute-accessibility gap, not an established playing
strength gain. No action-effect auxiliary, entity-join restoration or loss change.
Implementation: rl/model/{heads,encoder,player_model,config}.py; focused contracts
in tests/test_readout_input.py. The end-to-end privileged-partition test now opens
the readout and input scale so its policy equality cannot rely on the zero gate.

Validation: focused readout/dtype checks and GPU actor/learner equivalence and
end-to-end privileged-partition checks passed, together with resume-merge tests.
Zero-scale and legacy-tree tests preserve live logits exactly; all three banks
have live scale gradients, and opened scales send gradients into input features.
Focused Black/isort/Ruff and whitespace checks passed. All eight fast search
tests passed. The real-model search integration initially exhausted GPU memory
requesting another 4.14 GiB at its default rollout budget. The test now explicitly
uses two root and two inner chance samples, retaining depth-one/depth-two checks;
that bounded GPU rerun passed. Production search budgets are unchanged, and this
does not establish that the full production depth-two budget fits the GPU.
The final focused readout rerun also passed, including the disabled-gradient
control. No training-performance claim is made.

## Input-to-readout connection removed — 2026-09-08

User requested removal after reviewing the existing transformer residual path.
The trunk already returns input plus accumulated residual updates, without a
final normalisation. The added diagonal connection only reweighted that input
contribution for the policy; it did not supply an independently readable feature
bank or repair a missing identity path. Its sample-efficiency benefit was never
measured: no training run used the 768 new parameters. Removed the connection,
configuration, encoder/head plumbing and dedicated tests; restored the previous
privileged-partition test. Preserved pre-existing edits, including transition
consistency validity, and the independently justified search-test memory bound.
Local removal handle: runtime/readout-input-removal-2026-09-08/ contains the
pre-removal tracked diff and dedicated test (the diff also includes the preserved
transition_cons_valid edit, so do not apply it wholesale). No checkpoint or
training-process change. This is a user-directed cancellation of an untested
architecture hypothesis, not evidence that all input readout connections fail.
Post-removal validation: flat-readout and fast actor-layout tests passed; dtype
checks passed after completing the removal. Ruff and whitespace checks passed;
no connection symbols remain in model code or tests. No expensive GPU rerun was
needed for the restored architecture.

## Fresh replay stream and decision accounting — 2026-09-08

User requested accurate decision accounting and accepted the explicit fresh-stream
idea. Added `player_replay_fresh_fraction=0.125`: scheduled minimum first-use
chunk slots (one every two batches of four), oldest unseen first, uniform distinct
seen samples for the remainder, extra fresh permitted at startup/replay shortage.
Every first use/prefetch counts towards the existing global cap. To avoid a full-
buffer fresh-data deadlock, seen chunks can be evicted before exhausting the cap;
exhausted/retired entries are preferred, then oldest seen. Unseen entries remain
protected. Zero fraction restores the original uniform capped training sampler
and replacement policy. Read-only `sample(increment=False)` now also leaves draw
counters unchanged. This is a new intake/sampling distribution, not an exact
LASER reproduction or a proven strength improvement; fresh does not mean on-policy.

Reference: Schmitt et al., ICML 2020, https://proceedings.mlr.press/v119/schmitt20a.html
used seven replay samples per online sample and a distinct online stream. Here the
quota is in chunks and actor lag remains. Fast producers can drive realised reuse
below eight. Existing own-trajectory/eval isolation and feedback identity/cap
contracts remain in place; no cross-population intake or priority correction added.

Admission counts exclude terminal/padded/bootstrap-only rows; forced actions count.
The learner emits its existing acted-mask count. Host counters distinguish admitted,
sampled/prefetched, processed and finite-applied decision appearances and updates,
with applied/admitted reuse and updates-per-fresh-decision ratios. Chunk and decision
first-use fractions are separate. All cumulative counters explicitly say session,
with a lifetime-step origin; old checkpoints start new accounting, not invented
historical totals. Legacy frame_count and league/checkpoint pacing are untouched.
No dashboard publication or training restart.

Definitions (moved here from rl/online/README.md when that file was deleted in
the 2026-09-09 tidy). A decision is a real acted row, forced single-option
actions included; terminal rows, terminal padding and a nonterminal
bootstrap-only final row do not count, so overlapping chunks count each
admitted decision once (`rl/environment/utils.acted_rows`). The two player
perspectives are separate decisions, not games or simulator turns.

| metric | meaning |
|---|---|
| `player_batch_decisions` | acted rows in this learner batch (the learner mask) |
| `player_decisions_admitted_session` | fresh decisions inserted into replay |
| `player_decisions_sampled_session` | decision appearances drawn from replay, prefetch included |
| `player_decisions_processed_session` | decision appearances in completed learner calls, skipped updates included |
| `player_decisions_applied_session` | decision appearances in finite applied updates |
| `player_updates_processed_session` / `_applied_session` | completed calls / applied updates |
| `player_decision_reuse_session` | applied decision appearances / admitted fresh decisions |
| `player_updates_per_fresh_decision_session` | applied updates / admitted fresh decisions |
| `player_replay_fresh_chunk_fraction_session` | first-use chunks / sampled chunks |
| `player_replay_fresh_decision_fraction_session` | first-use decision appearances / sampled decision appearances |
| `player_replay_evicted_mean_reuses` | mean consumed uses of evicted chunks |

Exact counters for the current store session from
`player_accounting_start_lifetime_step`; a new process, a restore or a store
clear starts a new session (no lifetime totals are inferred). The log worker
records completed calls in order, while admission and sampling snapshots run
ahead of the logged call; read ratios after warm-up and derive windowed rates
from counter differences within one session. Fresh chunks are taken in
admission order, replay chunks uniformly without replacement within a batch;
"first use" means never sampled, not on-policy (actor and queue lag remain).
The fraction is measured in chunks, so decision fractions differ with chunk
length, and it is a preference, not a floor: unavailable fresh slots use
replay, which is what lets a full seen buffer exhaust its cap and admit
arrivals without early eviction or deadlock.

Validation: 64 focused fast replay/chunk tests passed, including positive controls
for forced actions versus policy masks, the 29-decisions/48-rows padding example,
chunk overlap, cap1/2/8, fractional fresh quotas, full-buffer progress, off-mode
seeded sampling, delayed feedback after early replacement, skipped updates and
prefetch accounting. Focused Black/isort/Ruff and whitespace checks passed. No
real-model forward or strength experiment; prior/WM generalisation is unmeasured.
Revert surfaces: rl/online/{buffer,config,decisions}.py, learner construction,
train-step scalar/log-worker accounting, tests/test_fresh_replay.py and the added
mask-equivalence test in tests/test_chunking.py. No commit or launch.

## Fresh-stream main-run restart — 2026-09-09

User authorised rebooting the run after implementation. Confirmed no active
learner/GPU workload, then launched `bash start.sh --load-mode checkpoint
--init-ckpt /home/joseph/Documents/porygon2/ckpts/gen9/ckpt_01889162`.
Main learner PID 469229, tmux `train`; log
`runtime/learner_20260909_034708.log`. Full checkpoint step 1,889,162 and
16 historical league entries restored, W&B irqeetfg resumed. Startup config
confirms fresh fraction .125 and reuse cap/ceiling 8. Both additional lattice
shapes compiled; actual updates advanced. The script's final tmux attach failed
because the tool terminal lacked clear support; the session and learner remained
alive, verified independently. No relaunch was needed.

W&B verification at lifetime step 1,889,204: accounting origin 1,889,162,
admitted decisions 10,099; sampled/prefetched appearances 5,458; processed and
applied appearances both 5,130; applied updates 42; latest skipped-update flag 0.
First-use sampled chunk/decision fractions .4833/.4857 are startup observations,
not the configured minimum or steady-state ratios. No learner/log-worker errors
in the bounded startup review. Training remains running. These verify startup
and counter wiring, not strength or world-model generalisation improvement.

## Fresh-stream first 18.1k-update read — 2026-09-09

Read-only review of completed local irqeetfg W&B records: 18,149 updates from
1,889,163 through 1,907,311, zero skips. Latest 2k window: realised chunk reuse
2.334, decision reuse from counter differences 2.325, fresh decisions117.89/s,
applied decision appearances274.05/s, updates2.494/s. Previous session's actual
last2k (1,887,161–1,889,160) measured4.373updates/s. Cap remains8, but first-use
share across this restart is~42.9%; early eviction/intake changed realised reuse
substantially beyond merely reserving the12.5% minimum.

EMA SimpleHeuristic wins since restart: plainT.5 447/682 (65.54%), plainT1
187/341 (54.84%), searchT1 186/340 (54.71%). Prior saved session review through
1,888,347:616/990 (62.22%),235/495 (47.47%),240/486 (49.38%). Latest200 games
per current actor: plainT.5 pooled260/400, plainT1 116/200, search117/200.
Encouraging plain-policy point estimates, not a matched causal reuse experiment;
checkpoints move and continuation/changed data distributions confound attribution.
No established search advantage or strength-per-hour improvement.

Prior saved last2k versus current last2k: actual-outcome R2 .1524→.1855,
early outcome-.1471→-.0881; replay value-target fit .9238→.7392; normalised
entropy .5134→.3331; learner-actor KL .00677→.01821; ISR ESS .99491→.95156.
Replay posterior pooled delta gain .10295→.15377, prior-MODE gain
.05494→-.05937. Consistency loss .86175→.76546. These are changing-distribution
training diagnostics, not frozen held-out prior expectations; no WM improvement
claim. Fresh/replayed value errors currently .07971/.07862, small aggregate gap.

Reproducible current reader/results: runtime/replay_stream_review_20260909.py,
runtime/replay-stream-review-20260909.json; previous baseline remains
runtime/consistency-ablation-01861967/latest-health.json. No forwards, new games,
coefficient edits or restart. Training remains active.

## Switching decline after fresh-stream restart — 2026-09-09

Read-only current W&B review through1,912,505 (23,343 completed learner records).
Prior saved last2k / first2k after restart / latest2k: voluntary-switch fraction
of acted replay rows .10082/.09213/.04304; per-legal-switch-cell probability on
move-and-switch-choice rows .04228/.03600/.01716. The latter is not total switch
mass. Voluntary examples per batch11.564/10.6535/4.633; fraction of chunks with a
voluntary example .87113/.85613/.59763. Forced rows17.289/17.1465/16.533, so the
observed drop is not simply disappearance of forced-switch requests. Current
pooled voluntary fraction .04310; current voluntary appearances11.63/s versus
25.90/s in first2k. Prior last-window batch counts times separately measured
4.373updates/s suggest~50.6 appearances/s before restart (approximate windows).

Normalised action entropy .51337/.47887/.31447; conditional macro entropy
.53677/.47704/.26797; modality-uniform KL .95373/1.04528/1.41371. Voluntary-switch
raw ISR .99184/1.04843/.96834 versus move .99927/.99392/1.00250; below-one switch
fraction .56367/.47396/.55821. These aggregate means do not establish severe
importance suppression or measure exact clipped attenuation. Switch-row KL
.00994/.01677/.02468. Switch-head RMS remains nonzero .00465/.00472/.00437.

Interpretation: move-heavy concentration and declining switch-example supply are
measured. Fresh first-use quota neither reserves switches nor guarantees useful
switch advantages. Its early eviction reduced effective reuse to~2.3; feedback
between fewer switches and fewer labels is plausible, not a causal diagnosis.
No current advantage-sign/per-loss switch-gradient decomposition establishes
whether PG/critic labels, shared gradients or state-distribution change drives the
shift. The modality restorer is finite regularisation, not a hard switch floor.
Do not infer strategically justified switching decline from SimpleHeuristic wins
or automatically increase its coefficient. No model forwards or training changes.
Local reader/results: runtime/switch_review_20260909.py,
runtime/switch-review-20260909.json.


## Retain replay visits and diagnose switch pressure — 2026-09-09

User rejected treating switch decline as benign merely because heuristic wins
improved, and requested deeper diagnosis and better sample efficiency. Audit
found a concrete implementation defect in the fresh-stream experiment: allowing
any seen occupant to be evicted made the configured cap a weak upper bound
controlled by producer speed. At steps1910089–1912088 the mean evicted reuse was
2.3238 with cap8, decision reuse2.286, and first-use chunk share~43%. This proves
lost retention budget, not that every additional visit would improve learning.

Latest inspected switch window1913445–1915444: voluntary-switch row fraction
.04625 versus prior .10082; voluntary appearances12.80/s versus approximately
50.6/s before the restart (nearby baseline windows, not a matched timing trial).
Per-legal-switch-cell probability .01900 versus .04228. Voluntary-switch raw
ISR .91563, move1.00673; below-one switch fraction .63952. These are sampled
batch averages, not an estimate of exact clipped attenuation or causal effects.
The post-stop checkpoint01916493 has switch_bias=-.1585254 versus -.1525829 at
01889162 (delta -.0059425): a direct odds multiplier exp(delta)~.9941 at fixed
features. This scalar alone cannot account for the much larger behavioural
change. State-dependent readouts/features, state distribution and advantages
remain candidates. Parameter norms do not identify their contributions.

Correction: retain chunks until their cap is exhausted or explicit KL retirement.
Keep FIFO first-use scheduling, but fill unavailable scheduled fresh slots from
eligible replay. That resolves the full-seen-buffer producer/consumer deadlock
without early eviction. Fraction is now an availability-dependent preference,
not a hard minimum; at fraction1 with cap8, retention wins when fresh is absent.
No loss, optimiser, cap, batch-size, architecture or regularisation coefficient
was changed. Backpressure acts directly at learner.enqueue_traj, not through
an added unbounded trajectory queue. Admission lag can still rise, so watch KL.

Added loss-only switch diagnostics: coefficient-weighted dL/d(common switch
logit shift) for PG, entropy, reference KL, modality KL and actor backward KL;
positive suppresses switching under gradient descent. Reuses executable loss
functions with JVPs, no extra model forward. Taken-switch/stay PG contributions,
signed raw/normalised advantages and sample counts use actual choice rows.
Total switch marginal distinguishes cell probability from modality mass.
Actual signed switch-bias gradient and applied delta expose Adam's scalar route.
These f32 instruments do not attribute shared-feature motion or guarantee exact
agreement with bf16 backward; they add no policy force.

Validation: focused replay/chunk/learner-gate checks passed. Saturated-arrival
regressions span caps1/2/8 and fresh fractions.125/.25/1, assert every eviction
exhausts its cap, and prove a full-seen buffer progresses. Two small loss-only
SPO/PPO checks compare each diagnostic to autodiff through actually shifted
logits, with positive controls, forced-switch and masked rows. Scoped Black,
isort, Ruff and whitespace checks passed. No real-model test while learner live.

Evaluation plan: hold coefficients fixed for two 10k reference-snapshot cycles
unless correctness/non-finite problems require stopping. Verify eviction reuse
matches the effective controller cap, all finite updates apply, and policy
mismatch does not force sustained cap reductions. Judge switching jointly with
signed gradient terms and switch advantages; do not impose a scripted switch
rate or retune the restorer from probability alone. Judge sample efficiency as
strength gained per admitted decision, alongside wall time, not reuse alone.
A same-start-checkpoint control is required for a causal learning-efficiency
claim; sequential before/after windows are confounded by training progress.
Graceful stop saved full checkpoint01916493; resuming it with this correction.
Revert surfaces: buffer replacement/readiness/sampling, fresh-fraction docs,
training/switch_telemetry.py, train-step diagnostics and their focused tests.


Live validation of retention correction: resumed W&B run
`irqeetfg` ( local directory run-20260909_070430-irqeetfg) from01916493;
log runtime/learner_20260909_070003.log, learner PID480367. Both additional
shape variants compiled and learner updates completed. At1917041,548 updates
had applied with zero skips; evicted mean reuse exactly8, effective cap8.
Latest200 updates measured4.652/s, KL.00819, ESS.9564. This is startup evidence,
not a settled throughput or strength comparison. Window applied/admitted reuse
can exceed8 while drawing down previously admitted inventory (latest9.49);
that is not a per-chunk cap violation. Eviction usage verifies the cap here.

First diagnostic window1916517–1916716: weighted switch-logit PG derivative
+.01115, of which +.01006 came from taken switches; entropy-.000945,
magnet-.000103, modality-.008354, actorKL+.0000254; total+.001774.
Actual bias gradient+.001786 agrees closely with the f32 directional estimate.
Mean voluntary-switch raw advantage-.09092, normalised-.35527. Second inspected
window1916842–1917041: PG+.009151 (switch+.008207, stay+.000944), entropy-.000960,
magnet-.000410, modality-.008209, actorKL+.0000627; total-.0003650 versus actual
bias gradient-.0003608. Raw switch advantage-.08711, normalised-.23533.
These are batch means over correlated replay visits; count means4.35/4.745.
Total switch mass on choice rows.05429/.05539, voluntary acted-row fraction
.04070/.04069. There is no recovery claim from these short windows.

This establishes persistent PG pressure AGAINST sampled switching, chiefly from
switch examples themselves, not merely lack of examples. The existing restorer
is active and nearly cancels it; total scalar loss pressure changes sign between
windows. It does not establish whether switch choices are bad or privileged
V-trace targets undervalue them. Raw negative advantages rule out batch centring
as the sole explanation. Follow-up should compare the switch advantage against
realised outcomes and deployable-critic controls before altering regularisation.
More reuse may amplify inaccurate labels, so retained visits are not proof of
sample-efficient learning. No coefficients changed and no further restart.
Local inspection helper/results: runtime/switch_retention_health_20260909.py,
runtime/switch-retention-health-20260909.json. No committed changes.

Definitions of the switch diagnostics (from the deleted rl/online/README.md;
`rl/online/training/action_telemetry.switch_loss_telemetry`):
`player_switch_logit_grad_{pg,entropy,magnet,support}` are the
coefficient-weighted directional derivatives of each actor-loss term for
raising every switch logit together with the features held fixed — positive
suppresses switching under gradient descent, negative encourages it — f32
JVPs of the executable loss functions on their shared policy-row
denominator, attributing no shared-feature update (`support` replaced the
`modality` entry with the uniform KL, 2026-09-09);
`player_switch_logit_grad_actor_total` their sum.
`player_switch_logit_grad_pg_taken_{switch,stay}` restrict the PG term to
the move-and-switch choice rows; `player_choice_{switch,stay}_adv_raw` /
`_adv_normalised` are the signed advantages on those rows (sampled-action
advantages, not matched counterfactual values), `_count` beside them so an
empty subset is not a zero mean. `player_switch_mass_choice` is total switch
mass on choice rows (not the per-cell probability); `player_switch_bias_gradient`
the pre-clip gradient of the switch bias and `player_switch_bias_applied_delta`
its update after Adam and the non-finite gate.

## Paired switch-advantage audit — 2026-09-09

User explicitly requested computing privileged/deployable/realised-outcome
signals on the same switch transitions after rejecting speculative explanations.
Existing telemetry did not preserve the paired quantities. Added
training/advantage_audit.py: both heads use compute_player_targets with identical
isr/config/batch, from existing target-model outputs (no added model forward).
For each head, record raw V-trace advantage, realised discounted outcome minus
its own baseline, and that residual with the same outer clipped rho. The paired
TD-minus-rho-MC difference cancels the baseline and measures bootstrapped-return
versus outcome disagreement; behaviour continuation mismatch still remains.
Outcomes are observational, never counterfactual action labels or a loss input.

Only first-use choice rows contribute, excluding padding, bootstrap, done,
forced switching and unknown outcomes. Horizons1–5/6–15/16–40/41+ use terminal
row index = game_length-1 and absolute decision index = offset+local row.
Publish additive counts/sums rather than sparse batch means, preserving exact
window pooling. Within-game correlation remains: row counts are not independent
game sample counts. Controls include stays, head-specific MC baselines,
head-to-head and TD-to-MC sign disagreement, outcome SSE and mean rho.

Validation:15 focused audit/target tests passed. Hand-computable critics produce
opposite target signs on the SAME row while the outcome residual is positive;
tests cover gamma1/.9, outer rho.5, all distance bins, replay exclusion,
unknown outcomes, forced rows and bootstrap/terminal masks. Black/isort/Ruff and
whitespace checks passed. No full model forwards/tests alongside the learner.
Graceful stop saved01920075; resuming that full checkpoint with diagnostics only.
No loss, coefficient, architecture, sampler or target-estimator changes in this
step. Readout script runtime/paired_advantage_review.py pools first-use sums.


Paired audit result (steps1920076–1921758,1,683 finite updates):1,115 unique
first-use voluntary-switch rows versus15,918 stay rows. Pooled privileged TD
advantage-.08020, deployable-.07968; outcome-minus-own-baseline residuals
-.15792/-.15890. With matching outer rho, privileged outcome residual-.14301:
TD-minus-rho-MC+.06281. Both heads agree; replacing privileged with deployable
changes negative to positive on19/1115 rows (1.70%), reverse23/1115 (2.06%).
Replay-inclusive switch advantage-.08045 over8,892 appearances agrees with
first-use-.08020, so repeated visits do not create the current negative signal.

By remaining decisions (N,privTD,publicTD,privMC):1–5=(129,-.03557,-.03502,
-.03332);6–15=(412,-.08229,-.08158,-.07978);16–40=(530,-.09075,-.09055,
-.26448);41+=(44,-.06424,-.06195,+.02875). The positive6–15 outcome residual
seen at386 switch rows disappeared with accumulation and was negative in the
subsequent-row read after1920640. The41+ bucket changed sign and is too small
for a stable conclusion. Rows remain correlated within games; no independent-
game confidence claim. Mean outer rho.9273, behaviour-continuation mismatch
not fully corrected. Outcome residuals are not counterfactual action advantages.

Conclusion: no evidence here for a privileged-head-specific switch penalty or
pooled hidden positive outcome credit. In the largest distant bucket the TD
signal is LESS negative than the realised residual. This rejects neither shared
conditional critic errors nor policy/continuation defects; do not claim switches
are intrinsically bad, or that collapse is fixed. No coefficients changed.
Report: runtime/switch-advantage-audit-20260909.md; frozen raw results:
runtime/paired-advantage-review-1921758.json. Learner continues with audit enabled.

Definitions (from the deleted rl/online/README.md;
`rl/online/training/action_telemetry.paired_advantage_audit`):
`player_adv_audit_{switch,stay}_{all,1_5,6_15,16_40,41_plus}_*` compares the
two estimators on the same first-use choice rows; horizons count actor
decisions to the terminal reward from whole-game length and chunk offset;
forced switches, terminal/padded/bootstrap-only rows, replay visits and
unknown outcomes are excluded. Sum `_count` and every `*_sum` over a window
before dividing — never average sparse per-batch means; rows within a game
stay correlated. For each of `public` and `privileged`: `td` is that head's
V-trace advantage (rho thresholded, c raw — the learner's own since the
2026-09-09 tidy), `mc` the discounted realised behaviour outcome minus the
head's own value, `rho_mc` the same under `td`'s outer truncated weight, so
`td - rho_mc` isolates bootstrap disagreement with the observed outcome with
the baseline cancelled (the behaviour continuation is not corrected to the
target policy; this is no counterfactual switch-versus-stay read). The other
sums record value, outcome, squared outcome error, mean outer weight and
paired sign disagreements. Outcomes enter these diagnostics only.

## Privileged-head usefulness audit — 2026-09-09

Follow-up asked why opponent-private knowledge gives little accuracy gain.
The paired first-use outcome audit supplies a direct error comparison (not just
advantage similarity): switch MSE public.781756 vs privileged.780656, relative
reduction.14%; stay MSE.812982 vs.810043, reduction.36%. These are outcome errors
on choice rows, not all states or an unrevealed-information-conditioned sample.
Latest500 updates1922271–1922770: mean absolute head gap.02309; TD-label R2
public.870445/privileged.872182. Those R2 panels compare both heads to the SAME
bootstrapped win_returns, not independent final outcomes.

Code audit: service serialises opponent's current private request; this is
hidden-state access, not the future action or terminal outcome. Encoder pools
private-sheet tokens then applies16x16 hard straight-through categorical codes;
only their code-table embeddings enter the secret rows read by VALUE_CLS.
The code is supervised through privileged value loss. Belief label gradients
are stopped; the hidden-token belief-label change did not alter critic inputs.
There is no direct hidden-attribute reconstruction requirement on this code.
The historical Sept5 ledger already recorded the full-code public-token shortcut.

Current code is not trivially dead: mean/min group perplexity5.408/3.367,
logits/embedding gradient norms.00559/.02026; trained kernel/table RMS
.07873/.08879. These establish activity, not retention/use of useful secrets.
Leading unproven mechanism is a value-supervised discrete bottleneck together
with shared self-bootstrapped labels preserving a public-feature solution.
Do not claim same labels force equal heads: useful private information could
still improve prediction under identical labels. No ablation proves this cause.
A conditional hidden-sheet intervention/outcome-error comparison is needed to
measure reliance, and a reconstruction probe can distinguish retention from
readout use. No restart, coefficient or architecture changes in this follow-up.


## Privileged attention routing measured — 2026-09-09

User requested attention weights rather than another hypothesised intervention.
Gracefully paused learner at01929574; ran existing COLLECT_INTERMEDIATES hooks
on that checkpoint's EMA target encoder, on recorded self-play from01889162
(collection-920/selfplay.pkl):256 sides,260 chunks,7,377 acted rows,5,429 choice
rows,724 voluntary switches. No new live model instrumentation or source change.
All six opponent-private rows valid throughout scored data. This is an older
fixed state distribution, not current on-policy games.

VALUE_CLS total opponent-private attention by block (mean over heads):
all decisions .49/1.63/2.02/2.53/1.71/5.92%; voluntary switches
.49/1.62/2.04/2.55/2.13/6.69%. Uniform valid-key references13.18/13.09%.
Final-layer head4 is an exception:15.41% all /17.22% switches; other final-layer
heads2.34/1.92/4.02% all. First layer own private sheets34.26%, field19.52%,
public entities18.91%, versus opponent-private.49%. This measures weak direct
private reads on most heads, not absence of every private read.

Private-query rows themselves assign56.36% to private keys in block1, then
2.40/6.40/6.25/6.31/7.23%; their private-key uniform reference13.48%. Remaining
attention reads policy-visible rows. Residuals retain information, so this does
not prove secret content gets erased. Nor do weights alone establish causal
value influence: value norms/output projection and later mixing matter.

Checks: exactly zero policy-to-private attention; nonzero private-query reads;
valid six-row input; query sums within bf16 tolerance (max raw error.014725),
renormalised for reported masses; terminal/padding/bootstrap exclusion and
named sequence slices. Frozen probe finished successfully before restart.
Report and raw data: runtime/privileged-attention-probe.{md,json,png,svg};
script runtime/privileged_attention_probe.py. Resuming01929574 unchanged.

Attention-probe restart verified: runtime/learner_20260909_080752.log, W&B
run-20260909_081202-irqeetfg;36 finite updates through1929610, zero skips.
Model/losses unchanged; COLLECT_INTERMEDIATES was confined to the offline probe.

## Removal ledger — 2026-09-09 search eval actor

Deleted the depth-1 expectimax eval slot (`eval_search_slots`, its
`-search` thread and the `search-*` per-game eval logs) and the disabled
depth-2 slot (`eval_search_depth`), and re-cut the slate to exactly two
slots against the simple heuristic, both EMA params at T=1:
`EvalActor-simpleheuristic-plain-t1-0` (the policy sampled exactly as the
training actors sample it) and `EvalActor-simpleheuristic-thresholded-1`
(the same policy with every legal cell below `player_prune_threshold`
= .005 removed and the rest renormalised at sampling, DeepNash's
`FineTuning._threshold` guard included, via the new traced
`HeadParams.prune_threshold`; 0.0 is bit-identical and is what every
training actor passes). The two T=0.5 slots retire with them: T=.5 is not
comparable across head parameterisations (2026-08-29) and the live read
moves to an offline diagnostic. Series are keyed by thread name, so the
`-0`/`-1`/`-t1-2`/`-search-3` series end at the restart that carries this.

Measured reason: at ckpt_01861967 depth-one expectimax moved root KL by
.000026–.000101 and left both inspected bad actions (Psychic Noise into a
revealed Soundproof; the Decidueye case) ranked first in 16/16 seeds — the
arm was pricing nothing. `rl/model/search.py`, the model's `cfg.search`
branch and `actor_params_view(search=True)` STAY for the offline readers
(`rl/offline/harness.py configure_search`, `search_ablation.py`,
`search_samples_probe.py`) and their tests. Revert handle: tag
`pre-eval-slate-2026-09-09` (the slot construction in `rl/online/main.py`
and the two config fields). The eval-leak gate `should_push_trajectory`
gains `tests/test_guards.py` — the thresholded slot samples a distribution
no training actor uses and was the first eval case the gate had no test
for.

Validation: `tests/test_prune_policy.py` (threshold 0 bit-identical to the
sampling form; removal + renormalisation; the untouched-above-the-line
positive control; the all-below guard; zero gradient through a removed
cell's own logit; a real-model actor forward bit-identical under
`HeadParams()` vs `HeadParams(prune_threshold=0.0)` with a .1 control that
changes `log_prob` and no metric), `tests/test_guards.py`. Not claimed:
any strength effect — nothing here changes what trains.

## Addition ledger — 2026-09-09 flat support telemetry and applied deltas

Observers only, no loss force, no model change. `legal_support_telemetry`
(`rl/online/training/move_telemetry.py`, called beside
`switch_loss_telemetry`) reads each real decision row's legal cells as flat
complete actions: min and median cell probability, legal-cell count, the
fraction of legal cells below .01/.005/.001, and the same split switch vs
move (`player_support_*`). `applied_delta_telemetry` (`telemetry.py`)
generalises `player_switch_bias_applied_delta` to the readout leaves a
support force acts on directly — `switch_bias`, the move `query`, the
target `key` and `local_tgt` projections — as rms of the post-clip,
post-revert update (`player_applied_delta_rms_*`); gradient norms say what
was asked, these say what moved. `player_learner_actor_ess` and
`player_learner_actor_ratio_tail_gt2` read the learner/behaviour ratio, a
different population from the target/behaviour `player_isr_ess` /
`player_rho_clip_frac` v-trace consumes. These are the exposure instrument
for the support hinge and the calibration input for the v-trace threshold
that follow; the panels land with those. Validation:
`tests/test_move_telemetry.py` (hand-computed rows with a forced row and a
masked row, permutation invariance, and the lifted-cell positive control).

## Removal ledger — 2026-09-09 actor backward-KL force

Deleted `config.player_kl_loss_coef` (.05) and its term
`player_kl_loss_coef * loss_actor_backward_kl` from the player loss — the
sampled k3 KL(learner || behaviour) on the taken action, outside the
`player_pg_coef` bracket. Not an ablation: it penalised the learner for
moving away from the behaviour policy, which is exactly the direction a
support force pushes when it lifts an action mu almost never takes; two
terms pulling opposite ways on the same cells, this one measured at
.00796 (`player_learner_actor_backward_kl`, `player_ref_kl` .01124
beside it) and doing no identified work. The estimator stays computed and
logged under both names; `player_switch_logit_grad_actor_kl` goes with
the force (a directional derivative of a term that no longer exists).

What still holds drift down: the SPO trust region (`player_ppo_clip` .2),
the magnet (`player_mag_coef` .2, `reg_params` snapping every 10k) and the
replay-reuse controller, which reads `player_learner_actor_forward_kl` —
a different estimator — against its .045 set-point and is untouched. In
expectation the two KLs combined into one KL toward normalised
reg^0.8 · behaviour^0.2 at weight .25
(`docs/representation-and-action-support-plan-2026-09-09.md` §5), so
what changes is the pull toward the behaviour policy's OLD mistakes, not
the trust region's existence. Revert handle: tag
`pre-actor-kl-removal-2026-09-09`. Validation:
`tests/test_switch_telemetry.py` (the four remaining terms still match
the derivative through shifted logits). Not claimed: any effect on
strength or staleness — read `player_learner_actor_forward_kl` and the
learner/behaviour ESS after the restart that carries this.

## Removal + addition ledger — 2026-09-09 flat support hinge

Deleted `loss.uniform_kl_modalities` and `player_uniform_kl_coef` (.025,
the modality-marginal zero-avoider live since 2026-08-31, last read
`player_loss_modality_kl` 1.264 → 1.203 holding `player_switch_mass_choice`
.07171) and put the FLAT SUPPORT HINGE in its place inside the
`player_pg_coef` bracket: over a row's legal cells as flat complete
actions, `(1/N) Σ_a max(0, log(tau_row / pi_a))`, `tau` .01,
`tau_row = min(tau, .5 / N)` (the feasibility clamp, panelled as
`player_support_n_tau_row` / `player_support_saturated_frac`), derivative
`active_fraction · pi_b − below_b / N` — bounded, zero-sum over the row,
no pi prefactor on the cell it lifts, exactly silent above the line. Why a
replacement and not a pair: at a starved action the two forces are the
same order and equally pi-independent, but the KL keeps pulling toward
uniform modality mass at every probability and pushes DOWN anything above
its target, while the hinge hands the choice back to the critic; a silent
hinge under a live KL would have proved nothing; and the KL's grouping
through `CELL_MODALITY_MASK` (MOVE vs WILDCARD) pulled P(tera | move)
toward one half, conditional derivative `c/M · (2q − 1)` — the flat form
has no hierarchy to do that with, which retires the "group tera with the
ordinary moves" question by deleting the grouping.

The axis sp75c failed on, taken deliberately: the row-form uniform KL
applied force at every probability and pinned `player_entropy_micro_taken`
at .93 (control .84, exploit halved). The hinge is silent above tau, so
discrimination above it is untouched; that panel (now on its own, currently
.4919) is the abort instrument. Accepted openly: a flat hinge does say
WHICH — it lifts ineffective legal moves along with useful ones. That is
the price of not letting the learner's current probabilities decide what
stays in contention.

Calibration: 5–6 bench cells at .01 induce a switch-mass floor of 5–6%,
just under the .07171 the KL was holding. `player_support_hinge_coef`
lands at 0.0 (exactly off) and is set by the offline screen
(`rl/offline/support_screen.py`, .001/.0025/.005 over recorded chunks:
the smallest that lifts the abandoned cells without a > 10% rise in
shared-encoder gradient rms) before the restart; never swept live (config
is a jit static argname). Revert handle: tag `pre-eval-slate-2026-09-09`
(the whole 2026-09-09 set reverts together — the effects train in, a
zeroed coefficient is not a revert). Validation:
`tests/test_support_hinge.py` (silent above tau / positive on a 1e-4 cell;
coefficient 0 and forced rows exactly off; illegal cells never scored;
permutation invariance; the derivative formula and its zero sum against
`jax.grad`; the zero subgradient at the hinge; the clamp binding on a
100-cell row and not on four; a ±1e4 saturating row finite and bounded),
`tests/test_switch_telemetry.py` (the hinge's directional term mirrors the
executable loss). `TestUniformKlModalities` goes with the term.

## Addition ledger — 2026-09-09 learner-side v-trace thresholding

`targets.thresholded_target_ratio`: the v-trace ratio pi_target(a) / mu(a)
is built from the TARGET policy with every legal cell below
`player_prune_threshold` (.005) removed and the rest renormalised — the
same `prune_log_policy` the `thresholded` eval slot samples from — so a
taken action the target has dropped below the line gets ratio 0 and
v-trace discards the row (no value target, no advantage; every earlier
row's trace is cut there). Applied to the π entering v-trace and nowhere
else: the learner ratio, the SPO surrogate, the magnet, the entropy term,
the hinge, `player_learner_actor_forward_kl` and `chunk_policy_mismatch`
read the raw policies, so the reuse controller's set-point keeps its
meaning; the training actors are untouched. Variance control on the
target estimator, not a policy force and not exploration.

Reference diff against `~/Downloads/rnad.py` (local): `FineTuning`
(:137) is elimination not a floor (`_threshold` :184–196, with the
all-below guard copied verbatim), applied inside `loss()` (:798,
`policy_pprocessed` → `v_trace` as `merged_policy`) and at acting (:1007);
`acting_policy` and the NeuRD loss's `pi` stay raw (:817, :836). Three
deliberate divergences, owned: always on from the restart (the reference
gates it on `from_learner_steps`, −1 = off, a late strength fix for a
converged policy — the divergence with the most weight behind it, and why
the discard rate is a revert trigger not a panel); .005 not .03; no 1/32
discretisation (at .005 it would not dominate the threshold, so dropping
it is coherent). One further difference, not a choice: our v-trace policy
is the target network's, so the discard decision is made on a lagged
distribution.

The hazard, measured not assumed: the continuation `c` is the same ratio,
so one zeroed row cuts the trace for every row before it in the chunk.
`player_trace_len_mean` / `_raw` (realised continuation length from each
policy row, thresholded vs raw) and `player_discard_{taken_frac,
legal_frac, position_mean}` read it live; the pre-restart cut measurement
on recorded chunks (`rl/offline/support_screen.py`) decides the
restriction pre-registered 2026-09-09: if > 5% of chunks carry a discard
before their midpoint, threshold rho only and leave `c` raw (split
`rho_t` / `c_t` at `targets.py`, merged 2026-08-21). `player_isr_ess` and
`player_rho_clip_frac` change meaning at this restart; their `_raw` twins
carry the comparable series. Revert (the whole set): `player_discard_taken_frac`
> 1% sustained over one 250k-fresh-decision window (≈ 1.7 h). Validation:
`tests/test_vtrace_threshold.py` (threshold 0 bit-identical; the .004 row
discarded with its raw ratio intact and the .40 rows untouched — the
positive control; the all-below guard; a discarded mid-chunk row zeroing
its own advantage, leaving a bootstrap-only value target, unchanged rows
after it and cut rows before it, with the trace-length twins reading the
cut; and the slow scope pin — learner ratio, surrogate, magnet, entropy,
hinge and forward KL bit-identical under threshold .3 vs 0 while the
v-trace ESS and the discard rate move). Not claimed: any effect on
strength; the set lands together and is read on the frozen cohort.

## Stop window — 2026-09-09 cut audit fired the rho-only restriction

The previous run had stopped itself at 13:08 on the host-RAM guard at
`ckpt_02014000` (step 2,014,000), so that is the stop checkpoint for the
whole set. `rl/offline/support_screen.py` on it, 32 self-play games → 64
chunks, 1,702 acted rows, threshold .005 on the target policy: .29% of
acted rows discarded, 7.8% of chunks carrying a discard, every one of
them before the chunk's midpoint (median first discard at 21% of the
chunk). Against the pre-registered 5% gate the restriction fires: rho
carries the threshold, c stays raw (`compute_player_targets`), so a
discarded row loses its own advantage and TD term and nothing else (its
value target still bootstraps through c) — with
c thresholded too, one chunk in thirteen would have lost credit assignment
for four fifths of its rows to one abandoned action. `player_trace_len_mean`
now reads the cut avoided and `_raw` the live trace.
`tests/test_vtrace_threshold.py` carries the positive control (the same
discard fed to c does cut every earlier row).

Frozen cohort built the same window (`rl/offline/tactical_cohort.py
collect`, 240 heuristic games at T=.5, seed 909, 162 wins, the
simulator's logs beside it) and read on ckpt_02014000: 531 states with an
immune damaging move and an effective alternative;
`ineffective_confident_mass` .144 at T=1 (whole-game bootstrap 95% CI
.081–.207), .159 at T=.5 (.079–.244); type immunity .122 over 470 states,
REVEALED-ability immunity .306 over 62 (T=.5: .130 / .371); 7
simulator-confirmed immune actions at mean recorded probability .686. This
is the before; the after is the same command on a later checkpoint.

The coefficient screen, same window, same 64 chunks, four compiled train
steps (TRAIN_STEP_JIT twice per batch from the restored state, 16
batches): pre-update `player_loss_support` .358 with 27.3% of legal cells
under tau (16.2% under .005, `player_support_min_prob` .0197,
`N·tau_row` .066, the clamp never binding); shared-encoder gradient norm
11.1699 / 11.1696 / 11.1700 / 11.1710 at 0 / .001 / .0025 / .005 (relative
1.0000–1.0001 against the 10% ceiling); `player_switch_logit_grad_support`
0 / −.00008 / −.00019 / −.00039 (restoring, the right sign); after ONE
restored-Adam step every exposure read is unchanged to the fourth decimal
at every coefficient (min .0197, below-.01 .272, below-.001 .048,
`player_entropy_micro_taken` .432, switch mass .0996). So the screen bounds
the cost at nothing and cannot see the benefit at one step by
construction; the pre-registered rule ("the smallest that lifts exposure
without > 10% encoder rms") has no smallest, and the top of the screened
range, .005, landed — recorded as a cost-side decision, not a measured
lift. If the live `player_support_active_fraction` does not fall from .27
over the first 250k fresh decisions the coefficient is too small for the
mechanism to be tested at all, and that is a separate restart, not a
verdict on the hinge.

Slow suite in the same window: everything green except
`tests/test_history_carry.py::test_server_mixed_group_matches_single_forwards`
(`np.squeeze` in `InferenceServer._run_group` meets an output leaf whose
leading axis is not the request's T=1), which fails IDENTICALLY at tag
`pre-eval-slate-2026-09-09` in a clean worktree — pre-existing, not this
set's; the inference server is not on the actor path (actors run on the
CPU direct path) and it is left open here. The scope pin in
`tests/test_vtrace_threshold.py` was corrected in passing: `player_loss_pg`
is not invariant to the threshold (its advantage is v-trace's, changed by
design); the ratio side is pinned through `player_ppo_clip_frac`.

Launch check, 2026-09-09 15:11 relaunch from ckpt_02014000 (wandb
irqeetfg resumed), read at lifetime_step 2,024,880: every flag live
(`player_support_hinge_coef` .005, tau .01, `player_prune_threshold` .005,
the two eval slots initialised, the two deleted coefficients absent from
the printed config), no non-finite skips or guard lines. `player_isr_ess`
.963 against `_raw` .986 — the threshold is live. `player_discard_taken_frac`
.024, above the 1% line from the first window (the offline audit on FRESH
self-play read .003: the live gap is replay staleness, taken actions the
target has since dropped); `player_discard_legal_frac` .203,
`player_support_active_fraction` .328, `player_switch_logit_grad_support`
−.0009, switch mass .045, `player_entropy_micro_taken` .447; trace length
12.3 had c been thresholded against 17.1 live. Eval after 217 games each:
plain-t1 .586, thresholded .526.

**User decision 2026-09-09: no revert regardless — the set rides out
overnight.** The pre-registered discard-rate trigger is therefore NOT
applied to this window; the overnight read is the verdict instrument, and
the cohort re-read on the next checkpoint the strength read.

## Support hinge coefficient .005 → .05, and the discard split — 2026-09-10

Overnight read of the relaunch (irqeetfg, steps 2.12M–2.23M): the hinge's
directional pull on the switch logits −.0005 to −.0013 against the retired
modality KL's −.0065 to −.0081 in the pre-restart segment and the policy
gradient's ±.02 swing; switch mass on choice rows .065–.092 (holding);
switch cells under .005 .12–.33 and move cells .11–.25, noisy, no trend —
the structural expectation when ~.08 of mass sits over ~5 bench cells, and
a per-cell floor the KL never held either (it was invariant to
within-modality redistribution by design). The hinge cannot push a cell
down (its tax on an above-line cell, coef · active_fraction · pi, is
~3e-5 against a ~1e-3 lift), so it was simply too small to test: coef / N
≈ coef · .15 holds a cell at the .005 line only against an adverse
normalised advantage of ~.15. User decision: raise to .05 — lift .0075
(holds against ~1.5), switch-axis pull ~ −.009 (the KL's order), encoder
gradient cost extrapolated from the screen ~0.1%. Lands at the NEXT
restart (static config); the overnight run stays at .005 by the user's
earlier decision. Same commit: `player_discard_taken_frac_{switch,move}`,
the discard rate by taken modality — a switch-side rate well above the
move-side one is the threshold acting on rare switches, the self-sealing
direction only the hinge resists. Validation: fast suite; the smoke
test's key list carries the two names.

Second guard trip of the day: the .005 relaunch stopped itself at 16:39 on
the host-RAM guard at `ckpt_02038000`, 24k updates in (the 13:08 trip came
84k after the 08:12 relaunch; the run's history is 120–150k between
trips — the interval is shortening and wants its own look). Cohort read on
ckpt_02038000 against the ckpt_02014000 baseline, same 531 states:
`ineffective_confident_mass` T=1 .144 → .133 (95% CI .081–.207 →
.082–.191, overlapping), T=.5 .159 → .133, revealed-ability states .306 →
.167 (62 states), simulator-confirmed immune actions 7 → 7. Descriptive:
24k updates at coefficient .005 with the threshold live, not attributable
to any one part. Relaunched from ckpt_02038000 with
`player_support_hinge_coef` .05 and the discard split (b542724); the
screen at .025/.05/.1 was skipped to keep the restart short and is owed at
the next stop.

## Removal ledger — 2026-09-09 per-chunk retention feedback

Deleted the per-chunk retention feedback built 2026-09-07/08 (`ca34c74`):
`player_replay_trajectory_mode` (observe/protect) and its `kl_threshold`,
`PlayerTrajectoryStore.apply_feedback` / `feedback_logs` and the seven
never-written fields behind them (`_retired`, `_last_feedback_visit`,
`_last_kl`, `_feedback_applied`, `_feedback_ignored`,
`_threshold_crossings`, `_retired_total`), `Trajectory.replay_{slot,id}`
and their `stack_batch` legs, `rl/online/training/replay.py`
(`chunk_policy_mismatch`, `consume_replay_feedback`), the train-step
`_player_replay_feedback` block, the log-worker call, and
`tests/test_trajectory_replay.py`. The fresh stream is untouched
(`player_replay_fresh_fraction`, oldest-id eviction over `_ids`,
`reuse_count`, the decision accounting).

Measured reason: the mode was `off` in every run since it landed, so the
whole path was dead code at the shipped config — yet the store still
allocated and reset the seven fields in three places, `apply_feedback`
duplicated the validity predicate that `consume_replay_feedback` re-spelled
(and the two disagreed: the store also rejected stale slot/id/visit, so
`player_replay_trajectory_threshold_crossings` and
`player_replay_chunk_above_threshold_frac` answered the same question
differently), and `feedback_logs` logged seven constant-zero scalars
(`player_replay_feedback_{applied,ignored}`,
`player_replay_trajectory_{threshold_crossings,retired_total,
retired_resident,observed_resident,eligible}`) every learner step. No
dashboard panel read any of them. Nothing here changes what trains.
Revert handle: tag `pre-tidy-2026-09-09` (the last commit carrying it) and
`ca34c74` (the landing). `player_replay_kl_target` STAYS — it is the replay
PI controller's setpoint (`workers.update_replay_controller`), which the
mode only borrowed.

Validation: `tests/test_fresh_replay.py` (chunks re-identified by an
admission tag on `game_length` instead of `replay_id`; the dynamic-cap half
of the deleted identity test kept as `test_fresh_stream_under_a_dynamic_cap`),
`tests/test_buffer.py`, `tests/test_chunking.py`; the train-step leaf dump
against the tag is owed at the next learner-free window with the rest of
the tidy.

## Probe — 2026-09-10 switch pair (why every head is flat within switching)

`rl/offline/type_probe.py` SWITCH PAIR block, read at the stop checkpoint
ckpt_02339569 over the frozen cohort (240 heuristic games, 18,607 legal
switch cells, 6,121 held out by chunk; the learner stopped for it). Ridge
reads of the type chart on (my legal switch candidate's private sheet row,
the opponent's active row), both labels: candidate row pre .486 / post .498
(OFFENSIVE, floor .542) and .483 / .525 (DEFENSIVE, floor .537); opp row
pre .496 / post .505 and .517 / .501; concat pre .487 / post .498 and .497 /
.518; random bilinear rp24 pre r +.14 / +.20, post +.03 / +.08. Every read
at or under the majority floor; only the PRE-trunk bilinear sees the chart
at all. Positive controls: the candidate's own primary type from its row
.916 pre → .637 post; the opponent's .946 → .869. The move × target control
in the same run: move row post-trunk acc .658 / r +.64 (floor .565), the
opponent's type readable from the move row at .664, the policy's mass on
immune moves .148 vs .389 uniform and on supereffective .716 vs .412 — the
2026-09-03 "post-trunk .50" read is stale for moves.

Reading: the trunk computes the move-vs-opponent matchup into the move row,
which the readout multiplies against the target row, and computes NOTHING
of candidate-vs-opponent into the sheet row, which the switch scalar reads
ALONE (`FlatActionReadout`: one bilinear for moves x targets, a scalar per
sheet row for switching). No head that reads the sheet rows — the retired Q
heads, the critic, the switch scalar — is given both sides of the pair, so
the failure is specific to switching across all of them. Justified: a
pairwise operand at the switch read over the PRE-trunk rows (types present
at .92 / .95; the chart is a rank-18 bilinear in type bits) —
switch(s) = w·c_s + q_raw(r_s)·k_raw(r_opp), q_raw zero-init. Not
established: the learned-bilinear ceiling on the switch pair (rp24 is an
accessibility bound only; the move table's .79 came from a learned fit), any
strength effect, or that the chart is the whole of a switch's value.
Write-up: docs/switch-pair-probe-2026-09-10.md (local); log
runtime/type-probe-switch/ckpt_02339569.log. Harness note: run probes in a
tmux window — the agent harness's memory guard killed two attempts with
23 GB free.

## Matched learned switch readouts — 2026-09-10

Follow-up to the switch-pair probe above, using frozen EMA parameters at
ckpt_02339569 on the same 240-game heuristic cohort. Removed terminal and
overlapping bootstrap rows: 18,372 unique game/step/candidate records.
Three 144/48/48 whole-game train/validation/test splits; learned rank-64
bilinear per effectiveness class, identical pair-head capacity (133,124
parameters), train-only feature standardisation, Adam .001, 100 epochs.
Validation selects epoch and L2 from {0, 1e-5, 1e-3, .01, .1}; the initial
narrow grid hit its early-epoch/strong-regularisation boundary and is saved.

Mean held-out offensive/defensive accuracy: majority .532/.552;
candidate-only post .533/.560; candidate-only pre .540/.558;
candidate×opponent post .538/.558; candidate×CLS post .532/.552;
candidate×opponent pre **.628/.640**. Balanced accuracy for the raw pair
is .465/.500 versus .297/.302 for the post pair. Raw-pair ordinal r is
.429/.447. The raw pair wins both accuracy measures in every split but
has .937/.939 training accuracy: generalisation remains a material gap.
Its selected L2 is .01 in every split. Many selected epochs are near the
100-epoch budget, so this is not a demonstrated optimisation ceiling.

Same-split fixed-alpha ridge controls: candidate type pre .885/post .630;
opponent type pre .956/post .858; opponent type from CLS .080. A synthetic
interaction-only task gives learned pair 1.000 vs candidate-only .249,
confirming the training implementation can learn a product. Four legal
cells have zero encoded HP ratio; the old rounded 1.000 alive check was
not exact and is not a proof of slot identity.

Interpretation correction to the preceding entry: a scalar of a contextual
row CAN depend on the opponent through attention; these probes establish
limited readout generalisation, not that the trunk computes NOTHING or that
all heads lack both sides. Current critics read CLS/VALUE_CLS, not the
switch sheet bank directly. The results favour a raw-pair experiment but
establish neither strength nor an exact .8–.9 type-chart decoder. Preserve
the existing learned scalar when adding zero-init terms for unchanged
initial behaviour. No production model or checkpoint was changed.

Reproduction: rl/offline/switch_readout_probe.py; local report
docs/switch-pair-learned-readouts-2026-09-10.md; frozen cache, per-split JSON,
initial grid, controls and logs under runtime/type-probe-switch/.
Focused Ruff/Black/isort checks and the synthetic positive/negative control
passed. GPU extraction/fitting completed; the small synthetic control used
CPU fallback, with no real-model CPU forward. No commit or training launch.

## Switch depth curve and intermediate supervision — 2026-09-10

User authorised depth probes and an isolated supervised intervention, with
results appended to docs/switch-pair-probe-2026-09-10.md section 10. Same
240-game cohort, 18,372 unique legal switch cells, three 144/48/48 whole-game
splits, frozen EMA checkpoint 02339569. Actual TrunkBlock extraction matched
the old depth-0/6 operands bit-for-bit; the shared frozen readout fitter
reproduced all endpoint test accuracies exactly.

Frozen paired offensive/defensive accuracy by depth 0–6:
.628/.640, .553/.577, .560/.572, .536/.568, .535/.556, .531/.553,
.538/.558. Largest adjacent drop at block 1, no intermediate peak above
the raw input. Candidate primary-type ridge accuracy declines monotonically
.885 → .780 → .738 → .715 → .694 → .657 → .630; opponent type stays
more accessible, .956 → .858. Candidate-only matchup reads stay near the
majority baseline. Training curves use validation-selected regularisation;
their decline does not establish a loss of memorisation capacity.

Intervention: freeze assembled inputs/embedders/history, train isolated
six-block trunk copies with the original policy/privileged mask. Final-layer
loss versus mean loss at layers 2/4/6, identical independent paired heads,
two label tasks, averaged head L2=.01, head Adam LR=.001, trunk LR grid
{1e-5,1e-4}, gradient clip 1, 20 epochs of identical 32-state batches.
Select both arms by final-layer validation cross entropy, then freeze and
refit fresh candidate-only and paired readouts at EVERY depth. All six
arm/split selections chose LR1e-5, epoch1; later training overfit validation.
All selected trunks changed in every block; this was not a closed-gradient
or unchanged-parameter control.

Final-depth fresh paired accuracy: frozen .5385/.5581, final-only
.5317/.5625, intermediate .5353/.5657. Balanced accuracy: frozen
.2972/.3019, final-only .2972/.3133, intermediate .2996/.3168.
Intermediate-minus-final-only balanced accuracy is +.00244/+.00345,
positive in 2/3 splits for each label: the registered directional gate
passes, but the effect is only **0.24/0.35 percentage points**, and the
depth-related accessibility decline remains largely unchanged. This does
not establish a substantial repair, adaptive-depth benefit, self-distillation
benefit, or policy strength. These are supervised offline type labels,
not a production auxiliary objective. No post-test tuning was performed.

Implementation rl/offline/switch_depth_probe.py reuses the existing frozen
fitter in switch_readout_probe.py. Protocol in local
docs/switch-depth-supervision-protocol-2026-09-10.md; all frozen/selected
features, per-split results, learning curves, checkpoints of the isolated
copies, parameter deltas and PNG/SVG figures under
runtime/type-probe-switch/depth/. 252 fresh readout results including the
unchanged depth-0 controls; 126 type controls. A layer-2 loss has nonzero
gradients in blocks1–2 and zero in blocks3–6. Privileged perturbations
leave every policy-readable depth identical, with a live public-opponent
perturbation positive control. Focused Ruff/Black/isort and whitespace
checks passed; plots visually inspected. No production parameter changes,
training restart, commit or modification of the dirty data/ps submodule.

## Six-layer attention routing visualisation — 2026-09-10

User requested measured per-layer attention and a 3D routing visualiser.
Read the production TrunkBlock's existing attention sow at EMA02339569
on the 5,643 cached switch-legal states from the same 240-game cohort.
All six layers/four heads, cohort means and six sampled states from distinct
games (seed910). Token means condition on valid queries; group means weight
valid query instances. The separate legal-switch summary gates sheet queries
on the 18,372 legal switch cells, whereas the visual's default sheet group
includes all valid sheet queries, including illegal switch targets.

Legal-switch sheet query mass, mean over heads: block1 request info38.88%,
CLS30.87%, field14.29%, opponent active3.85%. Input CLS is a learned
constant; input INFO contains only request type and number of actives
(encoder._assemble_sequence). Fixed values may still contribute useful
query-dependent updates; this does not prove an attention sink or the cause
of the type-readability decline. Opponent-active mass by block is
3.85/5.12/1.96/7.58/3.52/2.76%. Block4 head2 is specialised at23.74%,
versus3.04/1.36/2.16% in its other heads. Later field/move/history rows
are contextual, so their labels do not bound the information they carry.

Captured final candidate rows reproduce the saved frozen cache exactly
(max delta0). Forbidden attention max0. Raw bf16 query-sum error max.015686;
reporting renormalises valid queries, without changing the forward.
The visual draws key/value source→receiving row between consecutive depths,
with separately labelled unweighted residual skips. Attention probabilities
are not value-vector contribution norms or causal attributions; MLP effects
are not represented. Cohort averaging is not an attention rollout.

Implementation rl/offline/attention_routes.py. Local report in
docs/switch-pair-probe-2026-09-10.md section11. Self-contained interactive
fragment, editable template/builder, numeric summary, full-precision cohort
and six sampled matrices, and QA screenshots under
runtime/type-probe-switch/attention/. The inline token export stores top8
keys per query and uint16 weights; head means are calculated before
truncation, full group matrices are retained, and visible routes never
renormalise the displayed mass. Fragment size729kB, below the1MB limit.

Checked group/token, head, state, block, route count, camera sliders, drag
rotation, tooltips and node selection; no JavaScript errors or overflow at
736px/360px in light/dark. Screenshots inspected. Connected browsers were
unavailable, so QA used isolated local headless Chromium. Focused source
checks passed. Production parameters and service/training state unchanged;
no commit, package manifest change, or edit to the dirty data/ps submodule.

## First-block causal routing test — 2026-09-10

Frozen EMA02339569, same 240-game cohort / 5,643 states / 18,372 legal
switch cells. Remove INFO, CLS, both, or FIELD only from own-sheet queries
in block1 via the native mask (renormalising). Controls: attenuate the whole
attention output by one minus original INFO+CLS probability mass per query
(mean heads); bypass the whole block. Attenuation is probability-fraction
matched, not vector-norm matched. FIELD is one-source semantic control.
All later blocks unchanged. Three original game splits, same 100-epoch,
five-L2 paired readouts independently refit at depths1/6. 72 fresh fits;
12 baseline fits reused after bit-exact candidate/opponent cache reproduction.

Final balanced accuracy offensive/defensive: baseline .2972/.3019;
INFO removed .2894/.2958; CLS .2979/.3039; both .2811/.2813;
FIELD .2969/.3010; attenuation .2931/.3046; block bypass .2499/.2482.
Removing both worsens each label in all3 splits. Attenuation improves depth1
balanced accuracy .3801/.3921 to .3897/.4044 but does not consistently
improve final depth. No evidence to delete the high-attention INFO/CLS routes.
Bypass introduces distribution shift into later frozen blocks; it does not
price a five-block model trained from scratch. No policy-play claim.

Descriptive f32 reconstruction from bf16 values/weights/output kernel:
legal-sheet median INFO+CLS projected norm1.819, total attention3.213,
input residual7.039, whole block update59.316, approximate MLP58.939.
Median route/attention norm .571; route/input .256. Norms are non-additive
because routes can cancel. MLP magnitude dominates the first update, but
its causal effect on generalisation is not established by magnitude.

Register-token hypothesis prompted by user (Darcet et al., arXiv2309.16588):
valid-row median input norms differ sharply: CLS2.8, field4.3, sheets7.1,
INFO9.9, moves14.0, own history1041.0. History is already high-norm before
trunk; final CLS rises from160.7 at depth5 to1011.7 at depth6, but that final
output is never read by subsequent trunk attention. These are not sufficient
evidence for content-token repurposing or a register fix. Blocks apply
pre-RMSNorm, while the residual stream keeps heterogeneous scales; learned
per-channel scales do not guarantee equal normalised-output norms.

Native baseline and separate norm/capture endpoints exactly reproduce the
cache. Each targeted ablation changes candidate rows with zero first-depth
opponent delta; bypass depth1 exactly equals raw inputs. Initial combined
value instrumentation changed bf16 rounding and was rejected before fitting.
Descriptive reconstruction now runs separately from intervention extraction.
Focused Black/isort/Ruff and whitespace checks passed. No production weight
changes, training restart, commit or data/ps changes.

Code: rl/offline/first_block_causal.py and first_block_diagnostics.py.
Protocol docs/first-block-causal-protocol-2026-09-10.md; main report section12
in docs/switch-pair-probe-2026-09-10.md; all features/results/norms under
runtime/type-probe-switch/first-block-causal/.

## Group input normalisation — 2026-09-10 (landed opt-in, made unconditional the same day)

User requested normalisation per embedding group after the first-block norm
inspection. Added cfg.encoder.input_normalisation=False and the row-local
SequenceInputNormalisation at the end of encoder._assemble_sequence, after
identities and before the trunk. RMS over channels, then 1+zero-init group
channel scale; 12x256 f32 parameters only when enabled. Shared full-layout
scale bank on actor/learner; both teams' histories share HISTORY_ENTITY.
Zero invalid rows, no cross-row statistics, no privileged mixing. Initial
valid-row RMS approximately1 / L2 approximately16; learned group scales can
later change the norm. Dynamics target seam unchanged. Disabled or missing
config field is exact identity without new leaves; enabling on old weights
requires explicitly seeding the new leaf, not a transparent checkpoint toggle.
No production restart, checkpoint migration or training was performed.

Frozen sensitivity: same 240 games / three whole-game splits, depths0/1/6,
18 new paired fits with original100-epoch/five-L2 protocol. Balanced accuracy
original -> normalised: depth0 .4650/.4997 -> .4675/.4987;
depth1 .3801/.3921 -> .3921/.4126; depth6 .2972/.3019 -> .2959/.3076.
Final raw accuracy .5385/.5581 -> .5345/.5668; final CE1.0974/1.0609 ->
1.0856/1.0451. Early accessibility improves modestly, final balanced verdict
is mixed, and generalisation is not repaired. These are zero-scale frozen
checkpoint results, not training with normalisation. No learned group scales
or playing-strength measurements in this pass.

Five focused numeric tests cover equalisation, bf16/f32, padding, exact off
mode, actor/learner equality, privileged isolation with positive controls,
and live scale/directional-input gradients. Four full-model abstract dtype
checks cover enabled/disabled paths. Focused Black/isort/Ruff and whitespace
checks passed. Code: rl/model/{config,encoder}.py; offline runner
rl/offline/input_normalisation_probe.py; tests/test_input_normalisation.py.
Results in main switch-pair document section13 and
runtime/type-probe-switch/input-normalisation/. No commit or data/ps changes.

**Amended later on 2026-09-10 (magnitude, placement, no flag).** The user
asked for the trunk's inputs to be the size of a regular transformer's
embedding lookup while keeping the per-group separation, and for the norm to
be unconditional. `SequenceInputNormalisation` now lives in
`rl/model/modules.py` (imported by the encoder and the trunk), takes only
`num_groups`, and has no `enabled` field; `cfg.encoder.input_normalisation`
is deleted and the unnormalised matched control no longer exists in config.
The output is the RMSNorm's own: RMS 1 per valid row, no further constant.
Gemma's `.01 * sqrt(D)` form was tried and withdrawn the same day: measured
at init on our block (width 256, lecun blocks, block-1 update RMS 0.62
independent of the input scale under pre-norm), the input-to-first-update
ratio is 1.62 at RMS 1 (PaLM's N(0, 1) table, torch's default Embedding),
0.75-1.1 for Gemma at its own widths but 0.26 with its formula transplanted
to 256, 0.10 at L2 1 (this repo's variance_scaling embeddings), and 0.25
for GPT-2 once its 1/sqrt(2L) residual shrink is counted — an embedding
std alone does not transfer across block inits, the ratio does, and RMS 1
is inside the reference band with no invented number. The trunk's
registers pass through a one-group instance (`register_norm`), so nothing
enters at a magnitude of its own; a test pins that instance equal to the
full-layout bank at init. The frozen sensitivity table above was read at
this same RMS-1 scale.

## Four internal registers and normalised new-lineage defaults — 2026-09-10

User requested four ViT-style register tokens, then explicitly selected input
normalisation ON by default for a new lineage. Current model factory defaults:
encoder.input_normalisation=True; encoder.trunk.num_registers=4. False/zero
retain controls. Missing fields in legacy config objects retain False/zero.
No run was launched, checkpoint rewritten or existing lineage restarted.

Registers are four independent learned f32 vectors, normal std0.02 (untuned).
They are appended INSIDE Trunk, mixed through every block and discarded before
returning to heads/transition. Always valid, refreshed from parameters per
forward, no recurrent carry or direct target/readout. External layout remains
80 learner /73 actor; internal attention84/77. No protocol, head index, dynamic
target or transition/search sequence-size change. All original rows may read
registers; registers read only original columns readable by ALL original queries
(policy-observable rows in the current mask). Thus no secret->register->policy
route. Registers may read each other. Their initial distinct vectors break the
permutation symmetry of identical workspace tokens; no position bias is added.

When input normalisation is enabled, the register group has its own RMSNorm
shared-channel scale, effective1 at init, before entering the residual stream.
Adds1024 embedding parameters plus256 norm parameters at width256. Normalisation
reduces forward dependence on initial magnitude, but parameter-scale gradients,
relative optimiser updates, vector directions and epsilon near zero still matter.
The normaliser and register initialisation have not been compared in training.

The regular audited checkpoint merge can preserve original block leaves and
seed new register leaves. This changes behaviour even if old blocks are retained;
it is not a bit-identical resume claim. Implementation rl/model/{config,encoder,
trunk}.py; tests/test_register_tokens.py. Reference arXiv2309.16588; local main
switch-pair document section14. Prior section13 frozen normalisation measurements
remain the no-register checkpoint experiment, not evidence for this combination.

## Input-norm panels and the ckpt_02339569 merge audit — 2026-09-10

The live twin of the offline norm table: `trunk.group_row_l2` (per-step sum
of valid-row L2 and valid-row count per SequenceGroup, einsum at HIGHEST —
the default f32 einsum is TF32 on the GPU) rides out of `get_head_outputs`
as two learner-only fields and lands as `player_trunk_out_row_l2_<group>`
(every valid row of every valid step weighted once). Rows enter at RMS 1 =
L2 16 at width 256, so a group reading ~16 at the output is one the blocks
do not write; before the norm the history rows read ~1040 in and out.
Beside it `player_input_norm_scale_rms_<group>` (each group's row of the
zero-init channel scale — the only route by which the input disparity can
return) and `player_trunk_register_rms` / `_register_norm_scale_rms` on
the trunk drift panel. Panels: "Trunk output row L2 per group" (log y),
"Input norm group scale: drift from zero"; views re-saved.

Launch check for the relaunch from ckpt_02339569 (learner stopped there):
the manifest carries no register/norm field, so a checkpoint-mode resume
passes `check_manifest` strict and the by-path merge inits the new leaves
fresh. Audited offline with `merge_params(eval_shape(fresh), loaded)`:
kept fresh `encoder/input_normalisation`, `encoder/trunk/register_embeddings`,
`encoder/trunk/register_norm` AND — un-asked-for — `transition/dynamics_blocks/
register_embeddings` + `register_norm`, because `cfg.transition.block` is a
copy of `cfg.encoder.trunk` and inherited `num_registers = 4`. Set
`cfg.transition.block.num_registers = 0` explicitly (the registers are the
encoder trunk's workspace; giving g four of its own is a separate decision).
Nothing dropped. Config is rebuilt from code on resume, not from the
checkpoint's copy, so the defaults are what the code says: `num_registers`
4 on the encoder trunk, the norm unconditional. Expect the resume log line
`kept fresh init: 3 subtrees` naming exactly those three.


## Actor/learner parity at the normalised trunk boundary — 2026-09-10

After adding RMS-1 group input normalisation and four registers, the fresh-model
full actor/learner parity test failed despite f32/highest-precision forwards:
maximum entropy difference 1.25599, value-logit difference 0.232134. Input rows
agreed exactly and standalone encoder/trunk calls agreed within 9.1e-6; history
inputs agreed within 3.3e-7. Returning pre-trunk rows from a diagnostic forward
restored agreement. A barrier after the encoder did not help; an identity
`jax.lax.optimization_barrier` on assembled rows immediately before the trunk
reduced diagnostic entropy/value differences to 2.03e-6. Added at
`Encoder._batched_forward`; remove that line to reproduce the unprotected seam.
The exact compiler transformation remains unidentified. No tolerance or mask
was relaxed. Jitted reverse-mode identity was checked separately.

Final focused GPU validation: 18 passed across actor-sequence, register-token,
input-normalisation and privileged-partition tests, including the formerly
failing full-model comparison and f32/bf16 register gradients. Focused Ruff
passed. Diagnostics/log: runtime/register-validation/final-tests.log; main
report: docs/switch-pair-probe-2026-09-10.md, section 14. Historical checkpoint
impact, bf16 full actor/learner parity and throughput impact were not established.
No training restart or checkpoint rewrite was performed for this fix.


### Follow-up: XLA nested-concatenation root cause and upstream report — 2026-09-10

Supersedes the earlier unidentified-transformation diagnosis. The failing
Triton fusion combines a 73-row validity-mask concatenation with normalisation
and a final 77-row register append. Generated LLVM decodes the outer block ID
with divisor/modulus 77 but nested mask indices with 73, corrupting masking
after the first time tile. Extracted HLO independently fails against NumPy;
a ~45-line NumPy/JAX reproducer has max error 4.51070 vs 7.15e-7 with the barrier.
XLA's own Triton numerical verifier rejects it. Posted with user authorisation:
https://github.com/jax-ml/jax/issues/40588. RTX 3080 Ti, JAX/jaxlib 0.10.2.

Matched full-actor barrier-off/on controls: norm+4 registers entropy/value
maxima 1.25599/.232134; norm+0 registers 2.38e-6/2.38e-6; no norm+4 registers
7.15e-7/2.15e-6. The mathematical normalisation is sound; this combination
triggers faulty nested-concatenation code generation. Keep the local barrier;
neither the broad flags from related #39486 nor disabling a pass named
triton-fusion removed the failing kernel in the extracted control.

Impact qualification: minimal bf16 also fails, and failure onset varies with
chosen tile size. However the tested full bf16 learner forward has identical
policy entropy/value logits with/without the barrier; only five reduction
metric leaves differ (<=6.10e-5). Original failing actor test ran GPU f32 at
T=58; deployed actors default CPU. No historical training damage established,
and train-step gradients/other learner shapes remain unaudited. Details in
main switch-pair doc section 15; minimal example and HLO/LLVM evidence under
runtime/register-validation/hlo-investigation/. No restart or upgrade.
