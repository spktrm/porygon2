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

## History recurrence: the stacked associative form beside the loop — 2026-09-18

*(live behind `cfg.encoder.history_recurrence`, default `loop`; one survives
the ablation)* Commit `a4e632d`, plan
`~/.claude/plans/can-you-plan-all-quiet-horizon.md`.

**Problem.** `player_history_step_attn_grad_norm` 22 at 19k on z1o4bx1m (12 on
5esmkxl1) against the global clip of 10 — every module's update scaled by
~0.26 until the write gate opened (0.03 → 0.10 at ~30k). Mechanism: the step
attention's parameters are shared across every step AND read the memory they
feed, so their gradient is a coherent sum over the memory window (~50 steps at
retain 0.982) times the loop gain. The same loop was chaotic at init (09-13,
held by `HISTORY_RETAIN_BIAS = 4.0`), and `test_suffix_carry_replays_the_game_
within_bf16` fails on it at 0.052–0.066 against the 0.05 bound.

**The stacked form.** Layer 1: an input-gated associative scan (minGRU,
`GatedLinearCell` + `gated_linear_scan` restored from `c9a96c6`) over the 15
event rows. Attention: the step attention reads layer 1's states for all H
steps at once, addressed by the identities as before; the register rows carry
no state at the read, so their row is the learned register identity — a FIXED
QUERY, masked out of the keys (the 09-13 assessment declined register keys) —
a latent history-summary token. Layer 2: the same scan over event rows plus the
attended result, all 19 rows. No attention reads its own layer's memory: both
scans are `lax.associative_scan` (depth O(log H)), there is no loop to
contract, and the form runs with NO retain bias. Attention between hidden
states survives — layer 2 accumulates a read over layer-1 states that already
integrate every event up to the step — what is gone is the same-layer
memory → attention → memory loop.

**Measured at landing.** The `loop` arm is the old encoder: f32 forward
bit-identical to `9af2d58` on the ex.bin game (bf16 one ulp apart at step 0,
XLA refusing the restructured program). The `stacked` arm PASSES the carry
suite including the 0.05 replay bound, with no bias. Registers: a row's
layer-2 memory moves only that row at step 0 (isolation), its layer-1 memory
moves every group (the attention still reads across rows); registers are
queries only, a slot's layer-1 memory reaches the registers. Test-side:
f32 dots at the GPU default (TF32) make the associative scan over a prefix
and over the full window differ at ~2e-4 — the contracts run under
`jax.default_matmul_precision("float32")`.

**Ablation (Step 3), pre-registered.** `rl/offline/train.py --joint
--fresh-subtrees encoder/history_encoder --trunk-ckpt ckpts/gen9/ckpt_00036634
--history-recurrence {loop,stacked} --num-steps 20000`, seeds 0 and 1,
sequential (the decoded store needs ~23 GB), logs `runtime/ablation-history/`.
Deciding number, AMENDED before any comparison was read (the loop arm's own
eval at 5k showed why): Δ = eval_nats_per_token(stacked) − (loop) at 20k,
with eval_loss_flow beside it, against the loop arm's seed spread s: Δ ≤ s →
stacked wins; Δ > s → Δ nats is what the loop was worth, read beside the
actor-forward throughput and the divergence probe. `eval_loss_public_value`
was the plan's primary and is demoted: it sits at ln 2 (0.686 at 5k, R² 0.014)
for the offline public critic in EVERY run to date (heydhats 30-45k: -0.03 to
+0.01; LESSONS 09-16: .016 at 3k), so it cannot separate the arms.

**The offline public critic's pooled R² is near its ceiling — the claim
"the critic path is defective" is WITHDRAWN (second session, 2026-09-18).**
`critic_terms` pools one outcome label over every event step of a game (up to
512), and most of those steps are early-game states nobody can call. A
four-number logistic HP/faint-difference reader fitted on 400 games of shard
0 (held out every fifth game) reads, on held-out slices: every event step
(the trainer's weighting) loss 0.645 / R² 0.087; final step only 0.242 /
0.751; first half 0.687 / 0.012; second half 0.603 / 0.161. So the pooled
number's ceiling from raw HP is ~0.09 and the final-step 0.75 (my label check:
final-state potential r 0.865 with the outcome; the two perspectives of a
record carry opposite outcomes) is a different quantity. The joint loop-s0 arm
evaluates at 0.658 / 0.066 — about level with that reader; its train loss is
~0.644 from step 0 (the restored learner head already reads replay states
that well) and 20k joint steps moved held-out loss only 0.686 → 0.658. The
FROZEN-trunk runs (R² −0.03 to 0.02) sit BELOW the HP reader — that shortfall
is real and open. Checked in the path: label indexing per trajectory under
lax.map, inputs (the two perspectives' public caches differ only in SIDE and
about half the field rows), loss weight 1.0, the head trainable. Snapshots are
LATEST-touch (the 09-16 candidate list above calls them first-touch — stale).
Open: whether the model beats the HP reader per phase (final step vs 0.75,
second half vs 0.16) — needs phase-bucketed residuals in `critic_terms`, a
trainer change parked until the ablation's arms have all launched; and whether
batch size 1 (one game, one label sign, its mirror next) slows Adam on the head
(unmeasured; batch-8 runs scored no better). Baseline script: the second
session's scratchpad `ceiling.py`. The train-side `public_value_r2*` panels
are additionally a per-batch artefact (batch = one game, outcome variance 0 →
R² ≈ −1e8) and should pool like the eval does. The phase buckets landed
2026-09-18 evening (`critic_buckets`, b58aad4): first half / second half /
final step beside the turn-boundary split, each R² against its own bucket's
outcome variance — the per-phase read against the HP reader is now one
offline run away.

**VERDICT (Step 3 read, 2026-09-18 18:30) — stacked wins on the
pre-registered rule.** Eval at 20k, four arms:

| | loop s0 | loop s1 | stacked s0 | stacked s1 |
|---|---|---|---|---|
| nats/token (deciding) | 0.300 | 0.256 | 0.331 | 0.244 |
| loss_flow | 0.010 | 0.010 | 0.006 | 0.010 |
| public value R² | 0.064 | 0.098 | 0.083 | 0.106 |
| imagined R² | 0.924 | 0.886 | 0.896 | 0.865 |
| train nats/token | 0.295 | 0.313 | 0.265 | 0.343 |
| wall / 20k, H 512 | 42 min* | 33 min | 29 min | 22 min |

Δ = mean(stacked) − mean(loop) = 0.2875 − 0.278 = +0.010 nats against a loop
seed spread s = 0.044 → Δ ≤ s. Seed noise (0.044 loop, 0.087 stacked) is 4-8x
the form difference: the honest reading is "nothing measurable lost at this
budget", not "stacked is better". Public R² leans stacked (0.095 vs 0.081
mean), imagined R² leans loop (0.880 vs 0.905), flow is a tie. *loop-s0
overlapped a GPU diagnostic; the s1 pair is the clean throughput read, 1.5x.

Actor-forward bench (`rl/probes/history_bench.py`, bf16, fresh params,
median of 50, `runtime/ablation-history/bench.json`): H256 B1 20.0 → 1.66 ms
(12x), B16 28.5 → 7.65 (3.7x); H512 B1 37.4 → 2.02 ms (18x), B16 54.5 → 12.5
(4.4x). The loop's cost is sequential in H (doubling H doubles it); the
stacked form's barely moves with H and scales with B. Stacked compiles in
8-14 s per shape against the loop's cache hit — one-off.

Carry replay divergence on each arm's TRAINED encoder
(`PORYGON_TEST_CARRY_CKPT`, ex.bin game, worst policy / value log-prob diff
of carry-resumed vs full-window, bound 0.05): loop-s0 0.0208 / 0.0175;
stacked-s1 0.0144 / 0.0125 — both in class, the stacked with NO retain bias
(decision (c) of the plan, done by construction). Fresh params: stacked
passes, loop fails at value 0.0521 (the pre-existing 09-13 failure).

**Two findings the rule does not score, recorded before the deletion:**

1. *The stacked-trained trunk reads less of the long history.* Policy
   log-prob diff between the suffix-alone forward (last 32 steps, no carry)
   and the full-window forward, per request over the ex.bin game: loop-s0
   mean 0.39 / max 1.63; stacked-s1 mean 0.069 / max 0.22 — 5x less
   dependence on events older than the suffix, at the same event loss. Two
   readings, not separated: the loop's chaotic map amplifies any difference
   in memory content into O(1) output differences whether or not it carries
   predictive information (equal nats/token says the extra sensitivity
   bought no prediction); or the stacked form's long memory is harder to
   read as trained. The launch acceptance's history-attention panels and the
   100k pair probe are the next instruments.
2. *The stacked states are unbounded and large.* At 20k offline steps the
   carried layer-1 states read RMS 8.6 (max 60), layer-2 slot states RMS 4.8
   (max 29), against the loop's tanh-bounded (−1, 1). Harmless to the forward
   (`input_norm` before the attention, `SequenceNormalisation` at the trunk
   door) and to the f32 carry; it is why the carry test's state check and
   its control went RELATIVE (58cae1a): a fixed +1 shift moved the trained
   stacked policy 0.011 at the last request — a nudge on an RMS-9 state, not
   a deaf network. A bound on the cell's candidate (the layer is a convex
   combination, so |state| ≤ max |candidate|) is one line if it ever matters.

## Removal ledger — 2026-09-18 loop recurrence (NEW LINEAGE — param tree changes)

Tag `pre-history-loop-removal-2026-09-18`. The stacked form is the one
history recurrence after the ablation above; the loop and its knob go
together (a flag with no meaningful off is a comment with a runtime cost).
Every `history_encoder/sequence_step/*_gru` leaf is gone and
`initial_inner_memory` + `*_inner_cell` / `*_cell` are new, so no checkpoint
before this commit loads through `merge_params` with a trained history
encoder: scratch lineage.

| mechanism | why | revert |
|---|---|---|
| `HistoryGRUCell` (reset-after GRU, `{ir,hr,iz,hz,in,hn}` per token type) and the loop scan body `HistorySequenceStep.__call__(memory, inputs)` under `nn.scan(nn.remat(...))` | the memory-in-the-loop recurrence: chaotic without a retain bias (09-13), attention gradient a coherent sum over the memory window (the 22 / 12 grad hump), sequential in H (20 → 37 ms per doubling); the ablation priced it at +0.010 nats inside a 0.044 seed spread | `git show pre-history-loop-removal-2026-09-18:rl/model/history_encoder.py` |
| `HISTORY_RETAIN_BIAS = 4.0` | the constant that held the loop's chaos down; the stacked form replays the carry within bound with none (0.014 / 0.013 trained) | same |
| `cfg.encoder.history_recurrence`, `player_history_recurrence`, `Porygon2OfflineConfig.history_recurrence`, `--history-recurrence`, `recurrence_form()`, `step_key_mask(form)` → `STEP_KEY_MASK`, `invalid_history_carry(width, form)` → `(width)` | one form survives; the offline run name drops the form | same, plus `rl/offline/train.py`, `rl/online/{config,artifact,main}.py` at the tag |
| `player_history_slot_gate_rms` telemetry leaf (the GRU `iz`/`hz` kernels) | `player_history_slot_write_gate_rms` (the `entity_cell/gate` kernel) is the one gate panel; `scripts/wandb_views.py` follows | same |
| `tests/test_history_gru.py` | three of its contracts ported to `tests/test_history_cell.py` against the stacked step (norm and identities reach memory only through attention; cell weights separate by type, shared within type; identity changes reach memory only through the attention weights), the GRU-vs-Flax-reference and retain-bias tests deleted with the cell | same |
| `PORYGON_TEST_HISTORY_RECURRENCE`, `SESSION_FORM`, the `FORMS` fixture params | the test suite runs one form | same |

## Removal ledger — 2026-09-18 history tidy (structure-only, bit-identical)

Tag `pre-history-tidy-2026-09-18`. The ex.bin forward with z1o4bx1m's
ckpt_00036634 params before and after: log_policy, value and public-value
log-probs `np.array_equal` (run-to-run floor 0.0 — the GPU `segment_sum`
scatter is exact on this data). Plan: local
`~/.claude/plans/can-you-plan-all-quiet-horizon.md`.

| mechanism | why | revert |
|---|---|---|
| `HistoryAttentionPool` (`latent_queries`, `latent_cross`) + `Encoder.pool_history`, `cfg.encoder.history_pool` | the offline critic's pooled-latent probe; that critic was deleted 2026-09-16; instantiated in setup but never applied, so no params existed | `git checkout pre-history-tidy-2026-09-18 -- rl/model/history_encoder.py rl/model/encoder.py rl/model/config.py` |
| `NodeHistoryRead` (`diary_cross`, zero-init gate) + `Encoder.read_history_into_nodes` | the "photos query the diaries" read from before the flat trunk; no caller | same |
| `Encoder.history_slot_sides` | no caller | same |
| `PerSlotHistoryOutput.step_touched`, `.step_row_mask` | written, never read / a constant all-True whose two `&` were no-ops (`step_valid` stands in) | same |
| `_run_history_encoder`'s `edge_slot_ids`, `node_sides` returns | discarded by both callers | same |
| `node_content_cache` argument | always `== node_embedding_cache` | same |
| the `None` branches for `node_identity_cache` / `field_identities`, the second `scatter_step` vmap over the identity cache | one path, one scatter (content and identity concatenated, split after); tests pass zeros where they passed None | same |
| `resolve_initial`'s `register_states` tuple branch | unreachable — every constructor sets registers | same |
| `NUM_FIELD_ROWS` / `FIELD_ROW_*` / `SIDE_MINE` duplicates in history_encoder.py | `constants.py` / `identity.py` own them | same |

Side reading while here: `tests/test_history_carry.py::test_suffix_carry_
replays_the_game_within_bf16` FAILS on the tagged baseline too — the carry-vs-
window log-policy divergence reads 0.066 (0.051 after the tidy, same code
path, GEMM noise) against the 0.05 bound, on the current 19-row memory-in-the-
loop form with retain bias 4.0. LESSONS 09-13 set the bias on a 0.0215 slot-
memory read; the policy-level bound is not held on this lineage. This is the
number the stacked recurrence (plan Step 2) has to bring under the bound with
no bias at all.

## Switch logit over both teams from the public rows; every layer named — 2026-09-18

*(live, new lineage)* Two commits. `31f0af9`: every flax layer named
(memory `parameter-naming-convention`; the table is in the plan doc) — not
loadable against earlier checkpoints, so a scratch start. `b573d87`: the
switch block pairs the candidate with every mon on the field from the PUBLIC
rows — my active it replaces, my other active if present (doubles), and the
opponent's TEAM under a learned belief about who stands opposite next turn
(`OpponentTeamPairing`); the move block adds the same opponent-team term per
move, and its every-turn gradient trains the shared keys the one-in-eight
switch block reads through. Presence, life and being on the field come from
each entity's own public row, never the action mask, so the terms are live on
a forced switch and at team preview, where no enemy TARGET row is valid.

**Why (the 2026-09-10 probes, still the evidence).** The trunk computes
move-vs-opponent into the move row (post-trunk matchup read 0.66) and nothing
of candidate-vs-opponent into the sheet row (0.63 pre-trunk, 0.54 = floor post,
at every depth): the move readout multiplies the move row against the row it
would hit, the switch readout never had the opponent's row. The 09-11 pair
(sheet x the ally row it replaces) reached the opponent only through what
that row attended to; unmeasured on the lineage it ran on.

**Pre-registered.** Offline: `type_probe.py` switch-pair block +
`switch_readout_probe.py` on the 240-game heuristic cohort — the post-trunk
candidate x ENEMY_1 read off the 0.54 floor toward the raw pair's 0.63 by
100k; read beside `player_switch_opponent_query_rms`: flat while
`player_move_opponent_query_rms` climbs = the switch signal is not arriving
(credit is the bound, not routing). Strength: the advantage audit's
switch-minus-stay realised-outcome gap at matched horizon narrows from ~-0.07
toward 0 by 200k; switch mass >= 0.05; T=1 win rate >= the 5esmkxl1 curve at
matched lifetime steps (0.154 at 100k, 0.165 at 200k), confounded by the
fresh start and the entropy setting. Fallback if the read moves but the gap
does not: hypothetical-state evaluation (assemble the observation with
candidate c active). If the read does not move: revert to the 09-11 form and
reopen the depth probe on this lineage.

**Declined.** Sum over the four active TARGET rows (the plan's first draft:
invalid on forced switches and at preview). Sharing the opponent key with the
move x target pair (different rows post-trunk). Pairing against my own bench
(no dense gradient, the belief has no meaning on my side). Hand-coded
matchup features (game effects change them).

## PBRS information screen: the service potential adds ~nothing the critic lacks — 2026-09-18

`rl/probes/potential_information.py` on `ckpt_00320354` (the normalised-residual
run, entropy off), 32 self-play games at T=1, 1,993 decision rows. Per row:
$\Psi_t$, the $\lambda$-return (0.95, $\gamma$ 1) of the shaping rewards
$\Phi_{t+1} - \Phi_t$ over the rest of the game — what the channel adds to the
advantage at launch, times $\eta$ — against the critic's error to the realised
outcome $e_t = G - V(s_t)$. Output `runtime/pbrs/info_00320354.json` (local).

| rows | n | corr($\Psi$, e) | variance explained | $\eta^*$ | MSE ratio at $\eta$ .05 | at $\eta^*$ | mean $\Psi$ |
|---|---|---|---|---|---|---|---|
| all | 1993 | .14 | 1.9% | .36 | .995 | .981 | +.006 |
| real choice | 1546 | .18 | 3.3% | .51 | .995 | .976 | -.042 |
| taken switch | 128 | .27 | 7.4% | .86 | .994 | .998 | -.174 |
| stayed | 1418 | .17 | 3.0% | .49 | .995 | .976 | -.030 |

**Verdict: do not launch with this $\Phi$.** The potential's trace explains 2-3%
of the critic's residual variance (7% on switch rows, n = 128); at the screened
$\eta$ = .05 the shaped estimate is 0.5% better in squared error than the
critic alone, and the best $\eta$ (~.4, ten times the perturbation budget) buys
2%. Directionally it would move the switching problem the WRONG way: the HP /
alive balance falls over the turns after a voluntary switch (mean $\Psi$ -.17
on switch rows against -.03 on stays), so the channel would subtract ~.007 from
switch advantages relative to stays, a tenth of the audit's -.07 gap, in the
same direction. With $\gamma$ 1 the shaping telescopes, so this is not a
per-turn damage credit; it is the potential's read of the position at the trace
horizon, and after a switch that read is "less HP".

Side reading: the critic is pessimistic by .13 on these positions (mean e
+.13; mirror games average G = 0), while the online audit reads it optimistic
at long horizons — the online rows are league games, these are mirror.

**What would change the verdict.** A $\Phi$ that sees position rather than
HP: the offline trunk critic (a frozen forward, ~30% on the step, or an actor-
side forward) — the screen is one flag away from reading it, and should be
rerun on it before any plumbing.

## Entropy bonus off — 2026-09-18

*(live)* `player_ent_coef` 0.01 -> 0 on the normalised-residual lineage
(5esmkxl1), from `ckpt_00303473`. Floor (`player_uniform_kl_coef` 0.01) and
magnet (0.025) unchanged.

**Problem.** The policy stopped sharpening at ~60k: normalised action entropy
0.80 -> 0.76 and within-modality entropy 0.90 -> 0.84 over 60k-290k, win rate
at T=1 0.15 -> 0.19, against the control ijk4nyi4's 0.84 -> 0.63 entropy over
the same steps (bought with a switch-mass collapse to 0.016).

**Diagnosis, measured.** New panels `player_{sharpen,move_sharpen}_logit_grad_*`
(commit `d1aee9e`): the loss's gradient along "scale the logits about their
policy mean", split by term, coefficients included, positive = descent
flattens. Means over 12k steps from 291k (standard error of the pg mean
0.0002-0.0003, the others under 0.00003):

| term | all legal cells | among legal moves | switch direction |
|---|---|---|---|
| pg | -0.0160 | -0.0064 | +0.0028 |
| uniform-KL floor (0.01) | +0.0103 | +0.0028 | -0.0025 |
| entropy (0.01) | +0.0057 | +0.0025 | -0.0012 |
| magnet (0.025) | +0.0016 | +0.0011 | +0.0002 |
| net | +0.0015 | -0.00002 | -0.0007 |

The policy sits at a regularised equilibrium: among moves the three
regularisers cancel the policy gradient to the fourth decimal, the two halves
of the window agreeing. Sharpness is set by the ratio of the advantage signal
to the coefficients, not by training time. An earlier back-of-envelope in the
session ("a 0.01 floor is too weak to flatten moves") was wrong — the floor is
the largest single flattening force on the whole policy.

**Why entropy and not the floor.** Entropy's logit force is
$-c\,\pi_i(\log\pi_i + H)$: it carries the action's own mass, so it vanishes
on a dying action and cannot hold a floor (section 4, and the 08-31 ledger's
algebra; yhnfmjc7 ran entropy 0.01 with no floor and closed on switching at
0.005). The floor's force is $c\,(1/N - \pi_i)$. Per unit of flattening among
moves the floor holds switching up 0.0025/0.0028 = 0.88 against entropy's
0.0012/0.0025 = 0.47. Entropy came with the APPO reference, which has no floor;
the floor was added because entropy failed at the one job that matters here.

**Not the cause (measured the same day).** Replay overfitting: first-use vs
replayed value squared error 0.106 vs 0.096 at 280k (5% apart at reuse 4, ~12%
at reuse 8, steady; control the same), 0.106 vs 0.107 over 291k-303k; pooled
first-use value R2 0.72 (38 windows, 0.58-0.82). The normalised trunk: in/out
cosine 0.5-0.65 on every group, the control equally flat at matched steps to
60k. Reuse: the control also ran at cap ~7.8.

**Expected.** Flattening among moves falls ~39% (0.0064 -> 0.0039), switch
support ~32% (0.0037 -> 0.0025). The floor's own band for switch mass is
0.03-0.10 and the run sits at 0.107, so the floor is NOT raised to compensate.

**Acceptance, pre-registered, read at +20k and held to +40k.**
`player_entropy_micro_taken` below 0.82 (from 0.842) and
`player_move_sharpen_logit_grad_actor_total` back within 0.0005 of zero at the
new level; `player_switch_mass_choice` >= 0.05; T=1 win rate not below 0.17
(uninterrupted 200-game average). Control: this run's own 291k-303k window and
the banked ijk4nyi4 curve.

**Fallback.** Switch mass under 0.05: `player_uniform_kl_coef` 0.01 -> 0.015,
which restores the 0.0037 switch support at ~17% less move flattening than
today. Entropy is not restored — it is the inefficient half of that job.

**Declined.** Cutting the floor (it is what holds switching; three lineages
collapsed without it). Cutting the magnet first (smallest flattening term, and
it is the self-play anti-cycling piece). Lower reuse (no overfit gap to fix).

Revert: `player_ent_coef = 0.01`.

## Replay controller: actor KL 0.045 -> ESS floor 0.75 — 2026-09-17

*(live)* The reuse-cap PI loop now holds `player_learner_actor_ess` (normalised
effective sample size of the live-learner / behaviour importance ratios on the
replayed batch) above `player_replay_ess_floor = 0.75`, in place of holding
`player_learner_actor_forward_kl` under `player_replay_kl_target = 0.045`.
Still one-sided: it cuts reuse below the nominal 8 and recovers to it, never
above. Error is the lost fraction against its ceiling,
$(\mathrm{ESS} - f)/(1 - f)$, the same normalised form the KL error had, so the
gains carry over unchanged.

**Where 0.045 came from.** $\varepsilon^2/2$ at the July 2026 trust-region clip
$\varepsilon = 0.3$: for close policies $\mathrm{KL} \approx
\tfrac12\,\mathbb{E}[(r-1)^2]$, so 0.045 was the KL at which the typical
replayed sample sat on the clip edge and had its gradient zeroed. The
2026-09-14 provenance check (below, "Exact target provenance") found no
derivation in commit `ae2c1c3`; the ruler was recorded only in a session note.
It was never calibrated against strength.

**Why it stopped meaning anything.** The clip it was derived from is gone: SPO's
quadratic at eps 0.4 has an optimum at the band edge, not a zeroed gradient.
The sampled forward KL $(r-1) - \log r$ is driven by $r \to 0$ (actions the
learner has since dropped, which merely carry a small weight), while the rows
that distort the estimator are the LARGE ratios that v-trace truncates and the
behaviour-ratio clip at 2 caps; ESS $= \mathbb{E}[r]^2/\mathbb{E}[r^2]$ is
driven by exactly those, and is bounded, so one batch cannot throw a 3x error at
the loop the way the KL's 0.13 spikes did. For close policies the two carry the
same information, $\mathrm{ESS} \approx 1/(1 + 2\,\mathrm{KL})$: 0.045 is an
ESS floor of ~0.92.

**Reference numbers, normalised-residual run, 35k-71k, under the KL ceiling.**
Actor forward KL 0.03-0.07 with spikes to 0.11-0.13; the controller held the
cap at 4 of the nominal 8; `player_learner_actor_ess` 0.89-0.95;
`player_learner_actor_ratio_tail_gt2` 0-2%. The loop was halving reuse to
protect a ~10% effective-sample loss. Earlier lineages recorded actor KL
0.005-0.006 (section 6); behaviour age is still not logged per chunk, so
policy speed and lag are not separable as the cause of the 10x.

**0.75 is a judgement call, not a calibrated boundary** (user, 2026-09-17). The
accounting reuse x ESS puts the break-even for halving reuse near ESS 0.5; it
ignores truncation bias, the falling worth of a repeated pass over the same
chunk and the optimism of batch ESS under heavy tails, all of which argue
higher. The July plateau (capacity 2048, age ~2048 steps) has no recorded ESS,
so the one known failure does not calibrate it. The grounded number is owed:
strength-per-step against realised reuse across banked runs.

Expected on relaunch: ESS near 0.9 is above the floor, so the cap recovers to 8
(the PI state is not checkpointed — it restarts at nominal) and stays there
unless ESS at reuse 8 falls under 0.75. Watch `player_replay_max_reuses`,
`player_learner_actor_ess`, `player_learner_actor_ratio_tail_gt2`; actor KL
stays on its panel as a smoke alarm, not a set-point.

Revert: `git revert` this commit restores the KL signal and
`player_replay_kl_target`.

## Offline export sliced per history edge; batch 1 — 2026-09-16

**Export (a678310).** `service/src/scripts/offlineWorker.ts` now emits one
EnvironmentState per committed history edge instead of one per `|turn|`. An
edge commits inside the handler of the NEXT major arg (`|move|`, `|switch|`,
`|drag|`, `|replace|`, `|cant|`, `|faint|`) or of the bare `|` block separator
(the protocol's `|done|`, which follows every action's effect lines), or of
`|turn|`; the worker takes the slice BEFORE the committing line (after it the
state would already announce event k+1's move: a leak), labels it with the
index of the edge whose effects it holds (`INFO_FEATURE__HISTORY_STEP_COUNT`,
1-based; the feature was the windowed length and nothing in `rl/` read it), and
takes the turn boundary's slice AFTER the `|turn|` line as the live request is,
replacing a same-position slice; the terminal state stands for the last edge.
`requestCount` still advances once per turn so the edge features keep the live
distribution. Verified: 40 trajectories, labels exactly 1..N with the terminal
last. Corpus: 98,512 trajectories, **8,110,894 states = one per history step**
(2,760,044 per-turn before), 23 GB (9.3 before), 7 min at 4 workers, 4 replays
failed; the pool decodes it in ~20 s on 4 workers (the trainer reads only the
terminal state, so the store is unchanged at 3 GB). The per-turn export is at
`replays/shards/gen9randombattle-turnslices-20260916` (not deleted).

What an edge holds (traced on gen9randombattle-1090): edge k's block runs from
the line after the previous commit to the line before the k-th committing
line, so a `|turn|` line and the `|t:|` timestamp fold into the FOLLOWING
edge (the first move of the turn), and a RESIDUAL edge is a genuine
end-of-turn block (`|` … `|-status|brn` … `|upkeep`), not an artefact.

**Why:** the world model's per-event states were rebuilt from first-touch cache
snapshots (pre-effect for 10% of rows at a boundary); the exact slice is the
clean target and unifies the offline and online state construction. Not yet
consumed: the trainer's state source is still the scan's snapshots; switching
it is the next step, with the Step 1 parity test as the check.

**Batch 1 (31aac71).** One trajectory per step: the geometric bucket is the
game's own length (no padding to the longest of eight) and the step is a
sequential `lax.map` over trajectories anyway. Measured on the same trunk and
data: batch 8 0.46 s/step (2.2 updates/s, 17 trajectories/s); batch 1
**0.075 s/step at 99–100% GPU (13 updates/s, 13 trajectories/s)** — six
times the update rate at three quarters of the throughput. The first minutes
read 6% GPU: one compile per bucket shape (four at batch 1 where batch 8
always hit 512) plus the concurrent re-export's four workers on the host. The
epoch is the shuffled games flattened to trajectories (the game-wise cut,
truncated to the batch, would have trained one perspective only at batch 1).
Run heydhats resumed from the batch-8 run's step-3000 best (`ckpt_best_batch8_
step3000`); intervals rescaled (240k steps, eval every 5000 over 256).

---

## Removal ledger — 2026-09-16 offline tidy

Everything below existed at tag **`pre-offline-tidy-2026-09-16`** (commit
`06e3184`); `git checkout pre-offline-tidy-2026-09-16 -- <path>` brings a file
back. The pass was structure-only: the world-model trainer's per-batch terms on
a fixed batch and fixed params (ckpt_00373138) were dumped before the tag and
compared after each commit — 277 arrays, 0 differences, three times.

| mechanism | paths / symbols deleted | removed in | why it went |
|---|---|---|---|
| Separate offline critic architecture | `rl/offline/model.py` (`Porygon2OfflineCritic`: antisymmetric margin probe, survival / next-action / unseen-hazard / revealed-set heads, rating embed, `RelationalRounds`), `rl/offline/train.py` (old: losses, pair batching, ensemble, `_overlay_params`), `rl/offline/artifact.py` (potential Φ loader + uncertainty gate), the label two-thirds of `rl/offline/dataset.py`, `Porygon2OfflineConfig` (old), Makefile `ensemble` target, the wandb "Critic health" view | the tidy's second commit | no runtime consumer (the potential channel reads the service fit; `offline_critic_ckpt_path` was gone); no longer constructed against the 3-row field state; the world-model trainer already trained `public_v_head` on replays, i.e. the critic on the live model |
| `_overlay_params` | `rl/offline/train.py` (old) | same | duplicated `rl/online/artifact.py::merge_params` with union semantics (it ADDED checkpoint-only leaves — the trainer's tree carried the learner's private-path leaves); `overlay_whole` in the trainer wraps `merge_params` and refuses a loaded subtree that did not land whole |
| Decoded event-stream layer | `replays/shards/<format>/decoded/*.npz` + manifest, `rl/offline/event_stream.py` (`convert`, `EventStreamStore`), `rl/offline/world_model_data.py` (`WorldModelDataset`, the dead `shuffle_buffer_size` reader) | the tidy's fourth commit | a pure function of the terminal state cached on disk: it bought a 3 s startup over ~12 s at the price of a second format, manifest and refresh rule. `rl/offline/dataset.py::load_replay_store` decodes the export at startup (8 CPU-only spawn workers, 98,512 trajectories in 12 s, 3 GB RSS) bit-identically |
| `--max-history-steps` as a dead knob | — (kept, made LIVE at the store: the trailing window re-derives the labels on the window) | same | the store ignored it |

Two paid-for lessons from the pass: (1) importing the model package
initialises CUDA (`rl/environment/data.py` builds a pretrained-embedding
device array at import), so a spawn pool whose workers import it claims the
GPU per worker — the loader sets `CUDA_VISIBLE_DEVICES=""` in the environment
the children inherit for the pool's lifetime; (2) a script without an
`if __name__ == "__main__"` guard that calls the loader is re-imported by every
spawned worker, each re-runs it, dies on the bootstrap check and is respawned
without end (37 minutes, ~16 × 0.7 GB, a host-memory crash of the editor). The
default `decode_workers` is 8 for that reason.

---

## Public event world model, Steps 3–4: the offline trainer and the search read — 2026-09-16

Step 3 (`rl/offline/world_model_data.py`, `rl/offline/train_world_model.py`,
`Porygon2WorldModelConfig`, wandb view "Event world model"):
- One example per trajectory from the TERMINAL state only (its caches are
  the whole event stream); the per-|turn| states are not decoded. The
  history clip windows the (B, H) leaves to one geometric bucket and every
  (H,) label follows the same window. Batch dataclasses are `chex`
  dataclasses -- a plain `@dataclass` is not a pytree and `jax.tree.map`
  treats it as a leaf (the first smoke failure).
- `WorldModelTrainer` names its submodules `encoder` / `public_v_head` /
  `world_model` so `merge_params` resumes them by path; the encoder is
  overlaid from the learner checkpoint and frozen by `optax.multi_transform`
  (`set_to_zero`, so adamw's weight decay never touches a frozen leaf);
  `joint` flips the encoder label. `model.apply` takes `{"params": ...}`,
  not the bare tree (the second smoke failure: "collection params is
  empty in /encoder").
- Every metric is a ratio of POOLED sums over the batch (never a mean of
  per-trajectory ratios); the flow and control losses go through
  `pooled_group_loss` on the scaled difference (copy = 1); the EMA group
  scale lives in the trainer state and is written into the
  `world_model/delta_scale` parameter at save time, a parameter in name
  only (frozen label) so the actor's params view carries it.
- Eval-only reads sample `eval_samples` imagined next states per step:
  imagined-value R2 (sample mean vs the real next value) and CRPS of the
  sampled values, beside the mean control's |error|.

Step 4 (`rl/model/event_search.py`, the binding in `player_model.py`,
`harness.play_games(search_arm=...)`, `rl/offline/search_ablation.py`):
- The rollout samples one event in five decoder passes with a growing
  teacher-forced prefix; our own execution event carries the declared
  move once (`own_executed`); a newly revealed actor slot is ours only for
  our unplayed declared switch; the stop rule is a sampled new-turn bit,
  an own-side faint with a reserve, or END; padded scan steps carry the
  state. The root's active slots stand in for the whole rollout (the
  target-row rule after an imagined switch is approximate).
- The searching actor keeps POLICY_READABLE_ROWS + PUBLIC_CLS
  (`encoder.with_public_cls`; PUBLIC_CLS reads public rows only and has
  out-degree 0, so the policy's information set is unchanged), enumerates
  legal cells statically (`max_cells` 16, overflow zeroes the bonus),
  maps a switch cell to its private row's public slot (an unrevealed mon
  is the next slot to be revealed) and a move cell to its move id, and
  adds Q / temp to the readout's legal logits; the value-blind arm runs
  the same rollouts and adds 0. Contract (test_search_binding): at init
  the flow is the copy predictor, every leaf is the root's public value,
  Q is equal across cells and the root KL is exactly 0; opening the flow
  moves it.
- The service exposes no simulator or team seeds, so the search read's
  arms are INDEPENDENT games: Wilson intervals per arm and a
  two-proportion test, 400 games per arm to resolve ~7 pp. The plan's
  "paired seeds" wording was wrong and is corrected.

## Public event world model, second run's read (louxsi55) — 2026-09-16

Held-out at 3,000 steps (batch 8 trajectories, ~1,000 events per batch,
0.3 s/step, trunk ckpt_00373138 frozen): nats per token 1.28; touched
F1 .93; new-turn accuracy .79; flow loss on touched rows 0.28 (copy = 1)
vs the mean control 0.61 on the same rows; mean step on every row 0.27;
imagined-value R2 (sample mean vs the real next value under the same
head) 0.84, faint steps 0.69; CRPS flow 0.046 vs the control's |error|
0.057. The opponent's move among its revealed moves is at 3.5% -- the
960-way move head has not yet learned to read the actor's revealed row.
`untouched_delta_frac` 0.70: the mean residual on untouched rows is
load-bearing, not a corner. The run is deterministic (seed 0, streamed
shards): a relaunch with the same config reproduces every eval to three
decimals, which is how the first run's evals were recognised as the same
model. bc26c7f's message claims the head weight dropped to .02; that
edit never landed and the weight stayed 1.0 -- and it would have been a
no-op anyway: the head's params get no other gradient, so Adam's scale
invariance makes a weight on its loss inert up to eps.

The public critic finding. The learner's public critic (R2 .59 on
self-play requests) reads R2 -0.10 with CE 1.65 on per-event replay
states -- confident and wrong -- and the replay-trained head reaches
only .016 held-out after 3k steps (its per-batch train R2 swings
-.08 .. .29 with four games per batch, which the first run misread as a
collapse). The train and eval code paths agree exactly on the same
batches (probe), so this is the model, not the instrument: the value
reads above are consistency reads against a weak critic, and the plan's
Step 3 acceptance on `imagined_value_r2` is not yet a read on true
value. Candidates, in order: the head's own LEARNING RATE (four outcome
bits per step is the noise), longer training, then `joint` so the public
tier adapts to replay event states (INFO is a MOVE request with every
target legal, the snapshot is first-touch, human games are not
self-play).

## Public event world model, first run's defects — 2026-09-16

The first trainer run (wandb 55xsk5wm, 1,000 steps on ckpt_00373138)
read: kind NLL 1.70 -> 1.17 nats (the kind marginal is ~1.4, uniform
1.95), move 4.45 -> 2.42, touched F1 .94, new-turn accuracy .86, flow
loss 0.82 -> 0.56 and mean control 0.79 -> 0.55 (copy = 1). It also
exposed four defects, all fixed before the relaunch:
- The learner checkpoint's `player/params` component is the VARIABLES
  dict `{"params": tree}`; the trainer's guarded overlay
  (`if key in restored`) matched nothing and trained on a RANDOM encoder
  with a fresh public head -- which is why `loss_public_value` sat at
  ln 2 with R2 0.00 from step 50. The overlay now indexes `["params"]`
  and RAISES if the encoder or the head does not overlay exactly; the
  artifact is saved in the same layout. Rule: a guarded overlay is a
  silent no-op waiting to happen; assert the overlay.
- `nats_per_token` read 1.3e8: labels outside the grammar mask (a move
  event whose move token is unknown, id < SWITCH_IN) scored -log 0 = 1e9.
  Such labels leave the loss and are counted (`label_illegal_{actor,
  move,target}`).
- The eval crashed pooling (H, 55) per-row sums across history buckets;
  per-row sums are now (55,) per trajectory (summed over steps).
- `untouched_delta_frac` read 0.60: 60% of the true post-trunk difference
  energy sits on rows OUTSIDE the touched set -- the trunk mixes rows and
  every slot's history state moves each step. The plan's fallback
  applied: `imagine()` adds the mean step's deterministic residual on the
  untouched rows (one pass, no denoising) and the mean step trains on
  every valid row (its loss on the update rows stays the flow's control).

Operational: `pkill`/`pgrep -f` with a pattern that appears in the tool
shell's own command line kills that shell (exit 144); kill by a
bracketed pattern (`[t]rain_...`) or by pid from a separate command.

## Public event world model, Steps 1–2: the per-event public state and the model — 2026-09-16

Step 1 (72541ee, structure-only): `PUBLIC_SEQUENCE_ROWS` = the public
tier prefix + PUBLIC_CLS (55 rows, closed under the read mask);
`Encoder.encode_events` builds the trunk input at EVERY history step
from the scan's own products (`node_snapshots` as the public rows, the
slot's side/position/fainted read off the cache row via the scan's new
`node_row_index`, the step's field rows, the slot/field/register
snapshots as the history rows; a MOVE request with every target legal --
the replay convention) through the SAME `_public_parts` /
`_finish_sequence` the request path uses. Bit-check: the encoder forward
on the bundled trajectory (f32, highest precision, GPU) is identical to
cc999c1 over all 13 output leaves -- a check that took four attempts
because the live learner held 11.5 of 12.3 GB; a bit-check needs ~0.6 GB
and a learner-free window is the reliable way to run it (the CPU
platform is hook-banned even for a paired comparison).

Step 2 (`rl/model/world_model.py`, 02fc269), decisions not in the plan text:
- Seven KINDs, not six: DRAG is its own kind (a forced switch is not the
  owner's choice, and the search must not treat a dragged own mon as our
  declared execution). REPLACE (Illusion) maps to SWITCH, DETAILSCHANGE
  to RESIDUAL.
- The declared token is KEY-ONLY at position 0 of a six-token decoder
  sequence; the self-attention mask lets KIND and ACTOR read it always
  and MOVE / TARGET / TOUCHED only when `actor_is_mine` -- one mask, one
  invariant, pinned by `test_opponent_move_query_cannot_read_the_declared_token`
  with the own-side query as the control.
- Endpoint (x1) parameterisation, not velocity: with `out_proj` at zero
  the Euler sampler's last step lands exactly on the last endpoint
  prediction, so every sample is the copy predictor bit for bit at init
  for every step count (`test_every_sample_is_the_copy_predictor_at_init`).
  A velocity head at zero would leave the sample at the noise.
- The Euler loop is a LIFTED `nn.scan` over a function of the module:
  the endpoint net's params are created on first call, which a raw
  `lax.scan` body would trace (UnexpectedTracerError in `init`).
- The flow's normaliser is the pooled masked energy of the SCALED
  difference (copy = 1 whatever the EMA scale reads); the EMA group scale
  only sets the unit the flow works in and lives in the trainer state,
  saved as a scalar in the artifact.
- The two-mode contract test: a 16-wide 1-block model trained 300 steps
  on +-u differences -- the flow's 32 samples show both signs and sit
  within .35 of +-u on average while the mean control sits within .35 of 0.

## Public event world model, Step 0: event labels and the re-export — 2026-09-16

Plan approved 2026-09-16 (branch `public-event-world-model`, plan file
`~/.claude/plans/i-want-to-revist-lively-ember.md`): the latent world model
reopened as a model of the PUBLIC battle stepping one service edge at a
time on top of the trunk's 55 public output rows (public tier +
PUBLIC_CLS), major args decoded discretely, entity consequences as a flow
over the post-trunk difference gated by a predicted touched set, trained
standalone on human replay shards, read by depth-1 sampled rollouts
against a value-blind arm. User decisions: token decode for major args
only; diffusion over the difference from the start with the deterministic
mean step as the matched control; touched-set prediction before
denoising; human replays may train the world model, the public critic and
the public action model (a scoped amendment to the self-play-only rule:
eval-actor reads only, `public_v_head` feeds no policy target).

Wire facts that shaped the labels (`rl/offline/event_labels.py`):
- One history step is one COMMITTED edge; `|turn|` and `|done|` only
  commit the pending edge (`state.ts:2080-2092, 3441-3467`). There is no
  turn step.
- `TURN_ORDER_VALUE` is NEVER 0 on the wire: a turn's first committed edge
  reads 2 (the lead segment 1), later edges skip numbers. The turn
  boundary is a CHANGE in `TURN_VALUE` between consecutive valid steps.
  `FIELD_FEATURE__TYPE` is never written (dead column).
- The cache snapshots an entity's public row on its FIRST touch inside an
  edge (`Edge.updatePokemon`, `state.ts:1363`), so a step's cache row can
  be pre-effect: `snapshot_lag_frac` 0.100 over 895,510 (slot, boundary)
  pairs against the `|turn|` state's `public_team` (HP_RATIO / STATUS /
  FAINTED) -- under the plan's 0.20 fallback bar, the per-event
  PUBLIC_ENTITY source stays the cache snapshot.
- Execution is not submission: the own declaration in force is the turn's
  first own MOVE/SWITCH/CANT-with-a-move (a DRAG never declares), the
  answering SWITCH after an own FAINT, UNKNOWN otherwise -- 4.4% of steps.

Re-export (cc999c1, 4 workers, 4m40s at ~180 replays/s): 50,000 logs ->
98,512 trajectories / 2,760,044 per-turn states, identical to the July
manifest, 9.3 GB in 4 shards. The manifest now carries `export_commit`,
`num_history` and the five feature counts and `list_shards` refuses any
other layout (the July shards decoded to garbage after d9e6410's
structured mask). July shards kept aside at
`replays/shards/gen9randombattle-july-20260730`.

Audit (2,000 records = 4,000 games, 335,262 steps,
`runtime/event-audit-20260916.json`): KIND share MOVE .47 / SWITCH .235 /
DRAG .0015 / CANT .018 / FAINT .094 / RESIDUAL .17 / END .012; events per
turn mean 2.96, p90 4, p99 6 (`max_events` 8 covers p99); no game
reached the 512-step window; 30.9 decisions per game; own
cant-without-a-move 2.1% of own events.

## Interrupt checkpoint no longer skipped — 2026-09-16

Defect 2 of the 2026-09-11 health check, and it cost this morning's restart
~17k steps (the old learner ran to ~337k, Ctrl-C wrote nothing, the resume
took ckpt_00319999). Cause: the jitted train step donates the old train
state, and a Ctrl-C that lands between the donation and the rebinding of
the new state leaves run_state on deleted buffers, so the synchronous
interrupt save raised and was skipped; with a step in flight ~95% of the
time, that was the usual outcome. Fix: `DeferredInterrupt` in learner.py
installs a SIGINT handler on the main thread for the life of the loop;
the first Ctrl-C sets a flag and the loop raises KeyboardInterrupt at its
next safe point (top of the loop, or after the periodic tasks once the
step has rebound run_state), where the checkpoint reads whole state; a
second Ctrl-C raises immediately (the escape for a wedged step, with the
old skip message). The handler is restored in the loop's finally and
before the interrupt save, so a Ctrl-C during the write still aborts it.
`tests/test_deferred_interrupt.py` pins the three behaviours with
`signal.raise_signal`. Takes effect at the next learner start.

## Forward KL to uniform restored at .05 — 2026-09-16

User decision after the overnight control (ijk4nyi4, no floor): switch
mass on choice rows 0.42 → 0.01–0.02 by 140k and flat after, 0.003 at the
last logged batch before the restart at ~320k; switch rows per batch 1–3;
eval switching < 1%; importance ratio on switch rows 0.35–0.5 from 140k
(the truncation regime); winrate vs the heuristic 0.25 and rising WITHOUT
switching. The direct forces on the switch logit summed to ~0 for the whole
descent (pg +3e-3 vs entropy −1e-3 + magnet −2.3e-3 at 0.1–0.3 mass), so
the collapse ran through shared features; every term in the loss was
proportional to the mass it was defending.
`uniform_kl_rows` (KL(U_legal || pi), logit gradient pi − 1/k, bounded and
zero-sum) enters the pg bracket at `player_uniform_kl_coef = 0.05`, logged
as `player_loss_uniform_kl`, with `player_switch_logit_grad_uniform_kl`
beside the entropy and magnet terms. Coefficient: the fixed-point formula
p = c / (k (Delta + c)) with the run's k = 6.6 legal cells and measured
Delta = 0.052 (stay − switch advantage, 40k–120k) gives c ≈ .004–.015 for a
modality mass of .03–.10 (3 switch cells); the empirical calibration on
this codebase (o1rsldit at .07 held .02–.03 for 355k steps; .005 on
2026-09-13 held nothing) implies an effective Delta ~20x the panel's, the
shared-feature / replay-reuse / momentum routes the formula omits. .05 is
the calibration's number. Read `player_switch_mass_choice` after 30k
steps: below .03 double, above .10 halve. It is a floor and a container:
it holds the sample count (table: a .03 per-state edge needs ~5–10 switch
rows per batch to resolve over 1000 updates at adv std .30), it does not
make the switch advantage positive; that needs the critic (per-entity
labels) or the pool. Deploy-time tax ≤ c per decision across bad cells;
the thresholded eval actor prunes below .005.
Resumed from ijk4nyi4's latest checkpoint, not scratch: the term acts on a
collapsed policy, so the mass rising from .003 toward the floor IS the
read, against the flat line it would otherwise have stayed on.

## Layout ordered by tier — 2026-09-15

Same relaunch, user request: `SEQUENCE_LAYOUT` (and the `SequenceGroup`
ids) now list the public tier (PUBLIC_ENTITY, TARGET_SLOT, FIELD,
HISTORY_FIELD, INFO, HISTORY_ENTITY, HISTORY_REGISTER, PUBLIC_REGISTER),
then the private tier (CLS, PRIVATE_ENTITY, MOVE_SLOT, PREV_ACTION,
PRIVATE_REGISTER), then the learner-only partition (OPP_PRIVATE_ENTITY,
PRIVILEGED_REGISTER, PUBLIC_CLS, VALUE_CLS). CLS is row 54, VALUE_CLS 90.
Consequence: the actor's rows are the identity prefix
`arange(NUM_POLICY_READABLE_ROWS)` (81), asserted in constants, so the
HISTORY_ENTITY shift on the actor path and the `_first_dropped_row` / "a
head reads past the actor's prefix" machinery are gone. The encoder now
assembles the sequence from a dict keyed by group, in `SEQUENCE_LAYOUT`
order, checked against the kept rows' group ids -- the order exists once.
No offset is stable across this commit: anything that cached a row index
(offline dumps, probes with literal rows) is stale. Numbers move only via
the group-bias/scale bank index order; fresh lineage anyway.

## Trunk registers are layout rows, two per tier; channel-scaled row cosine — 2026-09-15

User decision after the register-highway question. The trunk no longer
appends registers: `num_registers`, `register_embeddings`, `register_norm`
and the derived read set (`all(read_mask, axis=q)`) are gone from
`rl/model/trunk.py`, and the trunk adds no rows of its own. Three layout
groups replace them, at the END so no offset moves: `PUBLIC_REGISTER`
(public tier), `PRIVATE_REGISTER` (private tier), `PRIVILEGED_REGISTER`
(secret tier: reads policy-readable | secret, read by VALUE_CLS and the
secret rows only, dropped from the actor with the rest of the learner-only
partition), `NUM_TRUNK_REGISTERS_PER_TIER = 2`. Sequence 85 → 91 rows,
groups 14 → 17, actor sequence 77 → 81. The registers are learned
embeddings in the encoder (`{public,private,privileged}_register_embeddings`)
and pass through the input norm with their own group scale and group bias
like every row; panels `player_{public,private,privileged}_register_rms`
replace the trunk register panels. Why: the old registers' information set
was a theorem about the mask (they read the keys every query may read,
which under the nesting happened to be the public tier) and only a test
kept it true; now it is a declaration in `SEQUENCE_LAYOUT` and the
privileged critic gets workspace of its own for the first time.
`tests/test_register_tokens.py` re-pinned: privileged registers reach
VALUE_CLS and nothing policy-readable, private ones reach the private tier
and no public row, public ones reach everything (the live-rows control),
and the actor's 81-row sub-sequence matches the learner's rows.
Same launch: `row_homogeneity` scales every channel to unit RMS over the
valid rows before the uncentred cosine and the centred participation. A
shared 30.0 in one channel over an orthonormal 8-row spread read cosine
> .99 raw and reads 1/9 scaled; eight rows sharing one direction at
magnitudes .5–4 still read 1.0. `player_trunk_row_cosine` is NOT
comparable across this commit; participation is unchanged on every closed
form the tests carry. Param paths moved (trunk → encoder), group count 17:
fresh lineage.

## Public tier in the read mask, and a public critic — 2026-09-15

User decision on the diagnostic read of yhnfmjc7 (voluntary switching 0.36
→ 0.009 by 197k, switch mass 0.11 → 0.005, no mass-independent term live
since 4a9e69c). Structure, numbers move:
- `SEQUENCE_READ_MASK` is now four nested tiers. PUBLIC (the 12 public
  entity views, the target slots, field, recurrent field, request info,
  history entities and registers: 52 rows) reads only itself; PRIVATE (CLS,
  my sheet, my move slots, PREV_ACTION: 25 rows) reads PUBLIC and
  itself; SECRET and VALUE_CLS as before. The trunk's shared registers read
  the keys every query may read, which under the nesting is exactly the
  PUBLIC tier -- they are public-tier memory now, not policy-readable
  memory. PREV_ACTION is private because it is my own slot's choice this
  turn, read by the doubles actor's second decision slot (HAS_PREV_ACTION
  is 0 on every singles request); INFO (request type, active count) and the targets are
  public because they are.
- `PUBLIC_CLS` (group 13, row 84, at the END of the layout so no offset
  moved) reads PUBLIC and itself, out-degree 0, learner-only, dropped from
  the actor's sequence like VALUE_CLS. `public_v_head` reads it, trained on
  the same win_returns and mask as the deployable and privileged critics at
  coefficient 1 (`player_public_value_head_loss_coef`); panels
  `player_public_value_head_r2`, `player_loss_v_win_public` beside their
  twins. It feeds no target. Its point: a value of the common-knowledge
  state that a human replay could also label with the SAME rows -- the
  precondition for training it off replays inside the no-human-signal rule
  (potential channel only) rather than with a separate offline model.
- Group count 13 → 14 changes the per-group norm scale/bias shapes: a
  strict resume from any earlier checkpoint fails; params-mode falls back
  to fresh init for those leaves. Launched as a fresh lineage.
- Same launch, separate commit: `player_lambda` 0.8 → 0.95 (credit
  horizon ~5 → ~20 requests, so the switch row's own trace reaches the
  realised outcome after it rather than V five steps out; the trace is
  still cut at the switch row by rho = pi_old/mu for the rows BEFORE it);
  league pacing doubled -- `add_player_min_frames` 2e5 → 4e5,
  `add_player_max_frames` 1.8e7 → 3.6e7, `minimum_historical_player_steps`
  5e4 → 1e5. There is no league off switch; `minimum_historical_player_steps`
  above `num_steps` is the off.
- Tests: `test_private_rows_are_invisible_to_public_rows_at_depth` (trunk,
  3 blocks, controls: private rows move, a public perturbation reaches
  public peers), PUBLIC_CLS in/out-degree pinned, slow
  `test_own_private_team_cannot_reach_the_public_critic` (deployable V
  moves as the control). Reference numbers to read at the relaunch:
  public R2 vs deployable 0.81 / privileged 0.83 at 197k on yhnfmjc7.

## APPO actor: old-policy snapshot and clipped surrogate — 2026-09-14

User-directed completion of the FootsiesGym alignment below, whose ledger
left one discrepancy open: "its actor is APPO with a separate target and
clipped V-trace surrogate; ours remains the score-function V-trace actor".
The player actor is now that APPO loss. The magnet, critics, replay,
builder, optimiser and actor publication are unchanged. Code and focused
checks are complete; the learner has NOT been restarted and no strength
verdict exists.

The reference is RLlib 2.49.0 `appo_torch_policy.APPOTorchPolicy.loss`
(the class `EMAgnetTorchPolicy` inherits; its `loss` is a verbatim copy plus
the magnet terms) with `appo/utils.make_appo_models`,
`torch_mixins.TargetNetworkMixin` and `APPO.training_step`, read against
IMPACT (Luo et al. 2020, arXiv 1912.00167). Discrepancies enumerated first:

- RLlib holds THREE distributions in the loss: the behaviour logits stored
  in the batch (mu), a separate `target_model` (pi_old) and the live model.
  `logp_ratio = clamp(mu/pi_old, 0, target_worker_clipping=2) * pi_live/mu`
  enters PPO's `min(A r, A clip(r, 1 +/- 0.4))`. V-trace's rho and c are
  `pi_old/mu` truncated at 1 (both thresholds 1.0), against the LIVE
  critic's values. We had one ratio, `pi_live/mu`, capped at one inside a
  stopped advantage, and a score-function loss on top.
- RLlib's target is `TargetNetworkMixin.update_target(tau=1.0)`: a hard
  copy, called at init and in `APPO.training_step` whenever sampled steps
  since the last copy exceed `num_epochs * minibatch_buffer_size = 1`, i.e.
  every driver iteration. FootsiesGym never overrides `tau`. With the
  learner thread keeping pace that is about one gradient step of lag, so
  in the reference configuration the PPO band is almost never active and
  the `mu/pi_old` cap does the work. IMPACT Algorithm 1 line 11 is the
  same hard copy every `t_target` SGD steps, `t_target` a multiple of
  `N * K` (buffer batches x replays), 4 x 2 = 8 for its discrete tasks.
  Ours copies the POST-update live parameters every
  `player_old_policy_snap_steps = 8` accepted updates: the paper's discrete
  setting, and our replay cap, so a chunk mostly meets one pi_old across
  its reuses. A snap-every-update setting makes pi_old the pre-update live
  parameters and the clip inert (`tests/test_appo_surrogate.py` pins the
  loss it reduces to).
- The old-policy tree is NOT an EMA and is a separate clock from the
  magnet (rate 3.75e-5, mean age ~26.7k updates): three parameter sets
  per policy, live / old-policy / magnet, only the first optimised.
- FootsiesGym's magnet forward is not detached (a wasted backward through
  the magnet) and its `kl` mean ignores sequence padding; ours keeps the
  stopped, masked f32 `reference_kl`. RLlib masks its losses after the
  fact (`reduce_mean_valid`) but forms the ratios on padded rows; ours
  zero invalid rows before the ratio is formed, so padding cannot leak
  through the clip.
- RLlib checkpoints neither target nor magnet (a restore seeds a random
  magnet). Ours persists `player/old_policy_params` beside `reg_params`;
  a checkpoint without it seeds a live copy, and the retired
  `player/target_params` EMA component is still ignored on load. The new
  name is deliberate: the old name carried a different mechanism.

Coefficients: `player_ppo_clip = 0.4` (RLlib `clip_param`, unchanged by
FootsiesGym), `player_behaviour_ratio_clip = 2.0` (`target_worker_clipping`),
`player_old_policy_snap_steps = 8`. `player_pg_coef`, entropy and magnet
coefficients are untouched. The advantage that multiplies the ratio is the
same stopped, truncated-rho V-trace advantage as before, now with
`rho = min(1, pi_old/mu)`: the V-trace `player_isr_*` panels therefore read
`pi_old/mu` from this change on, not `pi_live/mu` (`player_learner_actor_*`
still carries the live ratio). New panels: `player_ppo_clip_frac`,
`player_surrogate_ratio_mean`, `player_behaviour_old_ratio_mean` and
`_clip_frac` (RLlib's mean_IS and the cap's hit rate), `player_old_policy_age`
(updates since the last copy, 0 on a copy step). Section 1 of the Signal
health view carries them; rerun `scripts/wandb_views.py`.

Gradient shape, for the four questions: the surrogate's per-row force on
the taken logit is `r * A` inside the band and exactly zero past it in the
push direction, with `r <= clip * pi_live/mu` bounded by the cap; the clip
is what the removed score-function loss lacked as a bound on reuse. The
band is around pi_old, not the row's own mu, so with replay it is one
trust region per snapshot period rather than one per stale row. Nothing in
it refills a starved cell: the pi prefactor caveat stands unchanged.

Removed: `loss.vtrace_policy_loss` and its `tests/test_vtrace_loss.py`
(the pre-change tree has both; `git rm` the test). `switch_loss_telemetry`
now takes the behaviour and old-policy log-probs and differentiates the
real surrogate, so `player_switch_logit_grad_pg` reads the clipped loss.

Validation: 28 focused checks pass (the new surrogate tests, the
reference-update tests including the snapshot's exact-copy, separate-buffer
under donation, rejected-step and every-step cases, and the PPO objective
tests), run against the real `loss.py` and the extracted update function in
an isolated JAX harness. Black, isort, autoflake and Ruff pass on every
changed file. NOT run here: the real-model train_step smoke, the switch
telemetry JVP test, the checkpoint migration tests and the rest of the fast
suite; run `env/bin/python -m pytest tests/ -m "not slow"` and then the
slow suite when no learner is live, before any restart. For a later run:
hold `player_ppo_clip_frac` on a panel from the first step; a fraction
pinned near zero says the band is inert at this snap period, a fraction
climbing across each 8-step period is the reuse the band bounds. Acceptance
is matched evaluation strength, critic R2, replay KL/ESS and fresh switch
coverage together, as for the magnet change below.

## SPO on the APPO ratio — 2026-09-14

User decision, same day, before any relaunch: `appo_policy_loss` selects
`"spo"` (Xie et al., arXiv 2401.16025, `r A - |A| (r-1)^2 / 2 eps`, the
builder's objective) in place of PPO's `min`, on the SAME IMPACT ratio
`clip(mu/pi_old, 0, 2) * pi_live/mu` and the same stopped V-trace advantage.
`player_ppo_clip` stays 0.4, deliberately: only the objective moves, and
under SPO the number is where the restoring force balances A (an optimum,
not a boundary), so 0.4 is MORE permissive than PPO 0.4 and than the 0.2
the NashPG lineage ran SPO at on pi_live/mu. No SPO-vs-PPO A/B verdict
exists on any lineage; the 08-26 selector was never read.

Two properties the tests pin (`tests/test_appo_surrogate.py`): past the band
the gradient REVERSES rather than going flat; and on rows the mu/pi_old cap
places at `r = 2 pi_old/mu < 1 - eps` (behaviour took the action at more than
twice pi_old's probability) SPO pulls pi_live back UP toward `0.3 mu`
regardless of the advantage's sign, where PPO's min kept the raw term. That
is a bounded (<= |A|/4eps per row) pull toward stale behaviour policies on
exactly the rows `player_behaviour_old_ratio_clip_frac` counts -- read that
panel beside `player_surrogate_ratio_mean`; a cap fraction that is not small
means the anchor is live on a real share of rows. `player_ppo_clip_frac`
now reads the fraction of rows beyond the band, no longer a clip event.

## EMAgnet aligned to the FootsiesGym code variant — 2026-09-14

The user chose the executable code variant after the paper/code discrepancy
was explained. This supersedes the pending forward-KL and fresh-data-clock
adaptation below. That adaptation was never launched. Training remains on
its already-loaded code; this change has not restarted the learner.

The reference is FootsiesGym commit
`5fe9ead885325fe0011130c75b4ec53042e44dfa`, specifically
[emagnet.py](https://github.com/como-research/FootsiesGym/blob/5fe9ead885325fe0011130c75b4ec53042e44dfa/experimentation/experiments/rllib/components/emagnet.py#L247)
and the fixed [experiment configuration](https://github.com/como-research/FootsiesGym/blob/5fe9ead885325fe0011130c75b4ec53042e44dfa/experimentation/experiments/rllib/experiment.py#L266).
The reference discrepancies were read before implementation:

- It uses `KL(live || magnet)`, not the papers' opposite direction.
- Its EMA mutates during each loss call, before the optimiser step. Our
  pure JAX loss stays mutation-free, but the committed EMA uses the same
  pre-update live parameters, once per accepted optimiser update. Failed
  updates and extra diagnostic/initialisation calls do not advance it.
- Its actor is APPO with a separate target and clipped V-trace surrogate.
  Ours remains the existing score-function V-trace actor over replay.
- Our reference is detached, uses the learner's legal-action masks and f32
  distributions, starts as an independent live copy, and survives full
  checkpoint resume. The example constructs another model without an
  explicit initial copy in that file; this does not override our donation
  and restoration contracts.
- The numerical coefficients now match the example, but its reward scale
  is +/-10 and ours is +/-1, its learning rate is `6e-4` and ours `3e-5`,
  and the training batches/update estimators differ. This is neither a
  reward-scale conversion nor evidence of matched gradient strength,
  fresh-data staleness or reproduction of its unpublished paper experiments.

Current settings are `player_reg_ema_rate=3.75e-5`, `player_mag_coef=.04`,
`player_ent_coef=.006`, and `player_pg_coef=1`. The EMA rule is
`reference_next = (1-tau) * reference + tau * live_before_update` on every
accepted step, including batches entirely drawn from previously used replay.
Its retention half-life is about 18,484 accepted updates. Relative to the
post-update live parameters, mean parameter age tends to `1/tau`, about
26,667 updates. No rescaling by first-use fraction, batch size or replay cap
remains. Zero rate freezes the magnet; rate one copies pre-update live
parameters, preserving the source timing.

The first-use progress helper, tests for its discarded clock and its panel
are removed. `player_ref_kl` now measures `KL(live || reference)`;
`player_reg_ema_rate` records the fixed applied rate, zero on rejection.
Checkpoint schema, builder, critics, replay scheduling and actor publication
are unchanged. Exact f32 KL still re-normalises after bf16 promotion and
masks illegal entries before arithmetic. Its derivative is
`pi * (log(pi/reference) - KL(pi || reference))`; restoring gradients can
vanish near collapsed live actions. Finite full-support reference logits
give finite zero-sum logit gradients, but there is no uniform force bound
independent of reference log-probability gaps. Adam b1=.9 and shared-parameter
mean drift remain governed by the existing optimiser and observed by the
existing switch-gradient, pointer/encoder and applied-delta panels. No
entropy floor or neural convergence claim follows from this variant.

Validation: 51 focused reference-update, KL/target, switch-telemetry,
checkpoint-migration and V-trace-loss tests passed. Positive controls
distinguish pre-update from post-update averaging and reverse from forward
KL; tests retain exact rollback/freeze and repeated-donation checks.
Black and Ruff passed. No real-model forward, full train-step execution or
training restart ran. No runtime strength verdict is available. For a later
run, hold settings for one EMA half-life absent nonfinite updates or a
material strength regression; assess matched evaluation strength, critic
quality, replay KL/ESS, fresh action coverage and parameter drift together.

The exact pre-change source snapshot is
`runtime/emagnet-code-alignment-20260914/before-code-alignment.tar`; it
preserves the then-dirty files for a selective reversal without reverting
unrelated work to Git HEAD. The two dated entries below retain the discarded
clock and source-audit history.

## FootsiesGym EMAgnet source audit — 2026-09-14

The user supplied `como-research/FootsiesGym`'s legacy RLlib EMAgnet example.
Inspected commit `5fe9ead885325fe0011130c75b4ec53042e44dfa`, with source
copies under `/tmp/porygon2-footsies-audit`. This audit changes no training
code, coefficient or live process. It supplies concrete settings for a
related implementation and experiment, not the original EMAgnet benchmark
settings missing from the prior search.

The example's fixed experiment configuration sets
`magnet_learning_rate_schedule=(6e-4)/16=3.75e-5`,
`temperature_schedule=.04` (magnet KL strength), entropy `.006`,
`train_batch_size=4096`, and learning rate `6e-4`:
[experiment.py](https://github.com/como-research/FootsiesGym/blob/5fe9ead885325fe0011130c75b4ec53042e44dfa/experimentation/experiments/rllib/experiment.py#L266).
The `.005` EMA rate and `.1` temperature in the policy constructor are
fallbacks overridden by this configuration.

Its [loss](https://github.com/como-research/FootsiesGym/blob/5fe9ead885325fe0011130c75b4ec53042e44dfa/experimentation/experiments/rllib/components/emagnet.py#L247)
calls `action_dist.kl(magnet_dist)`, meaning `KL(live || magnet)`, opposite
to both the original EMAgnet paper and the FootsiesGym paper equations.
The EMA is mutated inside `loss()` before it returns, hence per loss
evaluation before the optimiser step, including evaluations of reused
data. It is not the original paper's post-epoch update or our first-use-row
clock. The class inherits APPO and includes a target-policy V-trace estimate
and clipped importance-ratio surrogate. Numerical EMA rates must therefore
be compared with update frequency and processed/new data volume, not alone.

The [FootsiesGym paper, Table 4](https://arxiv.org/html/2607.06514v1#A3.SS2)
reports actual experiment settings: EMA rate `1e-4`, entropy `.003`, magnet
KL strength `.5`, 48 parallel games, rollout length 64, eight epochs and
eight minibatches per epoch. Its equation uses `KL(magnet || live)`.
However, 'after each PPO update' leaves the exact EMA event ambiguous
without the experiment code. The repository's
[README](https://github.com/como-research/FootsiesGym/blob/5fe9ead885325fe0011130c75b4ec53042e44dfa/README.md#L186)
explicitly says its paper experiments did not use the supplied RLlib or
CleanRL examples and that their code is forthcoming. Do not combine the
paper's batch/epoch counts with this example's per-loss EMA mutation to
claim an exact fresh-data conversion. The two sources differ in their
coefficients as well as their objectives and training procedure.

Our pending forward-KL implementation still follows the papers' stated
direction. Its first-use-row EMA clock remains a separately documented
replay adaptation, with no direct implementation precedent established by
this repository. No new default or restart follows from this audit alone.

## EMAgnet reference with a fresh-data clock — 2026-09-14

User-directed replacement of the NashPG reference, followed by a request to
account for replay in fresh-data units. Code and focused checks are complete;
the learner has not been restarted for this change. No strength improvement
has been measured.

Reference discrepancies were enumerated before implementation. EMAgnet
[Eq. 1](https://arxiv.org/html/2606.23995v1#S3.SS1) uses
`KL(reference || live)`, opposite to NashPG's `KL(live || reference)`.
[Appendix B](https://arxiv.org/html/2606.23995v1#A2) updates the parameter EMA
after each PPO epoch, including repeated epochs over a rollout. Our actor
remains replay V-trace with raw stopped advantages; the magnet does not
construct critic targets or behaviour importance ratios. No official code
was linked by the paper or found in the focused search. Its rollout size,
epoch count and selected coefficients are insufficiently specified to
reproduce an exact fresh-data timescale. The paper reports log-uniform
search ranges `tau=[1e-5,.1]`, KL strength `[.01,32]`, residual entropy
`[1e-4,.1]`, and a choice of entropy annealing; these are not defaults or
winning settings ([Appendix D](https://arxiv.org/html/2606.23995v1#A4)).

The player now minimises `KL(stop(reference) || live)` over legal actions.
Both log distributions are normalised in f32 after promotion; illegal
entries are masked before normalisation. The per-logit force is
`player_pg_coef * player_mag_coef * (pi_live - pi_reference)`, bounded by
`.05` in absolute value with current coefficients and zero-sum over legal
actions. It retains restoring force when live mass vanishes but reference
mass remains. The moving reference can itself forget actions, so this does
not impose an exploration floor. Adam b1=.9 can still overshoot instantaneous
equilibria; zero-sum logit gradients do not prevent mean-logit drift induced
through shared parameters. Existing switch-direction PG/entropy/magnet JVPs,
pointer/encoder gradient norms and applied-delta panels remain the observers.

Initial coefficients stay `player_mag_coef=.05`, `player_ent_coef=.02`,
`player_pg_coef=1`. `player_reg_ema_rate=2e-4` replaces the snapshot interval.
This rate is our choice, not a paper result. It was initially chosen to match
the former 10,000-update snapshot's average parameter age; the user's later
fresh-data instruction changes its clock and therefore gives a longer memory
in optimiser updates when data is reused.

One nominal fresh batch means `batch_size * (player_chunk_length - 1)`
first-use valid policy rows: currently 252. For each learner batch:

```
fresh_batches = first_use_policy_rows / 252
tau_applied = -expm1(fresh_batches * log1p(-player_reg_ema_rate))
reference = (1 - tau_applied) * reference + tau_applied * post_update_live
```

First-use is replay's pre-increment `reuse_count == 0`; policy masks exclude
padding, terminal, bootstrap-only and singleton-action rows. Fixed nominal
units make the clock independent of shape trimming and variable chunk
lengths. Absent reuse metadata denotes fresh data, as in fixture/offline
batches. All-replay batches continue optimising the policy but leave the
magnet unchanged. Rate zero freezes it; rate one copies after updates with
fresh policy rows. Failed updates restore the whole state through `lax.cond`
and report zero applied rate.

Eight equal fresh-data fractions give `tau_applied=2.5002188e-5` per update
and the same old-reference retention as one nominal fresh batch. The base
half-life is about 3,465 nominal fresh batches. At full-length eight-use
sampling this is about 27,723 optimiser updates, with mean parameter age
about 39,995 updates; actual fresh decision counts determine the clock.
This preserves exponential retention per new policy row, not identical
averaged parameters or policy KL across different replay schedules. It is
an explicit replay adaptation, not EMAgnet's published per-epoch schedule:
later PPO epochs contain no first-use data but still advance its EMA.

Checkpoint schema and `reg_params` storage are unchanged. Scratch and
parameter-only loading initialise an independent live copy; full resume
preserves the saved reference, including an older NashPG reference used as
the initial EMA value. No extra clock counter is needed. Replay, V-trace,
critics, optimiser, actor publication and builder updates retain their
previous behaviour. The generic `player_ref_kl` metric now denotes
`KL(reference || live)` and must not be compared numerically with its old
direction without recomputation. `player_reg_snapped` is removed;
`player_reg_ema_rate` records the applied rate and `player_reg_fresh_batches`
records this batch's progress. The Signal health view was updated at
https://wandb.ai/jtwin/pokemon-rl?nw=8cvs1lfao19.

Validation: 53 focused reference-update, target, switch-telemetry,
checkpoint-migration and V-trace-loss checks passed across the seam runs.
The first reference test run exposed overly strict f32 expectations around
`expm1(log1p(...))`; numerical assertions now allow ordinary f32 rounding,
while rollback and freeze assertions remain exact. Ruff and diff checks
passed. No real-model forward, train-step execution or training restart was
performed. An independent review checked masking, donation, restoration and
the fresh-data clock.

For a later experiment, judge at matched fresh-policy-row budgets and retain
coefficients for at least one base half-life (about 873,278 new policy rows),
absent nonfinite updates or material strength regression. Compare matched
evaluation strength, replay KL/ESS, critic quality, fresh switch coverage,
choice entropy, magnet KL and shared-parameter drift; entropy alone is not
acceptance. The immediately following ledger records the prior snapshot
rule and coefficients. Reverting this mechanism requires restoring both
the reverse-KL helper and periodic-copy update; changing only the EMA rate
does not recover NashPG.

## Replay V-trace actor and one frozen NashPG reference — 2026-09-14

User-directed simplification, including initial regularisation coefficients.
This implementation was launched into a new lineage later on 2026-09-14;
startup/update validation below is not evidence of improved strength. The player
keeps replay (256 chunks, maximum eight uses, first-use fraction .125, existing
staleness/reuse controller), behaviour action log-probabilities and both
critics. Builder objectives and EMA updates remain unchanged.

Reference discrepancies were enumerated before implementation:

- The old player already used the raw live/behaviour ratio in its SPO/PPO
  surrogate. Its IMPACT worker/target correction was telemetry only; its
  separate EMA still supplied actual V-trace policy ratios and critic values.
- The previous advantage already included clipped importance weighting,
  then underwent batch centring/scaling and entered another ratio surrogate.
  The replacement is IMPALA's score-function actor loss with exactly one
  stopped weighted advantage: `-mean(log_pi_taken * stop(rho * advantage))`.
  It is a V-trace actor with NashPG's regularisers, not a verbatim copy of
  NashPG's public PPO implementation.
- Current live policy/critic predictions now construct detached f32 labels.
  Both rho and continuation use the raw legal policy ratio, capped at one;
  lambda .8 appears only in continuation. Actor bootstrap is `r + gamma *
  vtrace_next`, with no second lambda mixture. Terminal outcomes have no
  sampled action and use rho=1. A nonterminal final chunk row supplies its
  value bootstrap but no TD or loss. Categorical labels are projected once.
- The reference copies post-update live parameters every 10,000 successful
  optimiser updates, then freezes. Failed updates roll back optimiser,
  parameters, reference and counters together. Public NashPG defaults have
  1,000 inner collection updates with four epochs and four minibatches:
  16,000 Adam steps per reference, not 1,000 of our learner steps.
- Our Adam moments, epsilon 1e-5 and learning rate 3e-5 stay fixed. The public
  code uses a different collection/update scheme and LR 3e-4; its numerical
  coefficients cannot be transferred without accounting for advantage scale.

Player EMA state and its config, player PPO/SPO selector/clip and forward
uniform-KL coefficient are removed. Training target pruning is removed;
`player_prune_threshold` remains an evaluation-only intervention. Training
actors still sample their full legal policy. The frozen reference remains
separate from replay behaviour: it determines the regularised objective,
whereas behaviour probabilities correct sampled data.

Initial coefficients: `player_ent_coef=.02`, `player_mag_coef=.05`,
`player_pg_coef=1`, `player_reg_snap_steps=10_000`. W&B run `o1rsldit`, six
sampled `player_pg_adv_std` observations returned on 2026-09-14:
.2405, .2733, .2216, .2111, .2148, .2422. At representative std .25, the new
raw-unit magnet .05 is approximately old normalised-unit .2, while entropy
.02 is approximately .08. That raises entropy's relative influence after
removing the extra uniform penalty without importing the paper's .1 directly
onto a much smaller raw actor signal. This is a scale-based starting choice,
not an exact equivalence: current instead of EMA estimates, canonical actor
bootstrap, removed centring and Adam/shared critic gradients all matter.

The old sp75b/sp75c pair remains evidence against indiscriminate flattening:
uniform KL .05 improved switch mass but reduced the measured opponent win
rate .343 to .186 (one historical pair). It does not establish that all
entropy-only configurations fail. The older ledger's categorical statement
that no entropy coefficient can work is too strong: in a two-action fixed-gap
example, `p_bad = 1 / (1 + exp(gap / tau))` is positive for finite gap and
positive tau. Both expected policy-gradient and entropy forces carry a pi
factor; a small entropy gradient alone does not prove it cannot oppose PG.
The equilibrium may be extremely concentrated and recovery very slow; deep
neural, off-policy training has no guaranteed minimum action frequency.
Negative entropy already equals `KL(pi || U_legal) - log(num_legal)`.

For fixed finite reference logits the exact policy-space gradients are finite
and sum to zero over legal logits. The stopped return-scale actor advantage
is bounded on the terminal-only +/-1 channel (no potential channel). The
magnet's force depends on reference log-probability gaps, so it has no uniform
bound independent of the reference. Adam momentum b1=.9 can carry updates
past instantaneous equilibria; zero-sum logit gradients do not imply zero
parameter-induced mean drift. Existing switch-direction PG/entropy/magnet
JVPs, pointer/shared-encoder gradients and applied-delta panels are retained.
No new force controller or automatic coefficient adaptation is introduced.

EMA timing correction: with rate .001, mean parameter age is
`(1-.001)/.001 = 999` updates and half-life is about 693 updates. Copying that
EMA and freezing it for another 10,000 updates produced an approximate
1,000-to-11,000 mean-vintage lag, averaging 6,000 over a cycle. This was not
an 11,000-update EMA time constant or a direct policy-distance measurement.
The new live-source reference has no EMA component to that age.

The local R-NaD reference `/home/joseph/Downloads/rnad.py:872` uses
`jax.lax.cond` to rotate two references from its updated EMA target. At the
user's request our snapshot and nonfinite rollback also use `jax.lax.cond`,
replacing tree-mapped `jnp.where` selections with whole-pytree branches. The
snapshot still keeps one live-source reference. Neither API guarantees
separate physical output buffers or a runtime advantage; re-donation after a
snap and rejected updates are covered by focused tests. No speedup is claimed.

Full checkpoints now persist player live/reference/Adam/counters; legacy
player EMA files are explicitly ignored before decoding. Existing saved
references survive full resume; pre-reference and parameter-only restores
seed an independent live copy. Historical league entries retain their
explicit saved parameter key. New player publications/evaluation use live
parameters, with raw eval series named `main-*`; prior EMA eval scores are
not an exact matched control. The retired uniform-KL applied-update screen
fails explicitly and its exact source is archived at
`/tmp/porygon2-uniform-kl-screen-before-vtrace-20260914.py`. Historical
mechanisms remain in commit `0f6da86`, which also includes since-retired pair
heads: do not restore that commit wholesale over the existing dirty tree.

Validation: 124 focused target/loss/reference-update/checkpoint/league/
telemetry/inference-loading tests passed across their respective seam runs. A guarded
`jax.eval_shape` traced real-model initialisation and the complete learner
forward/gradient update for the bundled (58,1) fixture in 5.829 seconds,
peak RSS 655.7 MiB; lowering and compilation were explicitly forbidden.
No real-model forward or GPU train-step test ran alongside the live learner.
The updated Signal health dashboard was built and saved at
https://wandb.ai/jtwin/pokemon-rl?nw=tdumfdg18u1.

Acceptance for a later run: hold these static coefficients for two complete
reference periods (20,000 accepted updates), absent nonfinite updates or a
material performance regression. Read actor/behaviour KL against the existing
replay ceiling .045, raw-ratio ESS, fresh-game action coverage, choice entropy,
reference KL across its cycle, critic quality and matched evaluation strength
together. A prettier entropy curve alone is not acceptance. No runtime verdict
of training strength is claimed by this ledger.

User-authorised launch: `session-1789347450-main` / W&B `aa5pcqt9`, isolated
under `ckpts/gen9/lineages/vtrace-nashpg-20260914`. Params-only initialisation
inherits live player/builder weights from `ckpts/gen9/ckpt_00340000`
(340,000 player updates, 69,407,288 frames, parent W&B `o1rsldit`), with fresh
optimisers, copied reference, counters, league and replay. Old checkpoint files
and their W&B identity remain in place. New generic `--ckpt-subdir` scopes the
existing config field; params/scratch reject occupied explicit destinations,
checkpoint mode permits resume, and paths escaping the generation root or
combined with BR mode are rejected. 42 CLI/BR setup tests passed.

The old SIGINT stop at 10:49:47 hit the existing donated-state window: its
interrupt checkpoint was skipped because a state array had been deleted.
The final main.py message saying the checkpoint was saved was misleading;
the complete periodic checkpoint at 340,000 is the actual source. No newer
complete checkpoint or league snapshot was available. This loses unsaved
updates and must not be described as a lossless restart.

The new process loaded the source weights, filled replay and began applying
updates after its fixed shape precompilation. Reported non-active shape
compiles: (64,192) 85.5 seconds and (64,256) 34.8 seconds; the active (48,128)
shape compiled on its first real call. Beyond 100 applied batches, observed
throughput was roughly 4–5 updates/second. These are startup observations,
not a performance comparison with the prior learner. At logged step 281,
applied-update count was also 281, skipped-update metric 0, gradient norm
1.454, raw ratio ESS .9725, reference KL .01536 and realised replay reuse
7.922 (cap eight).

Source-weight hashes and commands are recorded in
`runtime/vtrace-nashpg-20260914/provenance.json`, `start.sh` (first launch only)
and `resume.sh` (checkpoint resume in the same subtree). A copy of provenance
is stored as the new checkpoint root's `lineage.json`.

ISR oscillation audit, same day: all 8,235 consecutive logged updates of
`aa5pcqt9` were read, with local evidence in
`runtime/vtrace-nashpg-20260914/isr-history.json` and `isr-oscillation.png`.
The below-one fraction uses the raw live/recorded-behaviour taken-action
ratio, a strict threshold, and a per-batch conditional denominator. Median
voluntary-switch count is five rows; 2.1% of batches have none and log zero.
These amplify jaggedness but do not explain the broad waves: after pooling
by matching row counts, updates 2,401–2,800 have switch below-one fraction
.067 and mean raw ratio 1.740, versus .951 and .606 at 3,201–3,600. Mean
choice-state switch mass is .0943 versus .0517 in those windows, though the
sampled states also change. Switch/non-switch below-one fractions correlate
at -.925 across complete 200-update bins. No reference snapshots occurred;
reuse cap stayed eight except two 200-update intervals at seven, starting
at 2,000 and 4,500. Neither explains the repeated waves as a periodic reset.

Coefficient-weighted common-switch-logit directional gradients average
PG +.0027005, entropy -.0021155, magnet -.0004891, net +.0000958. Positive
means gradient descent suppresses switching. PG is positive in every
400-update window; entropy is negative on every record; the magnet is
negative on 78.6% of records and resists some high-switch phases. This is
consistent with delayed policy/behaviour feedback under opposing objectives,
not identification of a unique oscillator: these JVPs hold features fixed,
omit Adam momentum/shared critic effects, and sample changing replay states.
Global raw-ratio ESS median .943 does not rule out subgroup attenuation.
Do not interpret the fraction alone as attenuation magnitude or a reason to
retune coefficients; mean clipped ratios would measure attenuation directly.
No learner settings changed for this diagnostic.

Replay/KL follow-up through update 19,634: exact history is stored in
`runtime/vtrace-nashpg-20260914/replay-kl-audit.json`; controller events and
the chart are beside it. Recomputing all 196 completed controller ticks from
their 100-record KL means reproduces every published tick cap. The sustained
14,001–15,000 actor-KL mean .07227 drove reuse towards six: cap 8->7 at
14,100 (sensor .05712), first 7->6 at 14,600 (.08760). The 18,001–19,000
mean is back to .04368, with cap alternating six/seven. At 19,600 the sensor
is .06410, cap six and realised reuse 5.882. Realised reuse is sampled chunks
divided by newly admitted chunks, so inventory/prefetch/admission timing
separates it from the current cap. No manual reuse or coefficient change.

The PI is not a hard threshold gate: its proportional term can reduce cap
when the KL mean rises but remains below .045 (17,300: .03854, cap 7->6).
For 2,001–10,000 versus 10,001–19,634, batch-mean overall KL rises
.03666->.04896, non-switch-conditioned KL .02865->.04475, while
switch-conditioned KL .06857->.06603. Preclip gradient norm falls
4.37->1.93, pointer-query/key applied-delta RMS stays roughly unchanged,
and no update is skipped. These are changing-batch diagnostics, not matched
state comparisons or proof of unchanged policy sensitivity. The reference
snap at 10,000 resets reference KL but does not immediately increase actor
KL; its causal role in later drift is unproven. Behaviour age is not logged
per sampled chunk, so current telemetry cannot separate actor/admission lag,
replay-state composition and policy movement as causes of the KL rise.

Exact target provenance checked on 2026-09-14: commit `ae2c1c3` (2026-07-30,
`improve replay eff`) introduced `player_replay_kl_target = 0.045` as a literal
in the former `rl/learner/config.py`. Its comment attributes the value to a
buffer-capacity plateau diagnosis, but no numerical derivation or comparison
against .05 was found in that commit or the relevant ledger/design records.
Treat .045 as an inherited heuristic, not a calibrated safety boundary. Later
KL/ESS measurements provide context for its scale, not its original derivation.

References: [IMPALA, section 4](https://proceedings.mlr.press/v80/espeholt18a/espeholt18a.pdf),
[NashPG](https://arxiv.org/html/2510.18183v3),
[public defaults](https://github.com/ntu-agents/nashpg/blob/main/conf/algorithm/nash_pg.yaml),
[JAX conditional semantics](https://docs.jax.dev/en/latest/_autosummary/jax.lax.cond.html).

## Pair critic removal and forward uniform KL — 2026-09-13

User-directed removal of the pairwise auxiliary critics and population
moments, replacing the support hinge with `KL(U_legal || policy)`. Both
pair heads, feature-moment banks, learner reductions/updates, auxiliary
scalar target payload, losses, configuration, probe and dashboard metrics
are removed. The deployable/privileged CLS critics and action readout remain.
Checkpoint resume merges by the current parameter/Adam paths, dropping old
heads while retaining shared parameters, targets, reference, optimiser,
counters and league. A small checkpoint-only unpickler recognises the two
retired scalar record names so old population pickles can be discarded
without keeping their implementation in the live model. Actual
`ckpts/gen9/ckpt_00060000/player/scalars` decoded successfully (60,000 steps,
12,945,896 frames). The committed pair-head baseline is recoverable from
`0f6da86`; the population-moment extension was uncommitted when removal was
requested, with its measurements and design preserved in the next ledger.

`uniform_kl_rows` scores the mean `-log pi` across each row's legal cells,
minus `log(N)`, in f32. Empty/singleton rows are zero, illegal cells are
masked before reduction and the learner uses its existing policy mask.
`player_uniform_kl_coef=0.005` replaces the hinge coefficient/tau/temperature
inside the policy-gradient bracket. Loss and switch-direction metrics are
`player_loss_uniform_kl` and `player_switch_logit_grad_uniform_kl`. Independent
legal-probability/pruning exposure metrics remain. The offline coefficient
screen is now `rl.offline.uniform_kl_screen`.

This restores the previously recorded extra actor regulariser; it does not
change the SPO/PPO selector, entropy/magnet, Adam or clipping order. Per-logit
force is `coef * (pi - 1/N)`, bounded and zero-sum with no probability-prefactor
on the restoring term. There is no direct force along the softmax-invariant
mean direction. Adam b1=.9 can carry motion through an equilibrium; the
existing pointer/shared-encoder parameter and applied-delta panels remain
the drift checks. Shared gradients reach the action readout and encoder;
no separate human-derived reward or action override is introduced.

Coefficient evidence and acceptance (not yet a post-change result):

- `runtime/replay-audit-01861967/analysis.json` records 19,512 voluntary
  switches / 96,622 observed move-or-switch actions = 20.19415868%. The
  requested 80% ceiling is **16.15532694%**, conservatively below the newer
  corpus's corresponding 16.423% ceiling.
- The prior flat-KL .05 experiment pinned within-modality entropy at .93
  and reduced matched BR win rate .343 to .186 (ledger around sp75c). The
  new .005 is one tenth that coefficient, selected for a smaller policy
  flattening force, not by balancing loss magnitudes.
- W&B run `57ctruuo`, step61,392, before this change: switch opportunity
  mass .02244, hinge switch derivative -.01049, policy-gradient switch
  derivative +.007368, within-modality entropy .7822. These are a single
  minibatch and are not human-comparable switch frequencies. At pg_coef1,
  the new KL's entire switch-shift derivative has magnitude at most .005,
  less than half that measured old restoring force.
- Fresh chosen-action counters `player_fresh_voluntary_switch_count` and
  `player_fresh_move_or_switch_count` sum before division; their fraction
  excludes forced/preview/wait/standalone, replay reuse and inactive rows,
  retaining singleton moves. They are a decision proxy: protocol logs omit
  some attempted moves cancelled by faint/flinch, so complete-game logs
  own the final human comparison.
- Pre-register the acceptance on T=1 complete games: the pooled voluntary
  switch fraction and its game-bootstrap upper 95% bound must be <= the
  16.1553% ceiling over at least 500 games; hold across two consecutive
  windows. Compare strength and within-move entropy at matched checkpoints
  as well. If the ceiling fails, screen .0025 and 0 against .005 in separate
  coefficient phases; do not add a switching heuristic or vary static JIT
  config within the live learner. No fixed coefficient is a hard switch cap.

The active learner was left running pending the user's restart choice.
No post-change training/gameplay coefficient sweep has been run; .005 is a
provisional evidence-based starting value, not a validated behavioural bound.
Focused validation: 25 loss/target/telemetry checks, 48 checkpoint/merge/BR
checks, 9 fresh-switch tests and 2 protocol-screen tests passed; Ruff,
Black and whitespace checks passed. A guarded `jax.eval_shape` traced
initialisation and the full gradient/train-step on the real T=58/B=1 fixture
in 7.49 seconds, peak host RSS 656.5MiB, without materialising model params
or allowing model lowering/compilation. The resulting 435 metrics included
all fresh-switch fields and no retired pair fields. Reproducer/log:
`/tmp/porygon2_abstract_pair_removal.py` and its `.log` sibling. This is an
arity/shape check, not numerical GPU validation.
Signal health dashboard refreshed with the retired metrics removed and
uniform-KL/fresh-switch panels. Real-model forwards are deferred while the
learner is live.

### Authorised restart and coefficient screen — 2026-09-13

The user subsequently authorised checkpoint/restart/tuning and asked whether
the extra old-loss training justified a fresh lineage. Chosen: a new experiment
record with the full trained state retained, rather than random initialisation;
no evidence established that the inherited policy should be discarded.
Graceful Ctrl-C saved `ckpts/gen9/ckpt_00128223` at 128,223 completed updates,
26,327,764 frames. W&B baseline `57ctruuo` finished. Its identity file was
archived to `runtime/uniform-kl-calibration-20260913/previous-wandb-runs.json`;
all checkpoint and league files remain available. The new W&B experiment is
`nm6i7f45`, named `uniform-kl-0.005-from-128223-main`, with explicit checkpoint
resume and the same counters, weights, targets, optimiser and league.

Four GPU tests passed: both real train-step paths (ordinary and potential
channel), actor/learner equivalence, and the privileged partition. Three
abstract precision checks and two protocol screen tests also passed. The
independent service must run with `service/` as its working directory: the
current worker reads `../constants/data.json`. A root-directory launch failed
before any games; only that task-owned process was terminated and restarted
from the correct directory. No unrelated process was stopped.

Frozen screen on eight T=1 self-play games: 18 chunks, four full batches,
11 voluntary switches / 481 observed move-or-switch actions = 2.2869%.
Both perspectives use the same frozen checkpoint here. No taken action was
discarded by the existing .005 v-trace pruning threshold. Each coefficient
used the same restored Adam state and recorded batches, two consecutive
updates per batch to read the first update's result. Compiled executables
were cleared between coefficient phases.

| KL coefficient | Encoder gradient norm | KL switch-shift derivative | KL after one update | Switch mass after one update | Within-taken-modality entropy after one update |
|---|---|---|---|---|---|
| 0 | 2.52317 | 0.0000000 | 1.013915 | 0.0339408 | 0.672210 |
| 0.0025 | 2.52272 | -0.0006851 | 1.013849 | 0.0339582 | 0.672161 |
| 0.005 | 2.52290 | -0.0013701 | 1.013621 | 0.0339636 | 0.672271 |
| 0.01 | 2.52298 | -0.0027403 | 1.013738 | 0.0339577 | 0.672211 |

The .005 setting stayed at the control's encoder gradient scale and entropy,
with a live restoring derivative and reduced KL. Single-update switch changes
are small and not monotonic after the restored Adam/bf16 computation; this
screen does not establish a trained-policy equilibrium. .005 is selected for
the live hold. Results: `runtime/uniform-kl-calibration-20260913/coefficient-screen.json`.

Direct restart reused the existing tmux panes, with a scoped `BATTLE_LOG_DIR`
on the service and explicit `--load-mode checkpoint --init-ckpt` on the
learner, avoiding `start.sh`'s broad W&B stop pass. Actual resume logs confirmed
that only the two retired head subtrees dropped from the three parameter
banks. Startup included slow per-leaf `jnp.copy` kernel compilation under the
learner environment (SIGUSR1 stack confirms `create_train_state`, not a hang),
then fixed-shape compilation. W&B confirmed step 128,543, uniform KL 1.066,
uniform-KL switch derivative -.001668 and skipped-update metric 0.

The acceptance clock starts **2026-09-13T11:33:29.699479Z**, after confirmed
publication at least through 128,500. Full protocol logs carry their start
time as the first `|t:|` timestamp: only games starting strictly after that
bound count, excluding old in-flight policies. Reports select the live
`main:p0gNN` side; the other side may be a historical league snapshot and
must not be pooled with main. The runtime reporter uses the existing parser,
completion-ordered 500-game windows and pooled-count game bootstraps; unknown
start times are excluded. Startup-only reports are diagnostic, not acceptance.

### Forward uniform KL hold passed — 2026-09-13

The .005 coefficient passed the pre-registered two consecutive 500-game
windows on the live main side, restricted to games starting after the verified
post-change publication bound. Exactly 1,000 games are selected; 1,098 were
available at the final refresh and the 98 later games are excluded from this
pre-registered result. Forced/pivot replacements and the historical opponent's
actions are excluded; the denominator is observed executed moves plus voluntary
switches, using the same parser as the human reference.

| Completion window (UTC) | Games | Voluntary switches / observed moves-or-switches | Rate | Game-bootstrap 95% interval | Ceiling |
|---|---|---|---|---|---|
| 11:33:32.275–11:38:34.846 | 500 | 298 / 10,195 | 2.923001% | 2.598030–3.263154% | 16.155327% |
| 11:38:35.355–11:43:39.038 | 500 | 247 / 10,451 | 2.363410% | 2.053500–2.683190% | 16.155327% |

Both upper bounds clear the requested 80%-of-human ceiling. Keep
`player_uniform_kl_coef=0.005`. This is an observed hold result, not a hard
constraint on future policies. The paired coefficient screen established local
update sensitivity; the full-game hold established the behavioural criterion.
No fresh-start model or new actor action heuristic was used.

The new learner was last verified at step 131,726 with skipped-update metric
0, uniform KL 1.384 and within-taken-modality entropy .6193 (individual logged
batches, not pooled hold summaries). A 64-row W&B history sample independently
verified the new fresh counters: 31 rows carried first-use decisions, totalling
21 chosen switches / 703 chosen move-or-switch decisions. No inference should
be drawn from their zero counts on reuse-only batches.

The service and learner remain running in the existing `train` tmux session,
experiment `nm6i7f45`; the temporary offline service and screen exited. Proof
is in `runtime/uniform-kl-calibration-20260913/`, including
`live-report-20260913T114442Z.json`, `final-switch-acceptance.json`,
`restart-provenance.json`, the coefficient-screen results and validation logs.
The runtime reporter's CPU controls include main-side identity swaps, forced
and pivot exclusions, pooled game-level resampling, incomplete-window/high-rate
negative controls and protocol start-time filtering. Evaluation strength is
reported separately; changing live snapshots, sparse raw-main eval interleaving
and the old run's EMA summary prevent a matched strength-improvement claim.

### Uniform KL adequacy review — 2026-09-13, step 148,710

The user's follow-up asks whether .005 is large enough. The switching ceiling
is an upper bound, so passing it cannot establish sufficient exploration or
an optimal coefficient. A 512-row W&B history sample from `nm6i7f45`, read at
12:40 UTC, compares the first 5k updates with the latest sampled window
(143,266–148,688; 126 logged batches). Mean legal switch-cell exposure below
the .005 pruning threshold rose from 13.55% to 37.66%; exposure below .001
in the latest window was only .208%. Mean within-taken-modality entropy was
.6838 initially and .6639 recently; raw uniform KL was 1.1768 and 1.3269.
These are means of sampled batch metrics on changing replay distributions,
not matched-policy effects or pooled action frequencies.

The latest window's coefficient-weighted switch-shift derivatives were
PG +.0020941, uniform KL -.0015846, entropy -.0005616 and magnet -.0001300,
with total -.0001821. The KL therefore supplies a material restoring force
(about 76% of PG's opposing mean), while the total direct actor derivative
still slightly favours switching. These diagnostics hold features fixed and
cannot attribute shared-feature or Adam updates; declining switch mass alone
does not prove that the KL is overwhelmed.

The latest two complete 500-game main-policy windows were 205/10,219 =
2.0061% (game-bootstrap 95% interval 1.7279–2.2743%) and 204/10,234 =
1.9934% (1.7240–2.2709%), completing 12:29:41–12:39:52 UTC. Both remain well
below the 16.1553% human-derived ceiling. This leaves room for a .01 trial,
judged on pruning exposure, within-modality discrimination and strength as
well as switching. It does not justify jumping to the historically harmful
.05 or treating increased switching as a strength improvement. No coefficient
change or restart was performed for this review. Evidence:
`runtime/uniform-kl-calibration-20260913/kl-adequacy-history-20260913.json` and
`latest-switch-windows-20260913T123957Z.json` in the same directory.

### Uniform KL target raised to 16% — 2026-09-13

The user explicitly replaced the former switching ceiling with a **16%
target**, requesting a smart coefficient estimate and minimum restarts.
Fresh main-side T=1 protocol windows completing 12:56:42–13:06:21 UTC were
141/9,729 = 1.4493% and 158/10,014 = 1.5778%; pooled 299/19,743 = 1.51446%.
These use executed moves plus voluntary switches, excluding forced/pivot
replacements and historical-opponent actions.

Selected `player_uniform_kl_coef=0.07`, from .005. Linear extrapolation to
.16 gives .052824. The approximate equilibrium
`coef * (uniform_switch_share - switch_fraction) = headwind * switch_fraction
* (1 - switch_fraction)` corrects for the restoring force weakening as mass
returns: assumed effective uniform switch shares .35/.40/.45/.50 give
.07940/.07225/.06756/.06425. Thus .07 centres the scenario range. These
shares are assumptions, not measured current-mask statistics; this is not
a causal fit or confidence interval. The one-update screen cannot identify
long-run response. The historical .05 discrimination/strength regression
remains relevant; increased switching alone is not a strength claim.

Graceful stop saved `ckpts/gen9/ckpt_00157138`, 157,138 updates and
32,049,328 frames. A single learner restart uses explicit full checkpoint
resume, retaining parameters, EMA targets, reference, Adam, counters and
league. The service remains running. Only the prior W&B identity file was
archived to the new task directory to separate the coefficient phase; all
trained state is retained. Thirteen focused uniform-KL/fresh-switch checks
passed. No new sampling heuristic or runtime controller was introduced.

Pre-registered operational target: 15–17% in each of two consecutive
500-game main-side windows, reporting game-bootstrap 95% intervals. Wait at
least 20,000 updates after the change (through 177,138) before a plateau-based
retune, and inspect the trend across reference snapshots; gross overshoot or
instability can justify earlier intervention. Filter games by start time
after verified changed-policy publication. Watch within-modality entropy,
strength, skipped updates, pruning exposure and shared-parameter drift.
The old reporter's below-ceiling `accepted` flag does not establish this
new target. Training outcome is pending; coefficient alone cannot enforce
an exact behaviour frequency.

Resume verified on new W&B run `o1rsldit` at step 157,262: coefficient .07,
skipped updates 0, uniform KL 1.245 and its switch-shift derivative -.01688.
Actor publication is verified through 157,250. The task's
`acceptance-start.json` records a conservative subsequent UTC start bound.
The CPU target reporter has four passing tests and requires both target-band
windows' 95% intervals to contain 16%. A thread heartbeat checks every
15 minutes and continues the authorised calibration; the first hold cannot
pass before step 177,138. One learner restart has occurred so far.

Evidence and operational state: `runtime/uniform-kl-target16-20260913/`
contains `baseline-games.json`, `baseline-game-records.json`,
`coefficient-estimate.md`, `restart-provenance.json` and `live-learner.log`.
The direct fallback is restoring the coefficient to .005 and resuming the
saved pre-change full checkpoint; any such rollback requires a measured
reason, not the prior ceiling criterion.

## Pair critic population moments — 2026-09-13

User requested an abstract, adapting reference population without retained
example teams. This supersedes the fixed-reference implementation recorded
below while retaining entity-local inputs, uniform living-pair aggregation
and the endgame-cancellation repair. Learning stopped at the full legacy
`ckpts/gen9/ckpt_00048083` checkpoint (run `57ctruuo`, frame 10,607,468,
manifest `local_reference_v1`). Actual checkpoint migration and one finite
update passed all three production shapes; the live restart is recorded
below separately from the earlier 40,226 restart.

`PopulationPairValueTerms` now learns bounded tanh features for unary hidden
units and cross/synergy queries and keys. It subtracts their running means
BEFORE the linear unary readout or bilinear interaction. Cross uses the
skew part and synergy the symmetric part, divided by `2*sqrt(qk_size)`;
there is no tanh after the pair product. If the means equal the current
feature expectations, product-reference pair marginals are zero. Actual
means track changing learned features and shared entity embeddings with
lag, so this is an approximate population decomposition. No causal credit,
Gaussian population, exact online identification or learning benefit is
claimed. The historical finite-example algebra check alone does not prove
these running estimates are accurate.

Both pair functions are bounded by `4*sqrt(qk_size)`, hence **64** at width
256. This is wider than the previous fixed-reference cross/synergy bounds
3/4; comparisons of raw contribution amplitudes across the boundary are
not like-for-like. Unary readout weights remain unconstrained. Query and
unary readouts start at zero with live upstream paths; keys are live.
Pair loss coefficient stays 1.0. No cosine penalty or additional reward
signal is introduced. Pair losses still bypass the trunk directly and can
change shared local embedders through their existing gradient paths.

`pair_population.py` stores five feature-mean banks indexed by head and
public/private visibility. The public head pools living entities from both
sides into its public channel. The sheet head keeps our private channel
separate from the opponent public channel. Learner and EMA-target weights
have separate population states. A forward reads the PREVIOUS means;
statistics from that forward update them only afterwards, with gradients
stopped. The population is fresh-decision/living-entity weighted, not
game-balanced: only first-use replay columns, valid nonterminal decision
rows and valid living entities contribute. The value mask excludes the
bootstrap overlap row; missing reuse provenance contributes nothing. The
existing evaluation-to-training gate still owns exclusion of eval games.

The provisional half-life is **65,536 fresh living-entity observations per
channel**, not learner updates or replay draws. At 12 contributors per
decision this is about 5,461 fresh decision rows; visibility channels have
different counts. Debiased exponential weighting makes the first nonempty
update its observed mean, leaves empty channels exactly unchanged, and
tracks normalised coverage mass in [0,1]. Mass is not an effective sample
size or confidence interval. Feature means remain in [-1,1]. The half-life
is a research assumption without tuning evidence; feature/encoder drift
and correlated decision samples remain calibration limitations.

The `population_moments_v1` manifest and `population_terms` namespace
coordinate persistence. Legacy 48,083 migration resets both auxiliary
heads, their corresponding optimiser leaves and empty population states;
shared model/target/regulariser parameters, matching optimiser moments and
global optimiser count, training/frame counters, builder and league are
preserved. Current-format restoration requires finite, bounded,
shape-compatible means and masses. Compatible parameter-only loads carry
main moments and copy them to seeded targets; incompatible heads and
moments reset together. Offline main/EMA diagnostics load the matching
checkpointed population and leave it frozen. Old exemplar assets and their
generator are removed from the active implementation.

Verified logs: 17 head tests, 7 population tests and 27 model/integration
tests passed. These cover product-marginal controls, symmetry/locality,
nonzero initial gradients, side/visibility weighting, inactive NaNs,
debiased startup, a 32,768-update bounded-state scan, fresh-row masks,
actor/learner policy equality, privileged partition, dtype and carry.
The earlier 38 history checks remain evidence for the unchanged history
encoder; they are not 38 additional new tests in this phase. All 22 focused
checkpoint tests passed, covering legacy migration, exact restoration,
invalid/missing moments, parameter-only resets and matching offline reads.
Their command/output transcript is `checkpoint-tests.txt` in the directory
below. Scoped Black/isort checks, Ruff and `git diff --check` passed.
Logs: `runtime/pair-population-abstract-20260913/` and
`runtime/pair-history-fix-20260913/pair-population-head-tests.log`.
The first production smoke harness exhausted GPU memory while retaining
multiple full train states for comparisons. The harness now compares host
copies and allocates states sequentially; this was a validation-memory
failure, not a demonstrated learner OOM. Its failed attempt is preserved in
`runtime/pair-population-abstract-20260913/resume-smoke-harness-memory.log`.

The sequential smoke passed (48,128), (64,192) and (64,256) in
65.70/62.84/41.32 seconds respectively, including compilation and the first
update, not steady-state throughput. Shared parameters matched the stopped
checkpoint bit-for-bit; both population banks were initially empty and
then updated. Applied public cross-query RMS was 3.228e-5/2.404e-5/2.404e-5.
Saving and restoring both moment banks was bit-exact. Reused replay left
both unchanged, and an injected non-finite update restored the entire
previous player state with zero reported population movement. Evidence:
`runtime/pair-population-abstract-20260913/resume-smoke.json` and its script
and log. Dashboard "Signal health" was refreshed with population panels.

Restart verified from 48,083 with `bash start.sh --load-mode checkpoint
--init-ckpt ckpts/gen9/ckpt_00048083`: the same W&B run `57ctruuo` reported
`resumed=True`. At the live check it had reached training/lifetime step
48,468, with skipped-update metric 0, public/sheet pair loss .2454/.1959,
and public cross-query applied RMS 2.385e-5. Population coverage mass was
.3680/.2074/.2026 for public-public/sheet-public/sheet-private; warm-up
therefore had not finished. The public channel consumed 246 fresh entity
observations on that reported update, with mean discrepancy .1037 and
actual mean movement .000732. Training-batch removal-MSE gains were
-.02412/.02581, mixed early behaviour rather than a usefulness verdict.
Live precompiles took 75.4/40.2 seconds for (64,192)/(64,256). Learner PID
717025 and service PID 717209 remained running with no logged traceback
or OOM. Live log: `runtime/learner_20260913_163154.log`. No commit or push.

Pre-register a warm-up of one observation half-life in every populated
channel (coverage mass at least .5), followed by **5,000 learner updates**
at unchanged coefficients unless correctness, non-finite updates or
material regression require intervention. Inspect population pre-update
mean discrepancy, actual mean movement, counts/mass and applied query/key
updates alongside pair losses and plain-T1 evaluation. After warm-up,
measure cross/synergy removal-MSE gains on games excluded from fitting,
with each checkpoint's population frozen and uncertainty grouped by game.
Existing live `*_mse_gain` panels describe training batches, not held-out
evidence. A larger pair variance or a nonzero term is not acceptance.
No improved learning or playing strength has been established. The local
design and acceptance record is
`docs/pair-population-moments-2026-09-13.md`; the preceding entry preserves
the retired design's measurements and original source/checkpoint handles.

## Pair critic locality repair and history audit — 2026-09-13

User authorised stopping learning, diagnosing and repairing the pair critics,
checking the new history encoder, and restarting. Run `57ctruuo` stopped
cleanly at `ckpts/gen9/ckpt_00040226`; its parameters, targets, optimiser,
regulariser state, scalars and league restored successfully. Pre-change code:
`0f6da86adf306e5608320ea0a902b738682bffd6`. Unrelated `data/ps` changes remain.

At sampled 30k–38k updates, public/sheet pair R² was .817/.821, but cross
variance divided by total prediction variance was only .00021/.00024.
Post-trunk unary rows could encode the whole board. Alive-roster grand-mean
centring did not identify main effects: uniform pair weights cancelled it,
and 1v1 cross and two-survivor symmetric synergy were necessarily zero.

Both heads now read entity-local embeddings before history, row identities
and trunk attention. Unary outputs subtract their fixed-reference mean;
cross/synergy functions are double-centred over that reference AFTER tanh.
Cross remains antisymmetric, synergy symmetric, with conservative absolute
bounds 3/4. Uniform alive-pair means replace learned selection. This defines
predictive effects relative to a specified product reference, not causal
credit or effects conditional on the actual correlated game distribution.
The sheet head still reads our private sheet and opponent public rows; its
common reference mixes the two information channels. No direct pair-loss
gradient enters the trunk, although shared entity embedders still train.
The CLS heads, policy, targets and loss coefficient (1.0) are unchanged.

The reference asset `rl/model/pair_reference.json` packages raw
observations from 32 distinct non-evaluation training games in the historical
checkpoint-01889162 self-play collection, seed 920. Existing held-out games
are excluded. Source SHA, exact selections and sampling seed 20260913 are
recorded; `scripts/make_pair_reference.py` reproduces the asset. Public uses
32 public samples; sheet uses 16 private and 16 public. References are encoded
once per trajectory outside the time vmap, with fixed shapes. Initialise
these shared embedders before that vmap: doing so afterwards leaked a tracer
under Flax intermediate capture. The dtype test exposed and verified the fix.
New manifests record the reference digest and `local_reference_v1` form.
The `local_terms` namespace resets only auxiliary heads and their moments
when merging the old checkpoint; all shared parameters matched bit-for-bit.

Frozen-embedding screen: 72 fitting and 24 held-out games, both perspectives
kept together, all reference games excluded, 1,000 fixed updates at 3e-4,
seed 20260913. Against an identically initialised unary-only arm, held-out
outcome MSE public 1.3205 -> 1.2850, sheet 1.3150 -> 1.2758. Game-bootstrap
95% improvement intervals [-.0606, .1246] / [-.0238, .0941] include zero.
Every arm overfit the small cohort (training R² .982–.997, held-out R² < 0).
This is no generalisation or playing-strength success. Pair gradients were
nonzero; held-out 1v1 cross RMS .1597/.1031 on 48 states. Probe parameters
were discarded. No coefficient sweep or minimum pair-variance quota.

History: 38 focused tests passed for the 19-row recurrence, separate
entity/field/register GRUs, chronological and request-count alignment,
padding, carry and rewrite/reset. Strengthened a vacuous identity-isolation
test with fixed Q and live V/output plus live-Q and muted-V controls; no
production history change was justified. All 162 checkpoint player leaves
were finite. On one bundled game (58 requests, 168 events), maximum bf16
suffix/full memory differences were .0456/.0121/.0201 for entity/field/
register state; f32 GPU differences .00181/.00133/.000884. Full-model
log-policy/value-log-probability errors .0143/.0428 were below the existing
.05 tolerance. Mean write fraction .0442, retention biases 3.975–4.028.
Nonterminal history reset changed policy total variation .17–.85% and value
log probability by up to .317; functional, with stronger critic effect in
this sample, not evidence of improved play.

Validation: 14 pair-head tests; reference provenance, positive-control
full-model locality and removal-MSE telemetry; privileged partition,
actor/learner equality, checkpoint merge and dtype checks passed. Actual
checkpoint restoration plus one finite update passed every production
shape, (48,128)/(64,192)/(64,256), with nonzero applied pair-query deltas.
Compile-plus-first-update times were 68.72/63.57/42.23 seconds respectively;
these are not steady-state throughput measurements. Ruff and diff checks
passed. Local reproducible scripts, reports and logs:
`runtime/pair-history-fix-20260913/`; detailed local plan:
`docs/pair-history-repair-2026-09-13.md`.

The earlier cross-share gate is superseded by the repaired definition.
Hold 50k fresh updates before judging usefulness. New `*_mse_gain` panels
measure error increase when removing cross, synergy or both on TRAINING
batches; positive is useful there, not held-out proof. Read alongside plain
T=1 evaluation. Sustained regression calls for a matched coefficient-zero
control, not repeated scale cuts. High all-row trunk cosine (.750 early,
.841 at 30k–39k; centred participation 10.98 -> 7.05) is not evidence that
this head caused it. No cosine penalty was introduced. Recover using the
pre-change code and checkpoint above, preserving unrelated work; no commit
or push was made for this task.

Restart verified: `bash start.sh --load-mode checkpoint --init-ckpt
ckpts/gen9/ckpt_00040226` restored the same W&B run (`resumed=True`). At the
post-start check, training/lifetime step was 40522, skipped-update metric 0,
public/sheet pair losses .1440/.1707 and public cross-query applied RMS
3.699e-5. Removal-MSE gains were -.002935/.03069: early mixed behaviour,
not an acceptance verdict. All live batch shapes compiled; startup's two
explicit precompiles took 82.7/41.0 seconds. Service and learner remained
running, with no logged traceback/OOM. Dashboard "Signal health" refreshed
with the new gain panels. Live log: `runtime/learner_20260913_152958.log`.

Subsequent user constraint: a permanently fixed sample is unacceptable as a
general design, especially for formats with player-built teams. The bank is
not a whitelist (feature-based heads can score unseen teams), but the archived
random-battle population is an unnecessary format-specific dependency. Keep
the local-input and endgame-cancellation repairs separate from this anchoring
choice. A rolling, same-format, training-only reference is a candidate, with
game-balanced sampling, fixed tensor capacity, independent scored games and
checkpointed/frozen evaluation references. It is not a drop-in refresh:
changing the reference changes auxiliary predictions at fixed parameters
under the current equations. Current-batch centring couples predictions to
batch companions; a null anchor does not identify population main effects.
Reference-free local scoring is possible with a weaker interpretation claim.
No replacement was implemented or run restarted during this design review.
Reference: Lengerich et al., AISTATS 2020, section 5.2,
https://proceedings.mlr.press/v108/lengerich20a.html.

Further user direction: model the reference population abstractly instead of
retaining representative examples. A sufficient-statistics design is possible:
for local features phi and population mean mu, the bilinear interaction
(phi(x)-mu)^T M (phi(y)-mu) has zero product-reference marginals. Skew/symmetric
M preserves cross/synergy symmetry. No Gaussian assumption or raw exemplars
are needed. NumPy algebra check against 2,048 explicitly averaged examples
matched within 1.34e-14, with nonzero interactions and stale-mean controls
(`runtime/pair-population-abstract-20260913/moment_check.py`). This verifies
algebra, not a production implementation or training benefit.

First moments are insufficient for the CURRENT nonlinear head: a zero-mean
population [-1,-1,2] has mean tanh -0.186387. Keep pair readout bilinear after
centring; any bounded nonlinear feature map must precede moment estimation.
Parameter-only bounds can preserve centring; per-pair tanh cannot. Means of
changing learned features also lag the current encoder. The existing local
EntitySumPool is linear in fixed masked feature blocks, including occupancy
features for biases, so their moments can be projected through current weights
without this representation-age error (apart from numerical precision).
The nonlinear unary MLP needs its own hidden-feature/output expectation
estimate, or a changed readout; its expectation cannot be inferred from the
input mean. These are explicit design constraints, not an implemented fix.
Population statistics should be format/visibility-specific, estimated from
training data rather than value-loss gradients, and checkpointed/frozen for
evaluation. Finite estimates and changing self-play still require calibration.
Background: https://arxiv.org/abs/1605.09522.

## Per-history-step register assessment — 2026-09-13

User requested stopping learning and inspecting the event-step attention for
register-token suitability. Ctrl-C stopped the learner, but the synchronous
interrupt checkpoint hit a donated/deleted buffer; the final main-process
"saved" message is misleading. Periodic `ckpt_00960000` remains available and
its raw learner parameters loaded successfully for this probe. Training stayed
stopped; no model change or register experiment was launched.

GPU-jitted production history path on first chunks of 100 historical tactical
cohort perspectives: 7,027 valid events (3,448 one-row, 3,567 two-row, 12
three-row). On 3,271 two-row/one-source events, head1 source→non-source mass
80.4%, non-source→source 51.1%; head2 self mass 99.8% for source and 97.8% for
non-source. Multirow normalised entropy head1/head2 .313/.0187; top weight
>95% on 59.9%/97.9% of queries. Head1's average is not uniform per event.
Attention-output/direct-message norm ratio median1.37, p99 2.84 (branch
magnitude, not causal playing value). Input row norms median5.42, p99 9.43,
max13.38; within-event largest/smallest ratio p99 2.06, max2.65. Padded-key mass
exactly zero. No extreme input-norm sink signature found in this cohort.

Declined adding registers on this evidence: StepAttention is ONE attention
layer, so fresh learned register keys/values offer a static fallback; discarded
register outputs cannot relay current-event information back to real rows.
Workspace needs a second layer or a new recurrent path. Distinct cross-row and
self routes are not evidence of pathology. Reference: ViT registers,
arXiv:2309.16588 (deep-network high-norm background-token mechanism). No training
A/B or causal policy intervention: benefit is unestablished, not falsified.
Historical perspectives are not independent games or current-policy samples;
no doubles conclusion. Local reproducible probe, raw arrays, connection plot,
and full report: `runtime/history-step-attention/`. No removal/revert handle
because production code is unchanged.

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

### Independent adversarial confirmation — 2026-09-10

At `b7d5225`, a fresh standalone run and numerical-verifier rejection reproduced
the nested-mask fusion defect. Independent row-wise float64 oracle and LLVM
pointer/predicate simulation predict every observed validity decision across
random/all-true/all-false/checkerboard runtime masks. Wrong-row counts:
1,242/480/0/240 out of 4,234; retained values accurate within 5.82e-7 and
registers bit-exact. All-true runtime masks can lose rows: faulty predicates
and wraparound addresses make this more than a row permutation. Fresh LLVM
is byte-identical to the prior standalone dump. Separate jitted calls and the
barrier both restore all masks; a matched precomputed-mask control also passes.
Fresh full f32 actor barrier pair: entropy/value differences 1.255993/.232134.
Standalone tracing-cache messages reflect first compilation/static-argument
changes, not Python side effects; the pure reproducer has no stateful trace
operations. Source-level compiler fix and historical learner impact remain
unestablished. Local report: `docs/xla-independent-audit-2026-09-10.md`;
raw evidence: `runtime/register-validation/hlo-investigation/adversarial-audit/`.
No production change, upgrade or restart in this audit.

## Python 3.13 + jax 0.11.1: the nested-concatenate miscompile is fixed upstream — 2026-09-10

`env/` rebuilt on Python 3.13.15 (the 3.11 venv that trained every lineage
to ckpt_02339569 is `env-py311-retired/`, gitignored). jax 0.11.0/0.11.1
require Python >= 3.12, so `pip index` from the 3.11 venv topped out at
0.10.2 and hid the release that matters: jax-v0.11.1 (2026-08-17) pins an
XLA 510 commits past openxla/xla 5d74754 "[XLA:GPU] Fix iteration space
bounds propagation for nested concatenates" (2026-07-31, the fix for jax
issue 39486), while jax-v0.10.2's XLA (2026-06-15) is 1265 commits behind
it. Measured with the issue-40588 reproducer
(`runtime/register-validation/hlo-investigation/bug_repro.py`) on this box:

| jax | barrier off | barrier on |
|---|---:|---:|
| 0.10.2, py3.11 and py3.13 | max err 4.51, bad steps 16..57 | 7.2e-7 |
| 0.11.1, py3.13 | 7.2e-7, no bad steps | 7.2e-7 |

So issue 40588 is the 39486 defect, already fixed in the current release,
not a distinct live bug; the encoder's `optimization_barrier` (b7d5225) is
now a no-op workaround and can go as its own structure-only commit once
the full-model compare (`compare.py --barrier 0`) agrees on 0.11.1.
Stack moved with it, latest of each on 3.13: flax 0.12.9 (0.12.6 called
`jax.core.get_opaque_trace_state`, removed in 0.11.0 — 48 fast-suite
failures, all that one AttributeError), optax 0.2.8, chex 0.1.92,
ml_dtypes 0.6.0, jaxtyping 0.3.11. `requirements.txt` is the exact freeze.
Venv lesson: a venv `mv`'d after creation keeps absolute paths in every
console-script shebang and in `activate` — `env/bin/pip` died with "bad
interpreter" and start.sh's `source env/bin/activate` would have fallen
through to the system python; fixed by sed, next time create it in place.

## Row identity: bias after the norm, per-row table deleted — 2026-09-10

Two changes to the sequence identity in `_assemble_sequence`, both a
fresh lineage (nothing migrated), landed on the same day as the input norm
(984b0f2) before its lineage launched.

**Order (3581269).** The group/row biases were added BEFORE
`SequenceInputNormalisation`. The norm divides a row by its own content
RMS, so the bias was divided too: at the measured content scales (CLS RMS
0.18, history rows ~66) the identity was 35% of a CLS row and 0.1% of a
history row, with the gradient into the bias shrunk by the same factor —
the per-group disparity the norm removes from the content, reproduced on
the identity channel. Now: normalise, then add the identity, then re-zero
invalid rows (token-plus-type form). `test_row_identity_is_added_after_the_input_norm`
pins it with the harness's zero history/field rows accounted for (they are
bias-only under either order; the rows with content discriminate).

**Per-row table deleted (this commit).** `sequence_row_bias`
(61 x 256, variance-scaling init RMS 0.0625, independent per row) was an
absolute slot embedding over a SET with a fixed layout. Measured on
ckpt_02339569 (2.34M steps, pre-norm lineage) as within-group spread of the
trained rows against the 0.0625 an untrained table keeps:

| group | rows | spread | | group | rows | spread |
|---|---|---|---|---|---|---|
| MOVE_SLOT | 16 | 0.066 | | PUBLIC_ENTITY | 12 | 0.140 |
| PRIVATE_ENTITY | 6 | 0.067 | | HISTORY_ENTITY | 12 | 0.100 |
| TARGET_SLOT | 17 | 0.073 | | HISTORY_FIELD | 3 | 0.201 |
| OPP_PRIVATE_ENTITY | 6 | 0.064 | | CLS / INFO / VALUE_CLS | 1 | n/a |
| PREV_ACTION | 2 | 0.044 (= init for 2 rows) | | | | |
| FIELD | 3 | 0.077 | | | | |

Six groups never learned a per-row identity in 2.34M steps while the group
bias trained in every group but PREV_ACTION (0.03-0.12 from zero), so the
gradient reached the tables and the per-row component had nothing to
carry: move, sheet and target rows are read POSITIONALLY by the action
readout, and the field triple and previous-action pair carry their own
biases inside their embedders. The three groups that did train are the
actives-first public order (which the ACTIVE feature already carries) and
the history rows. The encoder comment's claim that row i's bias "pairs"
history row i with public row i was false — two rows, two independent
vectors, nothing shared. The group bias is the only step-0 identity now and
takes the embedding init instead of zeros. If a history-to-public pairing
is ever wanted it is ONE slot embedding indexed by public slot i added to
both rows — a positional join at the same step, distinct from the retired
`entity_index_tag` (a join across the wire's stable index, 09-02).
Revert handle: this commit.

## Output sequence normalisation — 2026-09-11 (numbers move; merge relaunch from ckpt_00280000)

User asked for an output norm "like the input" after the ckpt_00280000
health check. `modules.SequenceInputNormalisation` renamed
`SequenceNormalisation` and instantiated a second time as
`encoder.output_normalisation`, applied in `_batched_forward` right after
the trunk: per-row RMS 1, then 1 + zero-init per-group channel scale, the
same 12x256 bank shape, invalid rows exactly 0, no cross-row statistics
(the privileged partition is untouched). The final norm every pre-norm
transformer carries (GPT-2 ln_f, LLaMA model.norm, ViT head norm); the
trunk had none. Grounding measured on ckpt_00280000 over 399 offline
steps: trunk-output row RMS by group CLS 9.8, registers 11.4, INFO 10.0,
FIELD 6.5, PUBLIC_ENTITY 5.6, PRIVATE_ENTITY 3.6, TARGET_SLOT 1.3,
MOVE_SLOT 0.99 — and block 5 alone took MOVE_SLOT from 2.05 to 0.99 and
TARGET_SLOT from 2.0 to 1.3 while doubling CLS 4.5 to 9.8, so the heads
read a 10x magnitude disparity the last block sets. Every head now reads
rows at RMS 1 x (1 + its group's scale).

Seams: the `trunk_out_group_l2` panels read the RAW trunk output
(computed in the encoder before the norm, learner-only, threaded through
the encoder's return) — after the norm they would read 1 for every group.
`trunk_row_cosine` reads the normed rows (per-row scale-invariant, so
identical at init). `harness.encode_policy_rows` and
`transition_probe.value_of` apply the output norm after their direct
`encoder.trunk` calls so the offline reads match what the live heads and
search read. Telemetry: `norm_scale_telemetry` reads both banks,
`player_{input,output}_norm_scale_rms_<group>`, one panel each.

Test: `test_every_row_leaves_the_trunk_at_unit_rms` (valid rows RMS 1 at
init, invalid 0; +1 on the output bank doubles every row; +1 on the
input bank changes content but leaves output RMS at 1). Found in
passing: `test_opp_private_team_cannot_reach_the_policy`'s positive
control reversed the opponent's six rows, which since the per-row bias
went (d5bb6a9) is a permutation of a set and moves no attention read —
it was failing on main; the control now copies mon 0 over all six rows.

Relaunch: checkpoint-mode param merge from ckpt_00280000 (the new bank
keeps fresh zero init, `player: 1 subtrees kept fresh init`); the heads
see their inputs rescaled at the merge, so expect a transient on the
policy/value losses. Revert handle: delete `output_normalisation` in
`encoder.setup` and the one application line; everything else is
structure.

## Human replay position potential — 2026-09-11 (research, no training change)

User explicitly requested deriving human-informed PBRS, then specified a
position-quality prior rather than action imitation, a learned override, and
exactly zero cumulative shaping. This authorises the investigation despite the
standing no-human-shaping rule; no production shaping path was restored.

8,000 eligible local gen9randombattle replay logs with metadata rating >=1900;
7,972 contain scored actions, 376,325 move/voluntary-switch rows. Whole-game
train/validation/test 5559/1242/1171; both perspectives together. Public-prefix
features, complex identity/type logs excluded; voluntary counts agree with the
existing battle_stats parser. Test11,415 switches:65.09% outgoing HP>=2/3;
switch rates33.80/17.96/12.62% from unfavourable/equal/favourable coarse STAB
matchups. These denominators are observed actions, not legal-choice-mask rows.
58.43% of chosen switches improve that matchup; no counterfactual benefit claim.

Fit human outcomes, not actions, with one hash-selected state/game. Features H:
public last-observed team HP balance (unseen slots full), N: alive balance,
M: difference in best-STAB log2 effectiveness in each direction (immunity floor
1/4; no moveset/ability/damage simulation). Experimental train-fit potential:
Phi=eta*tanh(.315472H+.082297N+.022844M). Material baseline:
eta*tanh(.313207H+.084851N). Test Brier material .218491, +M .218431;
paired difference -.000060,95% game CI[-.000729,+.000567]: incremental matchup
value NOT established. The four-feature active-HP extension won validation but
not test. Player-disjoint/chronological generalisation was not measured.

Critical negative control: over11,326 test switch turns with a next turn,
unscaled mean potential delta material-.05515 / +M-.03754, positive only
9.28%/33.18%. These potentials can initially penalise human switching too;
NOT a validated switching fix. Do not increase M merely to force positive signs.

At gamma1 use Psi(nonterminal)=Phi(history)-Phi(initial history), terminal
Psi=0 explicitly, F=Psi(next)-Psi(now): sum F=0 exactly. Whole-game centring,
all signed transitions, terminal debit; never centre each chunk or clip F.
A free shaped critic U can learn V-Psi; exact TD advantages then equal the
unshaped ones. No persistent heuristic agreement loss. Existing fixed [-1,0,1]
categorical heads cannot naively consume shaped returns outside their support;
implementation needs a derived residual/value parameterisation. No eta selected.

Report with derivation, pros/cons, limitations, implementation requirements and
PBRS references: docs/human-switch-pbrs-2026-09-11.md (local/gitignored).
Scripts, frozen manifest/rows/results: runtime/human-switch-pbrs/. Synthetic
parser controls include forced/pivot exclusion, future-move feature isolation,
benched healing and two-target HP. Numeric PBRS controls cover telescoping,
zero-return centring, corrected TD identity and positive-only clipping failure.
Scoped Ruff/Black passed. No model/GPU work, restart, checkpoint or coefficient
change. Research scripts/results are local artefacts; no production revert needed.

### Multiple-active extension of the position potential — 2026-09-11

User requested doubles generalisation. Added offline reference
`rl/offline/position_potential.py`, not a live reward path. Field matchup is
mean over enemy defenders of their maximum incoming pair pressure from own
actives, minus the reverse. Exactly the singles pair difference at1x1;
slot-permutation invariant, player-swap antisymmetric, bounded[-4,4]. It avoids
averaging a dangerous opponent away with a harmless one, but does not model
focus fire, spread damage, redirection, Protect or support. This aggregation
is a design hypothesis, not a measured doubles result.

Material uses6*(own roster fraction-opponent roster fraction), where6 records
the fit's reference-team size. Actual battling roster sizes, not preview pool
or active counts; fainted entries remain, unseen entries retain the public
full-HP convention. Four-mon fractional losses scale to the original six-mon
units. Singles-derived coefficients remain unchanged and unvalidated in doubles.

Episode-centred PBRS is unchanged. At gamma1, unchanged-field choice microsteps
get zero; never duplicate a resolved field reward across both active choices.
Nine focused tests passed: singles reduction, permutation/player symmetry with
live control, undiluted threat, four/six roster scale, bench/faint/empty handling,
microstep/terminal zero sum, corrected TD override, bounds/off mode/invalid data.
No model forward or simulator required. Scoped Black/Ruff/diff checks passed.
Existing doubles service slot-alignment defect is untouched; singles replay
parser still rejects doubles. Report extension/pros-cons/validation requirements:
docs/human-switch-pbrs-2026-09-11.md. No training restart or live shaping change.

### PBRS learner integration alternatives and eta recommendation — 2026-09-11

Documentation-only follow-up requested by user. Report now maps the proposed
reward/value changes to targets.py's scalar conversion/V-trace, train_step.py's
CE/PG, actor whole-game chunking, interfaces, head wiring and imagined consumers.
Extra scalar heads are OPTIONAL: existing three logits can predict a residual
on fixed widened support ±(1+eta), trained by CE on residual_target/(1+eta).
This preserves parameter count but abandons outcome-probability semantics and
requires transition/search consumer migration. Merely shifting an unchanged
outcome support by-Psi with consistently shifted targets cancels algebraically.

Alternative: two small scalar residual heads on CLS/VALUE_CLS, preserving both
raw categorical heads and their unshaped target passes; shared trunk is not an
independent control. Actor advantages must actually use the shaped estimator
for a direct PBRS intervention. No production choice or loss coefficient set.

Analytic initial-offset handling: raw_prior=Phi(nonterminal),0 at terminal;
start_offset=Phi(initial),0 at terminal; Psi=raw_prior-start_offset. Learn
R≈V-raw_prior, use shaped U=R+start_offset and original V=R+raw_prior. Network
need not reconstruct initial history. Terminal recorded value still learns the
payoff: only its potential is0, not its payoff/value label. Numeric scratch
check verified centred TD identity and zero sum with the actual done-row form.

User wants terminal dominance: recommend eta.05 for first trial, .025 fallback;
.10 only a justified later experiment, not a switch-rate response. This is a
bounds-based proposal, not measured optimum or applied config. Max increment/
terminal correction2eta=.10 at.05 (10% of unit payoff); held-out human switch
|delta|p95 .192472*.05=.00962, mean-.00188. Total shaping remains0 exactly.
Suggested offline same-estimator actor-logit gradient perturbation budget10%
RMS, explicitly a design criterion, not a guarantee from reward bounds. Include
actual advantage normalisation, absolute norms, shared-trunk/new-value-loss
costs, raw outcome evaluation and meaningful disabled estimator control.
No training modification/restart. Full alternatives, pros/cons and source map:
docs/human-switch-pbrs-2026-09-11.md.

## Fair-information service potential MCTS — 2026-09-11

User requested a service-only MCTS baseline with tests and explicitly selected
fair information. Added eval index3/potentialmcts; Python only registers its eval
name. No model inference, default eval selection, training reward or live run
changed. Source/usage/support: service/src/server/baselines/README.md.

Own request + public client snapshot only; no live sim/opponent/private opposing
request/submitted choice/RNG access. Reconstruct sampled worlds with own known
stats/HP/moves/PP, public boosts/ordinary tera/hazards and random-battle priors
for hidden sets. Coarse determinisation, not exact posterior/history recovery.
Open-loop decoupled UCT selects both players before resolving; terminal±1/0,
unit-scale potential at leaves. PBRS eta/centred differences do not enter search.
64-iteration/depth3-transition/100ms soft budget, up to3 sampled root worlds.

MCTS only ordinary Gen9 singles roots. Explicit SimpleHeuristic fallback for
preview/root replacement, doubles, weather/terrain, pseudo-weather/timed side
conditions, sleep/toxic counters, complex volatiles/identity and unsupported set
priors. Rollout-internal forced replacements work. Counters are worker-module
local, not W&B/per-game persistence; strength evaluation must price fallback
coverage. Potential coefficients now shared via rl/offline/position_potential_fit.json
between the Python reference and TypeScript leaf scorer.

Found/fixed simulator clone alias: toJSON() hands out the live log array and
fromJSON(object) reuses it. Reusing that object across rollouts grew logs until
its1000-unsent-line guard threw "Infinite loop". Cache JSON STRING snapshots;
fromJSON now parses fresh arrays.80-rollout log/HP isolation regression passes.

Validation:12 focused Vitest tests incl terminal/adversarial/leaf/deadline,
public-only getter guards with public-change positive control, sampler facts,
clone isolation, legal moves/forced replacements and real service decoder game.
Final inspected game18 MCTS decisions +5 explicit unsupported fallbacks, last
49 iterations/101ms. No strength claim. Nine Python potential tests passed;
TypeScript typecheck/focused ESLint/Ruff/Black/Prettier/diff checks passed.
No training restart or commit. Existing doubles alignment defect untouched.

### Potential-MCTS versus SimpleHeuristic paired screen — 2026-09-11

User requested head-to-head. Ran100 Gen9 random battles,50 independent seeded
team pairs; algorithms exchange fixed rosters/sides within each pair, same
battle seed. Unchanged64-iteration/depth3/100ms soft search budget, no learner,
no policy tuning during the sample. MCTS45 wins / SimpleHeuristic55, no draws,
truncations, failed games or rejected live choices. Pair-cluster percentile
bootstrap95% interval37–53% (50k resamples, seed9711); not evidence of a strength
advantage or conclusive inferiority. Pair outcomes:6 MCTS sweeps,11 heuristic
sweeps,33 split. MCTS24/50 wins as p1 and21/50 as p2.

Search used1221/2647 MCTS decisions (46.1%), or1221/2192 ordinary decisions
(55.7%). Fallbacks:455 request,386 pokemon state,176 status timer,161 species
prior,160 weather/terrain,25 side condition,33 rejected own sampled choice,
30 rejected opponent sampled choice. The63 reconstruction/rollout choice
rejections are search failures hidden by legal heuristic fallback, not live
invalid actions; investigate before expanding/promoting this baseline.
Mean successful search101.13ms and35.78 iterations; all MCTS decisions47.58ms
mean/102.73ms p95 versus SimpleHeuristic0.0868ms mean/0.1724ms p95.
Games took130.945s total. Voluntary switches167/2192 (7.62%) for hybrid MCTS
versus26/2192 (1.19%) heuristic. More switching did not establish better play
or more human switching. Cannot isolate leaf potential quality from shallow
search, sampled-world errors and fallback coverage with this comparison.

Reproducible runner:service/src/scripts/potential_mcts_h2h.ts. Local manifest,
packed teams/seeds, per-game decisions and summary under
runtime/potential-mcts-h2h-20260911/. No training changes. Runner typecheck,
ESLint and Prettier passed; all100 games exercised its actual battle path.

### Search-only potentials and Jaxcalibur-style PUCT — 2026-09-11

User authorised new search potentials while preserving original PBRS, ideally
70% versus SimpleHeuristic; subsequently requested PUCT like
https://jaxcalibur.github.io/#search. Enumerated differences from its public
rule before implementing: heuristic rather than learned priors/value, action
history rather than public-state hashes,3 coarse sampled worlds, small serial
budget, no learned surprise/revelation cutoff. Simulated opponent can therefore
overestimate its knowledge of our hidden set; actual policy still receives only
own request/public observations, never live opposing truth/action/RNG.

New search_potentials.ts registry retains original and adds material, strategic
(status/boosts/hazards), tactical (damage/speed), roster-coverage leaves. No
changes to Python PBRS potential or shared coefficient JSON. New search_priors.ts
contains uniform/tactical priors; tactical uses softmax with5% uniform mass.
New puct.ts samples weights max(Q−V,0)+cpuct/sqrt(max(1,N))*P for each side;
opponent Q/V reverse sign. Unvisited Q=V; no forced initial coverage. V is mean
backed-up return, units[-1,1], cpuct1 initial tested coefficient. Existing UCT
is preserved as a control within shared rollout/backups/disposal code.

Twelve configurations,40 tuning games each, seeds101–120, paired fixed teams
and battle RNG with algorithms exchanging sides/rosters.100ms soft budget:
UCT depth3 original19/material15/strategic19/tactical20/roster17 wins;
UCT depth1 original23/strategic14/tactical21. PUCT depth1 original+uniform18,
original+tactical30, tactical+tactical26; PUCT depth3 original+tactical23.
Selected/froze original leaf + tactical priors + PUCT cpuct1/depth1/256 cap/100ms.
The75% tuning score did NOT generalise to70%.

Fresh validation seeds1001–1100:120/200 wins (60%), pair-bootstrap95%CI54.5–65.5%,
50k resamples seed9711. Original UCT depth3/64cap/100ms matched control on first
100 games won43; selected PUCT won59 on that exact subset. Verified packed
teams/seeds/sides equal. Paired improvement16pp,95%CI+7 to+25pp. Of50 team pairs,
20 improved,5 worsened,25 tied. Combined selector/prior/depth improvement, not a
pure selector-only ablation. No70% claim and no changes after viewing heldout
outcomes. Those validation seeds are now spent for future candidate selection.

PUCT searched2428/4626 ordinary decisions (52.5%;44.0% of5524 requests), mean
100.63ms/57.78iterations per successful search.200 games256.45s.59 sampled
trapping-choice rejections, zero rejected live actions/failed games/truncations.
Voluntary switches177/4626=3.83%. UCT control100 games126.35s,1173 searches,
mean101.16ms/35.97iterations,153/2572=5.95% voluntary switches. More switching
was not the mechanism of improvement. Additional sampled failures at depth3
identified Revival Blessing's fainted replacement target; not fixed mid-run.

Baseline3 defaults now select tested PUCT configuration; baseline2 remains
learner eval default. New code takes effect after service rebuild; none launched.
Reproduction:service/src/scripts/potential_mcts_h2h.ts defaults to chosen config,
accepts explicit controls and seed offsets, persists resolved manifests and all
game records. Local780-game results/comparison:runtime/search-potentials-20260911/.
Docs:docs/search-potentials-2026-09-11.md (local), baselines/README.md (repo).
25 focused Vitest tests passed incl prior override by terminal wins, matching
pennies opponent-sign positive control, prior/leaf purity and real service game.
Typecheck/focused ESLint/Prettier/diff checks passed. No training restart,
learner shaping modification or commit. Retain all controls; do not equate this
adaptation's60% with Jaxcalibur's neural system or infer PBRS learning benefit.

### Service-side one-turn matrix regret matching — 2026-09-11

User asked to try CFR-style search service-side after discussing a one-turn
cached-payoff experiment. Reference examined: Zinkevich et al.2007
https://papers.nips.cc/paper_files/paper/2007/file/08d98638c6fcd194a4b1e6992063e944-Paper.pdf.
Explicit differences: ordinary regret matching on a restricted one-step Bayesian
game, not full-game CFR/safe re-solving; no recursive information-set traversal;
3 equally weighted sampled opposing worlds; top3 own actions shared across
worlds/top3 opposing replies per world; original potential at nonterminal leaves.
Opponent can condition on sampled own set; existing overknowledge of our hidden
set remains. No real opposing private data/submitted action/live RNG access.

regret_matching.ts: signed cumulative regrets, positive-part normalisation,
uniform zero-positive-regret policy, simultaneous updates, arithmetic average
strategies;1024 iterations. Own policy shared across worlds; opponent policies
world-specific. Tests expose false clairvoyant per-world solving and verify
an asymmetric mixed equilibrium, opponent response sign, dominance and gap.
matrix_search.ts: cached per-world payoffs; no missing-cell zero imputation.
Top3 action restriction is explicit, with legal/retained counts in diagnostics.
256 clone-call cap includes3 inspection clones;100ms soft deadline reserves5ms
for solving. Incomplete first sweep falls back with matrix_budget_incomplete.
Final action is sampled from averaged policy, not most-visited/argmax.

Caught deployment random draw coupled to observation-hash search RNG in a
preliminary49-game run (runtime/matrix-search-20260911/matrix/). Stopped own
process; results superseded, not used for selection. Corrected live sampling to
independent private cryptographic randomness, benchmark to separate seeded actor
streams (9712,pair+offset+1,game+1,side+1). Regression holds simulated payoffs and
root probabilities fixed while changing only deployment draw. Corrected
configuration frozen before fresh seeds3001–3050; no outcome-based retuning.

Fresh matched100 games each: matrix51 wins, PUCT56. Matrix95%CI42–60%, PUCT46–65%.
Verified packed teams/battle seeds/sides equal. Paired matrix−PUCT difference−5pp,
95%CI−17 to+7pp,50k whole-pair bootstrap resamples seed9711. No demonstrated
improvement or conclusive inferiority; do not generalise this to full CFR.
Neither meets70%. Keep PUCT/default baseline3 unchanged and retain matrix as
selection:"matrix" experiment via shared service factory/H2H runner.

Matrix1519/2396 ordinary searches(63.4%), PUCT1393/2312(60.3%). Matrix two
incomplete-coverage fallbacks/no sampled choice failures; PUCT six sampled
trapping-choice failures. Both zero live invalid choices, failed games, draws,
truncations. Matrix mean97.02ms/67.16 clone calls/64.16 transitions,26.62 cells,
2.96 retained own actions versus7.20 legal. Mean estimated-game gap0.000353 is
not whole-battle exploitability; only~2.4 payoff samples per cell. PUCT mean
100.66ms/59.10 simulations. Game totals153.29s vs145.95s. Voluntary switches
129/2396=5.38% vs90/2312=3.89%; increased switching did not establish improvement.
Investigate payoff estimates/action restriction before more regret iterations.

Local results/docs:runtime/matrix-search-20260911/{matrix-fresh,puct-fresh}/,
comparison.json, docs/matrix-regret-search-2026-09-11.md; repo baselines/README.md.
6 new tests plus25 existing=31 passed, including real service decoder battle.
Typecheck/focused ESLint/Prettier/diff checks passed. PBRS functions/coefficients,
learner rewards, training/live service unchanged. No rebuild, restart or commit.
Fresh seeds3001–3050 are now spent for future unseen-validation claims.

### 2026-09-11 — Service payoff reliability: response distribution before more worlds

40 paired PUCT/SimpleHeuristic games seeds5001–5020; public-only audit recorded
146 attempted roots,68 eligible,first2/game ->54. Supported nonterminal singles
slice excludes faint/pivot/request/status/weather boundaries; not whole-game or
doubles evidence. Offline16 worlds×16 chance draws, actual resolved public reply
used only as retrospective conditional diagnostic. 864worlds attempted,768built,
675support actual reply;46matched roots/29games. Original potential unchanged.
Next-potential RMSE:copy.08585,conditional.03400,initial tactical reply prior.06638,
adaptive production-budget PUCT first replay.07049,average4replays.07237.
Prior-minus-conditional MSE game-bootstrap95%CI[.00158,.00517]. These are scalar
potential forecasts, not win calibration; adaptive opposing search need not model
SimpleHeuristic. Actual reply top3 coverage58.8%worlds; exactprior mass15.8%,
normal+Tera family31.6%. More regret iterations cannot restore missing responses.
Conditional hidden-world variance share2.3%; does NOT include uncertainty removed
by conditioning on actual move or excluded mechanics. Main residuals include
misses, speed ties, full paralysis. Restricted38-root disjoint-world ranking audit:
3world×2chance differsfrom reference6.89%,potential loss>.01 in2.66%;3×16 reduces
to4.00%/.55%. Finite conditional subset reference is not optimal policy truth.
PUCT action changed across4replays in26/46roots;medianoriginalarm visits32,mean
estimate repeatSD.0190. Deadline iteration variability also contributes.

Concretefix:mcts_simulator.setPriorSpecies maps Dex cosmetic formes and explicit
Polteageist-Antique equivalence tocanonical random-set lookup,then retains observed
species. All96failedworlds in6roots were faintedFlorges-Yellow/Polteageist-Antique/
Minior-Orange,notactiveopponentunknownsets. Same864seeds nowallbuild. Otherbattle
formes remain distinct; regressions coveridentity/HP/faint and noncosmeticcontrols.
Forecastmetrics deliberately pre-fix;noexpanded-sample/post-fixwin claim. Recommend
opposingactioncoverage/response modelling and chance replication before moreworlds;
potential-to-winning alignment/depth still untested. Localdoc:
docs/payoff-reliability-2026-09-11.md;data runtime/payoff-reliability-20260911/.
Added public recorder and offline conditional/prior/PUCT replay tools. PBRS,
learner/live service untouched; no rebuild/restart/commit.
Validation:37 focused service tests passed including full decoder battle;
TypeScript no-emit and focused ESLint passed. All864saved world constructions
succeeded afterfix. No post-fix forecast or strength claim.

## Removal ledger — 2026-09-11 ActionEnum (structure-only, bit-identical)

`ActionEnum` (41 values: 16 move sources, 6 reserves, 17 target slots and
two ALLY_i_SWITCH pseudo-slots) is deleted from proto/service.proto. Its two
ALLY_i_SWITCH values named no sequence row and no cell, and read as if the
model had ally-switch embeddings -- the confusion that surfaced while
designing a switch bilinear this date. In its place, three slot vocabularies
whose ascending value lists ARE the old slot lists position for position:
`MoveSlot` (16), `ReserveSlot` (6) and `TargetSlot` (17, keeping the
never-legal zero slot and DEFAULT as slots so no mask bit, cell or readout
row moves; deleting them is a separate, number-moving change). Both sides
build the lists from these (`_slot_values` / `slotValues`).

Previous action: `INFO_FEATURE__PREV_ACTION_SRC/TGT` (an ActionEnum pair
built by `cellToEnumPair` from the request kind and ally half) become one
`INFO_FEATURE__PREV_ACTION_CELL`, the block cell taken; a request with no
choice records the DEFAULT standalone cell (`NO_CHOICE_CELL`). The encoder
names the cell by the rows its logit is read from -- `heads.chosen_bank_rows`
over this step's pre-trunk private/move/target rows, plus the existing
src/tgt tags -- so the 41-row `prev_action_embeddings` table is deleted
(user call: the rows are described relative to SEQUENCE_LAYOUT). Not
circular: those rows are built from this step's features alone. InfoFeature
is renumbered (fields after the pair shift down one). A lead and a battle
switch of the same mon are now one previous action (one cell), where the
enum pair told them apart.

Deleted with it: `numActionFeatures` / `NUM_ACTION_FEATURES`,
`ALLY_SWITCH_(SRC_)INDICES`, `cellToEnumPair`, `TEAM_PREVIEW_TGT`,
`lastMaskKind` / `lastMaskActiveSlot`, the constants.py partition assertion,
and the packed-grid replay-shard fallback (`get_action_mask`'s branch,
`_cells_from_packed_grid`, its test) with `EnvironmentState` field 2, now
`reserved`. **replays/shards no longer decodes** (user call): rebuild it
before the offline critic program is next used. `simple_heuristic.ts`
decodes cells directly (`decodeCell`); its choices are identical by
construction -- team index = switch cell, ally = the mask's active_slot,
the same (ally, move, target, wildcard) for move cells, pass/DEFAULT at
-1e6 and every other standalone cell at -1e4.

Bit-identity: the prev-action rows are dead in singles (HAS_PREV_ACTION is
always 0, health check 2026-09-11), so nothing here reaches a number. The
local ex.bin fixture was converted in place (803 states, info column 12
dropped; the conversion refuses any state with a live previous action), and
the ckpt_00360000 EMA learner-side forward on it matched the pre-edit
capture exactly: 15 decoded input leaves and 73 output leaves, deterministic
XLA, with the pre-edit self-compare as the control; the only param leaf the
model no longer has is prev_action_embeddings. Test:
`test_prev_action_rows_are_the_rows_the_cell_names` (with the src/tgt tags
and group bias zeroed, each prev-action row equals the row it names exactly;
controls: the tags are live, and no previous action gives exact zeros).
Service: tsc clean, vitest 42 passed / 2 skipped. Revert handle: this
commit, and regenerate ex.bin with service/src/tests/ex.ts.

## Fix — 2026-09-11 previous action made live (numbers move)

`runner.ts` cleared the taken-action list (`actionEnumPairs`, now
`actionCells`) at the end of every request since 4c8836d (2026-08-31), so
each state was built from an empty list: HAS_PREV_ACTION was 0 on every live
and offline singles state, the two PREV_ACTION sequence rows were zeroed at
input and output, and `prev_action_embeddings` sat at init for the whole
lineage (health check 2026-09-11, defect 1). The per-request reset goes; the
per-battle reset in the constructor stays. Every state after a player's
first decision now carries the cell it last took (NO_CHOICE_CELL after a
request with no choice), which the encoder reads as the rows that cell's
logit came from (the ActionEnum removal above). Both rows are
policy-readable and carry nothing privileged -- the player's own last
choice, which it knows at deploy.

Invariant in `harness.playerController` (the vitest suite and the soak):
after a player's first decision, every state but a team-preview request
carries HAS_PREV_ACTION = 1 and the cell that controller last sent. Positive
control: the same invariant against the pre-fix runner failed in every
singles battle ("previous action lost: HAS_PREV_ACTION 0, PREV_ACTION_CELL
0, last sent N", N over leads, switches and move cells). Numbers move: two
more live rows in every singles state after the first decision. Revert
handle: this commit.

## Addition + removal ledger — 2026-09-11 switch pair readout (numbers move)

The switch block was one zero-init scalar per sheet row plus a single
context-free `switch_bias`: a candidate's logit could read only its own
post-trunk row. The 2026-09-10 switch-pair probes found the candidate-vs-
opponent matchup unreadable from that row (post-trunk pair at the majority
floor, raw pre-trunk pair .63), and the paired advantage audit (09-09)
found taken switches scoring below stays at every horizon -- consistent
with switches the readout cannot aim. The move block, by contrast, is a
bilinear against the target rows, and the trunk did learn to put the
move-vs-opponent matchup into the move row (.66 vs .57 floor).

Change (user call: "private embeddings @ [ally_switch_1, ally_switch_2]"):
ONE pair form in `FlatActionReadout` -- a bilinear between every source row
and every target row plus a scalar on each side, source-side `query`
zero-init -- now serves both pair blocks. Moves x targets keep their param
names (`query`/`key`/`local_src`/`local_tgt`). Switching is sheet rows x the
ALLY_i_TARGET row of the active slot being replaced (`decision_slot`: 0 in
singles, 1 for doubles stage 2): new `switch_query` / `switch_key`; the old
`switch` kernel stays as the sheet-side scalar under its own name (a merge
carries it); `switch_local_tgt`, a scalar on the ally row, is the
whether-to-switch level and replaces `switch_bias` (deleted, with
`switch_bias_telemetry` and its panel). There are no ally-switch rows (the
ActionEnum removal above deleted the pseudo-slots): ALLY_i_TARGET is the
row that carries my active mon i. It was masked off whenever no move
targets it -- every forced switch and team preview -- so the encoder now
also marks it valid whenever a switch cell is legal, gated by the format's
active count (singles never wakes ALLY_2). `CELL_BANK_TGT` for a switch
cell becomes that ally row (was the private row twice), so the transition
model's action rows and the previous-action rows see the pair too.

Four questions (CLAUDE.md): (1) bounded -- a bilinear of two RMS-normalised
rows times a learned kernel, the same form the move block has run since
08-29; (2) the softmax-mean direction is opposed by nothing new -- the pair
adds no loss term; (3) momentum: none beyond the move pair's; (4) shared
routes: the ally row is read by every switch cell of a row, the same
high-gain shape as a target column -- `player_switch_{query,key,local_tgt}
_rms` and `player_applied_delta_rms_switch_{query,local_tgt}` are panelled
from launch beside the move pair's.

Tests: `test_flat_readout` -- every logit still exactly 0 at init;
`switch_query` / `switch_local_tgt` live at step 1, `switch_key` frozen one
step and unfreezing when switch_query is nudged; the decision slot's ally
row moves all six switch cells while the other ally row moves none; a sheet
row still moves only its own cell. Merge relaunch, measured on
ckpt_00360000's EMA params over the bundled ex.bin: exactly `switch_query`,
`switch_key` and `switch_local_tgt` start fresh; the learned `switch_bias`
(-0.20) is dropped, and on the 52 states where both a switch and a move are
legal the mean switch probability goes .0524 -> .0644 (median .0458 ->
.0555, about exp(0.2)), every legal logit finite. Revert handle: this
commit.

## Change ledger — 2026-09-11 smooth support hinge (numbers move)

`support_hinge_loss` takes a `temperature` T (`player_support_temperature`,
0.1): each legal cell scores T * softplus(log(tau_row / pi_a) / T) in place
of max(0, log(tau_row / pi_a)); T = 0 is exactly the old hinge (no softplus)
and every existing hinge test still runs there. From
docs/porygon2_support_loss_recommendations.md, adopted at the user's call
over an objection recorded here so the verdict can be read later:

- The motivating harm (cells chattering across the kink) was never
  measured. On uddwfke8 at 250k-360k the cells under the line rested about
  20% below tau (mean ~.008 from loss / active fraction), held by a
  pi-proportional PG push against the hinge's constant lift -- a stable
  resting point below the line, not a pile-up at it -- where T = .1 gives
  .90 of the hinge's lift. Minibatch averaging over states already smooths
  the kink in what the optimiser sees.
- It is not a pure smoothing: the lift now reaches ABOVE tau (~10% of full
  at 1.25 tau, ~2% at 1.5 tau, ~.1% at 2 tau), so a cell the critic is
  indifferent to rests near 1.2-1.5 tau, effectively a slightly higher floor,
  and the 09-09 hinge's "exactly silent above the line" property -- chosen
  because a force at every probability pinned entropy_micro_taken in sp75c --
  is given up at small magnitude.
- It lands in the same relaunch as the output norm, the live previous
  action and the switch pair readout, so no strength or switching change
  after the relaunch can be attributed to it alone.

Kept: log-space (the rescue term on a starved cell stays pi-free, s -> 1),
the feasibility clamp, bounded (<= 1) zero-sum derivative
`mean(s) * pi_b - s_b / N`, tau .01, coefficient .05, and a HARD
`player_support_active_fraction` (cells actually below the line). Tests:
the derivative against jax.grad on a row with a cell at 1.2 tau (the soft
band exercised), 0 < smooth - hard <= T log 2 at T = .1 and .01, a cell at
1.2 tau lifted by the smooth loss and only pushed down by the hinge, illegal
cells and a +-1e4 saturating row at both temperatures. Read it by
`player_support_frac_below_p01` / `_p005` and the switch cells on the floor
at the ~430k check; revert by setting the temperature to 0.0 (exact) or by
reverting this commit.

## Addition ledger — 2026-09-11 PBRS potential channel (lands dark at eta 0)

**What.** The human-replay position potential (2cedbaf fit, H+N+M) enters
learning as a second v-trace channel beside the win channel. The service
(406efe0) writes the public-view unit potential into
INFO_FEATURE__STATE_POTENTIAL on every state: both sides from publicBattle,
own HP through the sim's shared percentage rule, so p1 == -p2 exactly (the
harness requires it). The learner: `player_potential_strength` (eta) > 0 runs
`scalar_vtrace` (e04483e, the structure-only extraction) on reward
d * Psi' - Psi, Psi = eta * Phi on live nonterminal rows, value
W = eta * potential_head (target net) forced 0 on done and padding rows, and
adds its advantage to pg_advantages before normalisation. The learner-only
potential_head (RegressionValueLogitHead, zero-init output, built only at
eta > 0) reads stop_gradient(CLS) and regresses its channel returns (MSE,
coefficient 1, live nonterminal rows). The win critics are untouched.

**Why this form** (user decisions 2026-09-11; derivation and review record in
docs/human-switch-pbrs-2026-09-11.md and the session plan):
- The channel's exact value is -Psi under any policy; with it every channel
  TD is 0 (PBRS invariance). A zero-init head starts at W = 0, where the
  channel adds -Psi_t + (1 - lambda) * sum_k lambda^(k-1) * Psi_(t+k) on
  fully on-policy rows, and converges to -Psi: the head's lag IS the shaping.
  Handing the head -Phi analytically would be inert.
- At a merge the channel's advantage equals a win critic shifted by
  +eta * Phi (max difference 1e-17 over 4,000 random thresholded games). The
  residual-critic alternative absorbs that by retargeting the trained win
  critic -- support migration, introduced clipping, a (1 + eta) merge
  rescale, a rollback that reinterprets rather than restores. Declined.
- Uncentred on purpose: the per-game shaping sum is -eta * Phi(h0),
  |.| <= .0046 at eta .05 (H = N = 0 at the first request, no team
  preview), action-independent. Centred rewards give the same learner only
  with the start offset ALSO on the value; centred with a plain head never
  goes inert (a residual eta * Phi(h0) on the final transition).
  tests/test_potential_channel.py pins both.
- Done rows: rho there is a sampled ratio on a no-decision row (the actor
  forwards the terminal state; thresholded_target_ratio has no done case).
  Forcing W = 0 makes the done-row channel TD exactly 0 whatever rho is.
- Declined: shaped reward with a win-only critic and no head (the July
  927d80d shape -- a permanent, non-overridable near-term-potential bias);
  a head with trunk gradients (a permanent auxiliary regression of the trunk
  onto a human-derived target).
- Lineage: 925b620 (2026-04-02) learned potential-head channel -> 357f357
  one-step F on the advantage (not invariant) -> 927d80d Phi-valued channel
  with its own lambda (July) -> retired August.

**Bounds are measured, not claimed.** The only analytic bound is the loose
2 * eta / (1 - lambda) = .50 at W = 0; observed .107 max over 4,000 random
thresholded games; the review's thresholded counterexample reached .132. The
head's gradient enters the GLOBAL clip norm (clip_by_global_norm 10; logged
player_gradient_norm 3.2-9.1), a coupling the stop_gradient reach test cannot
see -- read `player_potential_head_grad_share`.

**Verified at landing.** vitest 43 passed (one Illusion slot-alignment flake,
known-open, clean on 4 reruns); pytest fast 395 passed + 2 GPU-OOM failures
that reproduce on e04483e (a live learner held 9.4 of 12 GB); e04483e
bit-identical on ex.bin; 7 channel tests with controls; Python/TS fixture
parity. NOT yet run, all gated on a learner-free window:
tests/test_potential_slot.py (slot invariance, gradient reach), the eta > 0
train_step slow test, the slow suite, and the Step 3 gradient screen
(rl/offline/potential_screen.py) that gates any launch.

**Panels.** player_potential_{mean,std,switch_delta_mean} (eta-free; the
switch delta is descriptive, humans read -0.038); player_potential_adv_share
(should fall as the head fits; a floor is the unfitted part persisting);
player_potential_head_fit_r2 (the head against its exact target -Phi, the
distance from inert; pre-registered >= .9 by 20k); player_potential_head_r2;
player_potential_win_adv_corr; player_potential_adv_{switch,move};
player_potential_head_grad_share.

**Revert.** eta 0: the head is not built, the param merge drops its leaf, and
the learning rule is today's exactly (the win critic was never retargeted).
Or revert this commit and 406efe0 (the slot then reads 0 again).

**Found during verification, outside this change.** The WIN payoff's done-row
TD is rho_done * (r - V_done) with rho_done a sampled ratio, 0 when the
sampled cell is thresholded away. The fixed point is intact (V_done -> r while
E[rho] > 0) but the payoff is slowed and dropped on those samples. Candidate
separate fix: rho = 1 on done rows.

### PBRS screen on ckpt_00480000 — 2026-09-11 (launch at eta .05)

rl/offline/potential_screen.py, 32 self-play games from ckpts/gen9/ckpt_00480000
(the live run's newest; its Ctrl-C checkpoint was skipped, the known
interrupt-checkpoint defect), 64 chunks / 16 batches, 98.8% of rows carrying
the potential. eta 0 against eta .05 with the head fresh at 0:
- Policy-logit gradient: RMS perturbation .031 against the .10 budget (norms
  .475 -> .478, cosine .9995); voluntary-switch rows (54) .026, stay rows
  (1362) .035. PASSED with a 3x margin.
- Applied update, shared params: RMS perturbation .0047 (action_head .018,
  encoder .006, the rest <= .0013).
- Global norm under the clip at eta .05: FAILED as written (max 18.3 vs 10) --
  but the clip already binds at eta 0 on 8 of 16 batches (7.0 to 18.3): the
  criterion's premise (logged player_gradient_norm 3.2-9.1, clip inactive) was
  false for these on-policy batches. PBRS's own part: the global norm moves by
  x1.0002-1.0109; the head's share of it is .024-.142 (added in quadrature).
  The pre-registered fallback (give the head its own optax transform) was
  conditioned on the clip binding BECAUSE of the head; that trigger is not met.
- The channel's advantage std share at W = 0: .021-.053.
Decision: launch at eta .05 (the user asked for the relaunch once done); the
clip criterion's failure is recorded here rather than reinterpreted. Watch
player_potential_head_grad_share and player_potential_adv_share falling; the
clip binding at eta 0 on half the fresh batches is itself new information
about the lineage. Revert: eta 0.

## Addition ledger — 2026-09-12 pairwise entity critics (numbers move)

Two learner-only value heads beside the CLS critics (`rl/model/heads.py`
`PairValueHead`; plan `~/.claude/plans/help-me-construct-adding-agile-
jellyfish.md`), each a generalised additive model over 12 entity rows, my
side first:

    V = sum_mine u_i - sum_theirs u_j
        + sum_{i mine, j theirs} alpha_ij tanh(g(i,j) - g(j,i))
        + sum_{mine pairs} beta s - sum_{their pairs} beta s

u a per-mon MLP over the row plus a zero-init projection of its context
(the global field row beside its OWN side's); g one bilinear over all 12
rows read in both orientations (strength of i over j and pressure of j on i
are the one number, antisymmetric by construction); alpha a softmax over
ALIVE cross pairs of a second bilinear symmetrised (a fainted or absent mon's
pairs weigh exactly 0; concentrating on the decisive matchup is the intended
reading, so no entropy term); s = tanh(g_s(i,i') + g_s(i',i)) the same-side
synergy from one bilinear shared by both sides with its own symmetric
weights. Sharing every function across sides makes a side swap negate V
exactly (`tests/test_pair_value_head.py`). User's form (2026-09-12),
replacing the first draft's per-edge gates + entropy + squared-norm
penalties: identifiability comes from what each term is allowed to see,
not from a penalty. The `bilinear_pair` form is the action readout's,
hoisted (378243a, bit-identical).

Inputs. PUBLIC head: the post-trunk public rows with the post-trunk field
triple (user's call over pre-trunk rows: the pair term can see the roster
through attention, so its locality is nominal and the antisymmetry acts as
weight tying). PRIVATE head: both sheets through the ONE private embedder,
before either side bias and before the opponent's code (`encoder.
PairValueInputs`; the user's call over the discrete-code rows), so its two
sides are one representation. Alive = hit-point ratio token > 0 off the
wire.

Loss. MSE of each head's scalar on `PlayerTargets.scalar_returns` -- the
v-trace scalar the CLS critics' two-hot is built from, clipped as `two_hot`
clips so it equals `win_returns @ support` exactly -- under `value_mask`,
coefficient `player_pair_value_loss_coef` (1.0; 0 builds neither head).
Gradient LIVE into what each head reads: the public head shapes the trunk
through 12 entity rows (the CLS critic reaches it through one), and the
private head trains the private embedder directly where today only the
code's straight-through argmax does. CLS critics unchanged = the matched
control; value bootstraps keep `player_privileged_targets`.

Four questions: (1) bounded -- pair parts are convex combinations of
numbers in [-1, 1], unary an MLP over RMS-1 rows against labels in [-1, 1]
under MSE; (2) a value force, not a logit force; (3) Adam as for v_head,
every term smooth; (4) a public row is read by 6 cross and 5 same-side
pairs (the many-readers shape of a target column): per-function
query/key rms + applied-delta panels, `player_pair_value_{head}_gradient_
norm`, `player_trunk_out_row_l2_<group>`.

Init facts (tested): V == 0, weights exactly uniform over alive pairs;
queries and the unary's last kernel move at step 1; keys, the weight
scores and the context projection are frozen ONE step (gradients
proportional to the pair terms / queries, both 0) and unfreeze at step 2.
Each head 722,433 params (qk 256, unary hidden 256).

Panels: `player_pair_value_{public,private}_r2` on "Value R2 (main head)"
beside the CLS pair; part shares (var(part)/var(V)), signed partials,
weight entropies (normalised), unary cancellation (mean sum|u_i| / mean
|V|), |m| / |s| means, kernel rms, applied deltas. Offline:
`rl/offline/pair_value_probe.py` -- hit-point monotonicity of m by
one-bin finite differences (a derivative along the scalar column alone
would miss the one-hot half); violation fraction and share, a measurement
and not a loss (Flail, Reversal, Endeavor, berries).

Pre-registered (hold 100k after launch): both pair R2s > 0 and rising by
20k with the cross share off the floor; end of hold public pair R2 within
.05 of `player_value_head_r2`, private pair R2 >= public, matched control
not down > .02, eval winrate and steps/s within 5%. Fallbacks: control
regression -> halve the coef once, a second regression falsifies the
placement -> coef 0; cross share at the floor while R2 rises -> the
pre-trunk raw-row pair term added to g (probe-recommended two-term shape),
its own commit; material monotonicity violations beyond the known
exceptions -> the derivative penalty as its own numbers-move commit, the
user's call. Step 2 (after the hold): `player_privileged_targets` ->
`player_value_target_route` in {deploy, privileged, pair_public,
pair_private}, `compute_player_targets` taking a scalar bootstrap.

Reference numbers at landing: uddwfke8 CLS R2 at the stop for this
relaunch (637,000, ckpt_00637000): deploy .8173 / privileged .8481, gap
.0586 -- the 2026-09-01 gate PASSING. The 485k summaries had read .540 /
.513 (privileged below deployable for ~120k steps from ~365k), a dip the
run climbed out of, not a standing failure; recorded here because no row
carried it. Revert handle: this commit; `player_pair_value_loss_coef 0`
is the bit-exact off.

## Addition ledger — 2026-09-13 pair terms centred over the alive pairs (numbers move)

What the 100k hold read (uddwfke8 from ckpt_00637000, bins of 20k): both
pair R2 .77 -> .80 against CLS .80 -> .80, privileged .81 -> .80 — parity,
the private head not above the public. Part shares: public unary .21 ->
.60, cross .32 -> .03, synergy .39 -> .015; private cross .31 -> .40,
synergy .45 -> .006; unary cancellation private 1.4 -> 2.1, public .8 ->
1.0; weight entropies (normalised) .86-.93 throughout. Query kernel RMS grew
like a square root (.0046 / .0084 / .0107 / .0122 over the four 40k bins, a
random walk under Adam — sign-inconsistent gradients); the keys sat at their
lecun init (.062 -> .061), never moved.

Offline read (740k target params, 3,057 rows of the 182k lineage dump,
label = game outcome, scratch script — the outcome label puts every head
near .2, relative order is the read): CLS .204, privileged .211, public
pair .209, private pair .211. Public: V without the cross term .220, unary
alone .218 — the pair terms are noise on this label, cross std .09 vs
unary .43. Private: V without the cross term .052, cross alone (linear
refit) .200 — the cross term carries the fit, but corr(cross, unary) .75
and every matchup bucket sits at the same mean (type-disadvantaged -.163,
neutral -.127, type-advantaged -.142): a state-wide offset repeated over
all 36 pairs. corr(m, type-effectiveness asymmetry) .05 public / .03
private; corr(m, hit-point difference) .30 / .17; the softmax weights vs
|type asymmetry| -.09 / -.05 — noise: the pair actually on the field is
exactly as type-lopsided as the average alive pair (.742 vs .737 over 666
states), so the selection story is falsified. The within-state variance
split and the hit-point monotonicity probe were killed twice by the host
low-memory watchdog beside the live learner (9 GB RSS) — owed at a
learner-free window.

Diagnosis: the label is one scalar per state; each term of an additive
model learns only from the residual the others leave, and a unary over
POST-trunk rows (row i already carries j through attention) absorbs
sum_j g(i, j) entirely. The pair term found the cheapest thing left — the
per-side offset — and the weights, whose only gradient is a pair's
deviation from the weighted mean (d/d logit_ij = (V - y) alpha_ij (m_ij -
sum alpha m)), had nothing to move on. Sparsity penalties were declined
again (the 09-12 form dropped them once): a force where the task gives
none picks an arbitrary pair; L1 on the cross term is coefficient 0 with
a panel; L1 on the unary moves the offset into the pair term (the private
head already shows that shape). Not a temperature: `bilinear_pair` already
divides by sqrt(qk), and the logits are 0 at init by the zero query.

Change (heads.PairValueHead, `centred_over`): m and s are centred over the
alive pairs of the state (cross over the 6x6, synergy per side) — a hard
structural restriction, no coefficient: an offset cannot live in a pair
term, only in the unary, so whatever a pair term carries differs across
pairs, which is also the only gradient its weights can receive. The
weight queries (`cross_weight_query`, `synergy_weight_query`) take a LIVE
lecun init, found by the new test: a centred term under exactly uniform
weights is two zero factors (d/dm_ij = alpha_ij - 1/n = 0 and d/dalpha
proportional to m - mean m = 0) — an exact saddle the pair queries' zero
init cannot leave. V == 0 at init still (the pair queries are zero);
"weights uniform at init" is no longer a property. Side-swap
antisymmetry holds by construction (the mean is symmetric under the swap).
Each pair part is now bounded by two units, not one. No new parameters;
the weight queries resume from their checkpoint values (RMS .012, a
contrast already). Numbers move: the unary must re-absorb the offset the
pair terms carried, expect a transient on the pair losses at the relaunch.
The hit-point monotonicity probe reads the CENTRED m: stepping one mon
shifts every other pair by -delta/n, a known 1/n contamination of its
per-pair sign test.

What this does NOT do: create per-pair information. That needs a per-pair
label (which mon damaged or knocked out which, from the self-play history
— 36 labels per state, self-play derived, on the head not the trunk) or a
unary that cannot see the partner (raw rows). Moving only the cross term
to pre-trunk rows does NOT fix it either: it competes for the same
residual against a contextual unary and loses — the 09-10 probe's .63 was
a raw pair bilinear alone. Pre-registered read (hold 50k): weight entropy
(normalised) leaving .9 and public cross share off the floor (> .1) by
25k; pair R2 not below CLS - .02 at 50k. Otherwise the pair machinery is
cost without a read and the coefficient goes to 0. Revert handle: this
commit; the centring is one call per term.

## Removal ledger — 2026-09-12 latent world model, opponent code and dynamics rows (explored; open to revisit)

Explored, not failed. The latent world model (stochastic transition
g(h_t, u, z), latent action u, chance code z, K=2 unroll, recursive
decision/chance backup, MCTS), the opponent discrete code with its belief SSL
stack, and the dynamics target rows were removed in one pass because the
scope is too large while the base agent is not yet strong (49.5% vs
SimpleHeuristic at T=1, ~1M fresh chunks, Tera-by-turn-3 82% vs human 3%,
voluntary switches 13% vs 20%). The documents of the time are explicit that
the offline reads "are not independent falsifications of all world-model
architectures" and that "weak aggregate calibration and improved play can
coexist"; the privileged critic STAYS and now reads the raw private-sheet
latent (the same embedder output the code used to quantise). Long-form
retrospective, timeline, every measured number and the reusable pieces:
`docs/latent-world-model-retrospective-2026-09-12.md` (local, gitignored).

Reopen when: the base agent is clearly above the heuristic baseline at T=1
with human-scale behaviour statistics; a value-blind search control exists;
the pooled uncentred prior-expectation delta gain clears copy with an interval
excluding 0 (it straddled 0 on every checkpoint: −0.073 [−0.186, +0.040]
@1.80M, −0.056 [−0.151, +0.020] @1.835M, −0.064 [−0.105, −0.012] @1.889M);
or a format with more chance / hidden information makes the Step-1 instrument
read above its 0.10 bar (gen9 randbats read 0.051). For the belief code: a
label whose margin over the mon's own revealed row (measured 0.044) is worth
its cost.

Tag `pre-world-model-removal-2026-09-12` on 7d222e1 (branch
`pair-value-critic`). Recovery for any row:
`git checkout pre-world-model-removal-2026-09-12 -- <paths>`; each removing
commit is a single `git revert`.

| mechanism | paths deleted | removing SHA | recovery |
|---|---|---|---|
| offline probes + the unpaired interval stack (REMOVED) | `rl/offline/{transition_probe,event_probe,search_samples_probe,search_ablation,interval_features,interval_data,train_interval,direct_interval}.py`, `rl/model/interval_transition.py`, `tests/{test_search_samples_probe,test_interval_transition,test_interval_scaffold,test_interval_data,test_direct_interval}.py` | `ce5c3f0` | `git checkout pre-world-model-removal-2026-09-12 -- rl/offline/transition_probe.py rl/offline/event_probe.py rl/offline/search_samples_probe.py rl/offline/search_ablation.py rl/offline/interval_features.py rl/offline/interval_data.py rl/offline/train_interval.py rl/offline/direct_interval.py rl/model/interval_transition.py` |
| search + MCTS (REMOVED): the recursive decision/chance backup, `mcts_root` / `_guarded_cond`, `cfg.search`, `SearchOutput` / `PlayerActorOutput.search`, `search_bonus`, `actor_params_view(search=)`, the inference server's `--search` / `--search-depth`, the offline harness's search args | `rl/model/search.py`, `rl/model/mcts.py`, `tests/test_mcts.py`, `tests/test_search.py` (the `eval_game_logs` half rehomed to `tests/test_eval_game_logs.py`), edits in `rl/model/player_model.py`, `rl/model/config.py`, `rl/environment/interfaces.py`, `inference/{model,server}.py`, `rl/offline/harness.py`, `tests/test_actor_device.py` | `45f833e` | `git checkout pre-world-model-removal-2026-09-12 -- rl/model/search.py rl/model/mcts.py tests/test_mcts.py tests/test_search.py` + `git revert <sha>` for the edits |
| `TransitionModel` (REMOVED): `rl/model/transition.py` (latent action encoder / candidate generator, chance prior / posterior, `RowRead`, `imagine`, grounding / kind+done / terminal-outcome readers, K-step unroll), `transition_objectives.py` (exact decode loss, Plackett–Luce candidate targets), `categoricals.py`; `_forward_transition`, `dynamics_alignment`; `cfg.transition`; the `transition_*` output leaves, `OFFSET_LEADING_LEAVES` / `batch_out_axes`; the eight learner coefficients (`player_dynamics_coef`, `player_transition_{dyn,rep}_coef`, `_free_nats`, `_cons_coef`, `_value_trains_v_head`, `_decode_coef`, `_align_coef`); `train_step.{dynamics_losses,transition_losses,…}` + the learner `sampling` rng; `hp_input_rows`; `_TRANSITION_LEAVES` + the seven transition grad-norm panels; wandb section 3b | `rl/model/transition.py`, `rl/model/transition_objectives.py`, `rl/model/categoricals.py`, `tests/test_transition_model.py`, edits in `rl/model/{player_model,config,state_features}.py`, `rl/environment/interfaces.py`, `rl/online/{config,artifact}.py`, `rl/online/training/{train_step,telemetry}.py`, `rl/offline/potential_screen.py`, `scripts/wandb_views.py`, `tests/{test_train_step,test_privileged_partition,test_history_carry}.py` | `2c50b97` | `git checkout pre-world-model-removal-2026-09-12 -- rl/model/transition.py rl/model/transition_objectives.py rl/model/categoricals.py tests/test_transition_model.py` + `git revert <sha>` |
| opponent discrete code + belief SSL (REMOVED): `opp_code_logits` / `opp_code_embedding` / `_opp_code_rows` (the 16×16 straight-through code on the secret rows), `belief_alignment`, `OppCodeLabels`, `_hidden_code` / `_opp_code_labels` (the hidden-token label), `belief_head`, `species_belief`, `revealed_belief`, `cfg.encoder.opp_code` / `cfg.belief_head` / `cfg.revealed_belief`, `player_belief_coef` 0.25, the `opp_code` / `hidden_code` / `belief_*` output leaves, `train_step`'s belief block, `telemetry.{_OPP_CODE_LEAVES,code_usage_logs,belief_accuracy_logs,_code_marginal}`, wandb section 3 + the two opp-code drift panels | `tests/{test_hidden_code_label,test_belief_telemetry,test_revealed_control,test_species_control,test_opp_code_telemetry}.py`, edits in `rl/model/{encoder,player_model,config,constants}.py`, `rl/environment/interfaces.py`, `rl/online/config.py`, `rl/online/training/{train_step,telemetry}.py`, `scripts/wandb_views.py`, `tests/{test_privileged_partition,test_train_step,test_actor_sequence,test_model_forward,test_resume_merge}.py` | `0c4b8a5` | `git checkout pre-world-model-removal-2026-09-12 -- tests/test_hidden_code_label.py tests/test_belief_telemetry.py tests/test_revealed_control.py tests/test_species_control.py tests/test_opp_code_telemetry.py` + `git revert <sha>` |
| dynamics target rows (REMOVED): `DYNAMICS_TARGET_ROWS` / `DYNAMICS_GROUP_SLICES` / `NUM_DYNAMICS_ROWS`, `Encoder.dynamics_rows`, `PlayerActorOutput.dynamics_target` | edits in `rl/model/{constants,encoder,player_model}.py`, `rl/environment/interfaces.py`, `tests/test_actor_sequence.py` | `c4375c0` | `git revert <sha>` |

**Numbers move at the next relaunch** (checkpoint-mode by-path merge drops the
removed subtrees and their Adam moments; nothing is a manifest field): the
shared `v_head` loses the imagined-row CE it trained on under
`value_trains_v_head`; the global grad clip at 10 stops counting the
transition and code gradients, so every surviving parameter's effective step
changes; the privileged critic reads the raw sheet latent instead of the
16×16 code (strictly more information; `opp_code_*` params dropped). Watch
`player_priv_value_head_r2` against its standing gate
(`>= player_value_head_r2` from 20k on; at ckpt_00637000 it read .8481 vs
.8173). The actor forward is structurally untouched (`search.enabled` was
False; every removed encoder branch was under `cfg.train`). Slow suite and
the full-lattice `train_step` smoke owed at the next learner-free window.

**What was measured (compressed; each: what it measures, which way is good).**
Delta gain = 1 − SSE(pred vs V_{t+1}) / SSE(V_t vs V_{t+1}), copy = 0, higher
is better. Step 1 (mean head, ckpt_01220000, 3,340 transitions): value gap
0.051 on hp-moved rows vs copy 0.074, bar ≥ 0.10 — neither gate branch fired;
long-span cost nil (0.038 vs 0.045: semi-Markov diagnosis unsupported).
Launch check 1: content grounding head loss 17.8, grad norm 22 vs clip 10 →
delta form. Launch check 2: prob_switch 0.04 → 0.014 through an uncounted path
(next-policy loss through the live readout = KL(π_{t+1}‖π_t) on the real
trunk) → observer stop-gradients. Launch check 3: kl_free_frac 0.93 (F = 1
nat over 2 groups) → F 0.0625 + RowRead. Step-2b hold: kl 0.12–0.18 (band
0.5–3), post perplexity 2.6/16 (bar 3) — "price it by play". Step 3: root_kl
0.015–0.04 (band 0.05–0.5, "search agrees with π"), 3.4× latency; "Do 1"
probe: branch spread 0.039, oracle gain 0.014, root_kl floor ~0.02 at 64
draws — sampling is not the lever; value_r2 0.84 VACUOUS (copy scores high on
the t+1 label). B: out_proj_rms 0.017 pinned across three gradient regimes.
D (rep 0): kl 0.17 → 0.64, kept; post perplexity flat 2.6. Posterior sampling:
perplexity FELL 2.54 → 2.50, rich-get-richer falsified. Event probe: prior z
≈ best MLP on its own input; label ceiling ~0.25. Rescored calibration
(pooled, exact 256 codes, whole-game bootstrap): posterior +0.33/+0.34, prior
expectation −0.073/−0.056 straddling 0, switches +0.18 [+0.07, +0.24] above
copy, moves −0.11 below. Search win rate +5.8 pp [+2.1, +9.5] on evolving
checkpoints (the one positive search read); at 01861967 the arm moved root KL
by 1e-5 — pricing nothing. Latent actions: decode_acc 0.80, action_mi 1.35
nats, value_delta_r2 +0.18 at launch; depth-2 arm 11× depth 1, learner RSS
13.7–14.4 GB. MCTS pilots (plain / MCTS1 / MCTS2 / exp1 / exp2): 45/48/50/41/50
then 55/47/42/45/51 — within noise, 1/64 forced-coverage confound unremoved;
GPU latency 118 → 20 ms at depth 1 after the mctx-style layout, bit-identical.
Action-code audit: 87% reconstruction, MI 1.59 nats, 35.6% of same-move
Tera pairs aliased, predicted value gap 0.002. Interval stack: held-out −0.35
/ −0.29 / −0.59 at train +0.99 (fits, does not generalise); direct probes
select step 0; bypass paired +0.011 [−0.007, +0.029], inconclusive; teacher
KL improves while successor MSE worsens 8% (different targets). Consistency
preflight: 99.3–99.7% of the loss from absent PREV_ACTION rows (norms
2,477–8,256 vs 12–21) → endpoint-union validity. Belief: accuracy .847 vs
revealed-row control .803 vs species .551; above-marginal .442 / .399 / .146;
context margin .044; code params moved ~10% in 182k through the priv-value CE
alone; the cosine mean dynamics head never positive over 230k (−0.27 → −0.019).

**Reusable pieces (one line each; a paragraph each in the doc).** Zero-init
delta-form grounding head (starts at copy, loss 1 / gain 0). Observer
stop-gradients + frozen-clone head application + the inverted reach test.
`RowRead` per-row read + per-group free nats ("the floor never goes up").
Latent action alphabet with the EXACT finite-alphabet decode objective and
without-replacement autoregressive candidates (nothing masked past the root).
Conditional terminal-outcome head: B = (1−c)T + c Σ μ Q fixes the (1−c)V +
cE[Q] double count. Straight-through posterior sampling with a liveness panel.
mctx-style traversal/expansion split + `_guarded_cond` (singleton vmap keeps
batched arithmetic). Endpoint-union consistency validity ("the missing mask is
in the comparison loss"). Calibration accounting: uncentred delta gain, exact
code grid, whole-game bootstrap, modality splits. Dreamer-style
straight-through code as a learner-only bottleneck grounded by a value loss;
the hidden-token label (mask every token the public row shows) with its
species and revealed-row controls. The rename rule: a head whose meaning
changes under an unchanged shape is renamed or the by-path merge resumes it.

**Verdicts (quoted).** "~0.02 is the depth-1 operator's ceiling and sampling
is NOT the lever." "The all-rows interval STRADDLES 0 on both checkpoints: the
block on C and rung 2 stays." "Implementing the architecture is distinct from
training its parameters." "MCTS does not improve in this pilot … no
value-blind matched control was run, so do not attribute the complete gameplay
deficit to the world model." "They are not independent falsifications of all
world-model architectures … This does not prove the achievable gain is zero
here or validate the current world model." "Keep policy-gradient baseline;
test representation/action-effect calibration and value-blind search control
before promoting search." "The root representation … was learned for
policy/value prediction. Its sufficiency for a Markov transition model is not
guaranteed." Belief: "most predictability is available from the mon's own
revealed row." User, on why Step 2 proceeded past a failed Step-1 gate: "in
other formats it will absolutely be necessary."

Acceptance read after the pass (2026-09-12): the production actor forward
(Agent.step_player on the host CPU device, f32, `actor_params_view`, the
bundled example step, params from ckpt_00637000) is bit-identical across
all seven commits — 15 output leaves compared exactly against the dump
taken at the tag (`runtime/tidy-bitcheck/actor_bitcheck.py`). The learner
loss is not bit-identical by design (the notes above); the slow suite and
the full-lattice train_step smoke are owed at the next learner-free window.

### Amendment — 2026-09-12 pairwise critics: no field context, private head post-trunk (numbers move)

User call at the launch check ("the trunk should share the context as it
needs"): the head's zero-init context projection (each mon's own side's
field row beside the global one) and the private head's pre-trunk sheet
latents are gone. Both heads read POST-trunk rows -- the public head the
PUBLIC rows, the private head PRIVATE_ROWS then OPP_PRIVATE_ROWS (the
opponent-truth rows, raw sheet latents plus side bias since the code's
removal the same morning) -- and `PairValueInputs` is the two alive
flags only. The context existed for the pre-trunk private head, which had
seen no field; on post-trunk rows it was redundant. Consequences: the
private head's gradient now shapes the trunk through the secret rows
instead of training the private embedder directly, and its side-swap
negation is exact at the HEAD (the test) but nominal through the trunk
(my sheet rows never read theirs; theirs read mine). Each head 591,361
params (was 722,433). The first launch of the context form ran minutes on
fresh heads and was stopped; the relaunch is again from ckpt_00637000.

### Amendment — 2026-09-12 pairwise sheet critic uses opponent public rows

User-directed routing change: `pair_value_private` now reads post-trunk
`PRIVATE_ROWS` × `OPP_PUBLIC_ROWS`, with opponent validity and alive flags
from the public channel. Neither pairwise head reads opponent-private rows
or their HP flags. Motivation: prevent the auxiliary critic from placing
useful matchup computation exclusively in privileged representations absent
at deployment. The privileged CLS critic and return estimator are unchanged.
Head names and parameter shapes are preserved for checkpoint compatibility;
the sheet head now pairs different representation groups, so algebraic
head-side antisymmetry remains, but does not establish whole-network symmetry.
Unrevealed public opponents remain masked; this adds no inferred hidden slots.
The full-model partition test now requires both opened pairwise heads to be
invariant to opponent-private perturbations. No strength result or training
restart is claimed. Revert: restore the sheet head's opponent slice and
validity to `OPP_PRIVATE_ROWS`, and its opponent alive flags to private HP.


## Comment sweep — 2026-09-12 evidence migrated out of the code (tag `pre-comment-sweep-2026-09-12`)

The comment policy landed the same day: a comment carries the WHY only — a
non-obvious constraint, a deliberate deviation, a workaround — and never
narration, a restated name, a re-verified signature or change context. 445
narration lines went first, then 50 comments that made FALSE claims were
corrected or deleted (five still called the sequence 61 rows against the
code's asserted 80). What follows is the third pass: the measured evidence
that was living in the source, moved here so it survives the code it
annotates, which is what this file is for. The code kept the constraint and
lost the measurement; nothing was discarded. Subsections are keyed by the
file the evidence came from, so grep the path as well as the mechanism.

Pre-registered thresholds, gates and revert triggers stayed in the code:
they govern decisions not yet taken, so they are live constraints rather
than findings. So did algebraic forms, cross-file contracts, Showdown
protocol quirks and the statement of what each test's positive control
proves. Pairwise-critic figures are not restated here — they live in
"Addition ledger — 2026-09-12 pairwise entity critics" and its amendments.

# Migrated evidence — `rl/model/**` and `rl/environment/*.py`

Removed from code comments/docstrings on 2026-09-12. Every number, date, run
id and checkpoint name is verbatim from the source it was removed from.

### rl/model/modules.py — EntitySumPool, the type-legibility measurement

Measured 2026-09-03 (`rl/offline/type_probe.py` and the supervised ceiling
beside it): the same readout form reached held-out 0.60 on the
attention-pooled rows and 0.79 on summed ones, against 0.80 from the raw
multi-hots — the attention pool was eroding type legibility, not adding
within-entity interactions the trunk could use.

### rl/model/modules.py — SequenceNormalisation, why RMS 1 and what the output norm fixed

The final norm every pre-norm transformer carries (GPT-2's ln_f, LLaMA's
model.norm, ViT's head norm), which the trunk had without: on ckpt_00280000
the heads read CLS at RMS 9.8 against move rows at 0.99, the disparity this
norm removed at the input coming back at the output.

RMS 1 per row is what a regular transformer's embedding lookup gives —
torch's N(0, 1) table, the original Transformer's sqrt(d_model)-scaled
embedding, PaLM's — and it puts the input at 1.6x the first block's update at
init (Gemma's .01*sqrt(D) rows sit at 0.75-1.1x; measured 2026-09-10, block-1
update RMS 0.62 at width 256 under the pre-norm blocks, independent of the
input scale). Under those blocks a row's residual magnitude decides how much
any block can move it: unnormalised, the history rows entered at L2 ~1040
against CLS at 2.85 and a block update of ~60, so six blocks moved them 2% —
frozen input the trunk could read but never revise.

### rl/model/modules.py — `layer_norm` naming

The docstring claimed RMS for months while the code did not (it is
nn.LayerNorm, not the RMSNorm above).

### rl/model/modules.py — attention backend benchmark

DELIBERATELY the plain einsum, not jax.nn.dot_product_attention / cuDNN
flash. Measured 2026-09-01 (fwd+bwd ms, bf16, 4 heads x 64, B*T = 8192
tokens; einsum / dpa-xla / cudnn+dense-mask / cudnn+seq_lengths):

| seq | einsum | dpa-xla | cudnn+dense-mask | cudnn+seq_lengths |
|---|---|---|---|---|
| 64 | 0.12 | 0.27 | 0.54 | 0.32 |
| 128 | 0.16 | 0.42 | 0.46 | 0.53 |
| 256 | 0.68 | 0.70 | 0.73 | 0.58 |
| 512 | 1.02 | 0.72 | 1.30 | 0.75 |
| 1024 | 1.42 | 1.53 | 2.33 | 1.71 |
| 2048 | 4.11 | OOM | 4.23 | 3.09 |

At the trunk's 73-80 rows the einsum is 2-4x FASTER than every flash variant
(kernel overhead dominates), and the only variant that ever clearly wins
(cudnn+seq_lengths, from ~256 and decisively at 2048 where plain xla OOMs)
cannot express this module's SCATTERED validity mask. The softcap is not a
blocker for a future flash switch — max |pre-cap logit| on the trained model
measured 7.6 against the 50 cap (qk layer norm bounds it), so it is deletable
insurance.

### rl/model/trunk.py — what the flat trunk replaced

Replaces `RoundBlock` (2026-08-29), which carried three separate residual
streams — 48 Perceiver latents, 41 action slots, 4 value queries — wired
together by five individually-gated, block-masked attentions per round, four
rounds deep, at 3.69M parameters a round. Every route those masks encoded is
a subset of one all-pairs attention over the 80 rows the sequence now has,
and at 80 rows the trunk can simply carry them: 80 x 80 is 6.4k attention
cells against the 24k the old routing plus its two feeding cross-attention
reads paid, so the masks were buying nothing but their own complexity.

The ungated pre-RMSNorm design also retires the 2026-08-24 gate-contribution
finding structurally rather than by tuning it.

### rl/model/trunk.py — remat policy sweep

MEASURED, not assumed (2026-09-01 sweep, full train_step compiled at the
largest lattice entry (64, 256) x batch 4; XLA memory_analysis temp +
15-step timing): the step is memory-bandwidth-bound, so recomputing is
genuinely cheaper than storing — NO remat is both 3.8x the memory AND ~10%
SLOWER. Full table (trunk x entity pool, temp MiB / steps per sec):

| trunk + entity pool | temp MiB | steps/sec |
|---|---|---|
| nothing+nothing (this) | 796 | 12.26 |
| nothing+dots | 1071 | 12.65 |
| dots+nothing | 1244 | 12.32 |
| dots+dots | 1508 | 12.60 |
| none+nothing | 3019 | 11.03 |
| none+dots | 3402 | 11.29 |

The fastest fitting variant buys +3.2% for +275MiB, landing on the >=1.5GB
headroom boundary (the 12GB box peaked ~10.5GB all-in), so the cheapest
policy stays.

### rl/model/trunk.py — the silent sow incident

This scan lifted only params from a1c18ed to 2026-09-02 — so the block's
attention sow captured NOTHING and scripts/attn_probe.py never saw a trunk
attention. (The scan is stacked along the block axis, as the old round
trunk's scan did.)

### rl/model/trunk.py — group_row_l2 reference numbers

The live twin of the 2026-09-10 offline norm table
(first_block_diagnostics): every row now ENTERS at RMS 1, so the trunk's
output norm per group is the read of which rows the blocks write to —
unnormalised, the history rows sat at ~1040 in and out while CLS went 2.85
-> 1012.

### rl/model/config.py — the trunk that replaced RoundBlock

Replaces the four-round, three-stream, five-masked-attention RoundBlock on
2026-08-29: at 80 rows an all-pairs attention is 6.4k cells, so the block
masks that encoded the routing were buying nothing but their own complexity,
and the 48 latents they fed were a bottleneck between rows the trunk can now
simply carry. Depth is the knob: a block costs ~1.05M params and almost no
attention at this sequence length.

### rl/model/config.py — the action readout's parameter saving

The action readout (2026-08-29). Three small heads over named trunk rows —
a scalar per sheet row for switching, ONE bilinear for moves x targets, a
scalar per target row for pass/default — replacing the hierarchical
macro/micro stack that was instantiated twice, for a policy and for an
advantage head the policy did not read. 2.65M parameters became 0.13M.

### rl/model/constants.py — token types routed through the proto, tried and reverted

Routing them through the proto was tried on 2026-08-25 and reverted, because
it bought nothing an IntEnum does not (the count is derived either way)
while costing a generated-but-unused TypeScript enum and, worse, an extra
table row — protolint mandates a `___UNSPECIFIED` zero value, which would
have taken the token-type table from 12 rows to 13 with row 0 never indexed.

### rl/model/constants.py — why every NUM_* is a len()

On 2026-08-25 a token type was deleted and the literal `13` had to be
hand-edited to `12` — exactly the edit that silently leaves a dead embedding
row, or an out-of-range gather, when someone forgets.

### rl/model/constants.py — what ONE TOKEN PER THING replaced

ONE TOKEN PER THING (2026-08-29). Before this the board was unpacked into
189 attribute tokens — 10 or 11 per entity — and a Perceiver read compressed
them to 48 latents for a trunk that could not afford the rows. With entities
pooled to a vector each the whole board is 80 rows, the trunk carries them
directly, and the read, the latents and the separate action stream all go.

### rl/model/heads.py — the doubles slot-alignment defect rate

`SlotConditioning` keeps the MODEL side of doubles reachable and nothing
more; the plumbing outside it includes the ~75% slot-alignment defect in the
service (the figure removed from the docstring, which now names the defect
without the rate).

### rl/model/heads.py — FlatActionReadout replaced the hierarchical stack

Replaces the hierarchical stack — `MacroMicroHead` = per-modality queries,
five MLPs and five zero-init output layers, over a per-slot-group
`PointerLogits` grid with a stop-grad RMS gauge and three zero-init local
routes — which was instantiated twice, once for the policy and once for an
advantage head the policy did not read. 2.65M parameters became 0.13M.

`calculate_hierarchical_prior` was the uniform-at-init anchor while the head
was hierarchical and retires with it.

### rl/model/heads.py — the two-factor stall the init contract avoids

LESSONS.md 13: a learned grid behind a zero-init scale sat at lecun init for
60k steps. That is what "getting to exact zero WITHOUT re-creating the
two-factor stall" refers to.

### rl/model/heads.py — why the categorical value head is f32 from the head outwards

f32 from the head outwards (2026-08-24): the 1.0-weighted head was the one
rung still paying bf16 while the ladder heads were cast f32.

### rl/model/player_model.py — the fourth module that retired

The model was four modules, two of them the same class — a policy
ActionScoreHead and an advantage one over the same grid. The advantage head,
`compose_q` and the Retrace baseline it fed retired on 2026-08-29: the
policy had not read it since the NashPG switch, so it was a matched-control
observer for an architecture that no longer exists, and its last readings
are banked in the ledger.

### rl/model/encoder.py — typed action-slot groups as residual streams

Since 2026-08-17 the groups are not just decoder bookkeeping — each is its
own residual stream through the round trunk. (The round trunk is gone; the
remaining comment keeps only the canonical-partition note.)

### rl/model/encoder.py — why the entity embedders are lifted (nn.jit / nn.vmap)

The surrounding lifted `nn.jit` makes each embedder its own XLA
subcomputation instead of being inlined wholesale into the caller's graph —
smaller HLO and cheaper compiles: the retained-executable RAM lesson from
run 1326.

### rl/model/encoder.py — the target slot table that replaced two others

Replaces the separate pass_embeddings / target_embeddings tables
(2026-08-29): those two plus the four entity-derived targets were three ways
of saying "a thing a move can be aimed at", and the readout wants them as
one contiguous block it can score against.

### rl/model/encoder.py — the private sheet carried the opponent's side tag

The service writes ENTITY_PUBLIC_NODE_FEATURE__SIDE = isMySide(...), so row
1 of side_bias is MINE and row 0 is the opponent's — the sheet was carrying
the opponent's tag (fixed 2026-08-28).

### rl/model/encoder.py — what the OPP_PRIVATE rows carried before 2026-09-12

Until 2026-09-12 the learner-only OPP_PRIVATE_ENTITY rows carried a
Dreamer-style discrete code grounded by the privileged value loss, with a
belief head predicting it from the public rows; LESSONS "Removal ledger —
2026-09-12". They now carry the opponent's sheet latent from the same
private embedder as my own sheet.

### rl/model/encoder.py — field_side_bias vs pos_bias (the 2026-08-28 fix)

Until 2026-08-28 these two field tokens borrowed pos_bias rows 1/0, but
pos_bias is indexed by ENTITY_PUBLIC_NODE_FEATURE__ACTIVE (= scoreOrder,
{0, 2} in singles), so row 0 meant "benched pokemon" AND "opponent side
conditions" — one vector, two meanings, coupled gradients.

### rl/model/encoder.py — the CLS row replaced the value embeddings table

The CLS row replaces the 4-row value_embeddings_table and its
(4 * entity_size,) concat.

### rl/model/encoder.py — the per-row bias table, measured dead

There is no per-row table (2026-09-10): measured on ckpt_02339569 after
2.34M steps, the per-row table's within-group spread sat at its init noise
(0.06-0.08 against an init of 0.0625) for six of the twelve groups. The rows
are a set with a fixed layout, not a sequence, so only the per-GROUP bias
survives.

### rl/model/encoder.py — the entity pool's retired intra-entity attention

The `EntitySumPool` docstring carried the measurement that retired the
intra-entity attention block the masked sum replaces (see the
rl/model/modules.py EntitySumPool section above for the numbers).

### rl/model/encoder.py — what the trunk collapsed

See rl/model/trunk.py for why the three gated streams and their two feeding
cross-attention reads all collapse into one sequence at 80 rows.

### rl/model/encoder.py — output normalisation reference number

Every row leaves the trunk at RMS 1, rescaled per group, so the heads read
every row at one magnitude rather than at whatever the blocks' writes left
it — CLS at 9.8 against move rows at 0.99 on ckpt_00280000.

### rl/model/encoder.py — the private embedder once served the opponent's sheet

`_embed_private_entity` is the path for MY OWN private team rows (the
opponent's sheet, which this once also served, was deleted 2026-08-25).

### rl/model/encoder.py — the second board path before the entity pools

`_embed_public_entity` / `_embed_private_entity` are the SAME entity-local
pools the packed history cache runs on; before 2026-08-29 the current board
took a second path that emitted 10-11 raw attribute tokens per entity
instead.

### rl/model/encoder.py — history states used to be summed into the public rows

Until 2026-09-01 the history states were SUMMED into the public rows here
("entity i's 11th attribute token"). They are their own HISTORY_ENTITY rows
now.

### rl/model/encoder.py — entity_index_tag measured dead

No learned join key between a sheet row and its public row
(entity_index_tag, 2026-08-31 -> 2026-09-02): it never trained (rms 0.0634
-> 0.0661 over 182k steps, ~3% of the row's norm) and a public-only read
from the sheet row scored no higher after the trunk than before it, so the
tag joined nothing.

### rl/model/encoder.py — the previous-action gather before 2026-08-29

The previous action's rows are gathered out of this step's pre-trunk
private/move/target rows. Not circular: those rows are built above from this
step's features alone, where the pre-2026-08-29 gather read a sequence built
over these rows.

### rl/model/encoder.py — row validity vs the old grid

Move-row and target-row validity from the block mask is the same content the
old grid's any-over-both-axes gave.

### rl/model/encoder.py — why the identity is added after the input norm

The norm divides a row by its own content RMS, so a bias added before it is
divided too — 35% of a CLS row (content RMS 0.18) but 0.1% of a history row
(RMS ~66), with the gradient into the bias shrunk by the same factor.

### rl/model/encoder.py — the raw node snapshot the RL path used to discard

The latest raw node snapshot per entity is the TGN staleness fix the RL path
used to discard (only the offline critic read it; "the GRU-only readout
loses the latest node").

### rl/model/history_encoder.py — the four-column RELEVANT_ENTITY bug

All EIGHT columns the service writes (state.ts maxRelevant = 8). Until
2026-09-01 this listed only IDX0..3, so any step touching more than four
entities — spread moves, hazard cascades — had rows 5-8 silently dropped
before the scatter ever saw them.

### rl/model/history_encoder.py — what the minGRU replaced

That is what the GRU it replaces (2026-09-02) could not offer: its scan sat
on a ~26us/step dependency-latency floor that hoisting (-8.6%) and unrolling
could not move.

### rl/model/history_encoder.py — the deleted "gestalt" slot input

A mean over the other slots' states used to ride in the slot input too (the
"gestalt"); it was redundant with flat_field — which is fed the SUM of every
message — and with the trunk's read-time attention over the HISTORY_ENTITY
rows (deleted 2026-09-02).

### rl/model/capacity.py — the 1e-4 learning-rate collapse these probes caught

These probes are what caught the 1e-4 learning-rate collapse:
action-embedding srank fell to 0.27 by 13k steps while actor-KL sat quietly
at 0.002, so KL headroom was never evidence the LR could rise (LESSONS.md
5).

### rl/environment/interfaces.py — src_index / tgt_index

`src_index`/`tgt_index` lived on PlayerPolicyHeadOutput until 2026-08-31:
coordinates into the 41x41 scoring grid the wire Action used to carry.
`action_index` IS the wire action now — an index into the block space.

### rl/environment/interfaces.py — advantage and q

`advantage` and `q` lived on PlayerActorOutput until 2026-08-29: the
learner-only Q = V + A decomposition over the flat src x tgt grid, composed
in the model by heads.compose_q. The policy stopped reading it at the NashPG
switch, which left it a matched-control observer for an architecture that no
longer exists; its last readings are banked in the ledger.

### rl/environment/data.py — the 41x41 grid the block action space replaced

The 41x41 (src, tgt) grid this replaced kept ~82% dead cells purely so the
readout's scatter had somewhere to land.

### rl/environment/utils.py — the actor's geometric-bucket base

The actor path's geometric-bucket base for BOTH history axes is 32 (was 64,
2026-09-02): a carried request's suffix is ~3 steps / ~5 packed rows, so the
smallest bucket is what the carry path runs at.

### rl/environment/env.py — the leaked websockets

Offline harnesses construct one env per game; without `close()` every game
leaked a connection — 600 open sockets after the 2026-08-23 check.

### rl/environment/actor_stats.py — why the timing sink exists

Built for the 2026-09-02 actor-step decomposition (the system rate is
actor-bound and no panel said WHERE the actor's time went) — the baseline
the history-carry pass is judged against.

# Evidence migrated out of `rl/online/**` — 2026-09-12

Every block below was deleted from or trimmed in the source. `Removed:` is the original comment text verbatim; `Kept:` is what the code now says in its place (blank when the whole block went).

### rl/online/config.py — eval_baseline — saturated baselines and the retired eval slots
Removed:

```
experimental. Random/Default were saturated (93%/72% at 163k steps).
The slate itself is fixed at two slots (rl/online/main.py, 2026-09-09):
```

Kept in code:

```
experimental.
The slate itself is fixed at two slots (rl/online/main.py):
```

### rl/online/config.py — eval_baseline — the T=0.5 slots and the search eval actor
Removed:

```
play. The earlier T=0.5 slots and the search eval actor are gone
(LESSONS.md "Removal ledger — 2026-09-09 search eval actor").
```

Kept in code:

```
play.
```

### rl/online/config.py — eval_main_params_every — why not every game
Removed:

```
params by only ~1/player_ema_update_rate steps, so alternating every
game (the old behaviour) logged two near-duplicate series at half the
effective sample size each. 0 = EMA params only.
```

Kept in code:

```
params by only ~1/player_ema_update_rate steps. 0 = EMA params only.
```

### rl/online/config.py — unroll_length — the removed MAX_REQUEST_COUNT force-tie
Removed:

```
pre-split to this count), NOT a target length: the service's
MAX_REQUEST_COUNT force-tie at 96 requests was removed alongside the
chunked-unroll change (2026-08-16) — games now run to their natural
outcome (Showdown's turn-limit/endless-battle clauses and the
```

Kept in code:

```
pre-split to this count), NOT a target length: games run to their
natural outcome (Showdown's turn-limit/endless-battle clauses and the
```

### rl/online/config.py — player_chunk_length — the geometric bucket family's three OOMs
Removed:

```
Fixed-length chunked unrolls (2026-08-16): every stored trajectory is
```

Kept in code:

```
Fixed-length chunked unrolls: every stored trajectory is
```

### rl/online/config.py — player_chunk_length — what the bucket family cost
Removed:

```
sees ONE shape forever instead of a geometric bucket family (each
bucket was a separate compiled variant with its own workspace; the
first top-bucket batch ~20min into a session is what OOM'd
1786537634, the Aug-15 03:26 run, and the Aug-15 23:33 run alike).
Targets bootstrap at the cut from the critic
```

Kept in code:

```
sees ONE shape forever. Targets bootstrap at the cut from the critic
```

### rl/online/config.py — player_shape_lattice — date
Removed:

```
Static shape lattice for the learner batch (2026-08-20): a CHAIN of
```

Kept in code:

```
Static shape lattice for the learner batch: a CHAIN of
```

### rl/online/config.py — player_shape_lattice — the surprise-compile OOMs
Removed:

```
real history is ever dropped). This is NOT the geometric bucket
family that OOM'd three runs: that compiled a data-derived variant
per shape, with the first top-bucket batch arriving as a SURPRISE
compile ~20min in. Here the variants are a fixed, enumerated set —
```

Kept in code:

```
real history is ever dropped). NEVER a data-derived shape family:
the variants are a fixed, enumerated set —
```

### rl/online/config.py — player_shape_lattice — the Aug-20 fill measurement
Removed:

```
single-shape behaviour exactly. Combos chosen from the Aug-20
measurement (batch_size 4): batch-max chunk fill mean ~42 of 64,
history fill mean ~85 of 256 — retune from the player_shape_T/H
logs.
```

Kept in code:

```
single-shape behaviour exactly. Retune the combos from the
player_shape_T/H logs.
```

### rl/online/config.py — player_replay_fresh_fraction — provenance
Removed:

```
The fresh stream (2026-09-08, 224582c): the share of each batch's
```

Kept in code:

```
The fresh stream: the share of each batch's
```

### rl/online/config.py — player_replay_fresh_fraction — the archive reads
Removed:

```
restores uniform capped sampling exactly. Reads: LESSONS.md "Fresh
replay stream and decision accounting — 2026-09-08" and the
2026-09-09 first-18.1k-update read.
```

Kept in code:

```
restores uniform capped sampling exactly.
```

### rl/online/config.py — player_replay_kl_target — where 0.045 came from
Removed:

```
Ceiling: the actor-KL level the buffer-capacity plateau diagnosis
identified as the healthy/stale boundary. This is a pathology
```

Kept in code:

```
Ceiling: the actor-KL level that marks the healthy/stale boundary.
This is a pathology
```

### rl/online/config.py — main_player_update_steps — the params-cache working-set measurement
Removed:

```
directly sets the inference server's params-cache working set: at 10
(~6s of main training), main alone kept 5-10 versions live at once;
50 (~30s) collapses that to ~2, letting inference_params_cache_size
=12 cover the whole working set without LRU thrash. Staleness cost:
actors act on params up to ~30s old — measured actor-KL is 0.005-
0.006 vs the 0.045 replay target, ~5x headroom, and the replay-KL
controller cuts reuse if that ever stops being true.
```

Kept in code:

```
directly sets the inference server's params-cache working set: a
longer interval keeps fewer versions live, so
inference_params_cache_size covers the whole working set without LRU
thrash. Staleness cost: at 50 the actors act on params up to ~30s
old, and the replay-KL controller cuts reuse if the actor KL climbs.
```

### rl/online/config.py — add_player_max_frames — the 3e6 and 9e6 eras
Removed:

```
this clock only paces snapshots while the agent is NOT visibly
improving. At 3e6 (~11.5k steps) it filled the league with
~0.5-winrate near-copies of main (mirror play with extra staleness)
and made the stagnation clock hair-trigger. 9e6 (~44k steps at the
live batch shape) fired every add of irqeetfg to 640k — the dominant
gate never did — and was doubled 2026-09-04 (~88k steps) together
with the batch cull below.
```

Kept in code:

```
this clock only paces snapshots while the agent is NOT visibly
improving. Too short and the league fills with near-copies of main
(mirror play with extra staleness) and the stagnation clock goes
hair-trigger; 1.8e7 is ~88k steps at the live batch shape.
```

### rl/online/config.py — minimum_historical_player_steps — the mirror-only measurement
Removed:

```
populated league rather than pure mirror self-play — mirror-only runs
measured 93% vs Random but ~10% vs SimpleHeuristic at 163k steps,
the signature of self-exploiting policies that don't transfer to
stylistically alien opponents.
```

Kept in code:

```
populated league rather than pure mirror self-play, which produces
self-exploiting policies that don't transfer to stylistically alien
opponents.
```

### rl/online/config.py — br_stop_winrate — the standard-error calibration of 0.7
Removed:

```
games behind it (n=20 puts the SE at ~0.11, so 0.7 is a ~1.8 SE
signal — the old promotion-bar lesson). 0.0 = off; the CLI defaults
```

Kept in code:

```
games behind it. 0.0 = off; the CLI defaults
```

### rl/online/config.py — br_init — the pre-2026-08-30 default
Removed:

```
params verbatim (the pre-2026-08-30 behaviour — the probe searches
only the target's own basin, and its blind spot is the collapsed
switch axis it inits from). "head-reset" grafts a fresh-init
```

Kept in code:

```
params verbatim (the probe searches only the target's own basin, and
its blind spot is the collapsed switch axis it inits from).
"head-reset" grafts a fresh-init
```

### rl/online/config.py — br_init — what shrink-perturb was justified by
Removed:

```
br_perturb_frac (Ash & Adams, arXiv:1910.08475 — the ~179k
perturbation is the one event observed to revive collapsed switch
mass). "scratch" ignores the target's params entirely — recorded
```

Kept in code:

```
br_perturb_frac (Ash & Adams, arXiv:1910.08475).
"scratch" ignores the target's params entirely — recorded
```

### rl/online/config.py — br_perturb_frac — the 2026-08-30 calibration
Removed:

```
ancestor and rotates nothing. Measured calibration (2026-08-30,
vs ckpt_00254992): full-tree fresh/trained norm ratio 0.925, so
frac maps near-linearly onto direction — 0.75 lands at cos 0.40
to the target (0.34 predicted orthogonal; the excess is
structural, e.g. LayerNorm scales ~1.0 in both nets).
```

Kept in code:

```
ancestor and rotates nothing.
```

### rl/online/config.py — memory_diag_interval — the session that motivated it
Removed:

```
census + exact replay-buffer/league-cache byte counts. Added after
session 1786537634's RSS climbed 5.9->17GB (threads 478->775) with
no way to attribute it from wandb alone. 0 disables. Cost per tick
```

Kept in code:

```
census + exact replay-buffer/league-cache byte counts. 0 disables.
Cost per tick
```

### rl/online/config.py — actor_stats_log_steps — the history-carry baseline
Removed:

```
dict merge. The actor-step decomposition it feeds (service wait /
decode / history clip / inference, and the inference server's own
phases) is the baseline the history-carry pass is judged against.
```

Kept in code:

```
dict merge. It feeds the actor-step decomposition (service wait /
decode / history clip / inference, and the inference server's own
phases).
```

### rl/online/config.py — player_actor_device — the 2026-09-03 timing
Removed:

```
alone. Measured 2026-09-03 beside a live learner: per-actor CPU
inference 23-60 ms against 147 ms through the GPU server (76 of it
queue wait, none of it compute: the server's forward shared one
device stream with the train step). f32 because XLA:CPU only
```

Kept in code:

```
alone. f32 because XLA:CPU only
```

### rl/online/config.py — oom_guard_enabled — the crash that prompted it
Removed:

```
safety valve, not a leak fix — added after 1361 crashed, though that
specific crash turned out to be an unrelated websocket failure to the
game server, not RAM exhaustion. Checks available system RAM every
```

Kept in code:

```
safety valve, not a leak fix. Checks available system RAM every
```

### rl/online/config.py — representation-health probe — moved offline 2026-08-21
Removed:

```
NOTE the representation-health probe (dormant-unit fraction,
srank@0.99) moved OFFLINE 2026-08-21 — rl/model/capacity.py, run
against a saved checkpoint by tests/test_checkpoint_collapse.py. It
cost an extra encoder forward plus an eigendecomposition per probe
inside the train loop, and no training decision read it. The
per-step fresh-vs-replayed value-error gap below stays: it is
computed from tensors train_step already has.
```

Kept in code: nothing — the block was deleted.

### rl/online/config.py — player_adam / builder_adam — the b1 and eps lineage
Removed:

```
Learning params. Player b1 back to 0.9 (2026-08-26): the b1=0
detour was specific to the previous prefactor-free logit force
(momentum carried each push ~1/(1-b1) steps past the stiff
equilibria its analytic shifts created — the dx65cpwp runaway).
The player now runs the same trust-regioned PPO surrogate as the
builder, the exact case the pro-momentum argument was always
about; NashPG's own optimiser is AdamW with default moments.
Player eps 1e-5 (2026-08-31): the NashPG reference explicitly
overrides optax's 1e-8 (`optax.adamw(lr, eps=1e-5)`) and the
reference-diff ledger flagged it as "the one to test" — Adam is
scale-invariant, so a param whose gradient has gone tiny (a starved
switch cell's) still steps at ~full lr along a noise-dominated
direction, and eps is the ONLY damper; 1e-5 engages 1000x sooner.
Builder keeps 1e-8: the divergence concerned the player bracket.
```

Kept in code:

```
Learning params. The player runs the same trust-regioned PPO
surrogate as the builder, and NashPG's own optimiser is AdamW with
default moments, so b1 stays at 0.9.
Player eps 1e-5 follows the NashPG reference, which explicitly
overrides optax's 1e-8 (`optax.adamw(lr, eps=1e-5)`): Adam is
scale-invariant, so a param whose gradient has gone tiny (a starved
switch cell's) still steps at ~full lr along a noise-dominated
direction, and eps is the ONLY damper.
Builder keeps 1e-8: the divergence concerned the player bracket.
```

### rl/online/config.py — player_learning_rate — the 1e-4 collapse
Removed:

```
3e-5. A 1e-4 trial (Aug 2026, zany-leaf-1305) collapsed: pre-clip grad
norms 10-100x the clip, action-emb srank at 0.27 by 13k steps (vs
0.82 at 3e-5), value CE degrading and eval regressing from ~40k —
all while actor-KL sat quietly at 0.002, so KL headroom is NOT
evidence the LR can rise (the trust region bounds per-update policy
movement, not representation damage).
```

Kept in code:

```
KL headroom is NOT evidence the LR can rise: the trust region bounds
per-update policy movement, not representation damage.
```

### rl/online/config.py — player_lambda — the 1328 sweep and the retired bootstrap-gap readout
Removed:

```
Value-target lambda. AlphaStar's own choice: TD(lambda=0.8), a
short (~5-step) bootstrap horizon — they could afford heavy
bootstrapping because supervised init gave them a sane critic from
step one. This project starts from scratch AND the 1328 five-arm
sweep pointed the same direction (monotone lower-lambda-better,
confounded but directional), so 0.8 is adopted as-is. NOTE: the
lambda=1.0 MC-anchor row of the aux spectrum used to keep a live
bootstrap-bias readout (player_bootstrap_gap) on this
bootstrap-heavy target; the aux heads went 2026-08-21, so that
instrument is gone with them (LESSONS.md ledger).
```

Kept in code:

```
Value-target lambda. AlphaStar's own choice: TD(lambda=0.8), a
short (~5-step) bootstrap horizon. Lower = more bootstrapping and
less Monte-Carlo variance.
```

### rl/online/config.py — removed controllers — AdaptivityController and ExploitabilityController
Removed:

```
No adaptivity/entropy controller fields anymore. The
AdaptivityController was removed entirely 2026-08-13 (hard to tune,
harder to predict — see LESSONS.md 10
for the bug history). Its entropy sensors are still logged from
train_step (player_action_normalized_entropy,
player_normalized_modality_entropy); modality collapse (1330 died
at 0.08 on that axis) is now watched on the dashboard, not
auto-corrected.

No ExploitabilityController anymore (removed 2026-08-14, the last
adaptive hyperparameter loop — see rl/online/training/controllers.py's
module docstring). The replay KL target is fixed at
player_replay_kl_target; the worst-matchup win-rate it sensed still
exists as _should_add_new_player's "dominant" gate, it just doesn't
actuate anything.

Both fields below now serve main's VERIFICATION branch
```

Kept in code:

```
The replay KL target is fixed at player_replay_kl_target; the
worst-matchup win-rate lives on in _should_add_new_player's
"dominant" gate, which actuates nothing.

Both fields below serve main's VERIFICATION branch
```

### rl/online/config.py — exploit_ctrl_min_games_per_opponent — the 1338 false positive
Removed:

```
recent self), which looks exactly like a real hole (1338: two
snapshots 5.5k/26.9k steps old, win-rate never left 0.48-0.54 — a
false positive from exactly this).
```

Kept in code:

```
recent self), which looks exactly like a real hole.
```

### rl/online/config.py — player_kl_loss_coef — removed 2026-09-09
Removed:

```
(`player_kl_loss_coef`, the actor backward-KL force, was REMOVED
2026-09-09 -- LESSONS.md "Removal ledger — 2026-09-09 actor
backward-KL force".)
player_value_head_loss_coef: float = 1.0
```

Kept in code:

```
player_value_head_loss_coef: float = 1.0
```

### rl/online/config.py — privileged critic — dates
Removed:

```
The privileged critic (2026-09-01): trained beside the deployable head
```

Kept in code:

```
The privileged critic: trained beside the deployable head
```

### rl/online/config.py — player_privileged_targets — the pre-2026-09-01 estimator
Removed:

```
pre-2026-09-01 estimator (deployable head), the live fallback: the
```

Kept in code:

```
deployable-head estimator, the live fallback: the
```

### rl/online/config.py — player_potential_strength — provenance
Removed:

```
PBRS as a potential channel (2026-09-11; docs/human-switch-pbrs-
2026-09-11.md, LESSONS "PBRS potential channel"): eta, the scale on the
```

Kept in code:

```
PBRS as a potential channel: eta, the scale on the
```

### rl/online/config.py — player_potential_strength — the offline screen and the launch
Removed:

```
builds neither the head nor the channel -- today's learning rule.
Launched at .05 on 2026-09-11 from ckpt_00480000 after the offline
screen (rl/offline/potential_screen.py): logit-gradient RMS
perturbation .031 vs the .10 budget, shared-param update .0047 --
LESSONS "PBRS screen on ckpt_00480000".
```

Kept in code:

```
builds neither the head nor the channel -- today's learning rule.
```

### rl/online/config.py — player_pg_coef — date
Removed:

```
THE policy gradient (2026-08-26): NashPG (arXiv:2510.18183, TMLR
```

Kept in code:

```
THE policy gradient: NashPG (arXiv:2510.18183, TMLR
```

### rl/online/config.py — player_pg_coef — the section 5.4 ablation
Removed:

```
player_reg_snap_steps. Their section 5.4 ablation is the reason
for the operator choice: swapping PPO into the older reward-
transform framework closes most of its gap in larger games, i.e.
the inner update rule, not the regularisation cycle, was the
bottleneck.
The whole bracket shares this coefficient;
```

Kept in code:

```
player_reg_snap_steps.
The whole bracket shares this coefficient;
```

### rl/online/config.py — player_ppo_clip — the runaway class it replaced
Removed:

```
stiff equilibrium — the structural fix for the runaway class the
previous logit-force loss needed a force clip, centred logits and
b1=0 to contain.
```

Kept in code:

```
stiff equilibrium.
```

### rl/online/config.py — player_pg_objective — date
Removed:

```
(the smooth quadratic the builder also runs, 2026-08-30) or "ppo"
```

Kept in code:

```
(the smooth quadratic the builder also runs) or "ppo"
```

### rl/online/config.py — player_mag_coef — the 2026-08-26 rename
Removed:

```
argument. Called "forward" here until 2026-08-26; that was wrong by
this package's own convention (loss.py's approx_forward_kl is the k3
estimator for KL(actor || learner), reference first). Reverse =
mode-seeking, which is exactly why it cannot refill a dropped
modality (the removed support-anchor family was built for that; see
the note below player_ent_coef).
```

Kept in code:

```
argument. Reverse = mode-seeking, which is exactly why it cannot
refill a dropped modality.
```

### rl/online/config.py — player_mag_coef — the prefactor-free bet
Removed:

```
gradient is pi-prefactored — with the PPO surrogate there is no
prefactor-free refill force anywhere any more; the bet (theirs) is
that the magnet cycle plus entropy keep pi interior so starvation
never starts. switch_ratio through the 13k wire is the acceptance
```

Kept in code:

```
gradient is pi-prefactored, so it cannot by itself restore an
abandoned action. switch_ratio through the 13k wire is the acceptance
```

### rl/online/config.py — player_ent_coef — the removed per-axis split and dual temperatures
Removed:

```
Entropy bonus, differentiated — NashPG's ent_coef verbatim
(2026-08-30): the plain JOINT entropy over legal cells, one static
coefficient. Up to a constant this is the reverse KL to uniform.
The per-axis split (H(macro) + H(micro|taken), 2026-08-27) and the
SAC-style dual temperatures holding each at a normalised target
(2026-08-28) are removed with the forward-KL-to-uniform term — the
per-level entropies survive as OBSERVER panels only
(loss.factorised_entropies). Revert handles in the LESSONS.md
ledgers.
```

Kept in code:

```
Entropy bonus, differentiated — NashPG's ent_coef verbatim: the
plain JOINT entropy over legal cells, one static coefficient. Up to
a constant this is the reverse KL to uniform. The per-level
entropies are OBSERVER panels only (loss.factorised_entropies).
```

### rl/online/config.py — player_prune_threshold — the offline cut audit
Removed:

```
only: the trace ratio c stays raw, the pre-registered restriction
the offline cut audit fired (7.8% of chunks cut before the midpoint
against the 5% gate; targets.compute_player_targets). The
```

Kept in code:

```
only: the trace ratio c stays raw, the pre-registered restriction
(targets.compute_player_targets). The
```

### rl/online/config.py — player_support_tau — the bench-cell calibration
Removed:

```
factor of two is the hysteresis band. Calibration: 5-6 bench cells at
.01 induce a switch-mass floor of 5-6%, just under the .07171 the KL
was holding -- low enough that evidence, not the floor, sets the
resting level. Confirm against player_switch_mass_choice in the
```

Kept in code:

```
factor of two is the hysteresis band. The floor it induces must stay
low enough that evidence, not the floor, sets the resting switch
mass. Confirm against player_switch_mass_choice in the
```

### rl/online/config.py — player_support_temperature — provenance
Removed:

```
The hinge's smoothing width, in LOG-PROBABILITY space (2026-09-11, the
user's call on docs/porygon2_support_loss_recommendations.md; the hard
hinge's kink was not measured to cost anything first): each legal cell
```

Kept in code:

```
The hinge's smoothing width, in LOG-PROBABILITY space: each legal cell
```

### rl/online/config.py — player_support_hinge_coef — date
Removed:

```
The FLAT SUPPORT HINGE (2026-09-09, loss.support_hinge_loss): over a
```

Kept in code:

```
The FLAT SUPPORT HINGE (loss.support_hinge_loss): over a
```

### rl/online/config.py — support-anchor family — removed 2026-08-27
Removed:

```
The support-anchor family (forward KL toward a temperature-raised /
advantage-tilted reference; player_support_{coef,temperature,
adv_temperature}) was REMOVED 2026-08-27 after phases 1-4: every
mass-restoring variant either erased within-modality discrimination
(mode-covering targets + the snap ratchet) or taught the mean
switch's losing value. Replaced by the per-level ENTROPY terms above
(the PPO surrogate was split per-level in the same pass and
re-joined 2026-08-28 — see that revert commit) — see train_step's
policy bracket and the LESSONS.md ledgers for history and handles.
Snap period of the reference:
```

Kept in code:

```
Snap period of the reference:
```

### rl/online/config.py — player_reg_snap_steps — the continuous EMA it replaced
Removed:

```
re-clone every 10k for 25 outer rounds). Frozen between snaps —
the continuous EMA it replaced never reset, so the KL gap
compounded with policy speed (2wvnlsz3: ref_kl 2.07 nats). A
shorter period approaches an EMA magnet, which chases the policy
and degenerates into a short-horizon trust region (LESSONS 4).
```

Kept in code:

```
re-clone every 10k for 25 outer rounds). Frozen between snaps.
A shorter period approaches an EMA magnet, which chases the policy
and degenerates into a short-horizon trust region.
```

### rl/online/main.py — the battle-abort handler — the 2026-09-03 league-snapshot incident
Removed:

```
The abort is usually the SYMPTOM: the other side died
first (2026-09-03: league snapshots from a superseded
param tree raised on their first forward, p0 then sat
600 s on the service watchdog), and raising the abort
alone hid that for an hour. Log every other exception
before it goes.
```

Kept in code:

```
The abort is usually the SYMPTOM: the other side died
first. Log every other exception before it goes.
```

### rl/online/main.py — the eval slate — the deleted search eval actor and the head-parameterisation date
Removed:

```
The eval slate (2026-09-09): two slots against the same baseline,
both the EMA params at temp 1.0 -- the temperature the training
actors sample at, and the only one comparable across head
parameterisations (the flat readout's single division vs the
hierarchical head's two, 2026-08-29). `plain-t1` samples the policy
exactly as the training actors do; `thresholded` samples it with
every legal cell below player_prune_threshold removed and the rest
renormalised -- the distribution the learner's v-trace ratios are
built on, sampled nowhere else. wr(thresholded) - wr(plain-t1) on
the same checkpoint prices the threshold in play. The search eval
actor (depth-1 expectimax, 2026-09-06) was deleted here 2026-09-09:
its measured influence at 01861967 was root KL .000026-.000101 with
both inspected bad actions still ranked first in 16/16 seeds
(LESSONS.md "Removal ledger — 2026-09-09 search eval actor").
```

Kept in code:

```
The eval slate: two slots against the same baseline,
both the EMA params at temp 1.0 -- the temperature the training
actors sample at, and the only one comparable across head
parameterisations (the flat readout's single division vs the
hierarchical head's two). `plain-t1` samples the policy
exactly as the training actors do; `thresholded` samples it with
every legal cell below player_prune_threshold removed and the rest
renormalised -- the distribution the learner's v-trace ratios are
built on, sampled nowhere else. wr(thresholded) - wr(plain-t1) on
the same checkpoint prices the threshold in play.
```

### rl/online/main.py — InferenceServer construction — the deleted inference_* config fields
Removed:

```
(one team-build per game vs ~35 player steps). The server's
constructor defaults are the values the deleted inference_* config
fields held (b219d84). Under "cpu" there is no server: every actor
```

Kept in code:

```
(one team-build per game vs ~35 player steps). Under "cpu" there is
no server: every actor
```

### rl/online/main.py — the crash handler — the session 1786537634 postmortem
Removed:

```
exists so the finish() below can mark the wandb runs FAILED.
Letting the exception fly past an unconditional finish() left
session 1786537634's OOM crash showing as three cleanly-
"finished" runs, which sent the postmortem down the wrong path.
```

Kept in code:

```
exists so the finish() below can mark the wandb runs FAILED:
an exception flying past an unconditional finish() leaves the
runs showing as cleanly "finished".
```

### rl/online/training/learner.py — the static-config rule — run 1326's retained executables
Removed:

```
needs its own traced pytree argument, because retained
executables per distinct static value OOM-killed run 1326
(LESSONS.md 1).
```

Kept in code:

```
needs its own traced pytree argument: retained executables per
distinct static value are an OOM.
```

### rl/online/training/learner.py — league host_step seeding — the 2026-08-14 add storm
Removed:

```
league-management tick (the 2026-08-14 10:15 add storm; also
the p_{step:08} snapshot-dir overwrite hazard once the counter
caught up).
```

Kept in code:

```
league-management tick, plus the p_{step:08} snapshot-dir
overwrite hazard once the counter caught up.
```

### rl/online/training/learner.py — _update_hyper_controllers — removed with the coefficients it actuated
Removed:

```
No _update_hyper_controllers anymore: every coefficient a controller
actuated has since been deleted outright — the magnet KL
(2026-08-22) and UPGO with the single-action PG (2026-08-21) — see
LESSONS.md 10 and the removal ledgers. The replay
reuse-cap controller below is the one remaining per-log-tick loop.
```

Kept in code:

```
The replay reuse-cap controller below is the one remaining
per-log-tick loop.
```

### rl/online/training/learner.py — num_steps bounds train steps — the first BR run's early stop
Removed:

```
continues below burn iterations without training, which at
a small --num-steps ended the first BR run at 1891 of 5000
steps — replay warm-up alone is ~600 ticks/minute.
```

Kept in code:

```
continues below burn iterations without training, so at a
small --num-steps a run can end well short of its budget.
```

### rl/online/training/learner.py — logger.exception over print_exc — the shredded OOM traceback
Removed:

```
stderr and got shredded line-by-line into the concurrent bar
redraws (session 1786537634's OOM traceback was near-
unreadable in the captured console for exactly this reason).
```

Kept in code:

```
stderr and got shredded line-by-line into the concurrent bar
redraws.
```

### rl/online/training/learner.py — learner_steps_per_sec — the irqeetfg hand-read comparisons
Removed:

```
The SYSTEM rate: learner steps per wall second over the
drain interval — actor-bound today, and the number the
cross-run comparisons (4.17-4.41 on irqeetfg) were read
by hand from _timestamp deltas until now.
```

Kept in code:

```
The SYSTEM rate: learner steps per wall second over the
drain interval — actor-bound today.
```

### rl/online/training/workers.py — the straggler raise — the 2026-08-11 RAM/VRAM leak
Removed:

```
rebuild from starting on top of state a leaked
thread still holds (the 2026-08-11 RAM/VRAM leak) — at
```

Kept in code:

```
rebuild from starting on top of state a leaked
thread still holds — at
```

### rl/online/training/league_ops.py — the removed exploitability helpers
Removed:

```
(_measure_exploitability/_update_exploit_controller/_apply_exploit_
scale removed 2026-08-14 with the ExploitabilityController — the
worst-matchup win-rate signal still exists in _should_add_new_player's
"dominant" gate; it just doesn't actuate anything anymore.)
```

Kept in code: nothing — the block was deleted.

### rl/online/training/league_ops.py — best-response child runs — date
Removed:

```
Best-response child runs (2026-08-27). A BR run trains in its own
```

Kept in code:

```
Best-response child runs. A BR run trains in its own
```

### rl/online/training/run_state.py — replay_kl_target — the ExploitabilityController that used to scale it
Removed:

```
Fixed at config.player_replay_kl_target — the ExploitabilityController
that used to scale it was removed 2026-08-14 (last of the adaptive
hyperparameter loops; see rl/online/training/controllers.py's module
docstring).
```

Kept in code:

```
Fixed at config.player_replay_kl_target; nothing scales it.
```

### rl/online/training/diagnostics.py — the heap census — the 2026-08-18 fork jump
Removed:

```
(replay buffers, league cache) don't cover — e.g. the ~3GB the
2026-08-18 fork jump left unexplained by thread counts
and league cache alone. sys.getsizeof is shallow (a dict/list's
```

Kept in code:

```
(replay buffers, league cache) don't cover.
sys.getsizeof is shallow (a dict/list's
```

### rl/online/training/controllers.py — module docstring — the removed controller family
Removed:

```
``PILogController`` is the actuator — a PI update in log space with clipped
bounds. Every other controller this project built (lambda, adaptivity, magnet
watchdog, plasticity) has been removed; LESSONS.md 10 records what each one
measured and why it went, including the evidence that pulls both ways.
```

Kept in code:

```
``PILogController`` is the actuator — a PI update in log space with clipped
bounds.
```

### rl/online/inference.py — grouping by history bucket level — what the unsplit batches cost
Removed:

```
256) made the short game's forward pay the long game's
attention FLOPs — with ~12 actors at random game stages,
most batches contained one long game, so nearly EVERY step
ran at worst-case history length and the batching win
leaked away as padding compute. Splitting by level trades a
```

Kept in code:

```
256) would make the short game's forward pay the long
game's attention FLOPs. Splitting by level trades a
```

### rl/online/artifact.py — the pi_head manifest literal — its predecessors and the stale-literal incident
Removed:

```
"flat_bilinear_readout" (2026-08-29) = FlatActionReadout over ONE
flat sequence: one pair form for sheet rows x the ally row a switch
replaces (2026-09-11) AND for moves x targets, a scalar per target
row for the standalone actions, and a flat pre-RMSNorm trunk behind
it. Predecessors:
"action_score_grouped_micro" (2026-08-25, ActionScoreHead with
per-slot-group micro and per-modality macro, Q composed in
heads.compose_q), "hierarchical_two_rung" (2026-08-20) and
"privileged_two_rung" (2026-08-17).

BUMPING THIS IS NOT COSMETIC: the literal was left stale through
the 2026-08-25 redesign, so a pre-redesign checkpoint passed
check_manifest STRICT and would have been restored onto a
structurally different param tree. The manifest exists precisely to
stop that, and it only works if the literal moves with the head.

`q_head` sat beside it until 2026-08-29 and went with the advantage
head itself.
```

Kept in code:

```
"flat_bilinear_readout" = FlatActionReadout over ONE flat
sequence: one pair form for sheet rows x the ally row a switch
replaces AND for moves x targets, a scalar per target row for the
standalone actions, and a flat pre-RMSNorm trunk behind it.

BUMPING THIS IS NOT COSMETIC: a stale literal lets a checkpoint
from a different head pass check_manifest STRICT and be restored
onto a structurally different param tree. The manifest exists
precisely to stop that, and it only works if the literal moves
with the head.
```

### rl/online/artifact.py — reg_params — the continuous EMA it replaced
Removed:

```
NashPG reference policy pi_reg (2026-08-22; snap semantics
2026-08-25): a periodic hard SNAP of target_params, in place, every
config.player_reg_snap_steps, frozen between snaps. The magnet
KL(pi || pi_reg) in the policy objective is measured against it;
the snap bounds the log-ratio gap structurally — the continuous
EMA it replaces (1e-4, then 5e-5) never reset, and the compounding
gap drove the pgaijs6l/2wvnlsz3 grad-norm runaways. One param set:
a hard reset needs no crossfade pair, no 4th net.
```

Kept in code:

```
NashPG reference policy pi_reg: a periodic hard SNAP of
target_params, in place, every config.player_reg_snap_steps, frozen
between snaps. The magnet KL(pi || pi_reg) in the policy objective is
measured against it; the snap bounds the log-ratio gap structurally.
One param set: a hard reset needs no crossfade pair, no 4th net.
```

### rl/online/artifact.py — jitted init — the eager-init timing
Removed:

```
whole forward op by op and compiles every nn.scan separately --
~6-10 min on the training box (2026-08-24 slow-suite timing); one
compile is a fraction of that and persists in the compile cache.
```

Kept in code:

```
whole forward op by op and compiles every nn.scan separately; one
jitted compile is far cheaper and persists in the compile cache.
```

### rl/online/artifact.py — merge-by-path — date
Removed:

```
Every tree is merged BY PATH onto the fresh state's own (2026-09-02):
```

Kept in code:

```
Every tree is merged BY PATH onto the fresh state's own:
```

### rl/online/artifact.py — the loud scratch fallback — the 1335/1336 lost lineage
Removed:

```
2. No checkpoint found -> fall back to scratch, loudly. A bare print()
here is how 1335's ~300k-step lineage and its league got lost between
it and 1336 without anyone noticing — mode was "checkpoint" (a resume
was expected) and it silently became a fresh run instead. Still
auto-falls back (the launch entry point doesn't set LOAD_STATE_MODE
per-run), but now at warning level so it can't scroll by unnoticed.
```

Kept in code:

```
2. No checkpoint found -> fall back to scratch, loudly. A silent
fallback loses a whole lineage and its league: the mode was
"checkpoint" (a resume was expected) and it becomes a fresh run
instead. Still auto-falls back (the launch entry point doesn't set
LOAD_STATE_MODE per-run), but at warning level so it can't scroll by
unnoticed.
```

### rl/online/league.py — PLAYER_KEYS — one entry since the exploiter populations went
Removed:

```
identities falls out of the same statistics code. One entry since the
exploiter populations were removed (LESSONS.md 9); the tuple shape is
```

Kept in code:

```
identities falls out of the same statistics code. One entry today; the
tuple shape is
```

### rl/online/league.py — the provenance tag — one value since the exploiter populations went
Removed:

```
range. One value since the exploiter populations were removed
(LESSONS.md 9), but kept as a field: refs pickled by older revisions
```

Kept in code:

```
range. One value today, but kept as a field: refs pickled by older
revisions
```

# Migration — evidence removed from the tooling, offline, service and test trees

Every block below is the text removed from the code, verbatim. Where one
comment carried both semantics and evidence, the whole original comment is
reproduced so no removed wording is lost; only the evidence half left the
source.

---

## scripts/wandb_views.py

### scripts/wandb_views.py — eval slate, superseded series
Original comment:

    # The eval slate since 2026-09-09 (rl/online/main.py): two slots against
    # the simple heuristic, both the EMA params at T=1. `plain-t1` samples the
    # policy exactly as the training actors do; `thresholded` samples it with
    # every legal cell below player_prune_threshold removed and the rest
    # renormalised (HeadParams.prune_threshold) -- the distribution the
    # learner's v-trace ratios are built on. Their gap prices the threshold
    # in play. Series are keyed by the eval thread's name, so the earlier
    # `-0`/`-1` (T=0.5) and `-t1-2` series end at that restart.

Evidence moved: the slate was cut on 2026-09-09; because series are keyed by
the eval thread's name, the earlier `-0`/`-1` (T=0.5) and `-t1-2` series end
at that restart.

### scripts/wandb_views.py — "0 · At a glance" section, the 2026-08-30 redesign
Original comment:

    # NEED-TO-KNOW ONLY: is this run winning, is it healthy, is it
    # collapsing, is the critic calibrated, is it about to OOM.
    # Everything else is drill-down detail in the sections below.
    # 2026-08-30 redesign collapsed 16 sections -> 10 and pulled the
    # canonical copy of every metric that used to be duplicated
    # across 3+ sections up here (scripts/wandb_views.py history —
    # see the old panel list in git log if you need the pre-redesign
    # layout back).

Evidence moved: the 2026-08-30 redesign collapsed 16 sections -> 10 and
pulled the canonical copy of every metric that used to be duplicated across
3+ sections up here (scripts/wandb_views.py history — see the old panel list
in git log if you need the pre-redesign layout back).

### scripts/wandb_views.py — "Loss & non-finite gate"
Original comment:

    # player_update_skipped is the non-finite gate — a
    # poisoned update is permanent and the next periodic
    # save overwrites the last good checkpoint with it
    # (LESSONS.md), so this is checkpoint protection, not
    # just a numerics footnote. Never surfaced before this
    # redesign.

Evidence moved: the "(LESSONS.md)" pointer, and "Never surfaced before this
redesign." — the panel was added by the 2026-08-30 redesign. The mechanism
(a poisoned update is permanent and the next periodic save overwrites the
last good checkpoint with it, so this is checkpoint protection) stays in the
code.

### scripts/wandb_views.py — "Collapse watch: entropy axes & switch rate"
Original comment:

    # THE collapse watch panel — with the adaptivity
    # controller removed (2026-08-13) and the entropy-floor
    # dual controllers removed (2026-08-30), modality
    # collapse has no automated backstop, only these
    # eyes-on axes (1330 died at modality entropy 0.08;
    # 1328 gained strength at 0.18-0.26).

Evidence moved: the adaptivity controller was removed 2026-08-13 and the
entropy-floor dual controllers 2026-08-30; run 1330 died at modality entropy
0.08, while run 1328 gained strength at 0.18-0.26.

### scripts/wandb_views.py — "Value R2 (main head)", the privileged premise
Original comment:

    # THE privileged-premise discriminator (2026-09-01):
    # priv >= deploy from 20k is the gate; priv < deploy
    # sustained past 30k is the abort (the 2026-08-25
    # falsification re-run on its own instrument).

Evidence moved: the discriminator was registered 2026-09-01, and it is the
2026-08-25 falsification re-run on its own instrument. The pre-registered
gate and abort (priv >= deploy from 20k; priv < deploy sustained past 30k)
stay in the code as live constraints.

### scripts/wandb_views.py — "Within-taken-modality normalised entropy (abort instrument)"
Original comment:

    # The ABORT instrument for the flat support hinge
    # (2026-09-09), on its own panel: the sp75c row-form
    # uniform KL pinned this at .93 while the control fell
    # to .84 and halved the exploit. The hinge is silent
    # above tau, so this should hold its ~.49; rising
    # toward .93 with ineffective_confident_mass unmoved is
    # the whole-set revert.

Evidence moved: the flat support hinge landed 2026-09-09; the sp75c row-form
uniform KL pinned this instrument at .93 while the control fell to .84 and
halved the exploit. The pre-registered read (hold ~.49; rising toward .93
with ineffective_confident_mass unmoved is the whole-set revert) stays.

### scripts/wandb_views.py — "2 · Switch & critic evidence" section header
Original comment:

    # What is left of the critic section after the advantage head
    # retired (2026-08-29) and the one-step-label panels went with
    # the last of the Q machinery (2026-08-30), merged with Step 1
    # of docs/critic-weakness-analysis.md (2026-08-23): the per-row
    # JOINT statistics that judge every later step, from the
    # completed-game outcome carried on every chunk. NaN where a
    # batch has no rows in the slice (wandb skips them).

Evidence moved: what is left of the critic section after the advantage head
retired (2026-08-29) and the one-step-label panels went with the last of the
Q machinery (2026-08-30), merged with Step 1 of
docs/critic-weakness-analysis.md (2026-08-23).

### scripts/wandb_views.py — "Action readout: drift from init"
Original comment:

    # The flat readout's own way to fail, and it is the same
    # SHAPE as the dx65cpwp runaway these panels were built
    # for: the bilinear is a two-factor product with ONE
    # zero-init factor. query must leave 0 within ~200 steps
    # (its gradient is a rank-1 outer product of live rows);
    # key must leave lecun 0.0625 shortly after (its gradient
    # is proportional to query, so it is frozen for exactly
    # one step). Either still flat at 2k IS the stall.

Evidence moved: the flat readout's failure mode is the same SHAPE as the
dx65cpwp runaway these panels were built for. The init values and the
pre-registered ~200-step / 2k stall criteria stay in the code.

### scripts/wandb_views.py — "Trunk row cosine similarity"
Original comment:

    # Rows of the trunk's OUTPUT converging to one direction
    # (Noci et al. 2022 rank collapse): cosine rising toward
    # 1 / participation falling toward 1 is the alarm
    # (> 0.9 / < 4 pre-registered); ckpt_00182000 read
    # 0.173 / 10.9 offline, and the first live points
    # after that restart must match.

Evidence moved: ckpt_00182000 read 0.173 / 10.9 offline, and the first live
points after that restart must match. The Noci et al. 2022 citation and the
pre-registered alarm (> 0.9 / < 4) stay.

### scripts/wandb_views.py — "Trunk output row L2 per group"
Original comment:

    # 2026-09-10: every row enters at RMS 1 (L2 16 at 256),
    # so a group's OUTPUT L2 is what the six blocks wrote on
    # it. Unnormalised, the history rows sat at ~1040 in and
    # out (moved 2%) while CLS went 2.85 -> 1012; a group
    # pinned near 16 is one the trunk does not revise.

Evidence moved: the 2026-09-10 date, and the pre-normalisation reading —
unnormalised, the history rows sat at ~1040 in and out (moved 2%) while CLS
went 2.85 -> 1012.

### scripts/wandb_views.py — "Matched-V realised gap (vol switch − move) per V bin"
Original comment:

    # Realised outcome of voluntary switches minus moves at
    # matched V(s). Offline: pooled -0.147 -> matched
    # -0.048±0.054. Per-batch n is tiny; read smoothed and
    # with the n panel beside it.

Evidence moved: Offline: pooled -0.147 -> matched -0.048±0.054.

### scripts/wandb_views.py — "V(s) at voluntary switches vs moves"
Original comment:

    # Selection, directly: V at the states where switches
    # are taken vs where moves are (offline -0.04 vs +0.08).

Evidence moved: offline -0.04 vs +0.08.

### scripts/wandb_views.py — "V outcome R²: all / phase / after switch vs move"
Original comment:

    # Outcome calibration of the V head (offline 0.265 on
    # fresh on-policy games). prev_switch vs prev_move is
    # the post-switch pessimism read.

Evidence moved: offline 0.265 on fresh on-policy games.

### scripts/wandb_views.py — "Voluntary-switch rows per batch (absorbing floor = 1.0)"
Original comment tail:

    # to the 1.0 floor is legible: 6ta9hmp6 ran 60.4 (3k) ->
    # 3.9 (33k), halving every ~8k.

Evidence moved: 6ta9hmp6 ran 60.4 (3k) -> 3.9 (33k), halving every ~8k. The
APO (arXiv:2602.05717) citation and the absorbing 1.0 floor stay.

### scripts/wandb_views.py — "3c · Eval slate" section header
Original comment:

    # The eval slate: the policy as the training actors sample it
    # against the same policy thresholded at sampling, both EMA at
    # T=1. The search eval actor was deleted 2026-09-09 (LESSONS.md
    # "Removal ledger — 2026-09-09 search eval actor").

Evidence moved: the search eval actor was deleted 2026-09-09 (LESSONS.md
"Removal ledger — 2026-09-09 search eval actor").

### scripts/wandb_views.py — "4 · Critic quality & value" section header
Original comment:

    # Observer critic quality. The policy no longer reads a Q stack
    # (retired 2026-08-26/30; its link to return is the v-trace
    # advantage), but an action-flat critic still voids the matched
    # control and the starvation discriminators above.

Evidence moved: the Q stack was retired 2026-08-26/30.

### scripts/wandb_views.py — "Privileged value gap"
Original comment:

    # Mean |priv - deploy| expectation: the 2026-08-25
    # "worth 0.005 value units" number, re-measured live.

Evidence moved: this panel re-measures live the 2026-08-25 "worth ... value
units" number. Per the carve-out, the figure itself is NOT restated here —
it is recorded in LESSONS.md, "Addition ledger — 2026-09-12 pairwise entity
critics".

### scripts/wandb_views.py — "Position potential"
Original comment:

    # The unit position potential (eta-free, on the wire at
    # any strength). switch_delta is DESCRIPTIVE (human
    # replays: -0.038); adv_switch/move split the channel's
    # advantage by the taken modality; grad_share is the
    # head's part of the global clip norm.

Evidence moved: switch_delta on human replays: -0.038.

### scripts/wandb_views.py — "Value R2 calibration (fresh rows)"
Original comment:

    # Fresh-row calibration. Was framed as "Q fresh/replay
    # vs V fresh" pre-2026-08-30 — the Q side retired with
    # the Q head; only the V-fresh reading remains.

Evidence moved: the panel was framed as "Q fresh/replay vs V fresh"
pre-2026-08-30 — the Q side retired with the Q head; only the V-fresh
reading remains.

### scripts/wandb_views.py — "Action-head gradient norm"
Original comment:

    # Pre-clip grad norm per policy-head subtree, the
    # policy pathway's own gradient scale (the retired
    # Q-head pair stayed calm through both dx65cpwp
    # failures).

Evidence moved: the retired Q-head pair stayed calm through both dx65cpwp
failures.

### scripts/wandb_views.py — "6 · League", the payoff heatmap preset
Original comment tail:

    # titles and a diverging win-rate colour scale —
    # replaces both the old matplotlib MediaBrowser image
    # panel and the later confusion-matrix-preset hijack.

Evidence moved: the custom Vega-Lite preset replaces both the old matplotlib
MediaBrowser image panel and the later confusion-matrix-preset hijack.

### scripts/wandb_views.py — "Game length"
Original comment:

    # Whole-game length off terminal chunks' done rows — the
    # distribution to watch since the 96-request force-tie
    # was removed (2026-08-16).

Evidence moved: this is the distribution to watch since the 96-request
force-tie was removed (2026-08-16).

### scripts/wandb_views.py — "Applied update rms · action readout leaves"
Original comment tail:

    # force acts on (post-clip, post-revert rms): the switch
    # pair's query and ally-side scalar (which replaced the
    # switch_bias whose delta started the pattern) beside the
    # move pair's.

Evidence moved: the switch pair's query and ally-side scalar replaced the
switch_bias whose delta started the pattern.

### scripts/wandb_views.py — "Shape lattice combo (T, H)"
Original comment:

    # Which (chunk_rows, history_rows) combo of
    # player_shape_lattice a batch hit — relevant given the
    # shape-lattice OOM-guard history (the first bullet
    # under CLAUDE.md's "Invariants"): a
    # surprise top-bucket compile is what killed three runs
    # before the lattice was enumerated up front.

Evidence moved: the shape-lattice OOM-guard history (the first bullet under
CLAUDE.md's "Invariants") — a surprise top-bucket compile is what killed
three runs before the lattice was enumerated up front.

### scripts/wandb_views.py — "9b · Actor step timing" section header
Original comment tail:

    # means over the pool. Where an actor's step goes — the
    # system rate is actor-bound (learner alone ~3x faster), and
    # this is the baseline the history-carry pass is judged on.

Evidence moved: the learner alone is ~3x faster, and this panel set is the
baseline the history-carry pass is judged on.

### scripts/wandb_views.py — "History carry: suffix size per request"
Original comment:

    # The carry path's own read: steps / packed rows of
    # the suffix a request actually sends (ex.bin ~3 / ~5
    # per request against the 64/128+ a full window pads
    # to).

Evidence moved: on ex.bin the suffix is ~3 steps / ~5 packed rows per
request.

---

## scripts/register_wandb_charts.py

### scripts/register_wandb_charts.py — `_winrate_hex` provenance
Original docstring:

    """Red/gold/green hex for a win rate. Self-contained: the learner-side
    twin this used to mirror no longer exists, and this script has no
    jax/model deps by design."""

Evidence moved: the learner-side twin this used to mirror no longer exists.

### scripts/register_wandb_charts.py — the v3-to-v9 payoff-heatmap spec ledger
Original comment, removed in full and replaced by a constraint-only note:

    # v3-v6 (explicit color.scale.range hex array), v7 (color.scale.scheme +
    # domain + clamp), v8 (per-cell literal hex column + scale: null), and v9
    # (v8 with the template field key renamed off "color") were all confirmed
    # spec-correct -- via wandb's own GraphQL API re-fetching the stored spec
    # byte-for-byte, and via a neutral standalone Vega-Lite renderer
    # (vl-convert) producing exactly the intended red -> gold -> green -- yet
    # every one rendered as either an unrelated pink/black/blue palette or a
    # single flat colour in wandb's actual custom-chart panel (confirmed via
    # the downloaded panel SVG: every cell baked in with the identical literal
    # fill regardless of field name or data values). The common factor across
    # every failing version: the rect mark's fill was bound to a table FIELD
    # via "field" in the color encoding. The one encoding that DID render
    # correctly the whole time was the text mark's black/white choice, which
    # uses "condition"/"value" with NO "field" at all -- pure literal values
    # selected by a boolean test. v10 applies that same pattern to the rect:
    # a chain of "condition" tests against the (quantitative, field-bound)
    # winrate value picking a literal hex "value" per 5%-wide band, and a
    # fallback "value" for the top band. No field is ever bound directly to
    # color/fill -- only used inside test expressions -- which is the one
    # combination not yet tried.

The surviving constraint in the code: a mark's fill must never be bound to a
table FIELD via "field" in the colour encoding; only "condition"/"value"
pairs render.

---

## rl/offline

### rl/offline/harness.py — provenance of the harness
Original docstring sentence:

    Born from the critic-weakness check that needed a stub
    learner, a monkeypatched SERVER_URI and a sed'd copy of the service to
    exist at all.

### rl/offline/harness.py — the unresolved-battle rate behind `deadline_s`
Original docstring:

    Every game is bounded by `deadline_s` in play_games: the service has no
    turn cap yet, and a battle that never resolves (seen on 2026-08-23, ~2%
    of games) would otherwise pin a slot forever — stragglers are dropped,
    not waited on, and the count is logged.

Evidence moved: battles that never resolve were seen on 2026-08-23, ~2% of
games.

### rl/offline/config.py — forfeit composition of the replay corpus
Original comment:

    # Forfeit handling (measured on 50k rated gen9randombattle games, July
    # 2026: ~48% played out, ~41% conceded with the winner ahead, ~11%
    # forfeited with the winner NOT ahead on mons).

Evidence moved in full: measured on 50k rated gen9randombattle games, July
2026 — ~48% played out, ~41% conceded with the winner ahead, ~11% forfeited
with the winner NOT ahead on mons.

### rl/offline/config.py — the regularisation curve behind the AdamW settings
Original comment:

    # Learning params. Supervised training wants momentum, unlike the RL
    # learner's b1=0. Regularization is sized to constrain without eroding:
    # 1e-2 decay + 0.05 smoothing produced a peak-then-decay-to-plateau
    # accuracy curve (smoothed CE saturates once fit; decay keeps shrinking
    # the solution until CE re-engages — a stable equilibrium below the
    # peak). The structural defenses (antisymmetric probe, pair batching,
    # deep supervision) carry the anti-memorization burden instead.

Evidence moved: 1e-2 decay + 0.05 smoothing produced a peak-then-decay-to-
plateau accuracy curve (smoothed CE saturates once fit; decay keeps
shrinking the solution until CE re-engages — a stable equilibrium below the
peak).

### rl/offline/type_probe.py — the behavioural comparison that motivated probe E
Original docstring:

    Motivated by the 2026-09-03 behavioural comparison (ckpt_00240000 vs 2000
    human >=1900 replays): the model lands twice the human rate of IMMUNE hits
    (0.047-0.051 vs 0.024) and fewer supereffective ones (0.175 vs 0.195). Types ARE on
    the wire: `species.npy` / `moves.npy` are multi-hot attribute tables with a
    dedicated column per type (verified 2026-09-03), so each operand row carries
    its type bits pre-trunk...

Evidence moved: the 2026-09-03 behavioural comparison (ckpt_00240000 vs 2000
human >=1900 replays) found the model lands twice the human rate of IMMUNE
hits (0.047-0.051 vs 0.024) and fewer supereffective ones (0.175 vs 0.195);
the type columns in `species.npy` / `moves.npy` were verified 2026-09-03.

### rl/offline/kind_probe.py — the readings that motivated the kind probe
Original docstring:

    The
    readout consumes the rows AFTER six blocks, and the type ceiling read
    post-trunk rows as LESS legible than the assembled input (0.50 vs 0.60)
    while `player_trunk_row_participation` fell 7.1 -> 4.5 over irqeetfg.
    This asks, per block, whether that is the trunk squeezing the kinds into
    a shared subspace:

Evidence moved: the type ceiling read post-trunk rows as LESS legible than
the assembled input (0.50 vs 0.60) while `player_trunk_row_participation`
fell 7.1 -> 4.5 over irqeetfg.

### rl/offline/separation_probe.py — the 2026-08-27 collapse measurement
Original docstring:

    (docs: the 2026-08-27 measurement found within-row
    switch Q std 0.0196 vs move 0.0374 vs between-row 0.507 on the live
    checkpoint — the collapse's representational reading).

Evidence moved in full. The pre-registered gates (switch-group r >= 0.9 AND
std ratio >= 0.8 by step 1000; the SEEN-identity correction of 2026-08-27)
stay in the code.

### rl/offline/separation_probe.py — provenance of the batch-selection block
Original comment, removed in full:

    # Batch selection, moved here from rl/offline/overfit_probe.py when that
    # file retired with the Q head on 2026-08-29. This probe was its only
    # remaining consumer.

### rl/offline/separation_probe.py — the grid era's control rows
Original comment:

    # Controls are SEQUENCE rows that carry the named entity: my active's
    # ally-target row and the opponent active's enemy-target row (the
    # entity-derived target rows). The grid era read the 41-slot action
    # stream here; the flat trunk has no such stream, so the rows are named
    # off rl/model/constants like every head does.

Evidence moved: the grid era read the 41-slot action stream here; the flat
trunk has no such stream.

### rl/offline/tactical_cohort.py — the collection recipe followed
Original docstring:

    fixed forever
    after, so every checkpoint is read on the same contexts (the
    runtime/priority-audit-01861967 recipe).

Evidence moved: the cohort follows the runtime/priority-audit-01861967
recipe.

### rl/offline/separation_probe.py — the "pair" mode lookup control reading
Original docstring tail:

    Lookup labels measured
    2026-08-27: both architectures ~0.73-0.75 held-seen — lookup does not
    discriminate them.

### rl/offline/separation_probe.py — the seen-identity restriction's overlap measurement
Original docstring:

    Why the restriction exists (measured 2026-08-27): held-out labels are
    hashes of identity, so a species never seen in training is UNLEARNABLE
    by any architecture — and randombattle teams barely overlap across
    games (seen-frac 0.267 for switch cells vs 0.793 for move cells on the
    12-game cache), so unrestricted held-out r is overlap-capped and reads
    as an architecture gap that is actually a data artefact.

Evidence moved: measured 2026-08-27, seen-frac 0.267 for switch cells vs
0.793 for move cells on the 12-game cache. The reason the restriction exists
stays in the code.

### rl/offline/separation_probe.py — the train-r memorisation reading
Original docstring:

    routing species -> cell — measured 2026-08-27: the shared-stream
    architecture hits train r = 1.000 by step 100.

### rl/offline/separation_probe.py — probe D, the deleted entity_index_tag
Original docstring:

    History row i is `gru_state + node_snapshot` (+ its group and row bias)
    -- addends summed into one vector, the shape the 2026-09-01 pass
    deleted one level up -- and its join to public row i is positional.
    Until 2026-09-02 both also carried a shared `entity_index_tag`, measured
    at 0.028 of the other addends' rms on ckpt_00182000 and deleted for
    never training.

Evidence moved: the summed-addends shape is what the 2026-09-01 pass deleted
one level up; until 2026-09-02 both rows also carried a shared
`entity_index_tag`, measured at 0.028 of the other addends' rms on
ckpt_00182000 and deleted for never training.

### rl/offline/dataset.py — the ending composition of the replay corpus
Original docstring:

    Endings (measured on 50k rated gen9randombattle games, July 2026):
    - "played_out" (~48%): the loser's six mons all fainted — exact margin.
    - "conceded" (~41%): forfeit/timeout with the winner ahead on mons —
      the margin is the count at concession, a compressed lower bound on
      the played-out margin (concessions cluster at 1-3, played-out games
      reach 4-6 far more often).
    - "clamped" (~11%): forfeit/timeout with the winner NOT ahead (rage
      quit / timer / disconnect) — the position contradicts the result, so
      the ±1 margin is pure label noise.

Evidence moved: measured on 50k rated gen9randombattle games, July 2026 —
played_out ~48%, conceded ~41%, clamped ~11%; concessions cluster at 1-3,
played-out games reach 4-6 far more often. The ending definitions stay in the
code.

---

## tests/ and the peripheral trees

(Nothing was moved out of `embeddings/`, `inference/`, `scrape/`, `serve/`, `data/src/` or `heuristics/` — they carry mechanism and deployment contracts only.)

### tests/conftest.py — JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES in the test env
The learner's env sets JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=all (kernel + autotune caches alongside the executable cache). Under that flag every new executable rewrites the whole ~340 MB xla_gpu_kernel_cache_file: the four four tiny loss unit tests took 75 s with it and 6 s without (2026-08-24), and the fast suite as a whole 264 s.

### tests/conftest.py — jitted model init in the session fixture
Jitted init (2026-08-24): eager init dispatches the forward op by op and compiles each nn.scan separately -- it was ~6 min of the slow suite, paid again inside create_train_state (also jitted now).

### tests/test_train_step.py — eager vs jitted train_step cost
The learner's compiled train_step (donates the states; nothing below reads the pre-step ones). The eager function was ~25 min here.

### tests/test_train_step.py — file rename history
Was tests/test_train_step_q.py until 2026-08-29, when the Q head it was named for retired.

### tests/test_dtype_policy.py — the param_dtype/dtype conflation
Two properties, and they are different things (conflating them wasted a measurement on 2026-08-25).

### tests/test_br_init.py — the sp75 BR-init seed bug reading
The first sp75 probe drew its "fresh" component from the lineage seed the TARGET also grew from, so the perturb was a rewind along the target's own training path (measured cos 0.95 to the target at the BR's first checkpoint, 2026-08-30).

### tests/test_prune_policy.py — pruned-cell gradient magnitude
The removed cell's gradient is two cancelling log-sum-exp terms, zero to float32 rounding (measured -2.4e-10); the illegal cell's exactly.

### tests/test_model_forward.py — probe C baseline before the private truth channel existed
Probe C's baseline read was r ~ 0.00 precisely because this input did not exist.

### tests/test_chunking.py — the tail-window forward tolerance history
Exact equality is UNATTAINABLE here and the test used to demand it (atol=1e-5, failing at 0.023 since before 2026-08-25).

### tests/test_history_carry.py — the measured shape-noise floor for the carry bound
Under the opened readout the floor on log_policy reads ~0.14 (2026-09-02: carry worst 0.093 against it; value log-probs 0.022 against a 0.026 floor) -- the chunking test's 0.05 was calibrated on value log-probs alone.

### tests/test_history_carry.py — carry-vs-floor readings behind the 1.5x margin
The floor is ONE draw of the bf16 leading-dim class (the 256-row tail) and the carry path runs another (the 32-row suffix bucket), so the bound carries a margin over it -- read 0.093 vs 0.14 (2026-09-03) and 0.101 vs 0.096 (2026-09-05); the shifted-carry control below sits ~2.0 away, so the margin costs the test nothing.

### tests/test_history_carry.py — content-dependence of the batch-1-vs-2 GEMM noise
Batch 1 vs batch 2 is a bf16 GEMM leading-dim change, and the noise it makes is content-dependent (0.042 on the plain request, 0.067 on the carrying one, 2026-09-02, opened readout).

---

## service/src

### service/src/server/worker.ts — battle-lifecycle guarantees, 2026-08-23
Battle-lifecycle guarantees (2026-08-23). Before these, a battle that stopped producing states — a swallowed choose() error, an RQID mismatch thrown out of the stream loop, a partner that never reset — parked the client's receive forever. Both actor slots of that game were then silently lost for the rest of the run (~2% of games in the 2026-08-23 offline sweep).

### service/src/server/worker.ts — why the step watchdog is 10 min, not 2
10 min, not 2: a step legitimately stalls for the learner's lattice precompile (~9 min at launch, actors blocked behind the GPU lock) — the 2026-08-23 first launch aborted every battle in flight and restarted them.

### service/src/server/index.ts — the worker-throw that motivated isolate replacement
A worker that throws (2026-08-23: a TypeError in sendFinalState after a mid-battle destroy) used to be logged and left in the routing table, so every gameId hashing to it — 1/numWorkers of all games — waited on a dead isolate forever.

### service/src/server/index.ts — the old justification for the same-worker routing invariant
(This once carried a justification about the exploiter "three-population redesign" raising the odds; those populations were deleted 2026-08-21 and the league is one population again, but the routing invariant is unchanged and still load-bearing.)

### service/src/server/runner.ts — actionCells cleared per request
Clearing it with every request (4c8836d, 2026-08-31) held HAS_PREV_ACTION at 0.

### service/src/server/runner.ts — endBattleStream must be once-only
... and writeEnd() is async so a plain try/catch around it catches nothing — the rejection killed workers as an unhandled 'error' (see the 2026-08-13 service crash).

### service/src/server/runner.ts — the destroyed flag
`start()`'s loop exits when destroy() ends the stream and would otherwise build the final state on the torn-down battle -- an unhandled rejection that took the WORKER down with every other game on it (2026-09-03, at a watchdog abort: `Cannot read properties of null (reading 'team')`).

### service/src/server/state.ts — warn-once for unmapped enum keys
... the per-occurrence line printed thousands of times per session and buried real errors in the pane (the 2026-08-13 worker crash was nearly scrolled out by it).

### service/src/server/state.ts — the private truth channel vs the privateBattle candidate
NOT the privateBattle `candidate`: its hp is log-event-driven and reads 0/0 for a mon the log has not yet given a reading (measured: request "252/342" against candidate 0/0, caught by the harness truth invariant on its first run), and NOT the transform-unwrapped `pokemon`.

### service/src/server/state.ts — team-preview cells before 2026-08-29
Until 2026-08-29 the mask lit one cell per REMAINING position -- up to 7 -- while choiceFromAction ignored the target entirely, so the policy spread its mass over up to 7 exact duplicates of one choice and the micro-entropy cell count was inflated to match. One cell per candidate now, at the canonical column.

### service/src/tests/battle.test.ts — doubles slot-alignment violation rate
Doubles formats currently violate the slot-alignment invariant (~75% of battles; 622 hits over one ~3200-battle soak) — a pre-existing defect the old harness swallowed (its controller caught and console.error'd every invariant throw).

### service/src/tests/battle.test.ts — the Illusion/Zoroark false-positive rate behind retry: 2
retry: the slot-alignment assert has a documented false-positive class (Illusion/forme changes — and the gen9ou sample team is a Zoroark team), observed at ~1% of soak battles. Three independent failures (~1e-6 by chance) still fail the suite, so a systematic regression stays fatal.

### service/src/tests/harness.ts — what assertPrivateSideShape replaced
The private-side shape contract. Until 2026-08-25 these two buffers were decoded here only to feed the frozen-opponent-sheet invariants; when those were deleted with the privileged critic the decodes stayed behind an eslint-disable, i.e. a decode-does-not-throw smoke test wearing the costume of an assertion. This is the replacement: the encoder's own shape contract, which is what the python decoder (`rl/environment/utils.process_state`) reshapes against and will throw on if it ever drifts.

### service/src/tests/harness.ts — alignment-key false-positive rate
KNOWN ~0.1% false-positive class (2 in 1791 soak battles): a my-side Illusion mon's own index attaches to no public row until |replace|.

---

## Appendix — exact original text of the trimmed offline docstrings

Reproduced line for line so that no removed line is unaccounted for.

### rl/offline/harness.py — original opening paragraph

    critic questions can be answered against real trajectories without a
    training run. Born from the critic-weakness check that needed a stub
    learner, a monkeypatched SERVER_URI and a sed'd copy of the service to
    exist at all.

### rl/offline/config.py — original forfeit-handling comment

    # Forfeit handling (measured on 50k rated gen9randombattle games, July
    # 2026: ~48% played out, ~41% conceded with the winner ahead, ~11%
    # forfeited with the winner NOT ahead on mons).
    #
    # Drop games where the sign-clamp engages (forfeit/timeout with the
    # winner not ahead): the recorded result contradicts the position, so
    # every step's label is noise

### rl/offline/separation_probe.py — original probe preamble

    "can this ARCHITECTURE tell a modality's candidates apart?" before any
    training time is spent (docs: the 2026-08-27 measurement found within-row
    switch Q std 0.0196 vs move 0.0374 vs between-row 0.507 on the live
    checkpoint — the collapse's representational reading).

### rl/offline/separation_probe.py — original "pair" mode tail

    mode "pair" keys every identity-carrying cell by (candidate id,
    OPPONENT ACTIVE species) instead: the label then changes when the
    opponent changes, so a lookup of the candidate alone cannot fit it —
    only RELATIONAL routing (candidate x opponent, the matchup shape the
    deployment task actually needs) generalises. Lookup labels measured
    2026-08-27: both architectures ~0.73-0.75 held-seen — lookup does not
    discriminate them.

### rl/offline/separation_probe.py — original held-out gate paragraph

    memorise per-row-per-slot values through state features without ever
    routing species -> cell — measured 2026-08-27: the shared-stream
    architecture hits train r = 1.000 by step 100. Identity-keyed labels
    generalise to UNSEEN states only through genuine candidate routing

### rl/offline/separation_probe.py — original probe D paragraph

    History row i is `gru_state + node_snapshot` (+ its group and row bias)
    -- addends summed into one vector, the shape the 2026-09-01 pass
    deleted one level up -- and its join to public row i is positional.
    Until 2026-09-02 both also carried a shared `entity_index_tag`, measured
    at 0.028 of the other addends' rms on ckpt_00182000 and deleted for
    never training. This is the behavioural read: a ridge readout over the
    post-trunk history row

### rl/offline/tactical_cohort.py — original collect paragraph

    SimpleHeuristic at T=.5 with fixed seeds and pickles it — fixed forever
    after, so every checkpoint is read on the same contexts (the
    runtime/priority-audit-01861967 recipe). Point PS_SERVICE_URI at a
    service started with BATTLE_LOG_DIR set so the simulator's own `-immune`
    lines confirm the taken-move events.


## Shared side and active/bench identities — 2026-09-13

User-directed numerical change, replacing separate representation-specific
side tags with a common mine/theirs table. Revert baseline: `f2991ff`.
The audit found four independent side parameter banks (public, own sheet,
opponent sheet, field), plus ally/enemy target tags. Removed six width-D
vectors in total: private_side_bias, opp_private_side_bias, field_side_bias,
ally_target_bias and enemy_target_bias. The existing side_bias and pos_bias
tables remain; this is semantic consolidation, not a measured performance
ablation of those vectors.

`rl/model/identity.py` owns the mapping. Encoder content is normalised before
adding shared side/position identities, target-slot identity and group bias;
invalid rows are zeroed afterwards. Public and private entities get side plus
position (0 bench, 2 first active, 1 second active). Private positions join the
one-based ENTITY_IDX to PUBLIC_ORDER with a side check; unmatched/unrevealed
entries use bench. Entity history gets only the shared side tag. Current and
remembered side conditions both get the corresponding common side vector;
global field rows get neither side nor position.

All 17 target categories have explicit semantics, independent of whether
Pokémon content is present: ally categories get mine, foe categories get
opponent, ALL/ALL_ADJACENT get their sum once per side, AUTO/DEFAULT/UNSPECIFIED
get neither. Named active targets and slot-specific passes get the active
position. Target-category embeddings remain. Previous-action rows gather the
same semantic identity as their named source/target.

History event processing retains event-time side/position tags, while the
latest snapshot stream and carry now store untagged content. This prevents
a snapshot from adding an old active-position vector directly to a history
trunk row. The existing recurrent-state + latest-snapshot addition remains;
its older late-game motivation is recorded above, but this pass did not find
an isolated ablation demonstrating its benefit in the current architecture.

Validation: 36 focused fast checks passed for semantic mappings, all target
categories, private alignment, shared gradients, abstract full-model dtype/
shape tracing, history recurrence and the privileged trunk mask. A further
small history test passed for untagged snapshots, live event-identity effects
on recurrent states and snapshot carry. Updated real-model assembly tests
cover post-normalisation side additions and private position joins; these
were not run because the NVIDIA driver is unavailable. No training restart
or checkpoint migration was performed. This change is not bit-identical.


### RL history rows use recurrent memory alone — 2026-09-13 follow-up

Explicit user-directed removal of the latest-node addition to RL history
rows. The public entity rows already carry current observable state; the
latest event snapshot is overlapping content and can be older. No isolated
current-architecture benefit was established for the bypass. This is a
numerical architecture choice, not a claim of measured learning improvement.

RL history content is now only the aligned recurrent row state, normalised
before adding shared side and HISTORY_ENTITY group identity. Removed the
snapshot argument from `_assemble_sequence` and `_batched_forward`, and the
snapshot gather from `_history_inputs`: 12 width-D snapshot vectors per
request no longer contribute directly to the trunk's history rows. The
existing public event inputs and recurrent update are untouched, as are the
side/position identities from the preceding change. No learned parameters
were removed in this follow-up.

The standalone offline critic still reads latest-node snapshots via
`encode_history` and `read_history_into_nodes`, so its snapshot computation
and the shared carry API remain. They are not a dependency of RL history-row
content. The offline assembly harness now unpacks the current three-result
assembly API. Restore baseline: `f2991ff`; to restore only this bypass on top
of the shared-identity change, gather `encode_history`'s third result through
PUBLIC_ORDER and add it to the aligned recurrent state before input norm.

Validation: 25 focused fast checks passed, including snapshot-independent
history input alignment with a live-memory positive control, semantic
identity contracts and abstract full-network shape/dtype tracing. The new
real-model test compares history rows with normalised memory plus shared
side/group embeddings and checks invalid-row zeroing; GPU execution remains
unavailable because the NVIDIA driver cannot be reached. No restart, commit
or checkpoint migration performed.

### Entity-history current position identity — 2026-09-13 follow-up

User revised the earlier side-only choice: entity-history rows now receive
exactly the same side + current active-slot/bench identity as their aligned
public rows. `sequence_identities` computes that identity once and assigns
it to both groups, after content normalisation in the encoder. Position is
read from the current public ACTIVE column, not from an event snapshot.
History content remains recurrent memory alone. All bench entries share the
bench vector; this is a role association, not a unique entity identifier.

Validation: all 24 focused identity tests passed, including current-role
changes affecting the corresponding history rows, shared public/history
identity, field position-independence and privileged identity isolation.
Ruff and diff whitespace checks passed. The GPU assembly expectation was
updated to include position; GPU execution remains unverified. No parameters
added, commit made or training restarted. To revert this follow-up only,
assign side_embeddings[public_sides] to HISTORY_ENTITY_ROWS again.

### Observed relevant-record counts — 2026-09-13 audit

Read four archived singles cohorts: runtime/lineage_games_ckpt182000.pkl,
runtime/discrim_sides_ckpt224773.pkl, runtime/sep_games_ckpt238825.pkl and
runtime/transition_games_01220000.pkl. All stored NUM_ACTIVE values were 1.
Across 360 stored game-perspectives, deduplicating overlapping history
windows within each perspective by (TURN_VALUE, TURN_ORDER_VALUE), there
were 27,706 observed events: 13,076 with one relevant record, 14,554 with two,
76 with three, none with four or more. Maximum consecutive packed-start
index difference was also 3 in every cohort, providing a check beyond the
already capped NUM_RELEVANT field. This is a sample of archived windows,
not proof of a universal bound, exhaustive complete-game coverage, or a
measurement of doubles. No runtime evidence here justifies increasing the
8-entry capacity to 12. Three newer cohorts (mcts-readiness-01861967-games,
tactical-cohort/games, priority-audit-01861967/control-games) could not be
loaded directly because their pickles reference the removed SearchOutput
class; they were excluded rather than interpreted as empty data.

## Unified 19-row recurrent history and four trunk memories — 2026-09-13

User requested four global history registers, then clarified that entity,
field and register memories must read each other across time through one
12 + 3 + 4 attention sequence. Final implementation replaces the independent
slot/field recurrences with one shared HistorySequenceStep over 19 rows.
The initially drafted separate register read/recurrence was superseded before
any run or commit. This explicitly changes the former no-cross-slot/no-field-
from-slot contracts; their tests are replaced by positive tests of all three
source groups reaching all three destination groups through prior memory.

At each valid event, packed records are projected and scattered to their
12 stable entity slots; three field embeddings and four zero event-input
register rows complete the sequence. All 19 memories participate, including
untouched entities; field-only valid events also advance memory. The shared
step RMS-normalises previous memory + event input, adds group identity and
four persistent register identities, then performs live-initialised 19x19
self-attention. One shared sigmoid gate and linear candidate update every
row: h = (1-z)*h_previous + z*candidate. No GRU reset gate or tanh candidate.
The f32 retained-memory path bypasses RMSNorm; invalid steps preserve it
exactly. Initial memory is one learned (19,D) bank. No modality-specific
recurrence, field-message sum or pooled-field-to-slot path remains.

Unlike the previous input-only minGRU, gates/candidates now depend on prior
cross-row memory, so the recurrence is a chronological Flax scan with shared
parameters and per-step rematerialisation. The former associative scan and
its separation tests were removed at the user's direction; the recorded
~26us/step serial-GRU latency motivation is still relevant, and no runtime
speedup or playing-strength improvement is claimed. RMSNorm here is a
pre-attention/read-branch scale choice, not an experimentally established
requirement. A shared standard GRU driven by attention context is a possible
reference comparison now that parallel-scan eligibility is lost; it was not
implemented in this pass.

Four request-aligned register states enter the trunk as HISTORY_REGISTER,
with normalised content + group identity, no side/active position tag.
Sequence layout is 84 learner /77 actor rows, preserving previous row indices;
existing four internal trunk workspace registers remain (88/81 internal rows).
The information mask keeps the history registers policy-observable only.
Actor carry includes their four f32 memories; request-count alignment, invalid
carry reset, no-future-input dependence and suffix/full-history equivalence
are tested. Raw snapshots remain available to the standalone offline critic,
with its encode_history unpack updated; RL entity-history content remains
memory-only with current side/position identities.

History parameter telemetry follows sequence_step/attention and the shared
sequence_step/cell/gate; the existing slot_gate_rms name now reads the shared
cell. Attention statistics now cover 19 queries/keys rather than up to eight
event records, so historical entropy/source-mass curves are not directly
comparable. Register-group norm telemetry derives from SequenceGroup.

Validation: 62 distinct focused fast tests passed across history attention,
19-row recurrence/carry/gradients, abstract full-model dtype/shape and telemetry
paths, actor carry batching, row layout/readout, semantic identity and privileged
mask tests. GPU real-model tests were updated and collected, but cannot run
here because the NVIDIA driver is unavailable. No checkpoint migration,
training restart, commit or push. Baseline restore handle: f2991ff for the
pre-session network; the earlier side/position edits are a separate semantic
change to retain if reverting only this recurrence.

The relevant-record cap stays eight. In this final architecture, reducing it
to four would only shrink gather/projection/scatter work, not the now-fixed
19x19 attention matrix. The archived singles maximum of three is evidence
about those samples, not permission to silently truncate larger events.

### Type-specific GRUs and attention-only identities — 2026-09-13 follow-up

The user superseded the shared linear-candidate update above: retain the
joint 19-row attention but use a GRU with separate weights for entity,
field and register tokens, shared within each type and across history time.
The local installed Flax GRUCell is the reference: reset and retention
sigmoids, reset-after recurrent candidate projection, tanh candidate, and
h_next = (1-retain)*candidate + retain*h_previous. Input projections use
LeCun normal initialisation and recurrent projections use orthogonal
initialisation, with the reference bias placement. The deliberate precision
difference is f32 gate/candidate activations and memory mixing while dense
projections respect cfg.dtype. Telemetry's slot gate remains the WRITE
fraction (1-retain); its weight RMS now reads entity_gru/iz and entity_gru/hz.

The user also required side/position identities only in step attention.
Packed entity content and field content now reach the raw event path without
additive identities; the redundant explicit side one-hot in event_projection
is removed. Event side/position embeddings are separately gathered and
averaged per touched stable slot; field rows receive their shared side tags.
These tags, group identities and register identities are added AFTER the
attention branch RMSNorm. They are not directly added to the GRU's event
input or memory. Each type's GRU reads raw previous memory and raw event
features plus the shared attention output. Identities can therefore influence
what attention writes into memory, but repeated identity addition and
normalisation do not directly modify the retained-memory path. Trunk
side/position identity assembly is unchanged by this follow-up.

Validation: 21 focused tests passed across history_gru, history_registers,
history_encoder and dtype_policy. Tests match Flax f32 initial parameters,
outputs and gradients, exercise both gates and bf16 small-update retention,
prove separate type weights and shared within-type behaviour, and verify
live gradients through all six projections for all three GRUs. Muting
attention makes changed side/position tags inert for every memory type;
with attention live they affect memory. Carry/padding and cross-type reads
also pass, as do full-network abstract dtype/parameter/telemetry checks.
Focused Black/isort/Ruff and git diff --check passed. GPU numerical tests
remain unavailable due to the driver; no learning or speed benefit measured.
This changes checkpoint parameter structure; no migration or training restart
was performed. Restore reference remains f2991ff with earlier session changes
kept separately when reverting this recurrence only.

### History identity routing audit — 2026-09-13

Traced the current working-tree implementation after the type-specific GRU
change. Event content and latest-node snapshots share the same untagged
public embedding; the two cache arguments are now redundant in production.
Event-time side/position tags are averaged over relevant records per stable
entity slot. Untouched slots receive zero explicit semantic identity for that
event, even though all 19 memory rows participate in attention. Field side
tags, three recurrent group tags and four register tags are persistent.
The learned (19,D) initial memory is an additional row-specific starting
signal, not a persistent additive identity. Duplicate relevant records for
one slot average their roles, so differing positions would mix there; this
audit did not establish whether such records occur in live events.

StepAttention projects queries, keys AND values from the same normalised
memory-plus-event rows with all those tags added. Consequently the tags can
be copied through values into GRU memory, as well as influence attention
weights. The raw event and retained-memory paths have no direct tag addition.
At trunk entry, request-aligned history memory receives current side/position
and trunk group tags after normalisation. These current roles have different
time semantics from the event roles already encoded inside memory.

A candidate separation is Q/K from normalised content plus identity and V
from normalised content alone. This removes direct tag-valued writes, but
memory still depends on identity through attention weights. It is a numerical
experiment, not an equivalent refactor or demonstrated improvement. Persistent
entity roles would additionally require event-time role tracking through carry
and rewrites; current request positions must not be broadcast into past events.
No architecture change or bias removal was made by this audit.

Validation: 42 focused tests passed across history_gru, history_encoder,
history_registers and sequence_identity, including muted-attention isolation
with live positive controls, carry/padding and current-role mappings. These
tests establish routing contracts, not absence of learned conflation or playing
strength. No full-model GPU test or training action was performed.

### History identities address Q/K only — 2026-09-13 implementation

User authorised the audit's Q/K versus V separation. HistorySequenceStep
now retains normalised memory-plus-event content separately from the rows
with event side/position, group and register identities added. StepAttention
requires explicit value_rows: Q/K project the addressed rows, V projects
the normalised content. GRUs still read raw memory and raw events plus
attention output. Parameter paths, shapes and initialisers are unchanged;
the forward is intentionally numerically different. Prior identity-dependent
information in memory remains readable through V; this removes fresh direct
tag-valued writes, not every identity dependency in memory.

Validation: 22 focused fast tests passed across history_gru, history_encoder,
history_registers and dtype_policy. The new regression fixes attention
weights by zeroing Q while keeping V/output live: perturbing semantic, group
and register tags then leaves outputs bit-identical. Live Q gives changed
probabilities and memory; zeroing V changes the fixed-attention result,
proving the isolation test has a live value path. Existing carry/padding,
cross-group gradient and abstract bf16/f32 contracts pass. Focused Black,
Ruff and whitespace checks pass. Slow full-model GPU tests were excluded;
no playing-strength or runtime claim, training restart, commit or push.
Restore baseline: a33eea3 for the model and affected tests.

### The 19-row recurrence was chaotic at init — 2026-09-13 follow-up

The unified history recurrence above shipped with a positive Lyapunov
exponent. `test_suffix_carry_replays_the_game_within_bf16`,
`test_untruncated_tail_window_forward_matches_within_bf16` and two others
failed on the first GPU run of the slow suite (the driver was unavailable
when the recurrence landed, so none of them had ever executed).

The seed is bf16 GEMM shape noise, not a boundary bug. The actor's suffix is
rounded to a small geometric bucket and the learner's window is 256 steps, so
the batched precompute (`event_projection`, the node/edge/field embedders)
runs one GEMM per path at different leading dimensions. Measured at request 0,
where BOTH paths start from `initial_memory` and consume the same two steps
and no carry exists: 0.0083 in slot memory. `clip_history_suffix`,
`_last_step_index` and the edge-filter-free `step_valid` all agree — a
boundary bug would show a large difference immediately, not a creeping one.

Carry-vs-full-window slot memory over one archived game, by request:
0.0083 (r0), 0.0182 (r6), 0.0396 (r12), 0.1279 (r18), 0.5352 (r24), 1.5938
(r30), then flat 1.5-1.8 to r54. Roughly x1.2/step, saturating at the signal
scale — bounded (RMSNorm on the read branch, tanh on the candidate) but fully
decorrelated by mid-game. Worst log-policy divergence 1.10 against the test's
0.05 bound, with the measured floor at 0.0: the 256-row tail clip is
BIT-IDENTICAL, so there is no shape-noise budget to spend.

Positive control, `attn_out` zeroed: flat 0.001-0.002 across the whole game,
field and register EXACTLY 0.0000 (their event inputs are shape-invariant, so
all of their drift arrived through attention). The memory -> RMSNorm ->
19x19 attention -> GRU input -> memory loop is the amplifier. The per-slot
minGRU it replaced could not do this: input-only gates and candidates, scanned
in f32, so a per-step perturbation stayed one.

`attn_out` scale sweep (worst slot memory over the game): 0.0 -> 0.0020,
0.1 -> 0.0452, 0.25 -> 0.7500, 0.5 -> 1.3203, 1.0 -> 1.8672. A knife-edge
between 0.1 and 0.25, so NO init scale is robust — `attn_out` grows in
training and walks back across it. Declined on the repo's own rule that a
coefficient cut which only delays onset is falsified. Zeroing it also makes
the identity-routing contract vacuous (with a zero output projection the tags
provably cannot reach memory, which is what those tests exist to deny).

The fix is contraction, not injection: `HISTORY_RETAIN_BIAS = 4.0` on each
GRU's `iz` bias, so memory retains sigmoid(4) ~ 0.982 per event. Retain-bias
sweep at full lecun attention: 0.0 -> 1.8672, 1.0 -> 1.7656, 2.0 -> 0.4629,
3.0 -> 0.0332, 4.0 -> 0.0215. The published forget-bias 1.0 (Gers et al. 2000;
Jozefowicz et al. 2015) does NOTHING here — needing 3-4 is the measurement
saying this loop carries more gain than a classic gated recurrence, which is
what putting attention inside the recurrence buys. 3.0 is the first value
under the bound; 4.0 is the margin. `HistoryGRUCell.retain_bias` defaults to
0.0 so the Flax-reference equivalence test still compares against an
unmodified `nn.GRUCell`; the deviation is set at the call site.

The bias is LEARNED, so this is a safe start, not a guarantee — training can
walk it back toward chaos. `player_history_slot_gate_rms` reads the write
fraction 1-retain (~0.018 at init, was ~0.5), so the drift is observable.

No fast unit test reproduces this. `HistorySequenceStep` in isolation is
contractive at every width tried (32/64/128/256: ~0.19 at retain 0, ~0.31 at
retain 4) — the chaos is emergent from the real operating point: event
embeddings small against memory so attention reads mostly memory, live
identities, and the carry chained across ~60 requests against a full-window
run that restarts each time. A toy-width growth test would pass vacuously;
the real contract test is the existing slow carry test.

Two test defects surfaced alongside, both PRE-EXISTING at f2991ff (neither
`interfaces.py`'s `priv_value_head` default nor `test_actor_sequence.py`
changed): `assert getattr(actor_out, "priv_value_head", ()) == ()` compares a
dataclass whose leaves are all () to a tuple and can never hold, and the
`log_policy` half read a field that lives on `action_head`, so it returned the
() default and passed whatever the actor emitted. Both now assert leaf
emptiness on the right objects. The actor was never leaking either head.

`test_server_mixed_group_matches_single_forwards`'s control was recalibrated
from an absolute 1.0 to 3 * shape_noise. The plain-vs-carrying separation is a
property of the architecture: ~2.0 under the per-slot minGRU, 0.70 before this
fix and 0.72 after, so the retain bias did not cause the shrink — the rewrite
did. Worth its own read: the carry moves the policy about a THIRD as much at
init as it used to, which for a rewrite premised on memories reading each
other is the opposite of the intended direction.

## Trunk feature-norm growth and the nGPT normalised residual — 2026-09-17

**Measurement (run ijk4nyi4, the public-tier lineage from step 0).** The
plain pre-RMSNorm residual stream grows without bound over the run.
`player_trunk_out_row_l2_{private_entity,public_entity,cls}`, read before
the output norm, went 266 / 211 / 319 at 99k to 1129 / 1191 / 1637 at 557k,
linear, no plateau. On the 5,643-state switch cohort at `ckpt_00513639`
(the graceful-stop checkpoint) a sheet row enters the trunk at RMS 1.0,
leaves block 1 at 36 and block 6 at 76; per-block update RMS
`[35.9, 9.6, 17.5, 21.9, 9.6, 23.9]`; cos(input row, block-6 row) = 0.009,
the final row 0.96 aligned with the block-5 row and 0.58 with block 1's
write. Weights: trunk Frobenius norm 145 → 167 over 20k → 514k (+15%,
linear, block 1 carrying most of it), but top singular values x2.5-3
(block-1 ffw up-projection 4.0 → 8.6 at 280k → 11.1 at 514k;
`player_trunk_attn_out_rms` 0.063 → 0.079, `player_trunk_mlp_out_rms`
0.032 → 0.043). SwiGLU is cubic in weight scale, so a 2.5x spectral growth
is a 20x feature growth. The norm scales `(1+scale)` all sit at ~1, so
pre-norm absorbs the scale functionally; the cost is that the residual path
no longer carries the input -- block 1 is the de facto embedding (matches
the 2026-09-10 first-block causal test) -- and bf16 at RMS 76 has no
resolution for input-scale content. Cross-lineage, suggestive only: the
previous lineage's 280k checkpoint (older layout) read depth RMS
`[1.0, 1.7, 2.2, 2.8, 3.3, 5.4, 3.5]` and cosine 0.32. Scripts and raw
numbers: `runtime/type-probe-switch/attention/ckpt_00513639/`.

**Mechanism (branch `normalised-residual`, flag
`player_trunk_normalised_residual`, default OFF = bit-identical).** nGPT's
residual (Loshchilov et al. 2024, arXiv 2410.01131, eq. 10-11):
`row = unit_rms(row + alpha * (unit_rms(sublayer_out) - row))` per
sub-layer, alpha a per-block per-channel f32 leaf at 1/num_blocks = 1/6
(nGPT's stated rule "of order 1/n_layers"; their literal 0.05 is the 24-36
layer value). **At init the normalised trunk is far less history-sensitive
than the plain one**: in `tests/test_history_carry.py` a one-request-shifted
carry moves the fresh policy 0.025 (0.008 at alpha 0.05), where the test's
own comment records ~2.0 for the plain trunk (that figure was NOT
re-measured on the current code; the plain control is only known to clear
the 0.05 bar), because each sub-layer write is a sixth of an RMS-1 row
rather than an unbounded add. The carry fixture therefore opens the alphas
to 1/3 for its controls, as it opens the readout's zero paths (drift 0.027
policy / 0.041 value inside the 0.05 bf16 bar; 0.056 at 1/2).
(`encoder/trunk/blocks/{attention,ffw}_alpha`), RMS-1 rather than unit-L2
so the stream shares the row convention every row enters at and the
zeros-init pre-norms stay identity at init; PLUS nGPT's
`normalize_matrices`: every embedding-space vector of the six block kernels
projected to unit L2 at init and after every optimiser step
(`trunk.project_trunk_kernels`, called from `apply_player_gradients`;
`reg_params` / `old_policy_params` left alone). Deviations from nGPT,
deliberate: pre-RMSNorms kept (identity at init; their gain is the
per-input scale), no `s_qk` (QK-norm is on), no `s_u`/`s_nu`/`s_z` (nGPT's
ablation prices fixed scales at +0.11% loss), biases kept, no alpha scale
trick (the fallback if alphas stall). Panels: `player_trunk_alpha_*` (per
block), `player_trunk_kernel_col_norm_*`, and
`player_trunk_in_out_cosine_<group>` -- the replacement read once the L2
panel pins at sqrt(width) by construction. The flag reaches the ACTOR
through `artifact.player_model_config_for` (main.py and harness build the
actor config through it now); a learner-only flag would have run the
actor with plain residuals over alpha-trained weights.

**Test lesson.** `tests/test_public_sequence.py`'s request-vs-event parity
compared bf16 rows from two separately compiled programs at 1e-3, below
bf16's ulp at the values compared (0.02 at magnitude 4). It passed only
while the two programs' autotuned kernels agreed bit for bit; the flag's
program broke that for 22% of elements although the pre-trunk rows are
provably identical between flag on and off (params and both paths'
pre-trunk outputs bit-equal in a direct check). Under pytest the
persistent XLA kernel cache is off (`conftest.py`), which is why a direct
script passed and pytest failed. The test now reads the parity through an
f32 forward over the same f32 params, the `test_actor_sequence.py`
pattern. A bf16 cross-program equality test needs either f32 or a
tolerance at bf16's ulp.

**Instrument verified offline (Step 1, no run).** `harness.forward` over
the tactical cohort at `ckpt_00513639` (flag off) reads the new
`player_trunk_in_out_cosine_<group>` panel as private_entity 0.012 /
public_entity 0.119 / cls 0.071 / move_slot 0.113 / value_cls 0.132 /
prev_action 0.000; the probe's 0.009 was the switch-legal candidates'
sheet rows only, the panel is all six sheet rows over every valid step.
These are the control's banked cosine values.

**Pre-registered acceptance (scratch lineage, flag ON, launch config
uniform-KL 0.01 / magnet 0.025 / world model off; user 2026-09-17).**
By construction: `player_trunk_out_row_l2_*` = 16 ± 0.1, kernel column
norms 1.0, no non-finite gate trips. Mechanism: private-entity in/out
cosine ≥ 0.3 at 100k and not decaying by 200k (control 0.009); alphas off
0.05 by 33k, none collapsing to 0 unexplained. Claim: wr-t1 heuristic ≥
the ijk4nyi4 row at 100k (0.108) and 200k (0.172), expected parity at
100k and a lead at 200k; `player_switch_mass_choice` ≥ 0.05 at 200k
(ijk4nyi4 0.019). Offline at 200k: switch-readout `candidate_post`
balanced accuracy above ijk4nyi4's 200k checkpoint. **Control: the banked
lineages only (user decision) -- ijk4nyi4 ran 0-320k with no uniform-KL
floor, then 0.05, then 0.025, at magnet 0.05 throughout, so the
behavioural gap is the trunk change PLUS those objective changes; the
construction and cosine reads are within-run and need no control.**
Fallbacks: instability in the first 5k → `residual_alpha_init` 1/6, once;
cosine restored but wr-t1 below the control by > 0.03 at 200k → record
and retire; alphas → 0 with cosine at 1 → the nGPT alpha scale trick.
Declined: checkpoint-mode resume of 513639 (the trained blocks write 36x
the input; alpha 0.05 would make the trunk near-identity at step 1 and the
hold would judge recovery); LayerNorm Scaling (the signature is block-1
dominance, not inert deep blocks); skip paths (moot if the cosine
recovers); weight decay as the control (superseded by the reference
mechanism). Local plan: `docs/normalised-residual-2026-09-17.md`.
