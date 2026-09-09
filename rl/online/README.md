# Replay sampling and decision accounting

`player_replay_fresh_fraction` schedules preferred first-use slots when fresh
chunks are available. Its default `0.125` schedules one slot every two batches
of four. Startup and shortages of reusable chunks can increase the fresh share.
Fresh chunks are taken in admission order; replay chunks are sampled uniformly
without replacement within a batch. First use means never sampled before,
not strictly on-policy: actor and queue lag still exist.

Every draw, including first use and prefetch, consumes `player_replay_ratio`'s
per-chunk cap. Replacement requires exhaustion or explicit KL retirement. When
fresh data is unavailable, its scheduled slots use eligible replay instead:
the learner can exhaust seen chunks and admit arrivals without early eviction
or a full-buffer deadlock. The fraction is therefore an availability-dependent
preference, not a hard minimum. This matters especially for a fresh fraction
above the reciprocal of the reuse cap. It is measured in chunks; decision
fractions can differ because chunk lengths differ.

Set `player_replay_fresh_fraction=0.0` to restore uniform capped sampling.
The replay KL controller remains a protective cap controller. Retaining more
visits is not proof of improved sample efficiency: compare playing strength
against admitted decisions, while checking policy mismatch and throughput.

## Decision metrics

A decision is a real acted row, including forced single-option actions. Terminal
rows, terminal padding and nonterminal bootstrap-only final rows do not count.
Overlapping chunks therefore count each admitted player decision once. Two player
perspectives are separate decisions, not unique games or simulator turns.

| Metric | Meaning |
| --- | --- |
| `player_batch_decisions` | Acted rows in this learner batch, using the existing learner mask. |
| `player_decisions_admitted_session` | Fresh valid decisions successfully inserted into replay. |
| `player_decisions_sampled_session` | Decision appearances drawn from replay, including prefetch. |
| `player_decisions_processed_session` | Decision appearances in completed learner calls, including skipped updates. |
| `player_decisions_applied_session` | Decision appearances in successful finite learner updates. |
| `player_updates_processed_session` / `player_updates_applied_session` | Completed calls / successful updates. |
| `player_decision_reuse_session` | Applied decision appearances divided by admitted fresh decisions. |
| `player_updates_per_fresh_decision_session` | Successful updates divided by admitted fresh decisions. |
| `player_replay_fresh_chunk_fraction_session` | First-use chunks divided by all sampled chunks. |
| `player_replay_fresh_decision_fraction_session` | First-use decision appearances divided by all sampled decision appearances. |
| `player_replay_evicted_mean_reuses` | Mean consumed uses of evicted chunks. |

These are exact counters for the current replay-store session, starting at
`player_accounting_start_lifetime_step`. They reset on a new process/store or
store clear and are **not historical lifetime totals**. Checkpoint restoration
starts a new accounting session; no historical fresh-decision counts are inferred.
The log worker records completed calls in order. Admission and sampling snapshots
can be ahead of that logged call because actors and prefetch run asynchronously.
Ratios include initial buffer fill and should be interpreted after warmup; derive
windowed rates from counter differences within a single accounting session.

The existing `frame_count`, checkpoint schema and frame-based league scheduling
are unchanged. They remain unsuitable as unique-decision counters.

## Switch learning diagnostics

`player_switch_logit_grad_{pg,entropy,magnet,modality,actor_kl}` report the
coefficient-weighted directional actor-loss derivatives for raising all switch
logits together, holding model features fixed. **Positive suppresses switching
under gradient descent; negative encourages it.** These f32 diagnostics reuse
the actual loss functions and their common policy-row denominator. They do not
attribute shared-feature updates and may differ numerically from bf16 backward.
`player_switch_logit_grad_actor_total` sums the components.

`player_switch_logit_grad_pg_taken_{switch,stay}` restrict the PG contribution
to move-and-switch-choice rows. `player_choice_{switch,stay}_adv_raw` and
`_adv_normalised` report signed advantages on those same subsets; accompanying
`_count` metrics distinguish no examples from a mean of zero. These are sampled
action advantages, not matched counterfactual action values.

`player_switch_mass_choice` measures total switch mass on choice rows, unlike
the existing per-legal-switch-cell probability. `player_switch_bias_gradient`
records the actual pre-clip bias gradient and `player_switch_bias_applied_delta`
records its update after Adam and the non-finite gate. These separate loss
pressure from momentum on this scalar; they do not explain all policy movement.

## Paired switch-advantage audit

`player_adv_audit_{switch,stay}_{all,1_5,6_15,16_40,41_plus}_*` compares
estimators on the same first-use choice rows. Horizons count actor decisions
until the terminal reward, using whole-game length and chunk offset. Forced
switches, terminal/padded/bootstrap-only rows, replay visits and unknown final
outcomes are excluded. Sum `_count` and each `*_sum` over a window before dividing;
never average sparse per-batch means. Rows within a game remain correlated.

For each of `public` and `privileged`, `td` is that head's raw V-trace advantage,
`mc` is the discounted realised behaviour outcome minus its own value baseline,
and `rho_mc` additionally applies the same outer truncated importance weight as
`td`. Thus `td - rho_mc` isolates the bootstrapped-return disagreement with the
observed outcome while cancelling the baseline. This comparison does not fully
correct the behaviour continuation to the target policy. It is not a
counterfactual switch-versus-stay evaluation.

Other sums record value, outcome, squared outcome error, mean outer weight and
paired sign disagreements. Both target computations reuse existing model
predictions and the executable target function. Game outcomes only enter these
diagnostics; they never modify the training targets or loss.
