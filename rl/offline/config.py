import chex

from rl.config.common import AdamWConfig, BaseTrainingConfig


@chex.dataclass(frozen=True)
class Porygon2OfflineConfig(BaseTrainingConfig):
    """Config for offline (supervised) critic training on Showdown replays.

    Composes with the RL learner config through BaseTrainingConfig: shared
    format/logging fields live in the base, everything below is specific to
    the offline pipeline and can be tuned without touching RL training.
    """

    # Dataset params. Shards are written by service/src/scripts/offline.ts as
    # repeated [uint32-LE length][EnvironmentTrajectory proto bytes] records
    # under {dataset_dir}/{format_id}/.
    dataset_dir: str = "replays/shards"
    # Trajectories whose (zero-based) index modulo holdout_modulus is 0 form
    # the eval split — deterministic without a separate file listing.
    holdout_modulus: int = 20
    shuffle_buffer_size: int = 256

    # Forfeit handling (measured on 50k rated gen9randombattle games, July
    # 2026: ~48% played out, ~41% conceded with the winner ahead, ~11%
    # forfeited with the winner NOT ahead on mons).
    #
    # Drop games where the sign-clamp engages (forfeit/timeout with the
    # winner not ahead): the recorded result contradicts the position, so
    # every step's label is noise — and it's perspective-consistent,
    # side-differenced noise, exactly the signal shape the antisymmetric
    # probe is built to learn.
    drop_clamped_forfeits: bool = True
    # Concessions right-censor the margin (the loser quit at |m| down; the
    # played-out margin would have been at least that). > 0 spreads label
    # mass geometrically from the observed bin with this decay, up to the
    # winner's alive-mon count at concession (mons never return, so no
    # continuation could exceed it), countering the compression of
    # conceded margins toward ±1..3. 0 keeps the exact one-hot at the
    # concession margin.
    concession_censor_decay: float = 0.5

    # Auxiliary per-mon survival supervision. Target per (step, revealed
    # live mon): y = survival_discount**(steps until its next faint), 0 if
    # it never faints — a soft horizon carrying the timing information the
    # final-margin label destroys (every step of a trajectory shares one
    # margin label, so "faints next turn" and "faints in 20" are otherwise
    # indistinguishable to the objective). Predicted as a categorical over
    # NUM_SURVIVAL_BINS value bins, so the distribution separates "coin-flip
    # dies now" from "certain death in a half-life" without any dies-in-x
    # window; mons alive when a game ends without being played out are
    # right-censored (loss constrains only the mass above what the replay
    # witnessed). Heads are per-entity and side-agnostic — they enrich the
    # encoder's features but never touch the antisymmetric margin readout,
    # and they are discarded at RL consumption time (the learner's Φ path
    # calls the outcome readout only).
    #
    # Half-life = ln(2)/ln(1/discount) request steps (~6.6 at 0.9) — the
    # "imminent doom" scale the margin probe can't otherwise learn to see
    # before the faint bit flips.
    survival_discount: float = 0.9
    # Weight on the survival loss next to the margin CE (both start near
    # ln(num_bins), so 1.0 is balanced). 0 disables the aux entirely.
    survival_loss_weight: float = 1.0

    # Anticipation aux heads (see rl/offline/dataset.py::_action_targets):
    # all self-supervised from the replay's own event stream — the labels
    # are "what happened next", no reveal ontology or human annotation.
    #
    # Next-action head: per-slot categorical over the move vocab (+ "never
    # acts again") for the mon's next executed move. Scoring well before a
    # move is revealed requires a set posterior over the unseen slots.
    # CE starts near ln(vocab) ≈ 7 vs the margin CE's ln(13) ≈ 2.6, so the
    # default weight keeps its gradient from dominating at init.
    action_loss_weight: float = 0.25
    # Unseen-move hazard: discounted time until the mon next uses a move
    # unrevealed as of the current step — the latent-threat clock. Discount
    # 0.95 puts the half-life at ~13.5 request steps, the anticipation
    # scale, vs the survival head's ~6.6 reactive scale. Same bins,
    # censoring, and loss machinery as the survival head.
    unseen_discount: float = 0.95
    unseen_loss_weight: float = 1.0
    # Eventually-revealed set head: positive-unlabelled multi-label over
    # the move vocab (positives = moves revealed later than the current
    # step; negatives only for mons whose full move count was eventually
    # observed). Forces explicit set posteriors into the slot tokens.
    set_loss_weight: float = 0.5

    # Announced-state supervision (Φ_ann): each turn's announced state —
    # the pre-turn recurrent state advanced once with outcome-masked
    # messages (both players' revealed choices, chance unresolved) — is an
    # extra evaluation point trained with the SAME trajectory margin label
    # through the SAME antisymmetric readout (deep supervision; mirror
    # antisymmetry of Φ_ann follows automatically, and no new parameters
    # exist). Trained Φ_ann = E[final outcome | history, announced
    # actions], the quantity both the replay skill/luck decomposition and
    # dice-excised PBRS need. Both losses start near ln(13), so 1.0 is
    # balanced. The announced pathway runs iff this or
    # announced_distill_weight is > 0; with both 0, with_aux is applied
    # instead, reproducing pre-announced training exactly.
    announced_loss_weight: float = 1.0
    # Announced-state distillation: KL(stopgrad P_Φ(t) ‖ P_Φ_ann(t)) —
    # train the announced distribution toward the REALISED same-step
    # distribution. Rationale (measured 2026-07-30: announced-movement
    # ratio ~0.15): the shared terminal label gives an announcement almost
    # no marginal CE to earn — the outcome-resolved state predicts the
    # label at least as well — so SGD never builds the announcement →
    # consequence circuit. This target demands it per turn: forward KL's
    # minimiser is E[P_Φ(t) | announcement], the conditional mixture over
    # dice, which is exactly what the decision/dice decomposition and
    # dice-excised PBRS want Φ_ann to be. The target is stop-gradient, so
    # the realised head is never pulled toward Φ_ann; one-step and
    # on-distribution, so the policy-counterfactual cap on anticipation
    # (see memory: offline-critic-no-anticipation mechanism 1) does not
    # apply. The KL is logged as announced_distill_kl even at weight 0 —
    # measure the gap on a run before paying for closing it. Default off:
    # an extension beyond the settled announced-states design, A/B it.
    announced_distill_weight: float = 0.0

    # Elo conditioning: with probability rating_dropout, a game's rating
    # features are zeroed (per GAME, not per perspective — mirrored pairs
    # must stay exact mirrors for pair-aware batching). Trains the
    # "unknown" bucket that live self-play, pre-rating shards, and
    # counterfactual Elo sweeps all rely on, and stops the model leaning on
    # rating where the board already says everything.
    rating_dropout: float = 0.25

    # Batch iteration params
    batch_size: int = 8
    min_history_length: int = 64
    # Trajectories are padded to geometric time buckets in [lo, hi] to bound
    # JIT recompilations; longer games are truncated at the cap.
    min_trajectory_bucket: int = 32
    max_trajectory_length: int = 512

    num_steps: int = 50_000

    # Learning params. Supervised training wants momentum, unlike the RL
    # learner's b1=0. Regularization is sized to constrain without eroding:
    # 1e-2 decay + 0.05 smoothing produced a peak-then-decay-to-plateau
    # accuracy curve (smoothed CE saturates once fit; decay keeps shrinking
    # the solution until CE re-engages — a stable equilibrium below the
    # peak). The structural defenses (antisymmetric probe, pair batching,
    # deep supervision) carry the anti-memorization burden instead.
    adam: AdamWConfig = AdamWConfig(b1=0.9, b2=0.999, eps=1e-8, weight_decay=1e-3)
    learning_rate: float = 3e-4
    # Cosine-decay the LR to this fraction over num_steps, freezing the
    # found solution instead of letting late-run noise/decay erode it.
    lr_final_fraction: float = 0.1
    clip_gradient: float = 10.0
    # Caps target confidence, directly penalising the memorization mode
    # (huge |logit| on train games). 0 disables.
    label_smoothing: float = 0.01

    # Ensemble training for uncertainty-gated shaping: run K times with
    # ensemble_index = 0..K-1; each member trains on a disjoint per-game
    # split (shared holdout). -1 trains one model on everything.
    ensemble_index: int = -1
    num_ensemble_splits: int = 4
    # Train all num_ensemble_splits members simultaneously in one process
    # (--ensemble): stacked params/optimizer with a vmapped member step —
    # pure parallelism, gradients are never mixed across members — and one
    # shard pass routing each game to its member. Statistically equivalent
    # to K separate --ensemble-index runs, substantially faster on one
    # GPU, and shared-holdout evals log live gate metrics.
    train_ensemble: bool = False
    # Gate scale used only for eval-time gate logging in --ensemble runs;
    # mirror of the learner's potential_uncertainty_scale.
    eval_gate_scale: float = 5.0

    # Eval / checkpoint cadence
    log_interval_steps: int = 100
    eval_interval_steps: int = 2_000
    eval_batches: int = 32
    save_interval_steps: int = 5_000

    # Artifacts land in {artifact_root}/{format_id}/ckpt_{step:08}/ using the
    # same component layout as rl/learner/checkpoint.py, so the RL learner
    # can merge them via load_from_params.
    artifact_root: str = "ckpts/offline"

    # Resume offline training from a previous offline artifact, if set.
    resume_from: str | None = None


def get_offline_config() -> Porygon2OfflineConfig:
    return Porygon2OfflineConfig()