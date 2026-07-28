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