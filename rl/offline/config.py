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

    # Batch iteration params
    batch_size: int = 8
    min_history_length: int = 64
    # Trajectories are padded to geometric time buckets in [lo, hi] to bound
    # JIT recompilations; longer games are truncated at the cap.
    min_trajectory_bucket: int = 32
    max_trajectory_length: int = 512

    num_steps: int = 50_000

    # Learning params. Supervised training wants momentum, unlike the RL
    # learner's b1=0. Weight decay is deliberately strong: with a
    # high-capacity encoder and identity-rich embeddings, the failure mode
    # is confident per-game memorization (held-out loss exploding), not
    # underfitting.
    adam: AdamWConfig = AdamWConfig(b1=0.9, b2=0.999, eps=1e-8, weight_decay=1e-2)
    learning_rate: float = 3e-4
    clip_gradient: float = 10.0
    # Caps target confidence, directly penalising the memorization mode
    # (huge |logit| on train games). 0 disables.
    label_smoothing: float = 0.05

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
