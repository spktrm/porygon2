import os

import chex

from rl.config.common import AdamWConfig, BaseTrainingConfig


@chex.dataclass(frozen=True)
class Porygon2OfflineConfig(BaseTrainingConfig):
    """The offline trainer (rl/offline/train.py): the player model's public
    path frozen from a learner checkpoint, the public critic (and, with
    world_model, the event world model) trained on the replay shards."""

    dataset_dir: str = "replays/shards"
    holdout_modulus: int = 20
    batch_size: int = 8
    min_history_length: int = 64
    # Trailing window over the event stream per trajectory, applied by the
    # store at load (NUM_HISTORY 512 is the whole game on this corpus); a
    # smaller cap trades early events for memory when the GPU is shared.
    max_history_steps: int = 512
    # Processes decoding the export at startup (1 = in one worker). Each
    # imports the model package (~0.7 GB); 8 decode the corpus in ~20 s.
    decode_workers: int = 8
    # The learner checkpoint whose encoder + public critic are the frozen
    # substrate (required); resume_from restarts a world-model run.
    trunk_ckpt: str | None = None
    resume_from: str | None = None
    # Joint: the offline losses also train the encoder (Step 6 of the
    # world-model plan); off = observer, the trunk never moves.
    joint: bool = False
    # Off = the critic alone: public_v_head on every valid event state.
    world_model: bool = True
    num_steps: int = 30000
    learning_rate: float = 3e-4
    lr_final_fraction: float = 0.1
    clip_gradient: float = 10.0
    adam: AdamWConfig = AdamWConfig(b1=0.9, b2=0.999, eps=1e-8, weight_decay=1e-2)
    kind_loss_weight: float = 1.0
    actor_loss_weight: float = 1.0
    move_loss_weight: float = 1.0
    target_loss_weight: float = 1.0
    touched_loss_weight: float = 1.0
    flow_loss_weight: float = 1.0
    mean_loss_weight: float = 1.0
    terminal_loss_weight: float = 1.0
    # The head's parameters receive no other gradient, so under Adam this
    # weight is a no-op up to eps (scale invariance): the head's noise
    # lever is its learning rate. Kept at 1 so the logged loss is the CE.
    public_value_loss_weight: float = 1.0
    # EMA of the per-group RMS difference the flow is scaled by.
    scale_momentum: float = 0.99
    # Samples per state for the eval-only imagined-value reads.
    eval_samples: int = 8
    log_interval_steps: int = 50
    eval_interval_steps: int = 1000
    eval_batches: int = 32
    save_interval_steps: int = 5000
    artifact_root: str = "ckpts/offline"

    def shard_dir(self) -> str:
        return os.path.join(self.dataset_dir, self.format_id)
