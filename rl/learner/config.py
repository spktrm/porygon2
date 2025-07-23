import chex


@chex.dataclass(frozen=True)
class AdamConfig:
    """Adam optimizer related params."""

    b1: float
    b2: float
    eps: float


@chex.dataclass(frozen=True)
class MMDConfig:
    num_steps = 10_000_000
    num_actors: int = 32
    unroll_length: int = 108
    replay_buffer_capacity: int = 16

    # Batch iteration params
    batch_size: int = 4
    target_replay_ratio: int = 2

    # Learning params
    adam: AdamConfig = AdamConfig(b1=0.9, b2=0.999, eps=1e-8)
    learning_rate: float = 2.5e-4
    clip_gradient: float = 2.0
    tau: float = 1e-3

    # Vtrace params
    lambda_: float = 0.95
    gamma: float = 1.0
    clip_rho_threshold: float = 1.0
    clip_pg_rho_threshold: float = 1.0
    clip_ppo: float = 0.2

    # Loss coefficients
    value_loss_coef: float = 0.5
    policy_loss_coef: float = 1.0
    entropy_loss_coef: float = 0.05
    kl_loss_coef: float = 0.05


def get_learner_config():
    return MMDConfig()
