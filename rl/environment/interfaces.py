from dataclasses import field

from chex import dataclass
from jaxtyping import ArrayLike


@dataclass
class PlayerEnvOutput:
    # Standard Info
    info: ArrayLike = ()
    done: ArrayLike = ()
    win_reward: ArrayLike = ()
    fib_reward: ArrayLike = ()
    public_team: ArrayLike = ()
    revealed_team: ArrayLike = ()
    field: ArrayLike = ()

    # Private Info
    moveset: ArrayLike = ()
    private_team: ArrayLike = ()

    action_mask: ArrayLike = ()


@dataclass
class PlayerPackedHistoryOutput:
    public: ArrayLike = ()
    revealed: ArrayLike = ()
    edges: ArrayLike = ()


@dataclass
class PlayerHistoryOutput:
    field: ArrayLike = ()


@dataclass
class PlayerActorInput:
    env: PlayerEnvOutput = field(default_factory=PlayerEnvOutput)
    packed_history: PlayerPackedHistoryOutput = field(
        default_factory=PlayerPackedHistoryOutput
    )
    history: PlayerHistoryOutput = field(default_factory=PlayerHistoryOutput)


@dataclass
class RegressionValueHeadOutput:
    logits: ArrayLike = ()


@dataclass
class CategoricalValueHeadOutput:
    logits: ArrayLike = ()
    log_probs: ArrayLike = ()
    entropy: ArrayLike = ()
    expectation: ArrayLike = ()


@dataclass
class PolicyHeadOutput:
    action_index: ArrayLike = ()
    log_prob: ArrayLike = ()
    entropy: ArrayLike = ()
    log_policy: ArrayLike = ()


@dataclass
class PlayerPolicyHeadOutput(PolicyHeadOutput):
    src_index: ArrayLike = ()
    tgt_index: ArrayLike = ()


@dataclass
class PlayerActorOutput:
    value_head: RegressionValueHeadOutput = field(
        default_factory=RegressionValueHeadOutput
    )
    action_head: PlayerPolicyHeadOutput = field(default_factory=PlayerPolicyHeadOutput)


@dataclass
class PlayerAgentOutput:
    actor_output: PlayerActorOutput = field(default_factory=PlayerActorOutput)


@dataclass
class PlayerTransition:
    env_output: PlayerEnvOutput = field(default_factory=PlayerEnvOutput)
    agent_output: PlayerAgentOutput = field(default_factory=PlayerAgentOutput)


@dataclass
class BuilderEnvOutput:
    species_mask: ArrayLike = ()

    ts: ArrayLike = ()
    done: ArrayLike = ()

    cum_teammate_reward: ArrayLike = ()
    cum_species_reward: ArrayLike = ()

    target_species_probs: ArrayLike = ()


@dataclass
class BuilderHistoryOutput:
    species_tokens: ArrayLike = ()
    packed_set_tokens: ArrayLike = ()


@dataclass
class BuilderActorInput:
    env: BuilderEnvOutput = field(default_factory=BuilderEnvOutput)
    history: BuilderHistoryOutput = field(default_factory=BuilderHistoryOutput)


@dataclass
class BuilderActorOutput:
    conditional_entropy_head: RegressionValueHeadOutput = field(
        default_factory=RegressionValueHeadOutput
    )
    value_head: RegressionValueHeadOutput = field(
        default_factory=RegressionValueHeadOutput
    )
    species_head: PolicyHeadOutput = field(default_factory=PolicyHeadOutput)
    packed_set_head: PolicyHeadOutput = field(default_factory=PolicyHeadOutput)


@dataclass
class BuilderAgentOutput:
    actor_output: BuilderActorOutput = field(default_factory=BuilderActorOutput)


@dataclass
class BuilderTransition:
    env_output: BuilderEnvOutput = field(default_factory=BuilderEnvOutput)
    agent_output: BuilderAgentOutput = field(default_factory=BuilderAgentOutput)


@dataclass
class Trajectory:
    builder_transitions: BuilderTransition = field(default_factory=BuilderTransition)
    builder_history: BuilderHistoryOutput = field(default_factory=BuilderHistoryOutput)

    player_transitions: PlayerTransition = field(default_factory=PlayerTransition)
    player_packed_history: PlayerPackedHistoryOutput = field(
        default_factory=PlayerPackedHistoryOutput
    )
    player_history: PlayerHistoryOutput = field(default_factory=PlayerHistoryOutput)


@dataclass
class SamplingConfig:
    temp: float = 1.0
    min_p: float = 0.0
