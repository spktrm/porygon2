from typing import NamedTuple

from jaxtyping import ArrayLike


class PlayerEnvOutput(NamedTuple):
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
    wildcard_mask: ArrayLike = ()


class PlayerHistoryOutput(NamedTuple):
    public: ArrayLike = ()
    revealed: ArrayLike = ()
    edges: ArrayLike = ()
    field: ArrayLike = ()


class PlayerActorInput(NamedTuple):
    env: PlayerEnvOutput = PlayerEnvOutput()
    history: PlayerHistoryOutput = PlayerHistoryOutput()


class ValueHeadOutput(NamedTuple):
    logits: ArrayLike = ()
    log_probs: ArrayLike = ()
    entropy: ArrayLike = ()
    expectation: ArrayLike = ()


class PolicyHeadOutput(NamedTuple):
    action_index: ArrayLike = ()
    log_prob: ArrayLike = ()
    entropy: ArrayLike = ()


class PlayerActorOutput(NamedTuple):
    value_head: ValueHeadOutput = ValueHeadOutput()
    action_head: PolicyHeadOutput = PolicyHeadOutput()
    wildcard_head: PolicyHeadOutput = PolicyHeadOutput()


class PlayerAgentOutput(NamedTuple):
    actor_output: PlayerActorOutput = PlayerActorOutput()


class PlayerTransition(NamedTuple):
    env_output: PlayerEnvOutput = PlayerEnvOutput()
    agent_output: PlayerAgentOutput = PlayerAgentOutput()


class BuilderEnvOutput(NamedTuple):
    species_mask: ArrayLike = ()

    ts: ArrayLike = ()
    done: ArrayLike = ()

    cum_teammate_reward: ArrayLike = ()
    cum_species_reward: ArrayLike = ()


class BuilderHistoryOutput(NamedTuple):
    species_tokens: ArrayLike = ()
    packed_set_tokens: ArrayLike = ()


class BuilderPerformance(NamedTuple):
    count: ArrayLike = ()
    n_wins: ArrayLike = ()


class BuilderActorInput(NamedTuple):
    env: BuilderEnvOutput = BuilderEnvOutput()
    history: BuilderHistoryOutput = BuilderHistoryOutput()


class BuilderActorOutput(NamedTuple):
    value_head: ValueHeadOutput = ValueHeadOutput()
    species_head: PolicyHeadOutput = PolicyHeadOutput()
    packed_set_head: PolicyHeadOutput = PolicyHeadOutput()


class BuilderAgentOutput(NamedTuple):
    actor_output: BuilderActorOutput = BuilderActorOutput()


class BuilderTransition(NamedTuple):
    env_output: BuilderEnvOutput = BuilderEnvOutput()
    agent_output: BuilderAgentOutput = BuilderAgentOutput()


class BuilderTrajectory(NamedTuple):
    transitions: BuilderTransition = BuilderTransition()
    history: BuilderHistoryOutput = BuilderHistoryOutput()
    performance: BuilderPerformance = BuilderPerformance()


class PlayerTrajectory(NamedTuple):
    transitions: PlayerTransition = PlayerTransition()
    history: PlayerHistoryOutput = PlayerHistoryOutput()


class SamplingConfig(NamedTuple):
    temp: float = 1.0
    min_p: float = 0.0
