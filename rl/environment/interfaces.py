from typing import NamedTuple

from jaxtyping import ArrayLike


class PlayerEnvOutput(NamedTuple):
    # Standard Info
    info: ArrayLike = ()
    done: ArrayLike = ()
    win_reward: ArrayLike = ()
    fib_reward: ArrayLike = ()
    public_team: ArrayLike = ()
    field: ArrayLike = ()

    # Private Info
    moveset: ArrayLike = ()
    private_team: ArrayLike = ()
    metagame_token: ArrayLike = ()

    action_type_mask: ArrayLike = ()
    move_mask: ArrayLike = ()
    switch_mask: ArrayLike = ()
    wildcard_mask: ArrayLike = ()


class PlayerHistoryOutput(NamedTuple):
    nodes: ArrayLike = ()
    edges: ArrayLike = ()
    field: ArrayLike = ()


class PlayerHiddenInfo(NamedTuple):
    species_tokens: ArrayLike = ()
    packed_set_tokens: ArrayLike = ()


class PlayerActorInput(NamedTuple):
    env: PlayerEnvOutput = PlayerEnvOutput()
    history: PlayerHistoryOutput = PlayerHistoryOutput()
    hidden: PlayerHiddenInfo = PlayerHiddenInfo()


class HeadOutput(NamedTuple):
    action_index: ArrayLike = ()
    log_prob: ArrayLike = ()
    entropy: ArrayLike = ()


class PlayerActorOutput(NamedTuple):
    v: ArrayLike = ()
    metagame_log_prob: ArrayLike = ()

    action_type_head: HeadOutput = HeadOutput()
    move_head: HeadOutput = HeadOutput()
    switch_head: HeadOutput = HeadOutput()
    wildcard_head: HeadOutput = HeadOutput()


class PlayerAgentOutput(NamedTuple):
    actor_output: PlayerActorOutput = PlayerActorOutput()


class PlayerTransition(NamedTuple):
    env_output: PlayerEnvOutput = PlayerEnvOutput()
    agent_output: PlayerAgentOutput = PlayerAgentOutput()


class BuilderEnvOutput(NamedTuple):
    species_mask: ArrayLike = ()

    ts: ArrayLike = ()
    done: ArrayLike = ()
    metagame_token: ArrayLike = ()

    cum_teammate_reward: ArrayLike = ()
    cum_species_reward: ArrayLike = ()


class BuilderHistoryOutput(NamedTuple):
    species_tokens: ArrayLike = ()
    packed_set_tokens: ArrayLike = ()


class BuilderActorInput(NamedTuple):
    env: BuilderEnvOutput = BuilderEnvOutput()
    history: BuilderHistoryOutput = BuilderHistoryOutput()
    hidden: PlayerHiddenInfo = PlayerHiddenInfo()


class BuilderActorOutput(NamedTuple):
    v: ArrayLike = ()
    metagame_log_prob: ArrayLike = ()

    species_head: HeadOutput = HeadOutput()
    packed_set_head: HeadOutput = HeadOutput()


class BuilderAgentOutput(NamedTuple):
    actor_output: BuilderActorOutput = BuilderActorOutput()


class BuilderTransition(NamedTuple):
    env_output: BuilderEnvOutput = BuilderEnvOutput()
    agent_output: BuilderAgentOutput = BuilderAgentOutput()


class Trajectory(NamedTuple):
    builder_transitions: BuilderTransition = BuilderTransition()
    builder_history: BuilderHistoryOutput = BuilderHistoryOutput()

    player_transitions: PlayerTransition = PlayerTransition()
    player_history: PlayerHistoryOutput = PlayerHistoryOutput()
    player_hidden: PlayerHiddenInfo = PlayerHiddenInfo()


class SamplingConfig(NamedTuple):
    temp: float = 1.0
    min_p: float = 0.0
