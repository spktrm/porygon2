from typing import NamedTuple

import jax
from jaxtyping import ArrayLike


class PlayerEnvOutput(NamedTuple):
    # Standard Info
    info: ArrayLike = ()
    done: ArrayLike = ()
    win_reward: ArrayLike = ()
    public_team: ArrayLike = ()
    field: ArrayLike = ()

    # Private Info
    moveset: ArrayLike = ()
    private_team: ArrayLike = ()

    action_type_mask: ArrayLike = ()
    move_mask: ArrayLike = ()
    switch_mask: ArrayLike = ()
    wildcard_mask: ArrayLike = ()


class PlayerHistoryOutput(NamedTuple):
    nodes: ArrayLike = ()
    edges: ArrayLike = ()
    field: ArrayLike = ()


class PlayerActorInput(NamedTuple):
    env: PlayerEnvOutput = PlayerEnvOutput()
    history: PlayerHistoryOutput = PlayerHistoryOutput()


class PlayerActorOutput(NamedTuple):
    v: ArrayLike = ()
    action_type_logits: ArrayLike = ()
    move_logits: ArrayLike = ()
    switch_logits: ArrayLike = ()
    wildcard_logits: ArrayLike = ()


class PlayerAgentOutput(NamedTuple):
    action_type: ArrayLike = ()
    move_slot: ArrayLike = ()
    switch_slot: ArrayLike = ()
    wildcard_slot: ArrayLike = ()
    actor_output: PlayerActorOutput = PlayerActorOutput()


class PlayerTransition(NamedTuple):
    env_output: PlayerEnvOutput = PlayerEnvOutput()
    agent_output: PlayerAgentOutput = PlayerAgentOutput()


class BuilderEnvOutput(NamedTuple):
    tokens: jax.Array = ()
    mask: jax.Array = ()
    done: jax.Array = ()


class BuilderActorOutput(NamedTuple):
    logits: ArrayLike = ()
    v: ArrayLike = ()


class BuilderAgentOutput(NamedTuple):
    action: ArrayLike = ()
    actor_output: BuilderActorOutput = BuilderActorOutput()


class BuilderTransition(NamedTuple):
    env_output: BuilderEnvOutput = BuilderEnvOutput()
    agent_output: BuilderAgentOutput = BuilderAgentOutput()


class Trajectory(NamedTuple):
    builder_transitions: BuilderTransition = BuilderTransition()
    player_transitions: PlayerTransition = PlayerTransition()
    player_history: PlayerHistoryOutput = PlayerHistoryOutput()


class SamplingConfig(NamedTuple):
    temp: float = 1.0
    min_p: float | None = 0.05
