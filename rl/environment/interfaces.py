from typing import NamedTuple

import jax
from jaxtyping import ArrayLike


class PolicyHeadOutput(NamedTuple):
    logits: ArrayLike = ()
    policy: ArrayLike = ()
    log_policy: ArrayLike = ()


class PlayerEnvOutput(NamedTuple):
    # Standard Info
    info: ArrayLike = ()
    done: ArrayLike = ()
    win_reward: ArrayLike = ()
    public_team: ArrayLike = ()
    field: ArrayLike = ()

    # Private Info
    moveset: ArrayLike = ()
    action_type_mask: ArrayLike = ()
    move_mask: ArrayLike = ()
    switch_mask: ArrayLike = ()
    private_team: ArrayLike = ()


class PlayerHistoryOutput(NamedTuple):
    nodes: ArrayLike = ()
    edges: ArrayLike = ()
    field: ArrayLike = ()


class PlayerActorInput(NamedTuple):
    env: PlayerEnvOutput = PlayerEnvOutput()
    history: PlayerHistoryOutput = PlayerHistoryOutput()


class PlayerActorOutput(NamedTuple):
    v: ArrayLike = ()
    action_type_head: PolicyHeadOutput = PolicyHeadOutput()
    move_head: PolicyHeadOutput = PolicyHeadOutput()
    switch_head: PolicyHeadOutput = PolicyHeadOutput()


class PlayerAgentOutput(NamedTuple):
    action_type_head: ArrayLike = ()
    move_head: ArrayLike = ()
    switch_head: ArrayLike = ()
    actor_output: PlayerActorOutput = PlayerActorOutput()


class PlayerTransition(NamedTuple):
    env_output: PlayerEnvOutput = PlayerEnvOutput()
    agent_output: PlayerAgentOutput = PlayerAgentOutput()


class BuilderEnvOutput(NamedTuple):
    tokens: jax.Array = ()
    mask: jax.Array = ()


class BuilderActorOutput(NamedTuple):
    head: PolicyHeadOutput = PolicyHeadOutput()
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
