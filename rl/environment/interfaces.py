from typing import NamedTuple

from jaxtyping import ArrayLike


class EnvStep(NamedTuple):
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


class HistoryStep(NamedTuple):
    nodes: ArrayLike = ()
    edges: ArrayLike = ()
    field: ArrayLike = ()


class PolicyHeadOutput(NamedTuple):
    logits: ArrayLike = ()
    policy: ArrayLike = ()
    log_policy: ArrayLike = ()


class ModelOutput(NamedTuple):
    v: ArrayLike = ()
    action_type_head: PolicyHeadOutput = PolicyHeadOutput()
    move_head: PolicyHeadOutput = PolicyHeadOutput()
    switch_head: PolicyHeadOutput = PolicyHeadOutput()


class ActorStep(NamedTuple):
    action_type_head: ArrayLike = ()
    move_head: ArrayLike = ()
    switch_head: ArrayLike = ()
    model_output: ModelOutput = ModelOutput()


class ActorReset(NamedTuple):
    tokens: ArrayLike = ()
    log_pi: ArrayLike = ()
    entropy: ArrayLike = ()
    key: ArrayLike = ()
    v: ArrayLike = ()


class TimeStep(NamedTuple):
    env: EnvStep = EnvStep()
    history: HistoryStep = HistoryStep()


class Transition(NamedTuple):
    timestep: TimeStep = TimeStep()
    actor_reset: ActorReset = ActorReset()
    actor_step: ActorStep = ActorStep()
