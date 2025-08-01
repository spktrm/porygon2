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
    action_mask: ArrayLike = ()
    private_team: ArrayLike = ()


class HistoryStep(NamedTuple):
    nodes: ArrayLike = ()
    edges: ArrayLike = ()
    field: ArrayLike = ()


class ModelOutput(NamedTuple):
    v: ArrayLike = ()
    log_pi: ArrayLike = ()
    entropy: ArrayLike = ()
    action_head_action: ArrayLike = ()
    move_head_action: ArrayLike = ()
    switch_head_action: ArrayLike = ()


class ActorStep(NamedTuple):
    model_output: ModelOutput = ModelOutput()


class ActorReset(NamedTuple):
    tokens: ArrayLike = ()
    log_pi: ArrayLike = ()
    entropy: ArrayLike = ()
    key: ArrayLike = ()
    v: ArrayLike = ()


class TimeStep(NamedTuple):
    rng_key: ArrayLike = ()
    env: EnvStep = EnvStep()
    actor_step: ActorStep = ActorStep()
    history: HistoryStep = HistoryStep()


class Transition(NamedTuple):
    timestep: TimeStep = TimeStep()
    actor_reset: ActorReset = ActorReset()
