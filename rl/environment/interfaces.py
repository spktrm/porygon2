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
    my_actions: ArrayLike = ()
    legal: ArrayLike = ()
    private_team: ArrayLike = ()


class HistoryStep(NamedTuple):
    nodes: ArrayLike = ()
    edges: ArrayLike = ()
    field: ArrayLike = ()


class ModelOutput(NamedTuple):
    pi: ArrayLike = ()
    v: ArrayLike = ()
    log_pi: ArrayLike = ()
    logit: ArrayLike = ()


class ActorStep(NamedTuple):
    action: ArrayLike = ()
    model_output: ModelOutput = ModelOutput()


class ActorReset(NamedTuple):
    tokens: ArrayLike = ()
    log_pi: ArrayLike = ()
    key: ArrayLike = ()
    v: ArrayLike = ()


class TimeStep(NamedTuple):
    env: EnvStep = EnvStep()
    history: HistoryStep = HistoryStep()


class Transition(NamedTuple):
    timestep: TimeStep = TimeStep()
    actor_step: ActorStep = ActorStep()
    actor_reset: ActorReset = ActorReset()
