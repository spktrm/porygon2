from typing import NamedTuple

from jaxtyping import ArrayLike


class EnvStep(NamedTuple):
    # Standard Info
    info: ArrayLike = ()
    done: ArrayLike = ()
    win_reward: ArrayLike = ()
    public_team: ArrayLike = ()
    current_context: ArrayLike = ()

    # Private Info
    moveset: ArrayLike = ()
    legal: ArrayLike = ()
    private_team: ArrayLike = ()


class HistoryStep(NamedTuple):
    entities: ArrayLike = ()
    relative_edges: ArrayLike = ()
    absolute_edges: ArrayLike = ()


class ModelOutput(NamedTuple):
    pi: ArrayLike = ()
    v: ArrayLike = ()
    log_pi: ArrayLike = ()
    logit: ArrayLike = ()


class ActorStep(NamedTuple):
    action: ArrayLike = ()
    model_output: ModelOutput = ModelOutput()


class TimeStep(NamedTuple):
    env: EnvStep = EnvStep()
    history: HistoryStep = HistoryStep()


class Transition(NamedTuple):
    timestep: TimeStep = TimeStep()
    actorstep: ActorStep = ActorStep()
