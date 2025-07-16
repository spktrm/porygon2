from typing import NamedTuple

from jaxtyping import ArrayLike

ArrayLike


class RewardStep(NamedTuple):
    win_rewards: ArrayLike = ()

    fainted_rewards: ArrayLike = ()
    scaled_fainted_rewards: ArrayLike = ()

    hp_rewards: ArrayLike = ()
    scaled_hp_rewards: ArrayLike = ()


class Targets(NamedTuple):
    advantages: ArrayLike = ()
    errors: ArrayLike = ()


class EnvStep(NamedTuple):
    # Standard Info
    ts: ArrayLike = ()
    timestamp: ArrayLike = ()
    draw_ratio: ArrayLike = ()
    valid: ArrayLike = ()
    draw: ArrayLike = ()
    turn: ArrayLike = ()
    game_id: ArrayLike = ()
    player_id: ArrayLike = ()
    seed_hash: ArrayLike = ()
    request_count: ArrayLike = ()
    public_team: ArrayLike = ()
    current_context: ArrayLike = ()
    # all_my_moves: ArrayLike = ()
    # all_opp_moves: ArrayLike = ()

    # Private Info
    moveset: ArrayLike = ()
    legal: ArrayLike = ()
    private_team: ArrayLike = ()
    heuristic_action: ArrayLike = ()

    # Reward
    rewards: RewardStep = RewardStep()


class HistoryContainer(NamedTuple):
    entities: ArrayLike = ()
    relative_edges: ArrayLike = ()
    absolute_edges: ArrayLike = ()


class HistoryStep(NamedTuple):
    major_history: HistoryContainer = HistoryContainer()


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
