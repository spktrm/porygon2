import chex


@chex.dataclass(frozen=True)
class RewardStep:
    win_rewards: chex.Array = ()
    fainted_rewards: chex.Array = ()
    switch_rewards: chex.Array = ()
    longevity_rewards: chex.Array = ()
    hp_rewards: chex.Array = ()


@chex.dataclass(frozen=True)
class EnvStep:
    # Standard Info
    ts: chex.Array = ()
    draw_ratio: chex.Array = ()
    valid: chex.Array = ()
    draw: chex.Array = ()
    turn: chex.Array = ()
    game_id: chex.Array = ()
    player_id: chex.Array = ()
    seed_hash: chex.Array = ()
    request_count: chex.Array = ()

    # Private Info
    moveset: chex.Array = ()
    legal: chex.Array = ()
    team: chex.Array = ()
    heuristic_action: chex.Array = ()

    # Reward
    rewards: RewardStep = RewardStep()


@chex.dataclass(frozen=True)
class HistoryStep:
    history_edges: chex.Array = ()
    history_entities: chex.Array = ()
    history_side_conditions: chex.Array = ()
    history_field: chex.Array = ()


@chex.dataclass(frozen=True)
class ActorStep:
    action: chex.Array = ()
    policy: chex.Array = ()

    # rewards
    rewards: RewardStep = RewardStep()


@chex.dataclass(frozen=True)
class TimeStep:
    env: EnvStep = EnvStep()
    actor: ActorStep = ActorStep()
    history: HistoryStep = HistoryStep()


@chex.dataclass(frozen=True)
class ModelOutput:
    pi: chex.Array = ()
    v: chex.Array = ()
    log_pi: chex.Array = ()
    logit: chex.Array = ()
    offline_logit: chex.Array = ()
    offline_pi: chex.Array = ()
    offline_log_pi: chex.Array = ()
