import chex


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

    # Private Info
    moveset: chex.Array = ()
    legal: chex.Array = ()
    team: chex.Array = ()

    # Reward
    win_rewards: chex.Array = ()
    fainted_rewards: chex.Array = ()
    switch_rewards: chex.Array = ()
    longevity_rewards: chex.Array = ()
    hp_rewards: chex.Array = ()

    # Public Info
    history_edges: chex.Array = ()
    history_entities: chex.Array = ()
    history_side_conditions: chex.Array = ()
    history_field: chex.Array = ()


@chex.dataclass(frozen=True)
class ActorStep:
    action: chex.Array = ()
    policy: chex.Array = ()

    # rewards
    win_rewards: chex.Array = ()
    fainted_rewards: chex.Array = ()
    switch_rewards: chex.Array = ()
    longevity_rewards: chex.Array = ()
    hp_rewards: chex.Array = ()


@chex.dataclass(frozen=True)
class TimeStep:
    env: EnvStep = EnvStep()
    actor: ActorStep = ActorStep()


@chex.dataclass(frozen=True)
class ModelOutput:
    pi: chex.Array = ()
    v: chex.Array = ()
    log_pi: chex.Array = ()
    logit: chex.Array = ()
