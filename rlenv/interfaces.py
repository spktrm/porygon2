import chex


@chex.dataclass(frozen=True)
class EnvStep:
    # Standard Info
    valid: chex.Array = ()
    turn: chex.Array = ()
    game_id: chex.Array = ()
    player_id: chex.Array = ()
    win_rewards: chex.Array = ()
    fainted_rewards: chex.Array = ()
    switch_rewards: chex.Array = ()
    hp_rewards: chex.Array = ()
    heuristic_action: chex.Array = ()
    heuristic_dist: chex.Array = ()
    prev_action: chex.Array = ()
    prev_move: chex.Array = ()

    # Private Info
    moveset: chex.Array = ()
    legal: chex.Array = ()
    team: chex.Array = ()

    # Public Info
    history_edges: chex.Array = ()
    history_nodes: chex.Array = ()


@chex.dataclass(frozen=True)
class ActorStep:
    action: chex.Array = ()
    policy: chex.Array = ()
    win_rewards: chex.Array = ()
    hp_rewards: chex.Array = ()
    switch_rewards: chex.Array = ()
    fainted_rewards: chex.Array = ()


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
    repr_loss: chex.Array = ()
