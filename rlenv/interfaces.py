import chex


@chex.dataclass(frozen=True)
class EnvStep:
    # Standard Info
    valid: chex.Array = ()
    turn: chex.Array = ()
    game_id: chex.Array = ()
    player_id: chex.Array = ()
    rewards: chex.Array = ()
    heuristic_action: chex.Array = ()
    heuristic_dist: chex.Array = ()
    prev_action: chex.Array = ()
    prev_move: chex.Array = ()

    # Private Info
    moveset: chex.Array = ()
    legal: chex.Array = ()
    private_side_entities: chex.Array = ()

    # Public Info
    turn_context: chex.Array = ()
    active_entities: chex.Array = ()
    public_side_entities: chex.Array = ()
    boosts: chex.Array = ()
    side_conditions: chex.Array = ()
    volatile_status: chex.Array = ()
    hyphen_args: chex.Array = ()
    terrain: chex.Array = ()
    pseudoweather: chex.Array = ()
    weather: chex.Array = ()


@chex.dataclass(frozen=True)
class ActorStep:
    action: chex.Array = ()
    policy: chex.Array = ()
    rewards: chex.Array = ()


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
