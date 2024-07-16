import chex


@chex.dataclass(frozen=True)
class EnvStep:
    # Standard Info
    valid: chex.Array = ()
    turn: chex.Array = ()
    game_id: chex.Array = ()
    player_id: chex.Array = ()
    rewards: chex.Array = ()

    # Private Info
    moveset: chex.Array = ()
    legal: chex.Array = ()

    # Mixture
    side_entities: chex.Array = ()

    # Public Info
    turn_context: chex.Array = ()
    active_entities: chex.Array = ()
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
