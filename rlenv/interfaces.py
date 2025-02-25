import chex


@chex.dataclass(frozen=True)
class RewardStep:
    win_rewards: chex.Array = ()

    fainted_rewards: chex.Array = ()
    scaled_fainted_rewards: chex.Array = ()

    hp_rewards: chex.Array = ()
    scaled_hp_rewards: chex.Array = ()


@chex.dataclass(frozen=True)
class EnvStep:
    # Standard Info
    ts: chex.Array = ()
    timestamp: chex.Array = ()
    draw_ratio: chex.Array = ()
    valid: chex.Array = ()
    draw: chex.Array = ()
    turn: chex.Array = ()
    game_id: chex.Array = ()
    player_id: chex.Array = ()
    seed_hash: chex.Array = ()
    request_count: chex.Array = ()
    public_team: chex.Array = ()

    # Private Info
    moveset: chex.Array = ()
    legal: chex.Array = ()
    private_team: chex.Array = ()
    heuristic_action: chex.Array = ()

    # Reward
    rewards: RewardStep = RewardStep()


@chex.dataclass(frozen=True)
class HistoryContainer:
    entities: chex.Array = ()
    relative_edges: chex.Array = ()
    absolute_edges: chex.Array = ()


@chex.dataclass(frozen=True)
class HistoryStep:
    major_history: HistoryContainer = HistoryContainer()


@chex.dataclass(frozen=True)
class ActorStep:
    action: chex.Array = ()
    policy: chex.Array = ()
    log_policy: chex.Array = ()

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
