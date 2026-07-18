import functools
import os
from collections.abc import Mapping
from pprint import pprint
from typing import Any, Callable, Literal

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb.wandb_run
from flax import core, struct
from flax.training import train_state

from rl.environment.interfaces import (
    BuilderActorInput,
    BuilderActorOutput,
    PlayerActorInput,
    PlayerActorOutput,
)
from rl.environment.utils import get_ex_builder_step, get_ex_player_step
from rl.learner import checkpoint
from rl.learner.league import MAIN_KEY, League
from rl.model.heads import HeadParams
from rl.model.utils import Params, ParamsContainer


@chex.dataclass(frozen=True)
class AdamWConfig:
    """Adam optimizer related params."""

    b1: float
    b2: float
    eps: float
    weight_decay: float


GenT = Literal[1, 2, 3, 4, 5, 6, 7, 8, 9]
SmogonFormatT = Literal["ou", "uu", "ru", "nu", "pu", "ubers", "randombattle"]
PolicyObjectiveT = Literal["spo", "ppo"]


@chex.dataclass(frozen=True)
class Porygon2LearnerConfig:
    num_steps = 5_000_000
    num_player_actors: int = 12
    num_builder_actors: int = 4
    num_eval_actors: int = 3
    unroll_length: int = 128

    # Batch iteration params
    batch_size: int = 4

    # Replay buffer params
    player_replay_buffer_capacity: int = 1024 * 2
    player_replay_ratio: int = 8
    builder_replay_buffer_capacity: int = 512
    builder_replay_ratio: int = 10
    # Fraction of replay buffer capacity that must be filled before training
    # starts. Valid range: [0.0, 1.0]. Defaults to 0.5 (50%).
    replay_buffer_min_fill_fraction: float = (
        player_replay_ratio * batch_size / player_replay_buffer_capacity
    )

    # Self-play evaluation params
    save_interval_steps: int = 20_000
    cloud_save_interval_steps: int = 100_000
    league_winrate_log_steps: int = 1_000
    main_player_update_steps: int = 10
    add_player_min_frames: int = int(2e5)
    add_player_max_frames: int = int(3e6)
    minimum_historical_player_steps: int = int(1e6)
    league_size: int = 16
    manage_league_interval: int = 10
    # Disk-backed league: max materialised opponents held in RAM at once, and
    # the UCB exploration coefficient governing which stay hot.
    league_cache_size: int = 16
    league_ucb_c: float = 1.0

    # Plasticity (shrink-and-perturb) params. Triggered when the main player
    # keeps failing to dominate its own league history: after
    # `plasticity_overdue_trigger` consecutive overdue-only league additions,
    # player params are interpolated toward a fresh init draw.
    plasticity_enabled: bool = True
    plasticity_overdue_trigger: int = 1
    # Fraction of the old weights kept (lambda). Higher = milder perturbation.
    plasticity_default_shrink: float = 0.5
    # Per top-level module overrides; the encoder holds expensive
    # representations, so it is perturbed more gently than the heads.
    plasticity_module_shrink: tuple[tuple[str, float], ...] = (("encoder", 0.5),)
    # Recovery gate: no further perturbations until the main player beats the
    # pre-perturbation snapshot at this winrate and the cooldown has elapsed.
    plasticity_recovery_winrate: float = 0.6
    plasticity_cooldown_frames: int = int(1e6)

    # Player magnet regularization (MMD-style). The policy is pulled toward a
    # fixed uniform magnet over legal actions: KL(pi || uniform-over-legal) is
    # per-state entropy regularization that scales with the legal set. The
    # magnet is deliberately stationary — a fixed anchor is what gives the
    # regularized self-play dynamics a stable fixed point (QRE), whereas an
    # EMA magnet chases the policy and degenerates into a short-horizon trust
    # region. The EMA target is reserved for the v-trace/IMPACT reference and
    # plays no regularization role. The coef sets the softness level.
    player_magnet_kl_coef: float = 0.05

    # Learning params
    adam: AdamWConfig = AdamWConfig(b1=0.0, b2=0.999, eps=1e-08, weight_decay=0)
    player_learning_rate: float = 3e-5
    builder_learning_rate: float = 3e-5
    player_clip_gradient: float = 10.0
    builder_clip_gradient: float = 10.0
    gradient_accumulation_steps: int = 1
    # Fast EMA target (IMPACT-style): supplies the clipped-target ratio in
    # the surrogate, the v-trace reference policy, and the value bootstraps,
    # so it must track the learner closely for stability under replay reuse.
    # (R-NaD likewise keeps a 1e-3 target purely for v-trace stability,
    # separate from its slow anchors.)
    player_ema_update_rate: float = 1e-3
    builder_ema_update_rate: float = 1e-3

    # Advantage estimation params
    player_gamma: float = 1.0
    player_alpha: float = 1.0
    player_lambda: float = 0.99

    builder_gamma: float = 1.0
    builder_alpha: float = 1.0
    builder_lambda: float = 0.99

    # Player policy objective: ratio-based surrogates with a trust region.
    player_policy_objective: PolicyObjectiveT = "spo"

    player_ppo_clip_threshold: float = 0.3
    builder_ppo_clip_threshold: float = 0.3

    # Advantage EMA normalization. When disabled, raw advantages are used;
    # the EMA statistics keep updating either way so re-enabling is smooth.
    player_advantage_ema_enabled: bool = True

    # Regularised reward params
    player_heuristic_advantage_coef_fn: Callable[[int], float] = lambda step: 0.0

    # Loss coefficients
    ## Player
    player_policy_loss_coef: float = 1.0
    player_kl_loss_coef: float = 0.05
    player_value_head_loss_coef: float = 1.0
    player_logit_norm_loss_coef: float = 0.0

    ## Decision-time search heads (rl/model/search.py). Learner-side aux
    ## losses only — actors never run these. Deliberately small defaults:
    ## the heads must reach useful prediction quality without reshaping the
    ## trunk away from the main objective; scale after validating held-out
    ## opponent-policy accuracy / q calibration.
    player_search_intent_prior_coef: float = 0.1
    player_search_my_policy_coef: float = 0.1
    player_search_q_coef: float = 0.1
    player_search_q_mean_coef: float = 0.01
    player_search_dynamics_coef: float = 0.1
    # SIGReg (LeJEPA): pins the chance posterior's batch marginal to
    # N(0, I) — the anti-collapse guarantee for the chance channel.
    player_search_sigreg_coef: float = 0.05
    player_search_sigreg_directions: int = 64

    ## Builder
    builder_value_loss_coef: float = 0.5
    builder_policy_loss_coef: float = 1.0
    builder_kl_loss_coef: float = 0.1
    builder_entropy_loss_coef: float = 0.01
    builder_conditional_entropy_loss_coef: float = 1.0
    builder_entropy_coef: float = 0.01
    builder_entropy_prediction_normalising_constant: float = 100
    builder_entropy_advantage_scale: float = 1e-3

    # Human
    builder_human_loss_coef: float = 1e-2

    # Smogon Generation
    generation: GenT = 9
    smogon_format: SmogonFormatT = "randombattle"

    # Logging params
    log_artifacts_online: bool = False


def get_learner_config():
    return Porygon2LearnerConfig()


class Porygon2PlayerTrainState(train_state.TrainState):
    apply_fn: Callable[
        [Params, PlayerActorInput, PlayerActorOutput, HeadParams], PlayerActorOutput
    ] = struct.field(pytree_node=False)
    init_fn: Callable[[jax.Array], Params] = struct.field(pytree_node=False)

    target_params: core.FrozenDict[str, Any] = struct.field(pytree_node=True)

    # Force these to be dynamic JAX arrays (PyTree nodes) instead of static Python scalars
    step_count: jax.Array = struct.field(
        default_factory=lambda: jnp.array(0, dtype=jnp.int32), pytree_node=True
    )
    frame_count: jax.Array = struct.field(
        default_factory=lambda: jnp.array(0, dtype=jnp.int32), pytree_node=True
    )

    ema_adv_mean: jax.Array = struct.field(
        default_factory=lambda: jnp.array(0.0, dtype=jnp.float32), pytree_node=True
    )
    ema_adv_std: jax.Array = struct.field(
        default_factory=lambda: jnp.array(1.0, dtype=jnp.float32), pytree_node=True
    )


class Porygon2BuilderTrainState(train_state.TrainState):
    apply_fn: Callable[
        [Params, BuilderActorInput, BuilderActorOutput, HeadParams], BuilderActorOutput
    ] = struct.field(pytree_node=False)
    init_fn: Callable[[jax.Array], Params] = struct.field(pytree_node=False)

    target_params: core.FrozenDict[str, Any] = struct.field(pytree_node=True)

    step_count: int = 0
    frame_count: int = 0


def create_train_state(
    player_network: nn.Module,
    builder_network: nn.Module,
    rng: jax.Array,
    config: Porygon2LearnerConfig,
):
    """Creates an initial `TrainState`."""
    ex_player_actor_inp, ex_player_actor_out = jax.tree.map(
        lambda x: jnp.asarray(x[:, 0]), get_ex_player_step()
    )
    ex_builder_actor_inp, ex_builder_actor_out = jax.tree.map(
        lambda x: jnp.asarray(x[:, 0]), get_ex_builder_step()
    )

    player_params_init_fn = functools.partial(
        player_network.init,
        head_params=HeadParams(),
        actor_input=ex_player_actor_inp,
        actor_output=ex_player_actor_out,
    )
    player_optimizer = optax.chain(
        optax.clip_by_global_norm(config.player_clip_gradient),
        optax.adamw(
            learning_rate=config.player_learning_rate,
            b1=config.adam.b1,
            b2=config.adam.b2,
            eps=config.adam.eps,
            weight_decay=config.adam.weight_decay,
        ),
    )
    if config.gradient_accumulation_steps > 1:
        player_optimizer = optax.MultiSteps(
            player_optimizer, config.gradient_accumulation_steps
        )
    initial_player_params = player_params_init_fn(rng)

    player_train_state = Porygon2PlayerTrainState.create(
        apply_fn=jax.vmap(player_network.apply, in_axes=(None, 1, 1, None), out_axes=1),
        init_fn=player_params_init_fn,
        params=initial_player_params,
        target_params=initial_player_params,
        tx=player_optimizer,
    )

    builder_params_init_fn = functools.partial(
        builder_network.init,
        actor_input=ex_builder_actor_inp,
        actor_output=ex_builder_actor_out,
        head_params=HeadParams(),
    )
    builder_optimizer = optax.chain(
        optax.clip_by_global_norm(config.builder_clip_gradient),
        optax.adamw(
            learning_rate=config.builder_learning_rate,
            b1=config.adam.b1,
            b2=config.adam.b2,
            eps=config.adam.eps,
            weight_decay=config.adam.weight_decay,
        ),
    )
    if config.gradient_accumulation_steps > 1:
        builder_optimizer = optax.MultiSteps(
            builder_optimizer, config.gradient_accumulation_steps
        )
    inital_builder_params = builder_params_init_fn(rng)
    builder_train_state = Porygon2BuilderTrainState.create(
        apply_fn=jax.vmap(
            builder_network.apply,
            in_axes=(None, 1, 1, None),
            out_axes=1,
        ),
        init_fn=builder_params_init_fn,
        params=inital_builder_params,
        target_params=inital_builder_params,
        tx=builder_optimizer,
    )

    return player_train_state, builder_train_state


def save_train_state(
    wandb_run: wandb.wandb_run.Run,
    learner_config: Porygon2LearnerConfig,
    player_state: Porygon2PlayerTrainState,
    builder_state: Porygon2BuilderTrainState,
    league: League,
):
    save_path = save_train_state_locally(
        learner_config, player_state, builder_state, league
    )
    if learner_config.log_artifacts_online and (
        player_state.step_count.item() % learner_config.cloud_save_interval_steps == 0
    ):
        wandb_run.log_artifact(
            artifact_or_path=save_path,
            name=f"latest-gen{learner_config.generation}",
            type="model",
        )


def save_train_state_locally(
    learner_config: Porygon2LearnerConfig,
    player_state: Porygon2PlayerTrainState,
    builder_state: Porygon2BuilderTrainState,
    league: League,
):
    save_path = os.path.abspath(
        f"ckpts/gen{learner_config.generation}/ckpt_{player_state.step_count:08}"
    )
    return save_state(save_path, learner_config, player_state, builder_state, league)


def save_state(
    save_path: str,
    learner_config: Porygon2LearnerConfig,
    player_state: Porygon2PlayerTrainState,
    builder_state: Porygon2BuilderTrainState,
    league: League,
):
    os.makedirs(save_path, exist_ok=True)
    player_components = dict(
        params=player_state.params,
        target_params=player_state.target_params,
        opt_state=player_state.opt_state,
        scalars=dict(
            step_count=player_state.step_count,
            frame_count=player_state.frame_count,
            ema_adv_mean=player_state.ema_adv_mean,
            ema_adv_std=player_state.ema_adv_std,
        ),
    )
    builder_components = dict(
        params=builder_state.params,
        target_params=builder_state.target_params,
        opt_state=builder_state.opt_state,
        scalars=dict(
            step_count=builder_state.step_count,
            frame_count=builder_state.frame_count,
        ),
    )
    checkpoint.save_train_state(
        save_path,
        learner_config,
        player_components,
        builder_components,
        league.serialize(),
    )
    return save_path


def _get_checkpoint_path(learner_config: Porygon2LearnerConfig) -> str | None:
    """Finds the most recent checkpoint folder."""
    save_path = f"./ckpts/gen{learner_config.generation}/"
    os.makedirs(save_path, exist_ok=True)
    return checkpoint.most_recent_ckpt_dir(save_path)


def _init_league(
    learner_config: Porygon2LearnerConfig,
    player_state: Porygon2PlayerTrainState,
    builder_state: Porygon2BuilderTrainState,
) -> League:
    """Creates a fresh League instance."""
    return League(
        main_player=ParamsContainer(
            player_frame_count=np.array(player_state.frame_count).item(),
            builder_frame_count=np.array(builder_state.frame_count).item(),
            step_count=MAIN_KEY,
            player_params=player_state.target_params,
            builder_params=builder_state.target_params,
        ),
        players=[],
        league_size=learner_config.league_size,
        cache_size=learner_config.league_cache_size,
        ucb_c=learner_config.league_ucb_c,
    )


def load_from_scratch(
    learner_config: Porygon2LearnerConfig,
    player_state: Porygon2PlayerTrainState,
    builder_state: Porygon2BuilderTrainState,
) -> tuple[Porygon2PlayerTrainState, Porygon2BuilderTrainState, League]:
    """
    No-op on state; simply initializes a fresh league.
    """
    print("Starting training from scratch.")
    league = _init_league(learner_config, player_state, builder_state)
    return player_state, builder_state, league


def load_from_checkpoint(
    ckpt_path: str,
    learner_config: Porygon2LearnerConfig,
    player_state: Porygon2PlayerTrainState,
    builder_state: Porygon2BuilderTrainState,
) -> tuple[Porygon2PlayerTrainState, Porygon2BuilderTrainState, League]:
    """
    Full restoration: loads params, opt_state, step counts, and league.
    """
    print(f"Loading checkpoint from {ckpt_path}")
    ckpt_data = checkpoint.load_full(ckpt_path)

    print("Checkpoint data:")
    ckpt_player_state = ckpt_data["player_state"]
    ckpt_builder_state = ckpt_data["builder_state"]
    ckpt_league_bytes = ckpt_data["league"]
    player_scalars = ckpt_player_state["scalars"]
    builder_scalars = ckpt_builder_state["scalars"]

    # Debug prints (scalars only — heavy arrays excluded)
    pprint(player_scalars)
    pprint(builder_scalars)

    # Restore League
    if ckpt_league_bytes is not None:
        league = League.deserialize(ckpt_league_bytes)
    else:
        # Fallback if league is missing in ckpt
        league = _init_league(learner_config, player_state, builder_state)

    # Fully replace player state
    player_state = player_state.replace(
        params=ckpt_player_state["params"],
        target_params=ckpt_player_state["target_params"],
        opt_state=ckpt_player_state["opt_state"],
        step_count=player_scalars["step_count"],
        frame_count=player_scalars["frame_count"],
        ema_adv_mean=player_scalars["ema_adv_mean"],
        ema_adv_std=player_scalars["ema_adv_std"],
    )

    # Fully replace builder state
    builder_state = builder_state.replace(
        params=ckpt_builder_state["params"],
        target_params=ckpt_builder_state["target_params"],
        opt_state=ckpt_builder_state["opt_state"],
        step_count=builder_scalars["step_count"],
        frame_count=builder_scalars["frame_count"],
    )

    # The league file holds only refs + stats; install the live main player
    # from the restored state so opponents have someone to be ranked against.
    league.update_main_player(
        ParamsContainer(
            step_count=MAIN_KEY,
            player_frame_count=int(player_scalars["frame_count"]),
            builder_frame_count=int(builder_scalars["frame_count"]),
            player_params=jax.device_get(player_state.target_params),
            builder_params=jax.device_get(builder_state.target_params),
        )
    )

    return player_state, builder_state, league


def merge_params(fresh: Params, loaded: Params) -> tuple[Params, list[str]]:
    """Overlay checkpoint params onto a freshly initialized tree.

    Keys present in both trees with matching leaf shapes take the loaded
    (trained) value; keys only in the fresh tree (newly added modules) keep
    their random/zero init; keys only in the checkpoint (removed modules)
    are dropped; shape mismatches fall back to fresh init. Returns the
    merged tree plus the paths that kept their fresh initialization, so a
    resume across architecture changes is auditable.
    """
    kept_fresh: list[str] = []

    def _merge(fresh_node, loaded_node, path: str):
        if isinstance(fresh_node, Mapping):
            out = {}
            for key, fresh_child in fresh_node.items():
                child_path = f"{path}/{key}"
                if isinstance(loaded_node, Mapping) and key in loaded_node:
                    out[key] = _merge(fresh_child, loaded_node[key], child_path)
                else:
                    out[key] = fresh_child
                    kept_fresh.append(child_path)
            return out
        fresh_shape = getattr(fresh_node, "shape", None)
        loaded_shape = getattr(loaded_node, "shape", None)
        if fresh_shape is not None and fresh_shape == loaded_shape:
            return loaded_node
        kept_fresh.append(f"{path} (shape {loaded_shape} -> {fresh_shape})")
        return fresh_node

    return _merge(fresh, loaded, ""), kept_fresh


def load_from_params(
    ckpt_path: str,
    learner_config: Porygon2LearnerConfig,
    player_state: Porygon2PlayerTrainState,
    builder_state: Porygon2BuilderTrainState,
) -> tuple[Porygon2PlayerTrainState, Porygon2BuilderTrainState, League]:
    """
    Params only: merges ckpt params into the freshly initialized trees, so
    modules added since the checkpoint keep their fresh init and everything
    else keeps its trained weights. Sets BOTH params and target_params to
    the merged tree. Resets opt_state and counts (by keeping the input
    state's version of those) and starts a fresh league.
    """
    print(f"Loading (merging) params only from {ckpt_path}")
    loaded_player_params = checkpoint.load_component(ckpt_path, "player", "params")
    loaded_builder_params = checkpoint.load_component(ckpt_path, "builder", "params")

    player_params, player_kept_fresh = merge_params(
        player_state.params, loaded_player_params
    )
    builder_params, builder_kept_fresh = merge_params(
        builder_state.params, loaded_builder_params
    )
    for name, kept in (("player", player_kept_fresh), ("builder", builder_kept_fresh)):
        if kept:
            print(f"{name}: {len(kept)} param subtrees kept fresh init:")
            for path in kept:
                print(f"  {path}")

    # target_params gets the same merged tree: leaving it at fresh init
    # would hand v-trace a garbage reference policy for ~1/ema_rate steps.
    player_state = player_state.replace(
        params=player_params, target_params=player_params
    )
    builder_state = builder_state.replace(
        params=builder_params, target_params=builder_params
    )

    # Initialize a fresh league since we are effectively starting a new run with existing weights
    league = _init_league(learner_config, player_state, builder_state)

    return player_state, builder_state, league


def load_train_state(
    learner_config: Porygon2LearnerConfig,
    player_state: Porygon2PlayerTrainState,
    builder_state: Porygon2BuilderTrainState,
    mode: Literal["scratch", "checkpoint", "params"] = "checkpoint",
) -> tuple[Porygon2PlayerTrainState, Porygon2BuilderTrainState, League]:

    latest_ckpt = _get_checkpoint_path(learner_config)

    # 1. Force Scratch
    if mode == "scratch":
        return load_from_scratch(learner_config, player_state, builder_state)

    # 2. No checkpoint found -> Fallback to Scratch
    if not latest_ckpt:
        print("No checkpoint found. Defaulting to scratch.")
        return load_from_scratch(learner_config, player_state, builder_state)

    # 3. Load Params Only
    if mode == "params":
        return load_from_params(
            latest_ckpt, learner_config, player_state, builder_state
        )

    # 4. Standard Checkpoint Load (Default)
    return load_from_checkpoint(
        latest_ckpt, learner_config, player_state, builder_state
    )
