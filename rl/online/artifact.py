"""RL train-state lifecycle and checkpoint artifacts.

The online mirror of rl/offline/artifact.py: the single boundary through
which trained RL products are created, saved, restored and merged.
Checkpoints are written with a manifest.json capturing the architecture
capabilities (entity size, decision slots, policy-head variant) so loads
across architecture changes fail with a sentence instead of a pytree
error — the same fail-loudly convention as the offline critic's
announced-states manifest flag. rl/checkpoint.py stays the shared
low-level serialisation beneath both.
"""

import functools
import json
import logging
import os
from collections.abc import Callable, Mapping
from pprint import pformat
from typing import Any, Literal

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import core, struct
from flax.training import train_state
from tqdm import tqdm

from rl import checkpoint
from rl.config.common import AdamWConfig  # noqa: F401 (re-export convenience)
from rl.environment.interfaces import (
    BuilderActorInput,
    BuilderActorOutput,
    PlayerActorInput,
    PlayerActorOutput,
)
from rl.environment.utils import get_ex_builder_step, get_ex_player_step
from rl.model.config import get_player_model_config
from rl.model.heads import HeadParams
from rl.model.utils import Params, ParamsContainer
from rl.online.config import Porygon2LearnerConfig
from rl.online.league import MAIN_KEY, League

MANIFEST_NAME = "manifest.json"

logger = logging.getLogger(__name__)


def _model_capabilities(learner_config: Porygon2LearnerConfig) -> dict:
    """Architecture facts a checkpoint consumer must agree on. pi_head is a
    literal because the variant is a code fact, not a config value — bump
    it when the head class changes."""
    model_cfg = get_player_model_config(learner_config.generation, train=True)
    return dict(
        generation=learner_config.generation,
        smogon_format=learner_config.smogon_format,
        entity_size=int(model_cfg.entity_size),
        num_decision_slots=int(model_cfg.num_decision_slots),
        pi_head="action_score_grouped_micro",
        # The advantage head is structural — always present, so the
        # manifest records the ARCHITECTURE variant, and a checkpoint whose
        # variant string differs must go through load-mode "params"
        # (fresh-inits the head, carries everything else).
        #
        # "action_score_grouped_micro" (2026-08-25) = ActionScoreHead —
        # per-slot-group micro q/k + per-group zero-init type_scale and
        # local src/tgt routes, per-MODALITY macro MLP and out layer, ONE
        # rung, Q composed in heads.compose_q. Predecessors:
        # "hierarchical_two_rung" (2026-08-20, owned ActionAdapter +
        # shared flat MacroMicroHead, plus a second privileged rung) and
        # "privileged_two_rung" (2026-08-17, flat cond-MLP bilinear grid).
        #
        # BUMPING THESE IS NOT COSMETIC: both literals were left stale
        # through the 2026-08-25 redesign, so a pre-redesign checkpoint
        # passed check_manifest STRICT and would have been restored onto a
        # structurally different param tree. The manifest exists precisely
        # to stop that, and it only works if the literal moves with the
        # head class.
        q_head="advantage_grouped_micro",
    )


def read_manifest(ckpt_path: str) -> dict | None:
    """The checkpoint's manifest, or None for pre-manifest checkpoints."""
    try:
        with open(os.path.join(ckpt_path, MANIFEST_NAME)) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def check_manifest(ckpt_path: str, learner_config, strict: bool) -> None:
    """Compares a checkpoint manifest against the current architecture.

    strict (checkpoint-mode resume): any mismatch raises — a full state
    restore requires an identical architecture. Non-strict (params-mode):
    mismatches print, since merge_params handles them field by field.
    Pre-manifest checkpoints pass silently either way."""
    manifest = read_manifest(ckpt_path)
    if manifest is None:
        return
    expected = _model_capabilities(learner_config)
    mismatched = {
        key: (manifest.get(key), value)
        for key, value in expected.items()
        if key in manifest and manifest[key] != value
    }
    if not mismatched:
        return
    detail = ", ".join(
        f"{k}: ckpt={ckpt!r} current={cur!r}" for k, (ckpt, cur) in mismatched.items()
    )
    if strict:
        raise ValueError(
            f"checkpoint at {ckpt_path} was written by a different "
            f"architecture ({detail}); resume with load mode 'params' to "
            "merge, or start from scratch"
        )
    tqdm.write(f"manifest deltas vs current architecture ({detail}) — merging params")


class Porygon2PlayerTrainState(train_state.TrainState):
    apply_fn: Callable[
        [Params, PlayerActorInput, PlayerActorOutput, HeadParams], PlayerActorOutput
    ] = struct.field(pytree_node=False)
    init_fn: Callable[[jax.Array], Params] = struct.field(pytree_node=False)

    target_params: core.FrozenDict[str, Any] = struct.field(pytree_node=True)
    # R-NaD reference policy pi_reg (2026-08-22; snap semantics
    # 2026-08-25): a periodic hard SNAP of target_params, in place, every
    # config.player_reg_snap_steps (frozen between snaps; launch snapshot
    # through the NeuRD warm-up). The reference penalty
    # -eta*(log pi - log pi_reg) in the policy objective
    # is measured against it; the snap bounds the log-ratio gap
    # structurally — the continuous EMA it replaces (1e-4, then 5e-5)
    # never reset, and the compounding gap drove the pgaijs6l/2wvnlsz3
    # grad-norm runaways. One param set instead of rnad.py's prev/prev_
    # crossfade pair: same delta_m reset, no 4th net.
    reg_params: core.FrozenDict[str, Any] = struct.field(pytree_node=True)

    # Force these to be dynamic JAX arrays (PyTree nodes) instead of static Python scalars
    step_count: jax.Array = struct.field(
        default_factory=lambda: jnp.array(0, dtype=jnp.int32), pytree_node=True
    )
    frame_count: jax.Array = struct.field(
        default_factory=lambda: jnp.array(0, dtype=jnp.int32), pytree_node=True
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

    # Jitted init: eager flax init of the player model dispatches the
    # whole forward op by op and compiles every nn.scan separately --
    # ~6-10 min on the training box (2026-08-24 slow-suite timing); one
    # compile is a fraction of that and persists in the compile cache.
    player_params_init_fn = functools.partial(
        jax.jit(player_network.init),
        head_params=HeadParams(),
        actor_input=ex_player_actor_inp,
        actor_output=ex_player_actor_out,
    )
    initial_player_params = player_params_init_fn(rng)
    player_optimizer = optax.chain(
        optax.clip_by_global_norm(config.player_clip_gradient),
        optax.adamw(
            learning_rate=config.player_learning_rate,
            b1=config.player_adam.b1,
            b2=config.player_adam.b2,
            eps=config.player_adam.eps,
            weight_decay=config.player_adam.weight_decay,
        ),
    )
    player_train_state = Porygon2PlayerTrainState.create(
        apply_fn=jax.vmap(player_network.apply, in_axes=(None, 1, 1, None), out_axes=1),
        init_fn=player_params_init_fn,
        params=initial_player_params,
        # Deep-copied: params and target_params must not share buffers, or
        # donating the train state to the jitted train step fails with a
        # duplicate-donation error on the first step.
        target_params=jax.tree.map(jnp.copy, initial_player_params),
        reg_params=jax.tree.map(jnp.copy, initial_player_params),
        tx=player_optimizer,
    )

    builder_params_init_fn = functools.partial(
        jax.jit(builder_network.init),
        actor_input=ex_builder_actor_inp,
        actor_output=ex_builder_actor_out,
        head_params=HeadParams(),
    )
    builder_optimizer = optax.chain(
        optax.clip_by_global_norm(config.builder_clip_gradient),
        optax.adamw(
            learning_rate=config.builder_learning_rate,
            b1=config.builder_adam.b1,
            b2=config.builder_adam.b2,
            eps=config.builder_adam.eps,
            weight_decay=config.builder_adam.weight_decay,
        ),
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
        # Deep-copied for the same donation-aliasing reason as the player state.
        target_params=jax.tree.map(jnp.copy, inital_builder_params),
        tx=builder_optimizer,
    )

    return player_train_state, builder_train_state


def _ckpt_root(learner_config: Porygon2LearnerConfig) -> str:
    """Root directory for this process's checkpoints. Flat: one training
    run means one resumable checkpoint tree, with no namespacing needed to
    keep two from colliding."""
    return f"./ckpts/gen{learner_config.generation}/"


def save_train_state_locally(
    learner_config: Porygon2LearnerConfig,
    player_state: Porygon2PlayerTrainState,
    builder_state: Porygon2BuilderTrainState,
    league: League,
    controller_bytes: bytes | None = None,
):
    save_path = os.path.abspath(
        os.path.join(
            _ckpt_root(learner_config),
            f"ckpt_{player_state.step_count:08}",
        )
    )
    return save_state(
        save_path,
        learner_config,
        player_state,
        builder_state,
        league,
        controller_bytes,
    )


def save_state(
    save_path: str,
    learner_config: Porygon2LearnerConfig,
    player_state: Porygon2PlayerTrainState,
    builder_state: Porygon2BuilderTrainState,
    league: League,
    controller_bytes: bytes | None = None,
):
    player_components = dict(
        params=player_state.params,
        target_params=player_state.target_params,
        reg_params=player_state.reg_params,
        opt_state=player_state.opt_state,
        scalars=dict(
            step_count=player_state.step_count,
            frame_count=player_state.frame_count,
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
    return write_checkpoint_components(
        save_path,
        learner_config,
        player_components,
        builder_components,
        league.serialize(),
        controller_bytes,
        step_count=int(np.asarray(player_state.step_count)),
        frame_count=int(np.asarray(player_state.frame_count)),
    )


def write_checkpoint_components(
    save_path: str,
    learner_config: Porygon2LearnerConfig,
    player_components: dict[str, Any],
    builder_components: dict[str, Any],
    league_bytes: bytes,
    controller_bytes: bytes | None,
    step_count: int,
    frame_count: int,
) -> str:
    """The actual checkpoint disk I/O, given already-host-side components.

    Split out of save_state so a background writer thread
    (Learner._checkpoint_writer_worker) can do this work from a queued
    payload without needing live TrainState/League objects — those aren't
    safe to hand across threads once the training loop has moved on to
    donating/mutating them for the next step.
    """
    os.makedirs(save_path, exist_ok=True)
    checkpoint.save_train_state(
        save_path,
        learner_config,
        player_components,
        builder_components,
        league_bytes,
        controller_bytes,
    )
    manifest = dict(
        step_count=step_count,
        frame_count=frame_count,
        **_model_capabilities(learner_config),
    )
    with open(os.path.join(save_path, MANIFEST_NAME), "w") as f:
        json.dump(manifest, f, indent=2)
    return save_path


def _get_checkpoint_path(learner_config: Porygon2LearnerConfig) -> str | None:
    """Finds the most recent checkpoint folder under this process's root."""
    save_path = _ckpt_root(learner_config)
    os.makedirs(save_path, exist_ok=True)
    return checkpoint.most_recent_ckpt_dir(save_path)


WANDB_RUNS_NAME = "wandb_runs.json"


def _wandb_run_info_path(learner_config: Porygon2LearnerConfig) -> str:
    return os.path.join(_ckpt_root(learner_config), WANDB_RUNS_NAME)


def save_wandb_run_info(
    learner_config: Porygon2LearnerConfig,
    group: str,
    run_ids: dict[str, str],
) -> None:
    """Persists this session's wandb identity (group + run id) next to the
    checkpoints, so a checkpoint-mode restart resumes the same wandb run
    instead of opening a fresh session. Still a dict so a runtime file
    written before the single-run collapse still loads."""
    path = _wandb_run_info_path(learner_config)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(dict(group=group, runs=run_ids), f, indent=2)


def load_wandb_run_info(learner_config: Porygon2LearnerConfig) -> dict | None:
    """The saved wandb session ({"group": ..., "runs": {pop: id}}), or None.

    Only meaningful alongside a resumable checkpoint: if the checkpoint
    tree is gone (archived, fresh start) the training lineage is gone too,
    and resuming the old wandb runs would append an unrelated lineage's
    metrics onto them — so a missing checkpoint invalidates the file.
    """
    if _get_checkpoint_path(learner_config) is None:
        return None
    try:
        with open(_wandb_run_info_path(learner_config)) as f:
            info = json.load(f)
    except (OSError, ValueError):
        return None
    if not isinstance(info, dict) or not isinstance(info.get("group"), str):
        return None
    runs = info.get("runs")
    if not isinstance(runs, dict):
        return None
    return info


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
            # Host copies, never the live state arrays: actors' device_put of
            # an already-on-device tree is a no-op, so handing out live
            # buffers has actors running inference on memory the donated
            # train step deletes.
            player_params=jax.device_get(player_state.target_params),
            builder_params=jax.device_get(builder_state.target_params),
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
) -> tuple[
    Porygon2PlayerTrainState,
    Porygon2BuilderTrainState,
    League,
    bytes | None,
]:
    """
    No-op on state; simply initializes a fresh league.
    """
    tqdm.write("Starting training from scratch.")
    league = _init_league(learner_config, player_state, builder_state)
    return player_state, builder_state, league, None


def load_from_checkpoint(
    ckpt_path: str,
    learner_config: Porygon2LearnerConfig,
    player_state: Porygon2PlayerTrainState,
    builder_state: Porygon2BuilderTrainState,
) -> tuple[
    Porygon2PlayerTrainState,
    Porygon2BuilderTrainState,
    League,
    bytes | None,
]:
    """
    Full restoration: loads params, opt_state, step counts, league and the
    host-side controller state.
    """
    tqdm.write(f"Loading checkpoint from {ckpt_path}")
    check_manifest(ckpt_path, learner_config, strict=True)
    ckpt_data = checkpoint.load_full(ckpt_path)

    tqdm.write("Checkpoint data:")
    ckpt_player_state = ckpt_data["player_state"]
    ckpt_builder_state = ckpt_data["builder_state"]
    ckpt_league_bytes = ckpt_data["league"]
    player_scalars = ckpt_player_state["scalars"]
    builder_scalars = ckpt_builder_state["scalars"]

    # Debug prints (scalars only — heavy arrays excluded)
    tqdm.write(pformat(player_scalars))
    tqdm.write(pformat(builder_scalars))

    if ckpt_league_bytes is not None:
        league = League.deserialize(ckpt_league_bytes)
    else:
        # Fallback if league is missing in ckpt
        league = _init_league(learner_config, player_state, builder_state)

    player_state = player_state.replace(
        params=ckpt_player_state["params"],
        target_params=ckpt_player_state["target_params"],
        # Checkpoints from before the reference policy existed seed it
        # from the target (KL 0 at resume; the next snap takes over from there).
        reg_params=ckpt_player_state.get(
            "reg_params", jax.tree.map(jnp.copy, ckpt_player_state["target_params"])
        ),
        opt_state=ckpt_player_state["opt_state"],
        step_count=player_scalars["step_count"],
        frame_count=player_scalars["frame_count"],
    )

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

    return (
        player_state,
        builder_state,
        league,
        ckpt_data.get("controllers"),
    )


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
) -> tuple[
    Porygon2PlayerTrainState,
    Porygon2BuilderTrainState,
    League,
    bytes | None,
]:
    """
    Params only: merges ckpt params into the freshly initialized trees, so
    modules added since the checkpoint keep their fresh init and everything
    else keeps its trained weights. Sets BOTH params and target_params to
    the merged tree. Resets opt_state and counts (by keeping the input
    state's version of those) and starts a fresh league.
    """
    tqdm.write(f"Loading (merging) params only from {ckpt_path}")
    check_manifest(ckpt_path, learner_config, strict=False)
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
            tqdm.write(f"{name}: {len(kept)} param subtrees kept fresh init:")
            for path in kept:
                tqdm.write(f"  {path}")

    # target_params gets the same merged tree: leaving it at fresh init
    # would hand v-trace a garbage reference policy for ~1/ema_rate steps.
    # Deep-copied so params/target_params share no buffers — required for
    # donating the train state to the jitted train step.
    player_state = player_state.replace(
        params=player_params,
        target_params=jax.tree.map(jnp.copy, player_params),
        reg_params=jax.tree.map(jnp.copy, player_params),
    )
    builder_state = builder_state.replace(
        params=builder_params,
        target_params=jax.tree.map(jnp.copy, builder_params),
    )

    # Initialize a fresh league since we are effectively starting a new run with existing weights
    league = _init_league(learner_config, player_state, builder_state)

    # Controllers start fresh too: params-mode is a new run, and the old
    # lineage's training dynamics (controller EMAs, league stats) do not
    # carry over.
    return player_state, builder_state, league, None


def load_train_state(
    learner_config: Porygon2LearnerConfig,
    player_state: Porygon2PlayerTrainState,
    builder_state: Porygon2BuilderTrainState,
    mode: Literal["scratch", "checkpoint", "params"] = "checkpoint",
) -> tuple[
    Porygon2PlayerTrainState,
    Porygon2BuilderTrainState,
    League,
    bytes | None,
]:
    """Loads the train state on a process (re)start. One checkpoint tree,
    no fork_from_ckpt/run_subdir indirection: every league snapshot is
    already a normal, permanent PlayerRef in the persisted League by the
    time this runs.
    """
    latest_ckpt = _get_checkpoint_path(learner_config)

    # 1. Force Scratch
    if mode == "scratch":
        return load_from_scratch(learner_config, player_state, builder_state)

    # 2. No checkpoint found -> fall back to scratch, loudly. A bare print()
    # here is how 1335's ~300k-step lineage and its league got lost between
    # it and 1336 without anyone noticing — mode was "checkpoint" (a resume
    # was expected) and it silently became a fresh run instead. Still
    # auto-falls back (the launch entry point doesn't set LOAD_STATE_MODE
    # per-run), but now at warning level so it can't scroll by unnoticed.
    if not latest_ckpt:
        logger.warning(
            "LOAD_STATE_MODE=%r but no checkpoint found under %s "
            "— starting from scratch.",
            mode,
            _ckpt_root(learner_config),
        )
        return load_from_scratch(learner_config, player_state, builder_state)

    # 3. Load Params Only (RL checkpoints across architecture changes)
    if mode == "params":
        return load_from_params(
            latest_ckpt, learner_config, player_state, builder_state
        )

    # 4. Standard Checkpoint Load (Default)
    return load_from_checkpoint(
        latest_ckpt, learner_config, player_state, builder_state
    )
