"""Step-6 intrinsic-capacity probe (docs/critic-weakness-analysis.md).

Can the Q head fit the ACTION axis at all? Overfit one fixed,
voluntary-switch-rich batch with the real train_step and watch the Q CE
approach its label-entropy floor, the per-modality R² (move / forced /
voluntary) and the action-axis resolution panels (q_action_var,
pivotal_frac, matched-V critic gap) — then read the same panels on a
held-out switch-rich batch for generalisation.

Two arms, same batch, same checkpoint:

  fixed   — labels frozen: target/reference EMA rates 0, NeuRD coef 0,
            R-NaD eta 0. Retrace labels come from target_params only, so
            this is pure supervised fitting of fixed 3-bin labels on
            fixed rows. Failure here ⇒ architecture / optimisation.
  live    — production config (EMA, NeuRD, R-NaD all on) so the labels
            move with the target net as they do in training. The gap to
            `fixed` is what the moving target costs.

Usage (learner down, the 8080 service up):

    env/bin/python -m rl.offline.overfit_probe ckpts/gen9/ckpt_00022641 \\
        --games 120 --batch 8 --steps 1500 --out /tmp/probe

Writes <out>/games.pkl (the played sides), <out>/<arm>.pkl (list of
per-eval log dicts, keys `step`, `split` in {train, heldout}) and a
terse progress line per eval to stdout.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import os
import pickle
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

from rl import checkpoint
from rl.environment.data import FLAT_MODALITY_MASK
from rl.environment.interfaces import Trajectory
from rl.environment.protos.service_pb2 import ModalityEnum
from rl.model.builder_model import get_builder_model
from rl.model.capacity import make_capacity_probe
from rl.model.config import get_builder_model_config, get_player_model_config
from rl.model.player_model import get_player_model
from rl.offline.harness import dump, flatten, load, play_games
from rl.online.artifact import create_train_state
from rl.online.config import get_learner_config
from rl.online.training.batching import stack_batch
from rl.online.training.train_step import TRAIN_STEP_JIT

logger = logging.getLogger(__name__)

_FLAT = np.asarray(FLAT_MODALITY_MASK)
_SWITCH_CELLS = _FLAT == ModalityEnum.MODALITY_ENUM__SWITCH
_MOVE_CELLS = _FLAT == ModalityEnum.MODALITY_ENUM__MOVE

# The panels printed per eval; everything train_step logs is pickled.
PRINT_KEYS = (
    "player_loss_q",
    "player_q_mse",
    "player_q_r2",
    "player_q_r2_move",
    "player_q_r2_switch_forced",
    "player_q_r2_switch_voluntary",
    "player_q_action_var",
    "player_q_pivotal_frac",
    "player_q_switch_move_gap",
    "player_mv_pooled_gap_critic",
    "player_mv_pooled_gap_realised",
    "learner_q_action_var",
    "learner_q_switch_move_gap",
    "learner_q_pivotal_frac",
    "learner_q_taken_voluntary",
    "learner_q_taken_move",
)


def trainable_rows(done: np.ndarray) -> np.ndarray:
    """The learner's q_mask along the time axis: rows up to and including
    the first done row (the terminal-copy padding after it repeats the
    terminal row with done=0), minus the bootstrap-only final row unless
    it is the game's own terminal, minus done rows themselves."""
    done = done.astype(bool)
    before_or_at_done = (np.cumsum(done, axis=0) - done) == 0
    rows = before_or_at_done.copy()
    rows[-1] &= done[-1]
    return rows & ~done


def voluntary_switch_rows(chunk: Trajectory) -> int:
    """Rows where a switch was taken with a legal move available, over the
    chunk's trainable rows (not done, not the bootstrap-only final row)."""
    env = chunk.player_transitions.env_output
    mask = np.asarray(env.action_mask)
    flat = mask.reshape(mask.shape[0], -1)
    idx = np.asarray(chunk.player_transitions.agent_output.actor_output.action_head.action_index)
    rows = trainable_rows(np.asarray(env.done).astype(bool))
    taken_switch = _SWITCH_CELLS[idx]
    has_move = (flat & _MOVE_CELLS).any(-1)
    return int((rows & taken_switch & has_move).sum())


def pick_batches(chunks: list[Trajectory], batch: int):
    """Top-`batch` chunks by voluntary-switch rows for the fixed batch, the
    next `batch` as the held-out batch (also switch-rich, so the held-out
    voluntary panels have rows)."""
    ranked = sorted(chunks, key=voluntary_switch_rows, reverse=True)
    train, held = ranked[:batch], ranked[batch : 2 * batch]
    logger.info(
        "fixed batch voluntary rows %s; held-out %s",
        [voluntary_switch_rows(c) for c in train],
        [voluntary_switch_rows(c) for c in held],
    )
    return stack_batch(train), stack_batch(held)


def arm_config(arm: str):
    base = get_learner_config()
    if arm == "live":
        return base
    if arm == "fixed":
        return dataclasses.replace(
            base,
            player_neurd_warmup_steps=0,
            player_neurd_coef=0.0,
            player_ref_eta=0.0,
            player_ema_update_rate=0.0,
            player_reg_ema_rate=0.0,
        )
    raise ValueError(arm)


def build_states(ckpt_dir: str, config):
    player_net = get_player_model(get_player_model_config(config.generation, train=True))
    builder_net = get_builder_model(get_builder_model_config(config.generation, train=True))
    player_state, builder_state = create_train_state(
        player_net, builder_net, jax.random.key(0), config
    )
    ck = checkpoint.load_full(ckpt_dir)["player_state"]

    def merge(fresh, saved):
        """Shape-tolerant load: leaves whose shape changed (the Q head's
        3-bin -> scalar readout, Step 3) keep their fresh init — the
        probe's question is whether the NEW head fits on a trained
        trunk. Logs what was skipped."""
        skipped = []

        def pick(path, a, b):
            if getattr(b, "shape", None) == getattr(a, "shape", None):
                return b
            skipped.append(jax.tree_util.keystr(path))
            return a

        out = jax.tree_util.tree_map_with_path(pick, fresh, saved)
        if skipped:
            logger.info("fresh-init (shape changed): %s", skipped)
        return out, bool(skipped)

    params, p_skip = merge(player_state.params, ck["params"])
    target, _ = merge(player_state.target_params, ck["target_params"])
    reg, _ = merge(player_state.reg_params, ck.get("reg_params", ck["target_params"]))
    # Adam moments mirror the param tree; a shape change resets them.
    opt_state = player_state.opt_state if p_skip else ck["opt_state"]
    player_state = player_state.replace(
        params=params,
        target_params=target,
        reg_params=reg,
        opt_state=opt_state,
        step_count=jnp.asarray(ck["scalars"]["step_count"]),
    )
    return player_net, player_state, builder_state


def make_learner_q_stats(player_net):
    """Action-axis panels read from the LEARNER params' Q_all — the logged
    q_action_var / pivotal_frac / switch_move_gap read the target net,
    which the `fixed` arm freezes by design."""
    from rl.environment.interfaces import PlayerActorInput
    from rl.model.heads import HeadParams
    from rl.online.training.targets import residual_q

    apply = jax.vmap(player_net.apply, in_axes=(None, 1, 1, None), out_axes=1)
    switch_cells = jnp.asarray(_SWITCH_CELLS)
    move_cells = jnp.asarray(_MOVE_CELLS)

    def stats(params, batch):
        pt = batch.player_transitions
        pred = apply(
            params,
            PlayerActorInput(
                env=pt.env_output,
                packed_history=batch.player_packed_history,
                history=batch.player_history,
            ),
            pt.agent_output.actor_output,
            HeadParams(),
        )
        done = pt.env_output.done.astype(bool)
        flat = pt.env_output.action_mask.reshape(*done.shape, -1).astype(bool)
        q_all = residual_q(
            pred.q_adv, pred.value_head.expectation, pred.action_head.log_policy, flat
        )  # (T,B,A)
        before = (jnp.cumsum(done, axis=0) - done) == 0
        final = jnp.arange(done.shape[0])[:, None] == done.shape[0] - 1
        rows = before & (~final | done) & ~done
        vs, vm = flat & switch_cells, flat & move_cells
        best_s = jnp.max(jnp.where(vs, q_all, -jnp.inf), -1)
        best_m = jnp.max(jnp.where(vm, q_all, -jnp.inf), -1)
        has_both = vs.any(-1) & vm.any(-1) & rows
        n_legal = jnp.maximum(flat.sum(-1), 1)
        mean_legal = jnp.sum(jnp.where(flat, q_all, 0.0), -1) / n_legal
        var_legal = jnp.sum(jnp.where(flat, (q_all - mean_legal[..., None]) ** 2, 0.0), -1) / n_legal
        idx = pt.agent_output.actor_output.action_head.action_index
        q_taken = jnp.take_along_axis(q_all, idx[..., None], -1)[..., 0]
        taken_switch = jnp.take(switch_cells, idx)
        vol = rows & taken_switch & vm.any(-1)
        mov = rows & ~taken_switch

        def avg(x, m):
            return jnp.sum(jnp.where(m, x, 0.0)) / jnp.maximum(m.sum(), 1)

        return {
            "learner_q_action_var": avg(var_legal, rows),
            "learner_q_switch_move_gap": avg(best_s - best_m, has_both),
            "learner_q_pivotal_frac": avg((best_s > best_m).astype(jnp.float32), has_both),
            "learner_q_taken_voluntary": avg(q_taken, vol),
            "learner_q_taken_move": avg(q_taken, mov),
            "learner_n_voluntary": vol.sum().astype(jnp.float32),
            "learner_n_rows": rows.sum().astype(jnp.float32),
        }

    return jax.jit(stats)


def _floats(logs) -> dict:
    return {k: float(v) for k, v in logs.items() if jnp.ndim(v) == 0}


def run_arm(arm: str, ckpt_dir: str, train_batch, held_batch, steps: int, every: int, out: str):
    config = arm_config(arm)
    player_net, player_state, builder_state = build_states(ckpt_dir, config)
    capacity = make_capacity_probe(player_net)
    learner_stats = make_learner_q_stats(player_net)
    records = []

    def evaluate(step):
        # Donation means a copy is consumed per eval; the update is discarded.
        for split, b in (("train", train_batch), ("heldout", held_batch)):
            ps = jax.tree.map(jnp.copy, player_state)
            bs = jax.tree.map(jnp.copy, builder_state)
            _, _, logs = TRAIN_STEP_JIT(ps, bs, b, config)
            rec = _floats(logs)
            rec.update(_floats(capacity(player_state.params, b)))
            rec.update(_floats(learner_stats(player_state.params, b)))
            rec.update(step=step, split=split, arm=arm)
            records.append(rec)
            print(
                f"[{arm} {split} {step:5d}] "
                + " ".join(f"{k.replace('player_', '').replace('learner_', 'L.')}={rec.get(k, float('nan')):.4f}" for k in PRINT_KEYS),
                flush=True,
            )

    t0 = time.time()
    evaluate(0)
    for step in range(1, steps + 1):
        player_state, builder_state, _ = TRAIN_STEP_JIT(
            player_state, builder_state, train_batch, config
        )
        if step % every == 0 or step == steps:
            evaluate(step)
            with open(os.path.join(out, f"{arm}.pkl"), "wb") as f:
                pickle.dump(records, f)
    logger.info("arm %s done in %.0fs", arm, time.time() - t0)
    return records


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--games", type=int, default=120)
    ap.add_argument("--pairs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--every", type=int, default=25)
    ap.add_argument("--arms", default="fixed,live")
    ap.add_argument("--out", required=True)
    ap.add_argument("--games-pkl", default=None, help="reuse played sides")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stdout)
    os.makedirs(a.out, exist_ok=True)

    games_pkl = a.games_pkl or os.path.join(a.out, "games.pkl")
    if os.path.exists(games_pkl):
        sides = load(games_pkl)
    else:
        target = checkpoint.load_component(a.ckpt, "player", "target_params")
        sides = play_games(target, n_games=a.games, pairs=a.pairs, tag="probe")
        dump(sides, games_pkl)
        del target
    chunks = flatten(sides)
    logger.info(
        "%d sides, %d chunks, %d voluntary rows",
        len(sides),
        len(chunks),
        sum(voluntary_switch_rows(c) for c in chunks),
    )
    train_batch, held_batch = pick_batches(chunks, a.batch)
    for arm in a.arms.split(","):
        run_arm(arm, a.ckpt, train_batch, held_batch, a.steps, a.every, a.out)


if __name__ == "__main__":
    main()
