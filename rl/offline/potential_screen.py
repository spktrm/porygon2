"""Step 3 of the 2026-09-11 PBRS plan: the matched offline gradient screen
that gates turning the potential channel on (LESSONS "PBRS potential
channel"). Never run beside a live learner: each strength is its own
compiled train step, on a stopped run's checkpoint.

From the SAME checkpoint, restored once at strength 0 and once at --strength
(the potential head fresh and therefore exactly 0 -- the launch state), on
every recorded batch:
- the policy-logit gradient of the learner's V-trace score-function loss,
  with its live-network estimates, ratios and masks held fixed: absolute norms, the RMS
  perturbation ||g_eta - g_0|| / ||g_0||, the cosine, and the voluntary-
  switch / stay split with row counts. The gradient is taken w.r.t. each
  row's legal log-policy, equal to the logit gradient on legal cells
  (softmax invariance);
- the learner's TRAIN_STEP_JIT at each strength: the global gradient norm
  against player_clip_gradient, the potential head's share of it, and the
  applied update's perturbation over the shared params (post-clip, post-Adam,
  the restored optimiser moments), per subtree.

Pre-registered acceptance: pooled RMS logit-gradient perturbation <= 0.10,
read beside the absolute norms (a design criterion, not a derived bound),
and the global norm under the clip at --strength on every batch. Fallback:
rescreen at 0.025; if that fails too, relaunch at 0 and record why.

    PORT=8081 MAX_WORKERS=2 MEMORY_STATS_PATH=/tmp/x.json \\
        node service/dist/server/index.js
    PS_SERVICE_URI=ws://localhost:8081 env/bin/python -m rl.offline.potential_screen \\
        --checkpoint ckpts/gen9/ckpt_00360000 --games 32 \\
        --out runtime/pbrs-screen/ckpt_00360000.json

service/dist must be built from 406efe0 or later (`npm run compile-base`,
with no training service running from it) so the chunks carry the
potential; the screen refuses chunks whose potential is 0 everywhere.
"""

import argparse
import dataclasses
import json
import logging
import os
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np

from rl.environment.interfaces import PlayerActorInput
from rl.model.heads import HeadParams
from rl.model.utils import legal_log_policy
from rl.offline.uniform_kl_screen import record_chunks, restored_states
from rl.online.config import Porygon2LearnerConfig
from rl.online.training.batching import stack_batch
from rl.online.training.loss import vtrace_policy_loss
from rl.online.training.targets import compute_player_targets, unit_potential
from rl.online.training.telemetry import action_axis_masks
from rl.online.training.train_step import TRAIN_STEP_JIT

logger = logging.getLogger(__name__)
LOGIT_BUDGET = 0.10


def policy_logit_gradient(player_state, batch, config: Porygon2LearnerConfig):
    """Differentiate the player's score-function loss with its labels fixed."""
    transitions = batch.player_transitions
    actor_input = PlayerActorInput(
        env=transitions.env_output,
        packed_history=batch.player_packed_history,
        history=batch.player_history,
    )
    actor_output = transitions.agent_output.actor_output
    learner = player_state.apply_fn(
        player_state.params, actor_input, actor_output, HeadParams()
    )
    behaviour_log_prob = actor_output.action_head.log_prob
    action_index = actor_output.action_head.action_index
    legal = transitions.env_output.action_mask
    ratio = jnp.exp(
        learner.action_head.log_prob.astype(jnp.float32)
        - behaviour_log_prob.astype(jnp.float32)
    )
    if config.player_privileged_targets:
        value_log_probs = learner.priv_value_head.log_probs
    else:
        value_log_probs = learner.value_head.log_probs
    potential_values = None
    if config.player_potential_strength > 0:
        potential_values = learner.potential_head.logits
    targets, _ = compute_player_targets(
        batch,
        value_log_probs,
        ratio,
        config,
        potential_values=potential_values,
    )

    def surrogate(log_policy: jax.Array) -> jax.Array:
        taken = jnp.take_along_axis(
            legal_log_policy(log_policy, legal), action_index[..., None], axis=-1
        )[..., 0]
        return vtrace_policy_loss(
            log_prob=taken,
            advantages=targets.pg_advantages,
            valid=targets.policy_mask,
        )

    gradient = jax.grad(surrogate)(learner.action_head.log_policy.astype(jnp.float32))
    axis = action_axis_masks(legal, action_index)
    switch_rows = targets.policy_mask & axis.taken_switch & axis.has_move
    stay_rows = targets.policy_mask & jnp.logical_not(axis.taken_switch)
    return gradient, targets.policy_mask, switch_rows, stay_rows


LOGIT_GRADIENT_JIT = jax.jit(policy_logit_gradient, static_argnames=["config"])


def applied_update(host_player, host_builder, batch, config):
    """The learner's own step (donating fresh device copies): the parameter
    delta it applies, and its logs."""
    player_state, builder_state, logs = TRAIN_STEP_JIT(
        jax.device_put(host_player), jax.device_put(host_builder), batch, config
    )
    delta = jax.tree.map(
        lambda new, old: np.asarray(new, np.float32) - np.asarray(old, np.float32),
        jax.device_get(player_state.params),
        host_player.params,
    )
    scalars = {
        key: float(np.asarray(value))
        for key, value in logs.items()
        if np.ndim(value) == 0
    }
    return delta, scalars


def squared_norm(tree) -> float:
    return float(sum(np.sum(np.square(leaf)) for leaf in jax.tree.leaves(tree)))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--strength", type=float, default=0.05)
    parser.add_argument("--games", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="gpu")
    parser.add_argument("--out", required=True)
    arguments = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)

    off = Porygon2LearnerConfig()
    on = dataclasses.replace(off, player_potential_strength=arguments.strength)
    host_off = restored_states(arguments.checkpoint, off)
    host_on = restored_states(arguments.checkpoint, on)
    chunks = record_chunks(
        off,
        host_off[0].params,
        arguments.games,
        arguments.seed,
        arguments.device,
        tag="potential-screen",
    )
    batches = [
        stack_batch(chunks[start : start + off.batch_size])
        for start in range(0, len(chunks) - off.batch_size + 1, off.batch_size)
    ]
    carried = np.mean(
        [
            np.mean(
                np.asarray(unit_potential(batch.player_transitions.env_output)) != 0
            )
            for batch in batches
        ]
    )
    if carried == 0:
        raise SystemExit(
            "every recorded potential is 0: the service predates 406efe0 "
            "(rebuild service/dist) -- the screen would measure nothing"
        )

    pooled = {name: 0.0 for name in ("base", "moved", "difference", "dot")}
    split = {
        side: {"rows": 0, "base": 0.0, "difference": 0.0} for side in ("switch", "stay")
    }
    per_batch = []
    shared_base = {}
    shared_difference = {}
    for index, batch in enumerate(batches):
        base, policy_rows, switch_rows, stay_rows = LOGIT_GRADIENT_JIT(
            jax.device_put(host_off[0]), batch, off
        )
        moved, _, _, _ = LOGIT_GRADIENT_JIT(jax.device_put(host_on[0]), batch, on)
        base, moved = np.asarray(base), np.asarray(moved)
        difference = moved - base
        pooled["base"] += float(np.sum(base**2))
        pooled["moved"] += float(np.sum(moved**2))
        pooled["difference"] += float(np.sum(difference**2))
        pooled["dot"] += float(np.sum(base * moved))
        for side, rows in (("switch", switch_rows), ("stay", stay_rows)):
            rows = np.asarray(rows)
            split[side]["rows"] += int(rows.sum())
            split[side]["base"] += float(np.sum(base[rows] ** 2))
            split[side]["difference"] += float(np.sum(difference[rows] ** 2))

        delta_off, logs_off = applied_update(*host_off, batch, off)
        delta_on, logs_on = applied_update(*host_on, batch, on)
        for subtree in delta_off["params"]:
            shared_base[subtree] = shared_base.get(subtree, 0.0) + squared_norm(
                delta_off["params"][subtree]
            )
            shared_difference[subtree] = shared_difference.get(
                subtree, 0.0
            ) + squared_norm(
                jax.tree.map(
                    np.subtract,
                    delta_on["params"][subtree],
                    delta_off["params"][subtree],
                )
            )
        per_batch.append(
            {
                "batch": index,
                "logit_rms_perturbation": float(
                    np.sqrt(np.sum(difference**2) / max(np.sum(base**2), 1e-30))
                ),
                "global_norm_off": logs_off["player_gradient_norm"],
                "global_norm_on": logs_on["player_gradient_norm"],
                "potential_head_grad_share": logs_on.get(
                    "player_potential_head_grad_share", float("nan")
                ),
                "potential_adv_share": logs_on.get(
                    "player_potential_adv_share", float("nan")
                ),
            }
        )
        logger.info("batch %d: %s", index, json.dumps(per_batch[-1]))

    logit_rms = float(np.sqrt(pooled["difference"] / max(pooled["base"], 1e-30)))
    clip = off.player_clip_gradient
    result = {
        "checkpoint": arguments.checkpoint,
        "strength": arguments.strength,
        "games": arguments.games,
        "seed": arguments.seed,
        "chunks": len(chunks),
        "batches": len(batches),
        "potential_carried_fraction": float(carried),
        "logit": {
            "norm_off": float(np.sqrt(pooled["base"])),
            "norm_on": float(np.sqrt(pooled["moved"])),
            "rms_perturbation": logit_rms,
            "cosine": pooled["dot"]
            / max(np.sqrt(pooled["base"] * pooled["moved"]), 1e-30),
            "split": {
                side: {
                    "rows": values["rows"],
                    "norm_off": float(np.sqrt(values["base"])),
                    "rms_perturbation": float(
                        np.sqrt(values["difference"] / max(values["base"], 1e-30))
                    ),
                }
                for side, values in split.items()
            },
        },
        "update": {
            "clip": clip,
            "rms_perturbation_by_subtree": {
                subtree: float(
                    np.sqrt(
                        shared_difference[subtree] / max(shared_base[subtree], 1e-30)
                    )
                )
                for subtree in shared_base
            },
            "rms_perturbation_shared": float(
                np.sqrt(
                    sum(shared_difference.values())
                    / max(sum(shared_base.values()), 1e-30)
                )
            ),
        },
        "per_batch": per_batch,
    }
    result["acceptance"] = {
        "logit_rms_perturbation_within_budget": logit_rms <= LOGIT_BUDGET,
        "global_norm_under_clip": all(
            row["global_norm_on"] < clip for row in per_batch
        ),
    }
    Path(arguments.out).parent.mkdir(parents=True, exist_ok=True)
    Path(arguments.out).write_text(json.dumps(result, indent=2))
    print(
        f"logit |g0| {result['logit']['norm_off']:.4g}  |g_eta| "
        f"{result['logit']['norm_on']:.4g}  rms perturbation {logit_rms:.4f} "
        f"(budget {LOGIT_BUDGET})  cosine {result['logit']['cosine']:.4f}"
    )
    for side, values in result["logit"]["split"].items():
        print(
            f"  {side:6s} rows {values['rows']:6d}  |g0| {values['norm_off']:.4g}  "
            f"rms perturbation {values['rms_perturbation']:.4f}"
        )
    print(
        f"update rms perturbation (shared params) "
        f"{result['update']['rms_perturbation_shared']:.4f}; global norm on "
        f"max {max(row['global_norm_on'] for row in per_batch):.3f} vs clip {clip}"
    )
    print("acceptance:", json.dumps(result["acceptance"]))


if __name__ == "__main__":
    main()
