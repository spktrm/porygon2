"""Offline screen for the flat support hinge coefficient, and the v-trace
threshold's cut audit, over recorded self-play chunks — the two
pre-restart measurements of the 2026-09-09 action-support plan.

Never run inside a live learner: config is a jit static argname, so each
coefficient here is its own compiled train step (four compiles, once), on
a stopped run's checkpoint.

Screen (per coefficient in --coefficients, 0 the baseline): the learner's
own TRAIN_STEP_JIT is run twice on every recorded batch from the same
restored state — the first step reads the pre-update policy's support,
the hinge's loss/active fraction and the gradient norms (the executable
loss, nothing re-derived); the second step, on the same batch, reads the
support the ONE applied update bought (min legal probability, the
fractions below the lines). The coefficient to take is the smallest that
lifts the abandoned cells without a > 10% rise in shared-encoder gradient
norm relative to 0; if none does, the hinge does not land.

Cut audit (threshold-only, no training): the target policy is thresholded
at player_prune_threshold on every recorded chunk and the discards
located — fraction of acted rows discarded, and the fraction of chunks
carrying a discard before their midpoint. Pre-registered 2026-09-09: above
5% of chunks cut before the midpoint, the threshold is restricted to rho
(the trace ratio c left raw) before the restart.

    env/bin/python -m rl.offline.support_screen --checkpoint ckpts/gen9/ckpt_N \\
        --games 32 --out runtime/support-screen/ckpt_N.json

with PS_SERVICE_URI pointing at a service (the offline one on :8081).
"""

import argparse
import dataclasses
import json
import logging
import os
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax  # noqa: E402
import numpy as np  # noqa: E402

from rl.environment.utils import acted_rows  # noqa: E402
from rl.model.builder_model import get_builder_model  # noqa: E402
from rl.model.config import (  # noqa: E402
    get_builder_model_config,
    get_player_model_config,
)
from rl.model.player_model import get_player_model  # noqa: E402
from rl.model.utils import prune_log_policy  # noqa: E402
from rl.offline import harness  # noqa: E402
from rl.online.artifact import create_train_state, load_from_checkpoint  # noqa: E402
from rl.online.config import Porygon2LearnerConfig  # noqa: E402
from rl.online.training.batching import stack_batch  # noqa: E402
from rl.online.training.train_step import TRAIN_STEP_JIT  # noqa: E402

logger = logging.getLogger(__name__)

FIRST_STEP_KEYS = (
    "player_loss_support",
    "player_support_active_fraction",
    "player_support_n_tau_row",
    "player_support_saturated_frac",
    "player_support_min_prob",
    "player_support_frac_below_p01",
    "player_support_frac_below_p005",
    "player_encoder_gradient_norm",
    "player_action_head_gradient_norm",
    "player_gradient_norm",
    "player_switch_logit_grad_support",
    "player_discard_taken_frac",
    "player_trace_len_mean",
    "player_trace_len_mean_raw",
    "player_isr_ess",
    "player_isr_ess_raw",
    "player_entropy_micro_taken",
    "player_switch_mass_choice",
)
SECOND_STEP_KEYS = (
    "player_support_min_prob",
    "player_support_median_prob",
    "player_support_frac_below_p01",
    "player_support_frac_below_p005",
    "player_support_frac_below_p001",
    "player_support_switch_min_prob",
    "player_support_move_min_prob",
    "player_entropy_micro_taken",
    "player_switch_mass_choice",
)


def restored_states(checkpoint: str, config: Porygon2LearnerConfig):
    """The learner's own restore: params, EMA target, reference, optimiser
    moments — so the applied update is the restored Adam's, not a fresh one."""
    player_net = get_player_model(
        get_player_model_config(config.generation, train=True)
    )
    builder_net = get_builder_model(
        get_builder_model_config(config.generation, train=True)
    )
    player_state, builder_state = create_train_state(
        player_net, builder_net, jax.random.key(0), config
    )
    player_state, builder_state, _, _ = load_from_checkpoint(
        checkpoint, config, player_state, builder_state
    )
    return jax.device_get(player_state), jax.device_get(builder_state)


def record_chunks(config, target_params, games: int, seed: int, device: str):
    sides = harness.play_games(
        target_params,
        games,
        pairs=4,
        tag="support-screen",
        seed=seed,
        opponent="self",
        temperature=1.0,
        device=device,
    )
    chunks = harness.flatten(sides)
    logger.info("%d games -> %d chunks", len(sides) // 2, len(chunks))
    return chunks


def screen_coefficient(coefficient, config, host_player, host_builder, batches):
    """Two consecutive train steps per batch from the same restored state:
    the first's logs read the pre-update policy and the update asked for,
    the second's the support the applied update bought."""
    config = dataclasses.replace(config, player_support_hinge_coef=coefficient)
    first = {key: [] for key in FIRST_STEP_KEYS}
    second = {key: [] for key in SECOND_STEP_KEYS}
    for batch in batches:
        # TRAIN_STEP_JIT donates the states: fresh device copies per batch.
        player_state = jax.device_put(host_player)
        builder_state = jax.device_put(host_builder)
        player_state, builder_state, logs = TRAIN_STEP_JIT(
            player_state, builder_state, batch, config
        )
        for key in FIRST_STEP_KEYS:
            first[key].append(float(logs[key]))
        _, _, logs = TRAIN_STEP_JIT(player_state, builder_state, batch, config)
        for key in SECOND_STEP_KEYS:
            second[key].append(float(logs[key]))
    return {
        "coefficient": coefficient,
        "pre_update": {key: float(np.mean(values)) for key, values in first.items()},
        "post_update": {key: float(np.mean(values)) for key, values in second.items()},
    }


def cut_audit(target_params, chunks, threshold: float, batch_size: int):
    """Where the threshold would discard, on the recorded chunks, with no
    training: the target policy thresholded and gathered at the taken
    action, per acted row."""
    discarded_rows = 0
    acted_row_count = 0
    chunks_with_discard = 0
    chunks_cut_before_midpoint = 0
    positions = []
    index = 0
    for prediction, batch in harness.forward(target_params, chunks, batch=batch_size):
        env = batch.player_transitions.env_output
        log_policy = prediction.action_head.log_policy
        legal = np.asarray(env.action_mask, bool)
        pruned = np.asarray(prune_log_policy(log_policy, legal, threshold))
        taken = np.asarray(
            batch.player_transitions.agent_output.actor_output.action_head.action_index
        )
        kept = (
            np.take_along_axis(pruned, taken[..., None], axis=-1)[..., 0]
            > np.finfo(pruned.dtype).min
        )
        group = chunks[index : index + legal.shape[1]]
        index += legal.shape[1]
        for column, chunk in enumerate(group):
            real = int(chunk.game_length[0] - chunk.game_step_offset[0])
            acted = acted_rows(env.done[:, column])
            discards = acted & ~kept[:, column]
            acted_row_count += int(acted.sum())
            discarded_rows += int(discards.sum())
            if discards.any():
                chunks_with_discard += 1
                first = int(np.flatnonzero(discards)[0])
                positions.append(first / max(real, 1))
                if first < real / 2:
                    chunks_cut_before_midpoint += 1
    return {
        "threshold": threshold,
        "chunks": len(chunks),
        "acted_rows": acted_row_count,
        "discarded_row_fraction": discarded_rows / max(acted_row_count, 1),
        "chunks_with_discard_fraction": chunks_with_discard / max(len(chunks), 1),
        "chunks_cut_before_midpoint_fraction": chunks_cut_before_midpoint
        / max(len(chunks), 1),
        "first_discard_position_median": (
            float(np.median(positions)) if positions else None
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--games", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="gpu")
    parser.add_argument("--coefficients", default="0,0.001,0.0025,0.005")
    parser.add_argument("--out", required=True)
    arguments = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)

    config = Porygon2LearnerConfig()
    host_player, host_builder = restored_states(arguments.checkpoint, config)
    target_params = host_player.target_params
    chunks = record_chunks(
        config, target_params, arguments.games, arguments.seed, arguments.device
    )
    audit = cut_audit(
        target_params, chunks, config.player_prune_threshold, config.batch_size
    )
    logger.info("cut audit: %s", json.dumps(audit))

    batches = [
        stack_batch(chunks[start : start + config.batch_size])
        for start in range(0, len(chunks) - config.batch_size + 1, config.batch_size)
    ]
    rows = []
    for coefficient in (float(value) for value in arguments.coefficients.split(",")):
        row = screen_coefficient(
            coefficient, config, host_player, host_builder, batches
        )
        rows.append(row)
        logger.info("coefficient %s: %s", coefficient, json.dumps(row))
    baseline = rows[0]["pre_update"]["player_encoder_gradient_norm"]
    for row in rows:
        row["encoder_gradient_norm_relative"] = row["pre_update"][
            "player_encoder_gradient_norm"
        ] / max(baseline, 1e-12)
    result = {
        "checkpoint": arguments.checkpoint,
        "games": arguments.games,
        "seed": arguments.seed,
        "chunks": len(chunks),
        "batches": len(batches),
        "cut_audit": audit,
        "screen": rows,
    }
    Path(arguments.out).parent.mkdir(parents=True, exist_ok=True)
    Path(arguments.out).write_text(json.dumps(result, indent=2))
    print(
        f"{'coef':>8} {'loss':>9} {'active':>7} {'enc_rel':>8} {'min_post':>9} {'p01_post':>9}"
    )
    for row in rows:
        print(
            f"{row['coefficient']:>8} "
            f"{row['pre_update']['player_loss_support']:>9.5f} "
            f"{row['pre_update']['player_support_active_fraction']:>7.4f} "
            f"{row['encoder_gradient_norm_relative']:>8.3f} "
            f"{row['post_update']['player_support_min_prob']:>9.6f} "
            f"{row['post_update']['player_support_frac_below_p01']:>9.4f}"
        )
    print("cut audit:", json.dumps(audit))


if __name__ == "__main__":
    main()
