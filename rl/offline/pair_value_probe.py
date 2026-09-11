"""Offline reads of the pairwise entity critics (2026-09-12) that the live
panels cannot afford: hit-point monotonicity of the cross-side pair term.

For every alive mon on the board the hit-point ratio token is stepped DOWN
by one of the 32 bins the state kernel encodes (the honest read: the model
sees bins, so a derivative along the scalar column alone would miss the
one-hot half) and the cross term m_ij re-read. Losing hit points should not
make my mon i stronger over any j (delta m_ij <= 0) nor make their mon j
weaker against any i (delta m_ij >= 0). Reported per head: the fraction of
alive cross pairs whose change has the wrong sign beyond a tolerance, and
the share of the total absolute change those violations carry. Known true
exceptions (Flail, Reversal, Endeavor, berry thresholds) are the reason this
is a measurement and not a loss (the plan's §3).

    env/bin/python -m rl.offline.pair_value_probe --ckpt <dir> --chunks <dump>

`--chunks` is a harness dump (rl/offline/harness.py dump/load) of recorded
chunks; the checkpoint must carry the heads (a fresh head reads 0 everywhere
and the probe says so).
"""

import argparse

import jax
import jax.numpy as jnp
import numpy as np

from rl.environment.data import MAX_RATIO_TOKEN
from rl.environment.interfaces import PlayerActorInput, PlayerActorOutput
from rl.environment.protos.features_pb2 import (
    EntityPrivateNodeFeature,
    EntityPublicNodeFeature,
)
from rl.model.heads import HeadParams
from rl.model.player_model import get_player_model
from rl.offline.harness import flatten, load, load_params
from rl.online.artifact import player_model_config_for
from rl.online.config import Porygon2LearnerConfig
from rl.online.training.batching import stack_batch

HP_STEP = MAX_RATIO_TOKEN // 32
PER_SIDE = 6
# Per head: (output leaf, (env leaf, hp column) for my rows, for theirs).
HEAD_INPUTS = {
    "public": (
        ("public_team", EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__HP_RATIO),
        ("public_team", EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__HP_RATIO),
    ),
    "private": (
        (
            "private_team",
            EntityPrivateNodeFeature.ENTITY_PRIVATE_NODE_FEATURE__HP_RATIO,
        ),
        (
            "opp_private_team",
            EntityPrivateNodeFeature.ENTITY_PRIVATE_NODE_FEATURE__HP_RATIO,
        ),
    ),
}


def _apply_fn(config: Porygon2LearnerConfig):
    net = get_player_model(player_model_config_for(config))
    return jax.jit(
        jax.vmap(
            net.apply,
            in_axes=(None, 1, 1, None),
            out_axes=PlayerActorOutput.batch_out_axes(),
        )
    )


def _stepped_input(actor_input: PlayerActorInput, leaf: str, column: int, slot: int):
    """The same input with `slot`'s hit-point token in `leaf` lowered by one
    bin (floored at 0); a mon already at 0 is fainted and its pairs weigh 0
    either way."""
    array = np.asarray(getattr(actor_input.env, leaf)).copy()
    # Public rows index the whole 12-row board; sheet leaves are 6 rows.
    array[..., slot, column] = np.maximum(array[..., slot, column] - HP_STEP, 0)
    return actor_input.replace(
        env=actor_input.env.replace(**{leaf: jnp.asarray(array)})
    )


def _cross(pred: PlayerActorOutput, head: str) -> np.ndarray:
    return np.asarray(getattr(pred, f"pair_value_{head}").cross, dtype=np.float32)


def _weights(pred: PlayerActorOutput, head: str) -> np.ndarray:
    return np.asarray(getattr(pred, f"pair_value_{head}").cross_weight, np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--chunks", required=True)
    parser.add_argument("--which", default="target_params")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    args = parser.parse_args()

    config = Porygon2LearnerConfig()
    params = jax.device_put(load_params(args.ckpt, args.which))
    if "pair_value_public" not in params["params"]:
        raise SystemExit("checkpoint carries no pairwise critic: nothing to read")
    apply = _apply_fn(config)
    chunks = flatten(load(args.chunks))

    totals = {
        head: {"pairs": 0.0, "violations": 0.0, "abs_change": 0.0, "bad_change": 0.0}
        for head in HEAD_INPUTS
    }
    for start in range(0, len(chunks), args.batch):
        batch = stack_batch(chunks[start : start + args.batch])
        transitions = batch.player_transitions
        actor_input = PlayerActorInput(
            env=transitions.env_output,
            packed_history=batch.player_packed_history,
            history=batch.player_history,
        )
        actor_output = transitions.agent_output.actor_output
        base = apply(params, actor_input, actor_output, HeadParams())
        live = ~np.asarray(transitions.env_output.done, dtype=bool)
        for head, (mine, theirs) in HEAD_INPUTS.items():
            base_cross = _cross(base, head)
            alive_pairs = _weights(base, head) > 0
            row_ok = live[..., None, None] & alive_pairs
            for side, (leaf, column), sign in ((0, mine, -1.0), (1, theirs, 1.0)):
                for slot in range(PER_SIDE):
                    if leaf == "public_team":
                        board_slot = slot + side * PER_SIDE
                    else:
                        board_slot = slot
                    stepped = apply(
                        params,
                        _stepped_input(actor_input, leaf, column, board_slot),
                        actor_output,
                        HeadParams(),
                    )
                    delta = _cross(stepped, head) - base_cross
                    if side == 0:
                        pairs = row_ok[..., slot, :]
                        change = delta[..., slot, :]
                    else:
                        pairs = row_ok[..., :, slot]
                        change = delta[..., :, slot]
                    # Losing hit points: my mon's edges should not RISE
                    # (sign -1), their mon's should not FALL (sign +1).
                    wrong = pairs & (sign * change < -args.tolerance)
                    totals[head]["pairs"] += pairs.sum()
                    totals[head]["violations"] += wrong.sum()
                    totals[head]["abs_change"] += np.abs(change[pairs]).sum()
                    totals[head]["bad_change"] += np.abs(change[wrong]).sum()

    for head, total in totals.items():
        pairs = max(total["pairs"], 1.0)
        abs_change = max(total["abs_change"], 1e-12)
        print(
            f"{head}: alive pair reads {int(total['pairs'])}, "
            f"violation fraction {total['violations'] / pairs:.4f}, "
            f"violating share of |delta m| {total['bad_change'] / abs_change:.4f}, "
            f"mean |delta m| {total['abs_change'] / pairs:.5f}"
        )


if __name__ == "__main__":
    main()
