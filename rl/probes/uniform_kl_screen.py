"""Historical uniform-KL screening helpers.

The applied-update coefficient screen was retired when the player switched
back to entropy and a frozen-reference KL on 2026-09-14. Its complete source
was archived at /tmp/porygon2-uniform-kl-screen-before-vtrace-20260914.py.
The protocol audit and restore/play helpers remain available to offline probes.
"""

import gc
import logging
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import numpy as np

from rl.environment.utils import acted_rows
from rl.model.builder_model import get_builder_model
from rl.model.config import (
    get_builder_model_config,
)
from rl.model.player_model import get_player_model
from rl.model.utils import prune_log_policy
from rl.offline import harness
from rl.online.artifact import (
    create_train_state,
    load_from_checkpoint,
    player_model_config_for,
)
from rl.online.config import Porygon2LearnerConfig
from rl.probes import battle_stats

logger = logging.getLogger(__name__)

RETIRED_SCREEN_MESSAGE = (
    "The uniform-KL applied-update screen was retired on 2026-09-14. "
    "The current player has no forward uniform-KL loss. Historical source: "
    "/tmp/porygon2-uniform-kl-screen-before-vtrace-20260914.py."
)


def clear_compiled_state():
    jax.clear_caches()
    gc.collect()


def protocol_switch_report(paths):
    moves = battle_stats.load_moves()
    games = []
    skipped = 0
    for path in paths:
        lines = path.read_text().splitlines()
        completed = any(
            line.startswith("|win|") or line in ("|tie", "|tie|") for line in lines
        )
        if not completed:
            skipped += 1
            continue
        game = battle_stats.parse_log(lines, moves)
        if game is None:
            skipped += 1
            continue
        games.append(game)
    switch_count = 0
    move_count = 0
    for game in games:
        for side in game.sides.values():
            switch_count += side.switch_kinds["voluntary"]
            move_count += sum(side.moves.values())
    decisions = switch_count + move_count
    if decisions:
        fraction = switch_count / decisions
    else:
        fraction = None
    return {
        "scope": "pre_update_checkpoint_selfplay",
        "completed_games": len(games),
        "skipped_logs": skipped,
        "voluntary_switch_count": switch_count,
        "observed_move_or_switch_count": decisions,
        "voluntary_switch_fraction": fraction,
        "human_reference_switch_count": 19512,
        "human_reference_decision_count": 96622,
        "human_reference_fraction": 19512 / 96622,
        "ceiling_fraction": 0.8 * 19512 / 96622,
    }


def restored_states(checkpoint: str, config: Porygon2LearnerConfig):
    """Restore the learner, EMA reference and optimiser moments."""
    player_net = get_player_model(player_model_config_for(config))
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


def record_chunks(
    config,
    player_params,
    games: int,
    seed: int,
    device: str,
    tag: str = "uniform-kl-screen",
):
    sides = harness.play_games(
        player_params,
        games,
        pairs=4,
        tag=tag,
        seed=seed,
        opponent="self",
        temperature=1.0,
        device=device,
    )
    chunks = harness.flatten(sides)
    logger.info("%d games -> %d chunks", len(sides) // 2, len(chunks))
    return chunks


def screen_coefficient(coefficient, config, host_player, host_builder, batches):
    raise RuntimeError(RETIRED_SCREEN_MESSAGE)


def cut_audit(player_params, chunks, threshold: float, batch_size: int):
    """Where the threshold would discard, on the recorded chunks, with no
    training: the supplied policy thresholded and gathered at the taken
    action, per acted row."""
    discarded_rows = 0
    acted_row_count = 0
    chunks_with_discard = 0
    chunks_cut_before_midpoint = 0
    positions = []
    index = 0
    for prediction, batch in harness.forward(player_params, chunks, batch=batch_size):
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
    if positions:
        first_discard_position_median = float(np.median(positions))
    else:
        first_discard_position_median = None
    return {
        "threshold": threshold,
        "chunks": len(chunks),
        "acted_rows": acted_row_count,
        "discarded_row_fraction": discarded_rows / max(acted_row_count, 1),
        "chunks_with_discard_fraction": chunks_with_discard / max(len(chunks), 1),
        "chunks_cut_before_midpoint_fraction": chunks_cut_before_midpoint
        / max(len(chunks), 1),
        "first_discard_position_median": first_discard_position_median,
    }


def main(argv=None):
    raise SystemExit(RETIRED_SCREEN_MESSAGE)


if __name__ == "__main__":
    main()
