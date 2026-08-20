"""Fixed-length chunked unrolls (2026-08-16): span arithmetic, the
trailing history window clip, and the bootstrap-row mask in
compute_player_targets."""

import jax.numpy as jnp
import numpy as np
import pytest

from rl.environment.interfaces import (
    PlayerEnvOutput,
    PlayerHistoryOutput,
    Trajectory,
    PlayerTransition,
)
from rl.environment.interfaces import PlayerPackedHistoryOutput
from rl.environment.protos.features_pb2 import FieldFeature
from rl.online.config import Porygon2LearnerConfig
from rl.online.player_actor import chunk_spans
from rl.online.targets import compute_player_targets


@pytest.mark.parametrize("chunk_length", [3, 4, 64])
@pytest.mark.parametrize("game_done", [True, False])
def test_chunk_spans_cover_every_step_exactly_once(chunk_length, game_done):
    stride = chunk_length - 1
    for num_steps in range(1, 4 * chunk_length + 3):
        spans = chunk_spans(num_steps, chunk_length, game_done)

        # Every span fits the game and only the last may be short.
        for start, end in spans[:-1]:
            assert end - start + 1 == chunk_length
        assert all(0 <= s <= e <= num_steps - 1 for s, e in spans)

        # Consecutive spans overlap by exactly the one bootstrap row.
        for (_, prev_end), (next_start, _) in zip(spans, spans[1:]):
            assert next_start == prev_end

        # PG-trained rows (every span row except its final one) tile the
        # game with no gaps and no double coverage.
        trained = [row for start, end in spans for row in range(start, end)]
        assert len(trained) == len(set(trained))
        if game_done:
            # All steps except the terminal row itself (done rows never
            # take PG loss anyway).
            assert sorted(trained) == list(range(num_steps - 1))
            assert spans[-1][1] == num_steps - 1
        else:
            # Capped no-done game: full chunks only; the dropped tail is
            # shorter than one chunk's trainable width.
            covered = len(trained)
            assert covered == min(
                (max(num_steps - 1, 0) // stride) * stride, num_steps - 1
            )
            assert num_steps - 1 - covered < stride + 1


def test_chunk_spans_short_game_is_single_span():
    assert chunk_spans(1, 64, True) == [(0, 0)]
    assert chunk_spans(40, 64, True) == [(0, 39)]
    # Exactly one full chunk.
    assert chunk_spans(64, 64, True) == [(0, 63)]
    # One step past a chunk: terminal partial reusing the overlap row.
    assert chunk_spans(65, 64, True) == [(0, 63), (63, 64)]
    # Capped game with no done drops the partial tail.
    assert chunk_spans(65, 64, False) == [(0, 63)]
    assert chunk_spans(127, 64, False) == [(0, 63), (63, 126)]


def _windows_fixture(valid_steps: int, rows_per_step: int, capacity: int = 512):
    """Consistent field+packed pair: step s owns packed rows
    [s*rows_per_step, (s+1)*rows_per_step), named by its
    RELEVANT_ENTITY_IDX columns — the contiguous-ascending layout the
    service's getHistory guarantees and clip_history_windows_tail relies
    on."""
    from rl.environment.protos.enums_pb2 import SpeciesEnum
    from rl.environment.protos.features_pb2 import EntityRevealedNodeFeature

    num_features = len(FieldFeature.keys())
    field = np.zeros((capacity, num_features), dtype=np.int32)
    field[:valid_steps, FieldFeature.FIELD_FEATURE__VALID] = 1
    field[:valid_steps, FieldFeature.FIELD_FEATURE__REQUEST_COUNT] = np.arange(
        valid_steps
    )
    field[:valid_steps, FieldFeature.FIELD_FEATURE__NUM_RELEVANT] = rows_per_step
    for k in range(rows_per_step):
        field[
            :valid_steps,
            FieldFeature.Value(f"FIELD_FEATURE__RELEVANT_ENTITY_IDX{k}"),
        ] = (
            np.arange(valid_steps) * rows_per_step + k
        )

    packed_rows = valid_steps * rows_per_step
    num_revealed = len(EntityRevealedNodeFeature.keys())
    revealed = np.zeros((2 * capacity, num_revealed), dtype=np.int32)
    revealed[
        :packed_rows, EntityRevealedNodeFeature.ENTITY_REVEALED_NODE_FEATURE__SPECIES
    ] = 7  # any real species token
    packed = PlayerPackedHistoryOutput(
        public_cache=np.zeros((2 * capacity, 1), dtype=np.int32),
        revealed_cache=revealed,
        edge_cache=np.arange(2 * capacity, dtype=np.int32)[:, None],
    )
    return PlayerHistoryOutput(field=field), packed


def test_joint_tail_clip_keeps_recent_rows_and_rebases_indices():
    from rl.environment.utils import clip_history_windows_tail

    history, packed = _windows_fixture(valid_steps=300, rows_per_step=2)
    out_hist, out_packed = clip_history_windows_tail(history, packed, 256)

    field = out_hist.field
    assert field.shape[0] == 256
    assert field[:, FieldFeature.FIELD_FEATURE__VALID].sum() == 256
    requests = field[:, FieldFeature.FIELD_FEATURE__REQUEST_COUNT]
    assert requests[0] == 44 and requests[255] == 299

    # Steps 44..299 own packed rows 88..599 -> start_row 88, 512 rows kept.
    assert out_packed.edge_cache.shape[0] == 512
    np.testing.assert_array_equal(
        out_packed.edge_cache[:, 0], np.arange(88, 600)
    )
    # Rebased references: each kept step's rows, shifted to the window.
    idx0 = field[:256, FieldFeature.FIELD_FEATURE__RELEVANT_ENTITY_IDX0]
    np.testing.assert_array_equal(idx0, np.arange(256) * 2)
    # The gather target matches: field row i names packed rows carrying the
    # original absolute indices of its own step.
    np.testing.assert_array_equal(
        out_packed.edge_cache[idx0, 0], (np.arange(44, 300)) * 2
    )


def test_joint_tail_clip_short_history_is_identity_with_padding():
    from rl.environment.utils import clip_history_windows_tail

    history, packed = _windows_fixture(valid_steps=100, rows_per_step=2)
    out_hist, out_packed = clip_history_windows_tail(history, packed, 256)
    field = out_hist.field
    assert field.shape[0] == 256
    assert field[:, FieldFeature.FIELD_FEATURE__VALID].sum() == 100
    assert field[99, FieldFeature.FIELD_FEATURE__REQUEST_COUNT] == 99
    assert field[100:].sum() == 0
    # No rebase: indices untouched, packed rows 0..199 then zero padding.
    np.testing.assert_array_equal(
        field[:100, FieldFeature.FIELD_FEATURE__RELEVANT_ENTITY_IDX0],
        np.arange(100) * 2,
    )
    assert out_packed.edge_cache.shape[0] == 512
    np.testing.assert_array_equal(out_packed.edge_cache[:200, 0], np.arange(200))
    assert out_packed.edge_cache[200:].sum() == 0


def test_joint_tail_clip_shrinks_field_window_to_fit_packed_budget():
    """Dense games (3 packed rows per step) cannot fit history_length
    steps into the 2x packed budget — the field window must shrink, as in
    the service's getHistory loop, never misalign."""
    from rl.environment.utils import clip_history_windows_tail

    history, packed = _windows_fixture(valid_steps=300, rows_per_step=3)
    out_hist, out_packed = clip_history_windows_tail(history, packed, 256)
    field = out_hist.field
    kept = int(field[:, FieldFeature.FIELD_FEATURE__VALID].sum())
    # 3 * kept rows must fit 512.
    assert kept == 170
    assert field[0, FieldFeature.FIELD_FEATURE__REQUEST_COUNT] == 300 - 170
    idx0 = field[:kept, FieldFeature.FIELD_FEATURE__RELEVANT_ENTITY_IDX0]
    np.testing.assert_array_equal(idx0, np.arange(kept) * 3)
    assert int(idx0.max()) + 2 < 512
    np.testing.assert_array_equal(
        out_packed.edge_cache[idx0, 0], (np.arange(300 - 170, 300)) * 3
    )


# real_model_and_trajectory: session-scoped fixture in conftest.py, shared
# with the other model-forward suites so the model initialises once.


@pytest.mark.gpu
@pytest.mark.slow
def test_untruncated_tail_window_forward_is_identical(real_model_and_trajectory):
    """The history scan is causal and requests align to history by
    REQUEST_COUNT value, so a trailing window that still starts at the
    game's first token — only the zero-padding capacity differs — must
    reproduce the full-history forward exactly. This is the property that
    makes per-chunk windows train exactly what the actor computed."""
    from rl.environment.protos.enums_pb2 import SpeciesEnum
    from rl.environment.protos.features_pb2 import EntityRevealedNodeFeature
    from rl.environment.utils import clip_history_windows_tail
    from rl.model.heads import HeadParams

    network, params, actor_input, actor_output = real_model_and_trajectory

    history_valid = int(
        np.asarray(
            actor_input.history.field[..., FieldFeature.FIELD_FEATURE__VALID]
        ).sum()
    )
    packed_valid = int(
        np.asarray(
            actor_input.packed_history.revealed_cache[
                ..., EntityRevealedNodeFeature.ENTITY_REVEALED_NODE_FEATURE__SPECIES
            ]
            != SpeciesEnum.SPECIES_ENUM___UNSPECIFIED
        ).sum()
    )
    # Window generous enough that no field step or packed row is dropped —
    # only the zero-padding capacity changes.
    window = max(history_valid, (packed_valid + 1) // 2)
    history_window, packed_window = clip_history_windows_tail(
        actor_input.history, actor_input.packed_history, window
    )
    windowed = actor_input.replace(
        history=history_window, packed_history=packed_window
    )

    full = network.apply(params, actor_input, actor_output, HeadParams())
    clipped = network.apply(params, windowed, actor_output, HeadParams())

    np.testing.assert_allclose(
        np.asarray(clipped.value_head.log_probs, dtype=np.float32),
        np.asarray(full.value_head.log_probs, dtype=np.float32),
        atol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(clipped.action_head.log_policy, dtype=np.float32),
        np.asarray(full.action_head.log_policy, dtype=np.float32),
        atol=1e-5,
    )


@pytest.mark.gpu
@pytest.mark.slow
def test_truncated_tail_window_forward_is_finite(real_model_and_trajectory):
    """A window that genuinely drops the oldest tokens (the burn-in
    approximation for deep-in-game chunks) must still produce finite,
    normalised outputs — requests older than the window read the h0
    state, the same fallback the service's own NUM_HISTORY windowing
    exercises."""
    from rl.environment.utils import clip_history_windows_tail
    from rl.model.heads import HeadParams

    network, params, actor_input, actor_output = real_model_and_trajectory

    history_window, packed_window = clip_history_windows_tail(
        actor_input.history, actor_input.packed_history, 32
    )
    windowed = actor_input.replace(
        history=history_window, packed_history=packed_window
    )
    out = network.apply(params, windowed, actor_output, HeadParams())
    log_probs = np.asarray(out.value_head.log_probs, dtype=np.float32)
    assert np.isfinite(log_probs).all()
    # bf16 log_softmax: normalisation holds only to ~3e-3.
    np.testing.assert_allclose(np.exp(log_probs).sum(-1), 1.0, atol=1e-2)
    assert np.isfinite(np.asarray(out.action_head.log_prob, dtype=np.float32)).all()


def _targets_batch(dones: np.ndarray):
    """Minimal batch for compute_player_targets: T x B dones, everything
    else neutral (uniform two-action masks, zero reward off-terminal)."""
    t_len, batch_size = dones.shape
    win_reward = np.zeros((t_len, batch_size, 3), dtype=np.float32)
    win_reward[..., 2] = dones.astype(np.float32)  # win lands on the done row
    action_mask = np.ones((t_len, batch_size, 2, 2), dtype=bool)
    env = PlayerEnvOutput(
        done=jnp.asarray(dones),
        win_reward=jnp.asarray(win_reward),
        action_mask=jnp.asarray(action_mask),
    )
    return Trajectory(player_transitions=PlayerTransition(env_output=env))


def test_final_row_is_bootstrap_only_unless_terminal():
    t_len, batch_size = 5, 2
    dones = np.zeros((t_len, batch_size), dtype=bool)
    # Column 0: game ends at row 2 (rows 3-4 are done-copy padding).
    dones[2, 0] = True
    batch = _targets_batch(dones)

    n_bins = 3
    value_log_probs = jnp.log(
        jnp.full((t_len, batch_size, n_bins), 1.0 / n_bins, dtype=jnp.float32)
    )
    isr = jnp.ones((t_len, batch_size), dtype=jnp.float32)

    targets, _ = compute_player_targets(
        batch, value_log_probs=value_log_probs, isr=isr,
        config=Porygon2LearnerConfig(),
    )
    value_mask = np.asarray(targets.value_mask)
    policy_mask = np.asarray(targets.policy_mask)

    # Terminal column: rows through the done row train value; padding after
    # it never does. The done row takes no PG loss.
    np.testing.assert_array_equal(value_mask[:, 0], [1, 1, 1, 0, 0])
    np.testing.assert_array_equal(policy_mask[:, 0], [1, 1, 0, 0, 0])
    # Mid-game column: the final row is the bootstrap row — no value or PG
    # loss there; its training signal belongs to the next chunk's row 0.
    np.testing.assert_array_equal(value_mask[:, 1], [1, 1, 1, 1, 0])
    np.testing.assert_array_equal(policy_mask[:, 1], [1, 1, 1, 1, 0])


def test_shape_lattice_trim_is_lossless():
    """The static shape lattice (2026-08-20): _chunk_required_shape reads
    each chunk's real content and _trim_to_lattice slices to the first
    fitting combo — never dropping a valid history step, and preserving
    the [-1] outcome reads (padding rows are terminal-step copies)."""
    from rl.online.learner import _chunk_required_shape, _trim_to_lattice

    lattice = ((48, 128), (64, 192), (64, 256))
    history, packed = _windows_fixture(valid_steps=50, rows_per_step=2, capacity=256)
    T = 64
    done = np.zeros(T, dtype=bool)
    done[20] = True
    # Terminal-copy padding convention: rows past the done row repeat it.
    win = np.zeros(T, dtype=np.float32)
    win[20:] = 1.0
    traj = Trajectory(
        player_transitions=PlayerTransition(
            env_output=PlayerEnvOutput(done=done, win_reward=win)
        ),
        player_packed_history=packed,
        player_history=history,
    )
    # done at row 20 -> 21 rows; 50 valid steps / 100 packed rows -> 50.
    assert _chunk_required_shape(traj) == (21, 50)

    trimmed = _trim_to_lattice([traj], lattice)[0]
    assert trimmed.player_transitions.env_output.done.shape[0] == 48
    assert trimmed.player_history.field.shape[0] == 128
    assert trimmed.player_packed_history.revealed_cache.shape[0] == 256
    assert trimmed.player_transitions.env_output.win_reward[-1] == win[-1]
    assert (
        trimmed.player_history.field[:, FieldFeature.FIELD_FEATURE__VALID].sum() == 50
    )

    # A mid-game chunk (no done row) pins the batch to full chunk length;
    # the chain still trims the history axis.
    full = Trajectory(
        player_transitions=PlayerTransition(
            env_output=PlayerEnvOutput(done=np.zeros(T, dtype=bool), win_reward=win)
        ),
        player_packed_history=packed,
        player_history=history,
    )
    assert _chunk_required_shape(full)[0] == 64
    out = _trim_to_lattice([traj, full], lattice)
    assert out[0].player_transitions.env_output.done.shape[0] == 64
    assert out[0].player_history.field.shape[0] == 192

    # Full-shape batches and single-entry lattices pass through untouched.
    assert _trim_to_lattice([traj], ((64, 256),))[0] is traj
