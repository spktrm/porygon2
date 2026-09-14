from pathlib import Path

import numpy as np
import pytest

from inference.model import load_checkpoint_params
from rl import checkpoint


def test_inference_loads_new_checkpoint_without_player_ema(tmp_path: Path) -> None:
    player = {"kernel": np.array([1.0, 2.0], dtype=np.float32)}
    builder = {"kernel": np.array([3.0, 4.0], dtype=np.float32)}
    checkpoint.save_param_snapshot(
        str(tmp_path),
        player_components={"params": player},
        builder_components={"target_params": builder},
    )
    container = load_checkpoint_params(str(tmp_path))
    np.testing.assert_array_equal(container.player_params["kernel"], player["kernel"])
    np.testing.assert_array_equal(container.builder_params["kernel"], builder["kernel"])
    assert not (tmp_path / "player" / "target_params").exists()

    # A retired component must not override or need decoding beside live weights.
    (tmp_path / "player" / "target_params").write_bytes(b"retired invalid pickle")
    restored = load_checkpoint_params(str(tmp_path))
    np.testing.assert_array_equal(restored.player_params["kernel"], player["kernel"])


def test_inference_does_not_silently_substitute_retired_player_weights(
    tmp_path: Path,
) -> None:
    checkpoint.save_param_snapshot(
        str(tmp_path),
        player_components={"target_params": {"kernel": np.ones(2)}},
        builder_components={"target_params": {"kernel": np.ones(2)}},
    )
    with pytest.raises(FileNotFoundError, match="player/params"):
        load_checkpoint_params(str(tmp_path))
