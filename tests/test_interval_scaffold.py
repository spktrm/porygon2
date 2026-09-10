"""The interval arms' shared loop (rl/offline/train_interval.run_arm) on a
stub update: the schedule, the finiteness gate and the files it writes."""

import json
from collections.abc import Callable
from pathlib import Path

import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rl.offline.train_interval import prepare_output, run_arm, train_read_subset


def _stub_arm(
    tmp_path: Path, steps: int, eval_steps: set[int], log_every: int, update: Callable
) -> tuple[Path, dict[str, jax.Array], list[dict], list[np.ndarray]]:
    output = prepare_output(tmp_path / "arm")
    calls = []

    def make_batch(indices: np.ndarray) -> dict[str, jax.Array]:
        calls.append(np.asarray(indices))
        return {"indices": jnp.asarray(indices)}

    def read_progress(
        step: int, params: dict[str, jax.Array], seen: np.ndarray
    ) -> dict[str, int | float | str]:
        return {
            "step": step,
            "split": "validation",
            "param": float(params["weight"]),
            "seen": int(seen.sum()),
        }

    params, records = run_arm(
        output,
        "stub",
        {"arm": "stub"},
        {"weight": jnp.float32(0.0)},
        None,
        update,
        make_batch,
        read_progress,
        np.arange(3, 8),
        num_rows=10,
        steps=steps,
        batch_size=2,
        seed=0,
        eval_steps=eval_steps,
        log_every=log_every,
    )
    return output, params, records, calls


def _step_update(
    params: dict[str, jax.Array],
    state: None,
    batch: dict[str, jax.Array],
    key: jax.Array,
) -> tuple[dict[str, jax.Array], None, dict[str, jax.Array]]:
    return {"weight": params["weight"] + 1.0}, state, {"loss": params["weight"]}


def test_run_arm_schedule_and_files(tmp_path: Path) -> None:
    output, params, records, calls = _stub_arm(
        tmp_path, steps=6, eval_steps={0, 4, 6}, log_every=3, update=_step_update
    )
    assert float(params["weight"]) == 6.0
    assert [record["step"] for record in records] == [0, 4, 6]
    # Evaluations read the params as updated so far; the step-0 read
    # carries no update logs, the later ones the last update's.
    assert [record["param"] for record in records] == [0.0, 4.0, 6.0]
    assert "loss" not in records[0] and records[1]["loss"] == 3.0
    assert all(record["seen"] <= 5 for record in records)
    assert len(calls) == 6 and all(len(draw) == 2 for draw in calls)
    assert json.loads((output / "manifest.json").read_text()) == {"arm": "stub"}
    lines = [
        json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()
    ]
    update_lines = [line for line in lines if "split" not in line]
    assert [line["step"] for line in update_lines] == [3, 6]
    assert sorted(path.name for path in output.glob("state-*.msgpack")) == [
        "state-000000.msgpack",
        "state-000004.msgpack",
        "state-000006.msgpack",
    ]
    state = flax.serialization.msgpack_restore(
        (output / "state-000004.msgpack").read_bytes()
    )
    assert state["step"] == 4 and float(state["params"]["weight"]) == 4.0
    assert not list(output.glob("*.tmp-*"))


def test_run_arm_refuses_a_finished_experiment_and_non_finite_updates(
    tmp_path: Path,
) -> None:
    _stub_arm(tmp_path, steps=1, eval_steps={1}, log_every=0, update=_step_update)
    with pytest.raises(FileExistsError):
        prepare_output(tmp_path / "arm")

    def diverging(
        params: dict[str, jax.Array],
        state: None,
        batch: dict[str, jax.Array],
        key: jax.Array,
    ) -> tuple[dict[str, jax.Array], None, dict[str, jax.Array]]:
        return params, state, {"loss": jnp.float32(jnp.nan)}

    with pytest.raises(FloatingPointError):
        _stub_arm(tmp_path / "second", 2, {2}, 0, diverging)


def test_train_read_subset_is_fixed_and_bounded() -> None:
    first = train_read_subset(np.arange(20), 5, seed=3)
    second = train_read_subset(np.arange(20), 5, seed=3)
    np.testing.assert_array_equal(first, second)
    assert len(train_read_subset(np.arange(4), 1663, seed=0)) == 4
    assert len(train_read_subset(np.arange(4), 0, seed=0)) == 0
