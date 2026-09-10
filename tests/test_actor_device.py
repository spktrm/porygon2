"""player_actor_device (2026-09-03): the direct Agent path and the
InferenceServer pad a request to the SAME joint history bucket shape — one
function, two call sites, both pinned here — and DeviceParamsCache is keyed
by container IDENTITY, so one host container is one device copy however
many actor threads read it, and two containers that share step and frame
counts (the eval thread's main and EMA pair) never alias."""

import threading

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from constants import NUM_HISTORY
from rl.environment.interfaces import PlayerActorInput, PlayerActorOutput
from rl.environment.utils import (
    ACTOR_HISTORY_MIN_LENGTH,
    _bucket_value,
    get_ex_player_step,
    joint_history_level,
    pad_history_to_level,
)
from rl.model.heads import HeadParams
from rl.model.utils import ParamsContainer
from rl.online.agent import DeviceParamsCache
from rl.online.inference import InferenceServer, _InferenceRequest


@pytest.fixture(scope="module")
def full_window() -> PlayerActorInput:
    actor_input, _ = get_ex_player_step()
    return jax.tree.map(lambda x: np.asarray(x[:, 0]), actor_input)


def _raw_lengths(
    actor_input: PlayerActorInput, history_len: int, packed_len: int
) -> PlayerActorInput:
    """A request at an UNBUCKETED shape (what an actor sends): the window's
    leading axes cut to the given lengths, content irrelevant here."""
    return actor_input.replace(
        history=jax.tree.map(lambda x: x[:history_len], actor_input.history),
        packed_history=jax.tree.map(
            lambda x: x[:packed_len], actor_input.packed_history
        ),
    )


def _shapes(actor_input: PlayerActorInput) -> tuple[int, int]:
    return (
        actor_input.history.field.shape[0],
        actor_input.packed_history.revealed_cache.shape[0],
    )


def _shape_probe_apply(
    params: dict,
    actor_input: PlayerActorInput,
    _placeholder: PlayerActorOutput,
    head_params: HeadParams,
    rngs: dict[str, jax.Array],
) -> jax.Array:
    # Stands in for the network: reports the shape the forward SAW, with
    # the [T=1] axis _run_group squeezes off each request's output.
    return jnp.asarray(_shapes(actor_input))[None]


def _container(player_params: dict) -> ParamsContainer:
    return ParamsContainer(
        step_count=0,
        player_frame_count=0,
        builder_frame_count=0,
        player_params=player_params,
        builder_params=None,
    )


@pytest.mark.parametrize(
    "history_len, packed_len",
    [
        (5, 9),  # both inside the smallest bucket
        (40, 70),  # history at level 1, packed at level 2 -> joint 2
        (33, 200),  # the packed axis alone drives the level
        (300, 40),  # the history axis alone drives the level
    ],
)
def test_direct_path_pads_to_the_servers_group_shape(
    full_window: PlayerActorInput, history_len: int, packed_len: int
) -> None:
    request = _raw_lengths(full_window, history_len, packed_len)
    level = joint_history_level(request, ACTOR_HISTORY_MIN_LENGTH)
    direct = _shapes(pad_history_to_level(request, level, ACTOR_HISTORY_MIN_LENGTH))
    # One joint level for both axes: the axis that did not drive the level
    # is padded UP to it, never to its own smaller bucket.
    assert direct == (
        _bucket_value(level, ACTOR_HISTORY_MIN_LENGTH, NUM_HISTORY),
        _bucket_value(level, ACTOR_HISTORY_MIN_LENGTH, 2 * NUM_HISTORY),
    )
    assert direct[0] >= history_len and direct[1] >= packed_len

    server = InferenceServer(player_apply_fn=_shape_probe_apply)
    served = _InferenceRequest(
        container=_container({}),
        rng_key=jax.random.key(0),
        actor_input=request.replace(
            env=jax.tree.map(lambda x: np.asarray(x)[0], request.env)
        ),
        done=threading.Event(),
    )
    server._run_group([served])
    assert tuple(int(x) for x in served.output.actor_output) == direct


def test_joint_level_never_splits_the_axes(full_window: PlayerActorInput) -> None:
    # The property the pad exists for: a request compiles one variant per
    # LEVEL, so two requests at the same joint level land on one shape even
    # when their raw axes fall in different per-axis buckets.
    same_level = [
        _raw_lengths(full_window, 5, 70),
        _raw_lengths(full_window, 70, 5),
        _raw_lengths(full_window, 100, 128),
    ]
    padded = {
        _shapes(
            pad_history_to_level(
                request,
                joint_history_level(request, ACTOR_HISTORY_MIN_LENGTH),
                ACTOR_HISTORY_MIN_LENGTH,
            )
        )
        for request in same_level
    }
    assert len(padded) == 1


def test_params_cache_is_one_copy_per_container() -> None:
    device = jax.devices("cpu")[0]
    cache = DeviceParamsCache(device, "player_params", size=2)
    first = _container({"w": np.ones(3, np.float32)})
    twin = _container({"w": np.ones(3, np.float32)})  # same counts, new object
    first_params = cache.get(first)
    assert cache.get(first) is first_params  # a hit is the SAME device copy
    assert first_params["w"].devices() == {device}
    twin_params = cache.get(twin)
    assert twin_params is not first_params  # identity, not (step, frame) key
    assert len(cache) == 2


def test_params_cache_evicts_least_recently_used() -> None:
    cache = DeviceParamsCache(jax.devices("cpu")[0], "player_params", size=2)
    first, second, third = (
        _container({"w": np.full(2, fill, np.float32)}) for fill in (1.0, 2.0, 3.0)
    )
    first_params = cache.get(first)
    cache.get(second)
    cache.get(first)  # refresh: `second` is now the least recent
    cache.get(third)
    assert len(cache) == 2
    assert cache.get(first) is first_params  # survived the eviction
    # `second` was evicted: its next read is a fresh transfer, not the old copy.
    second_again = cache.get(second)
    assert len(cache) == 2
    assert float(second_again["w"][0]) == 2.0
    assert cache.get(second) is second_again


def test_actor_parameter_view_preserves_required_branches() -> None:
    from rl.model.player_model import actor_params_view

    branches = {
        name: {"weight": np.ones(2)}
        for name in (
            "encoder",
            "action_head",
            "v_head",
            "transition",
            "dynamics_delta_head",
        )
    }
    variables = {"params": branches}
    plain = actor_params_view(variables)
    search = actor_params_view(variables, search=True)
    assert set(plain["params"]) == {"encoder", "action_head", "v_head"}
    assert search["params"]["transition"] is branches["transition"]
    assert plain["params"]["encoder"] is branches["encoder"]
    assert "dynamics_delta_head" in variables["params"]
    branches["slot_conditioning"] = {"weight": np.ones(2)}
    assert (
        actor_params_view(variables)["params"]["slot_conditioning"]
        is branches["slot_conditioning"]
    )
    with pytest.raises(KeyError):
        actor_params_view({"params": {"encoder": {}}})
    with pytest.raises(KeyError):
        actor_params_view(plain, search=True)


def test_actor_jit_shared_across_agents_and_historical_params(
    full_window: PlayerActorInput,
) -> None:
    from rl.model.heads import HeadParams
    from rl.model.player_model import actor_params_view
    from rl.online.agent import Agent

    traces = []

    def apply_probe(
        params: dict,
        actor_input: PlayerActorInput,
        placeholder: PlayerActorOutput,
        head_params: HeadParams,
        rngs: dict[str, jax.Array],
    ) -> jax.Array:
        traces.append(1)
        return (params["params"]["v_head"]["weight"] * head_params.temp)[None]

    request = _raw_lengths(full_window, 5, 9)
    agents = [
        Agent(
            apply_probe,
            device=jax.devices("cpu")[0],
            player_params_view=actor_params_view,
            player_head_params=HeadParams(temp=temperature),
        )
        for temperature in (1.0, 0.5)
    ]
    for index, agent in enumerate(agents):
        variables = {
            "params": {
                name: {"weight": np.ones(2, np.float32)}
                for name in ("encoder", "action_head", "v_head")
            }
        }
        variables["params"]["v_head"]["weight"] *= index + 2
        variables["params"][f"retired_{index}"] = {"unused": np.ones(index + 1)}
        result = agent.step_player(jax.random.key(0), _container(variables), request)
        np.testing.assert_array_equal(
            result.actor_output, (index + 2) * agent.player_head_params.temp
        )
    assert len(traces) == 1
    # Positive control: a genuinely different required parameter shape retraces.
    variables["params"]["v_head"]["weight"] = np.ones(3, np.float32)
    agents[0].step_player(jax.random.key(0), _container(variables), request)
    assert len(traces) == 2


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("search", [False, True])
def test_real_actor_parameter_view_is_bit_identical(
    real_model_and_trajectory: tuple[
        nn.Module, dict, PlayerActorInput, PlayerActorOutput
    ],
    search: bool,
) -> None:
    from rl.model.config import get_player_model_config
    from rl.model.heads import HeadParams
    from rl.model.player_model import actor_params_view, get_player_model
    from rl.model.utils import open_zero_init_paths

    _, variables, actor_input, actor_output = real_model_and_trajectory
    variables = open_zero_init_paths(variables, ["action_head", "dynamics_out_proj"])
    actor_input = actor_input.replace(
        env=jax.tree.map(lambda leaf: leaf[:1], actor_input.env)
    )
    actor_output = jax.tree.map(lambda leaf: leaf[:1], actor_output)
    config = get_player_model_config(9, train=False)
    config.search.enabled = search
    config.search.depth = 1
    apply_model = jax.jit(get_player_model(config).apply)
    arguments = (actor_input, actor_output, HeadParams())
    rngs = {"sampling": jax.random.key(9)}
    full = apply_model(variables, *arguments, rngs=rngs)
    projected = apply_model(
        actor_params_view(variables, search=search), *arguments, rngs=rngs
    )
    for expected, actual in zip(
        jax.tree.leaves(full), jax.tree.leaves(projected), strict=True
    ):
        np.testing.assert_array_equal(expected, actual)
    assert np.any(np.asarray(full.action_head.entropy) > 0)
    if search:
        assert np.any(np.asarray(full.search.root_kl) > 0)
