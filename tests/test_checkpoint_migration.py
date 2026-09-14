"""Retired auxiliary state is dropped while the training lineage resumes."""

import json
import pickle
from collections import namedtuple
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from rl import checkpoint
from rl.environment import interfaces
from rl.online import artifact
from rl.online.config import Porygon2LearnerConfig
from rl.online.league import MAIN_KEY, PlayerRef
from rl.online.training.league_ops import publish_live_params


@pytest.fixture(autouse=True)
def small_model_capabilities(monkeypatch: pytest.MonkeyPatch) -> None:
    model_config = SimpleNamespace(entity_size=4, num_decision_slots=1)
    monkeypatch.setattr(
        artifact, "get_player_model_config", lambda *args, **kwargs: model_config
    )


def player_parameters(value: float, retired_namespace: str | None = None) -> dict:
    parameters = {
        "params": {
            "encoder": {"history": {"kernel": jnp.full((2,), value)}},
            "action_head": {"kernel": jnp.full((2,), value + 1)},
        }
    }
    if retired_namespace is not None:
        for name in ("pair_value_public", "pair_value_private"):
            parameters["params"][name] = {
                retired_namespace: {"kernel": jnp.full((3,), value + 2)}
            }
    return parameters


def train_states(value: float = 0.0, retired_namespace: str | None = None):
    parameters = player_parameters(value, retired_namespace)
    optimiser = optax.adam(1e-3)
    player = artifact.Porygon2PlayerTrainState.create(
        apply_fn=lambda *args: None,
        init_fn=lambda key: parameters,
        params=parameters,
        reg_params=jax.tree.map(jnp.copy, parameters),
        old_policy_params=jax.tree.map(jnp.copy, parameters),
        tx=optimiser,
    )
    builder_parameters = {"params": {"kernel": jnp.full((2,), value)}}
    builder = artifact.Porygon2BuilderTrainState.create(
        apply_fn=lambda *args: None,
        init_fn=lambda key: builder_parameters,
        params=builder_parameters,
        target_params=jax.tree.map(jnp.copy, builder_parameters),
        tx=optimiser,
    )
    return player, builder


def write_retired_scalars(path: Path, scalars: dict) -> None:
    means_type = namedtuple(
        "PairFeatureMeans",
        "unary_hidden cross_query cross_key synergy_query synergy_key",
        module=interfaces.__name__,
    )
    population_type = namedtuple(
        "PairPopulationState", "means mass", module=interfaces.__name__
    )
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(interfaces, "PairFeatureMeans", means_type, raising=False)
        patch.setattr(interfaces, "PairPopulationState", population_type, raising=False)
        means = means_type(*(np.zeros((2, 2, 3)) for _ in range(5)))
        population = population_type(means, np.full((2, 2), 0.25))
        checkpoint._dump(
            str(path),
            dict(
                scalars,
                pair_population=population,
                target_pair_population=population,
            ),
        )


def write_checkpoint(tmp_path: Path, retired_namespace: str = "population_terms"):
    config = Porygon2LearnerConfig()
    player, builder = train_states(7.0, retired_namespace)
    gradients = jax.tree.map(jnp.ones_like, player.params)
    _, optimiser_state = player.tx.update(gradients, player.opt_state, player.params)
    player = player.replace(
        step_count=48_083,
        frame_count=720_000,
        opt_state=optimiser_state,
        reg_params=player_parameters(9.0, retired_namespace),
        old_policy_params=player_parameters(10.0, retired_namespace),
    )
    builder = builder.replace(step_count=23, frame_count=41)
    league = artifact._init_league(config, player, builder)
    league.add_player(PlayerRef(11, "historical-snapshot", 22, 0))
    league.games[MAIN_KEY, 11] = 12.5
    league.wins[MAIN_KEY, 11] = 7.25
    path = tmp_path / "ckpt_00048083"
    artifact.save_state(str(path), config, player, builder, league, b"controllers")
    checkpoint._dump(
        str(path / "player" / "target_params"),
        player_parameters(8.0, retired_namespace),
    )
    manifest = artifact.read_manifest(str(path))
    if retired_namespace == "population_terms":
        manifest["pair_value_form"] = "population_moments_v1"
        write_retired_scalars(
            path / "player" / "scalars", artifact.player_scalar_components(player)
        )
    else:
        manifest["pair_value_form"] = "local_reference_v1"
        manifest["pair_value_reference"] = "retired-bank-digest"
    (path / artifact.MANIFEST_NAME).write_text(json.dumps(manifest))
    return str(path), config, player, builder


def assert_tree_equal(actual, expected) -> None:
    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    for actual_leaf, expected_leaf in zip(
        jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
    ):
        np.testing.assert_array_equal(actual_leaf, expected_leaf)


@pytest.mark.parametrize("retired_namespace", ("population_terms", "local_terms"))
def test_full_resume_drops_retired_heads_and_preserves_lineage(
    tmp_path: Path, capsys: pytest.CaptureFixture, retired_namespace: str
) -> None:
    path, config, saved_player, saved_builder = write_checkpoint(
        tmp_path, retired_namespace
    )
    fresh_player, fresh_builder = train_states()
    restored, builder, league, controllers = artifact.load_from_checkpoint(
        path, config, fresh_player, fresh_builder
    )
    for component in ("params", "reg_params", "old_policy_params"):
        restored_parameters = getattr(restored, component)
        assert set(restored_parameters["params"]) == {"encoder", "action_head"}
        for name in ("encoder", "action_head"):
            assert_tree_equal(
                restored_parameters["params"][name],
                getattr(saved_player, component)["params"][name],
            )
    restored_adam = restored.opt_state[0]
    for moment in ("mu", "nu"):
        restored_moments = getattr(restored_adam, moment)["params"]
        assert set(restored_moments) == {"encoder", "action_head"}
        for name in ("encoder", "action_head"):
            assert_tree_equal(
                restored_moments[name],
                getattr(saved_player.opt_state[0], moment)["params"][name],
            )
    assert int(restored_adam.count) == 1
    gradients = jax.tree.map(jnp.ones_like, restored.params)
    restored.tx.update(gradients, restored.opt_state, restored.params)
    assert int(restored.step_count) == 48_083
    assert int(restored.frame_count) == 720_000
    assert int(builder.step_count) == int(saved_builder.step_count)
    assert int(builder.frame_count) == int(saved_builder.frame_count)
    assert_tree_equal(builder.params, saved_builder.params)
    assert_tree_equal(builder.target_params, saved_builder.target_params)
    assert not hasattr(restored, "target_params")
    assert_tree_equal(league.get_main_player().player_params, restored.params)
    assert_tree_equal(league.get_main_player().builder_params, builder.params)
    assert league.players[11].snapshot_dir == "historical-snapshot"
    assert league.games[MAIN_KEY, 11] == 12.5
    assert league.wins[MAIN_KEY, 11] == 7.25
    assert controllers == b"controllers"
    assert set(artifact.player_scalar_components(restored)) == {
        "step_count",
        "frame_count",
    }
    assert "pair_value_form" not in artifact._model_capabilities(config)
    output = capsys.readouterr().out
    assert "step_count=48083" in output
    assert "dropped" in output
    assert "pair_value_public" in output
    assert "Ignoring retired player/target_params" in output


def test_removed_record_types_decode_without_live_model_types(tmp_path: Path) -> None:
    path = tmp_path / "scalars"
    write_retired_scalars(path, dict(step_count=123, frame_count=456))
    with pytest.MonkeyPatch.context() as patch:
        patch.delattr(interfaces, "PairFeatureMeans", raising=False)
        patch.delattr(interfaces, "PairPopulationState", raising=False)
        with path.open("rb") as stream:
            with pytest.raises(AttributeError, match="PairPopulationState"):
                pickle.load(stream)
        restored = checkpoint._load(str(path))
    assert restored["step_count"] == 123
    assert restored["frame_count"] == 456
    assert isinstance(restored["pair_population"], tuple)


@pytest.mark.parametrize(
    ("name", "value"),
    (("pi_head", "unknown"), ("generation", 8), ("smogon_format", "ou")),
)
def test_auxiliary_removal_keeps_manifest_checks_strict(
    tmp_path: Path, name: str, value
) -> None:
    path, config, _, _ = write_checkpoint(tmp_path)
    manifest = artifact.read_manifest(path)
    manifest[name] = value
    (Path(path) / artifact.MANIFEST_NAME).write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=name):
        artifact.check_manifest(path, config, strict=True)


def test_params_mode_seeds_separate_reference_and_drops_retired_state(
    tmp_path: Path,
) -> None:
    path, config, saved_player, _ = write_checkpoint(tmp_path)
    fresh_player, fresh_builder = train_states()
    restored, builder, league, controllers = artifact.load_from_params(
        path, config, fresh_player, fresh_builder
    )
    expected = {
        "params": {
            name: saved_player.params["params"][name]
            for name in ("encoder", "action_head")
        }
    }
    for component in ("params", "reg_params", "old_policy_params"):
        assert_tree_equal(getattr(restored, component), expected)
    assert_tree_equal(builder.target_params, builder.params)
    assert not hasattr(restored, "target_params")
    assert int(restored.step_count) == 0
    assert int(restored.opt_state[0].count) == 0
    assert not league.players
    assert controllers is None
    restored = jax.device_put(restored)
    for component in ("reg_params", "old_policy_params"):
        for main_leaf, derived_leaf in zip(
            jax.tree.leaves(restored.params),
            jax.tree.leaves(getattr(restored, component)),
            strict=True,
        ):
            assert (
                main_leaf.unsafe_buffer_pointer()
                != derived_leaf.unsafe_buffer_pointer()
            )


def test_legacy_ema_component_is_not_deserialised(tmp_path: Path) -> None:
    path, config, saved_player, _ = write_checkpoint(tmp_path)
    (Path(path) / "player" / "target_params").write_bytes(b"retired invalid pickle")
    fresh_player, fresh_builder = train_states()
    restored, _, _, _ = artifact.load_from_checkpoint(
        path, config, fresh_player, fresh_builder
    )
    assert_tree_equal(
        restored.reg_params["params"]["encoder"],
        saved_player.reg_params["params"]["encoder"],
    )
    (Path(path) / "player" / "params").write_bytes(b"active invalid pickle")
    with pytest.raises(pickle.UnpicklingError):
        artifact.load_from_checkpoint(path, config, fresh_player, fresh_builder)


@pytest.mark.parametrize("component", ("reg_params", "old_policy_params"))
def test_checkpoint_without_a_derived_tree_seeds_independent_live_parameters(
    tmp_path: Path, capsys: pytest.CaptureFixture, component: str
) -> None:
    path, config, saved_player, _ = write_checkpoint(tmp_path)
    (Path(path) / "player" / component).unlink()
    fresh_player, fresh_builder = train_states()
    restored, _, _, _ = artifact.load_from_checkpoint(
        path, config, fresh_player, fresh_builder
    )
    assert_tree_equal(getattr(restored, component), restored.params)
    # The OTHER derived tree is restored from disk, not re-seeded: the
    # magnet and the target are separate clocks.
    other = {"reg_params": "old_policy_params", "old_policy_params": "reg_params"}[
        component
    ]
    assert_tree_equal(
        getattr(restored, other)["params"]["encoder"],
        getattr(saved_player, other)["params"]["encoder"],
    )
    assert f"Checkpoint predates player/{component}" in capsys.readouterr().out
    restored = jax.device_put(restored)
    for main_leaf, derived_leaf in zip(
        jax.tree.leaves(restored.params),
        jax.tree.leaves(getattr(restored, component)),
        strict=True,
    ):
        assert main_leaf.unsafe_buffer_pointer() != derived_leaf.unsafe_buffer_pointer()


def test_new_checkpoint_has_no_player_ema_and_roundtrips_reference(
    tmp_path: Path,
) -> None:
    config = Porygon2LearnerConfig()
    player, builder = train_states(7.0)
    player = player.replace(
        reg_params=player_parameters(9.0),
        old_policy_params=player_parameters(10.0),
        step_count=100,
    )
    builder = builder.replace(
        target_params={"params": {"kernel": jnp.full((2,), 11.0)}}
    )
    league = artifact._init_league(config, player, builder)
    path = str(tmp_path / "ckpt_00000100")
    artifact.save_state(path, config, player, builder, league)
    assert set((Path(path) / "player").iterdir()) == {
        Path(path) / "player" / component
        for component in (
            "params",
            "reg_params",
            "old_policy_params",
            "opt_state",
            "scalars",
        )
    }
    fresh_player, fresh_builder = train_states()
    restored, restored_builder, restored_league, _ = artifact.load_from_checkpoint(
        path, config, fresh_player, fresh_builder
    )
    assert_tree_equal(restored.params, player.params)
    assert_tree_equal(restored.reg_params, player.reg_params)
    assert_tree_equal(restored.old_policy_params, player.old_policy_params)
    assert_tree_equal(restored_builder.target_params, builder.target_params)
    assert_tree_equal(restored_league.get_main_player().builder_params, builder.params)


def test_publication_uses_live_parameters_without_a_player_ema() -> None:
    player, builder = train_states(7.0)
    player = player.replace(
        reg_params=player_parameters(9.0), step_count=jnp.array(123)
    )
    builder = builder.replace(
        target_params={"params": {"kernel": jnp.full((2,), 11.0)}}
    )
    run_state = SimpleNamespace(player_state=player, builder_state=builder)
    league = artifact._init_league(Porygon2LearnerConfig(), player, builder)
    publish_live_params(run_state, league)
    assert run_state.eval_snapshot.step_count == 123
    assert_tree_equal(run_state.eval_snapshot.main.player_params, player.params)
    assert_tree_equal(run_state.eval_snapshot.main.builder_params, builder.params)
    assert_tree_equal(league.get_main_player().player_params, player.params)
    assert not hasattr(run_state.eval_snapshot, "ema")
    for leaf in jax.tree.leaves(run_state.eval_snapshot.main.player_params):
        assert isinstance(leaf, np.ndarray)


@pytest.mark.parametrize("player_key", ("params", "target_params"))
def test_legacy_league_reference_keeps_its_selected_policy(
    tmp_path: Path, player_key: str
) -> None:
    path, config, player, builder = write_checkpoint(tmp_path)
    league = artifact._init_league(config, player, builder)
    reference = PlayerRef(123, path, 0, 0, player_key=player_key)
    materialised = league.materialize(reference)
    expected = checkpoint.load_component(path, "player", player_key)
    assert_tree_equal(materialised.player_params, expected)


def test_league_reads_new_snapshot_without_player_ema(tmp_path: Path) -> None:
    player, builder = train_states(7.0)
    path = str(tmp_path / "p_00000123")
    checkpoint.save_param_snapshot(
        path,
        player_components={"params": player.params},
        builder_components={"params": builder.params},
    )
    league = artifact._init_league(Porygon2LearnerConfig(), player, builder)
    materialised = league.materialize(PlayerRef(123, path, 0, 0))
    assert_tree_equal(materialised.player_params, player.params)


def test_resume_with_missing_active_scalar_fails_visibly(tmp_path: Path) -> None:
    path, config, saved_player, _ = write_checkpoint(tmp_path)
    scalars = artifact.player_scalar_components(saved_player)
    scalars.pop("step_count")
    checkpoint._dump(str(Path(path) / "player" / "scalars"), scalars)
    fresh_player, fresh_builder = train_states()
    with pytest.raises(KeyError, match="step_count"):
        artifact.load_train_state(
            config, fresh_player, fresh_builder, mode="checkpoint", ckpt_path=path
        )


def test_nonzero_checkpoint_requires_league_unless_explicitly_reset(
    tmp_path: Path,
) -> None:
    path, config, _, _ = write_checkpoint(tmp_path)
    (Path(path) / "league").unlink()
    fresh_player, fresh_builder = train_states()
    with pytest.raises(ValueError, match="missing league state"):
        artifact.load_from_checkpoint(path, config, fresh_player, fresh_builder)
    restored, _, league, _ = artifact.load_from_checkpoint(
        path, config, fresh_player, fresh_builder, reset_league=True
    )
    assert int(restored.step_count) == 48_083
    assert not league.players
