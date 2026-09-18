"""The one offline trainer: with the world model off it trains exactly the
public critic (the inverted reach), with it on the world-model subtree
joins -- the control that proves the flag reaches the parameters, the
optimiser labels and the loss."""

import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rl.model import world_model as wm
from rl.offline import dataset, train
from rl.offline.config import Porygon2OfflineConfig
from rl.offline.shards import check_shard_manifest, list_shards

SOURCE = "replays/shards/gen9randombattle"


def _params(world_model: bool) -> dict:
    params = {
        "encoder": {"kernel": jnp.zeros(2)},
        "public_value_head": {"kernel": jnp.zeros(2)},
    }
    if world_model:
        params["world_model"] = {
            "flow": {"kernel": jnp.zeros(2)},
            "delta_scale": jnp.ones(wm.NUM_PUBLIC_GROUPS),
        }
    return params


def test_param_labels_follow_the_params_tree_and_joint() -> None:
    labels = train.param_labels(_params(False), joint=False)
    assert labels == {
        "encoder": {"kernel": "frozen"},
        "public_value_head": {"kernel": "train"},
    }
    labels = train.param_labels(_params(True), joint=True)
    assert labels["encoder"] == {"kernel": "train"}
    assert labels["world_model"] == {
        "flow": {"kernel": "train"},
        "delta_scale": "frozen",
    }


def test_total_loss_is_the_critic_alone_with_the_world_model_off() -> None:
    metrics = {"loss_public_value": jnp.asarray(0.7)}
    config = Porygon2OfflineConfig(world_model=False, public_value_loss_weight=2.0)
    assert float(train.total_loss(config, metrics)) == pytest.approx(1.4)
    on = Porygon2OfflineConfig(world_model=True)
    assert set(train.loss_weights(on)) == {
        "loss_kind",
        "loss_actor",
        "loss_move",
        "loss_target",
        "loss_touched",
        "loss_flow",
        "loss_mean_all_rows",
        "loss_terminal",
        "loss_public_value",
    }
    with pytest.raises(KeyError):
        train.total_loss(on, metrics)


def _two_trajectory_batch(config: Porygon2OfflineConfig) -> dataset.ReplayBatch:
    shard = list_shards(SOURCE)[0]
    part = dataset._decode_range((shard, 0, 1, 0, config.holdout_modulus))
    store = dataset.ReplayStore(config, [part], check_shard_manifest(SOURCE))
    return dataset.collate(
        [store.example(0), store.example(1)], config.min_history_length
    )


def _init_and_terms(config: Porygon2OfflineConfig, batch: dataset.ReplayBatch):
    model = train.OfflineTrainer(
        train.trainer_model_config(config.joint, config.world_model)
    )
    scale = jnp.ones(wm.NUM_PUBLIC_GROUPS, jnp.float32)
    first = jax.tree.map(lambda x: jnp.asarray(x[0]), batch)
    params = jax.jit(
        lambda key, trajectory: model.init(
            key,
            trajectory,
            scale,
            key,
            False,
            0,
            method=train.OfflineTrainer.trajectory_terms,
        )
    )(jax.random.key(0), first)["params"]
    floor = model.cfg.world_model.scale_floor

    @jax.jit
    def terms(params, batch):
        pooled = train.batch_terms(
            model, params, batch, scale, jax.random.key(1), False, 0
        )
        metrics = train.pooled_metrics(pooled, scale, floor)
        return metrics, train.total_loss(config, metrics)

    metrics, total = terms(params, jax.tree.map(jnp.asarray, batch))
    return params, jax.device_get(metrics), float(total)


@pytest.mark.slow
@pytest.mark.gpu
@pytest.mark.skipif(not os.path.isdir(SOURCE), reason="replay shards not on this box")
def test_critic_only_reaches_public_v_head_alone() -> None:
    config = Porygon2OfflineConfig(
        batch_size=2, max_history_steps=64, min_history_length=64, world_model=False
    )
    batch = _two_trajectory_batch(config)
    params, metrics, total = _init_and_terms(config, batch)
    assert "world_model" not in params
    labels = train.param_labels(params, joint=False)
    assert set(jax.tree.leaves(labels["public_value_head"])) == {"train"}
    assert set(jax.tree.leaves(labels["encoder"])) == {"frozen"}
    assert "loss_flow" not in metrics
    assert np.isfinite(total)
    assert total == pytest.approx(
        config.public_value_loss_weight * float(metrics["loss_public_value"])
    )
    # Control: the world model on adds its subtree, its labels and its loss,
    # and the critic's term does not move.
    on = Porygon2OfflineConfig(
        batch_size=2, max_history_steps=64, min_history_length=64, world_model=True
    )
    params_on, metrics_on, _ = _init_and_terms(on, batch)
    assert "world_model" in params_on
    labels_on = train.param_labels(params_on, joint=False)
    assert "train" in set(jax.tree.leaves(labels_on["world_model"]))
    assert labels_on["world_model"]["delta_scale"] == "frozen"
    assert "loss_flow" in metrics_on
    # bf16 forward: the two arms are the same program over the same
    # parameters (a leaf-by-leaf check finds no difference), but the extra
    # world-model rows change what XLA autotunes, and that lands under
    # bf16's ulp (3.9e-3), not under 1e-5.
    np.testing.assert_allclose(
        metrics_on["loss_public_value"], metrics["loss_public_value"], rtol=5e-3
    )
