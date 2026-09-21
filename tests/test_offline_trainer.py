"""The one offline trainer trains exactly the public critic: the parameters
it builds, the optimiser labels and the loss."""

import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rl.offline import dataset, train
from rl.offline.config import Porygon2OfflineConfig
from rl.offline.shards import check_shard_manifest, list_shards

SOURCE = "replays/shards/gen9randombattle"


def _params() -> dict:
    return {
        "encoder": {"kernel": jnp.zeros(2)},
        "public_value_head": {"kernel": jnp.zeros(2)},
    }


def test_param_labels_follow_the_params_tree_and_joint() -> None:
    labels = train.param_labels(_params(), joint=False)
    assert labels == {
        "encoder": {"kernel": "frozen"},
        "public_value_head": {"kernel": "train"},
    }
    # The control: joint is what reaches the encoder.
    labels = train.param_labels(_params(), joint=True)
    assert labels["encoder"] == {"kernel": "train"}


def test_total_loss_is_the_weighted_critic_term() -> None:
    metrics = {"loss_public_value": jnp.asarray(0.7)}
    config = Porygon2OfflineConfig(public_value_loss_weight=2.0)
    assert float(train.total_loss(config, metrics)) == pytest.approx(1.4)
    assert set(train.loss_weights(config)) == {"loss_public_value"}


def _two_trajectory_batch(config: Porygon2OfflineConfig) -> dataset.ReplayBatch:
    shard = list_shards(SOURCE)[0]
    part = dataset._decode_range((shard, 0, 1, 0, config.holdout_modulus))
    store = dataset.ReplayStore(config, [part], check_shard_manifest(SOURCE))
    return dataset.collate(
        [store.example(0), store.example(1)], config.min_history_length
    )


def _init_and_terms(config: Porygon2OfflineConfig, batch: dataset.ReplayBatch):
    model = train.OfflineTrainer(train.trainer_model_config(), joint=config.joint)
    first = jax.tree.map(lambda x: jnp.asarray(x[0]), batch)
    params = jax.jit(
        lambda key, trajectory: model.init(
            key, trajectory, method=train.OfflineTrainer.trajectory_terms
        )
    )(jax.random.key(0), first)["params"]

    @jax.jit
    def terms(params, batch):
        metrics = train.pooled_metrics(train.batch_terms(model, params, batch))
        return metrics, train.total_loss(config, metrics)

    metrics, total = terms(params, jax.tree.map(jnp.asarray, batch))
    return params, jax.device_get(metrics), float(total)


@pytest.mark.slow
@pytest.mark.gpu
@pytest.mark.skipif(not os.path.isdir(SOURCE), reason="replay shards not on this box")
def test_the_trainer_reaches_public_v_head_alone() -> None:
    config = Porygon2OfflineConfig(
        batch_size=2, max_history_steps=64, min_history_length=64
    )
    batch = _two_trajectory_batch(config)
    params, metrics, total = _init_and_terms(config, batch)
    assert set(params) == {"encoder", "public_value_head"}
    labels = train.param_labels(params, joint=False)
    assert set(jax.tree.leaves(labels["public_value_head"])) == {"train"}
    assert set(jax.tree.leaves(labels["encoder"])) == {"frozen"}
    assert np.isfinite(total)
    assert total == pytest.approx(
        config.public_value_loss_weight * float(metrics["loss_public_value"])
    )
