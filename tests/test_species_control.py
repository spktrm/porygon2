"""The species-only matched control beside the belief head (2026-09-02).

Two contracts: its CE trains its own table and NOTHING else (so it cannot
touch the lineage it controls for), and it reads the PUBLIC row's species
token -- the disguise under Illusion -- never the private truth.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from rl.environment.protos.features_pb2 import (
    EntityPrivateNodeFeature,
    EntityRevealedNodeFeature,
)

pytestmark = [pytest.mark.gpu, pytest.mark.slow]


def _with_column(actor_input, leaf, row, column, value):
    import dataclasses

    env = actor_input.env
    table = jnp.asarray(getattr(env, leaf)).at[:, row, column].set(value)
    return dataclasses.replace(
        actor_input, env=dataclasses.replace(env, **{leaf: table})
    )


def test_species_control_gradient_stays_in_its_table(real_model_and_trajectory):
    from rl.model.heads import HeadParams

    network, params, actor_input, actor_output = real_model_and_trajectory

    def species_ce(params):
        out = network.apply(params, actor_input, actor_output, HeadParams())
        labels = jax.lax.stop_gradient(out.opp_code.astype(jnp.float32))
        ce = optax.softmax_cross_entropy(
            logits=out.species_belief_logits.astype(jnp.float32), labels=labels
        )
        return jnp.mean(ce)

    grads = jax.jit(jax.grad(species_ce))(params)
    flat = jax.tree_util.tree_leaves_with_path(grads)
    for path, leaf in flat:
        keys = tuple(entry.key for entry in path)
        if "species_belief" in keys:
            assert float(jnp.abs(leaf).max()) > 0.0, "control: the table trains"
        else:
            assert float(jnp.abs(leaf).max()) == 0.0, keys


def test_species_control_reads_the_public_species_only(
    real_model_and_trajectory, real_model_apply
):
    from rl.model.heads import HeadParams

    network, params, actor_input, actor_output = real_model_and_trajectory
    base = real_model_apply(params, actor_input, actor_output, HeadParams())
    matched = np.asarray(base.belief_matched)
    step, mon = np.argwhere(matched)[0]
    from rl.model.player_model import belief_alignment

    _, public_row_index = belief_alignment(
        actor_input.env.opp_private_team[step], actor_input.env.info[step]
    )
    public_row = int(public_row_index[mon])

    species_column = EntityRevealedNodeFeature.ENTITY_REVEALED_NODE_FEATURE__SPECIES
    current = int(actor_input.env.revealed_team[step, public_row, species_column])
    public_flipped = _with_column(
        actor_input, "revealed_team", public_row, species_column, current + 1
    )
    moved = real_model_apply(params, public_flipped, actor_output, HeadParams())
    assert not np.array_equal(
        np.asarray(base.species_belief_logits[step, mon], dtype=np.float32),
        np.asarray(moved.species_belief_logits[step, mon], dtype=np.float32),
    )

    private_column = EntityPrivateNodeFeature.ENTITY_PRIVATE_NODE_FEATURE__SPECIES
    private_current = int(actor_input.env.opp_private_team[step, mon, private_column])
    private_flipped = _with_column(
        actor_input, "opp_private_team", mon, private_column, private_current + 1
    )
    unmoved = real_model_apply(params, private_flipped, actor_output, HeadParams())
    np.testing.assert_array_equal(
        np.asarray(base.species_belief_logits, dtype=np.float32),
        np.asarray(unmoved.species_belief_logits, dtype=np.float32),
    )
    # Control: the private species DOES reach the code the control is
    # scored against, so the invariance above is not vacuous.
    assert not np.array_equal(
        np.asarray(base.opp_code[step, mon], dtype=np.float32),
        np.asarray(unmoved.opp_code[step, mon], dtype=np.float32),
    )
