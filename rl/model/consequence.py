"""The action-conditioned consequence model (2026-09-20): from the rows the
policy scored the taken action with, predict how the public state rows'
trunk outputs CHANGE by the next request.

Two predictors over the same 16 rows (12 PUBLIC_ENTITY + 3 FIELD +
STATE_VALUE_CLS), both starting at the copy predictor (a zero change):

`PairConsequence` is deterministic and deliberately no more expressive than
`FlatActionReadout`'s own pair form -- a low-rank bilinear of the conditioning
rows against each target row. A conjunction such as "Sleep Talk AND its user
is asleep" therefore has to live IN the move row for this head's loss to
fall, which is where the policy's logit can read it; an MLP over concatenated
rows would compute it privately and shape nothing. The same class under a
CLS-only conditioning is the state-only control.

`ConsequenceSampler` is the stochastic predictor: Gaussian noise enters the
one conditioning vector added to every token, so one pass is one sample
(engression; Shen & Meinshausen 2025). It is trained with the energy score,
which needs no posterior network and no KL term.

The target side is the learner's business (rl/online/training/loss.py): it is
always a stop-gradient of real trunk outputs, never an input here.
"""

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from ml_collections import ConfigDict

from rl.environment.interfaces import ConsequenceInputs, ConsequenceOutput
from rl.model.constants import NUM_PUBLIC_SLOTS, PUBLIC_STATE_ROWS, STATE_VALUE_CLS_ROW
from rl.model.trunk import Trunk

# The rows predicted, as layout rows: the state value row's read set and the
# row itself, so a value can be read off a predicted state with no trunk pass.
CONSEQUENCE_ROWS = np.append(PUBLIC_STATE_ROWS, STATE_VALUE_CLS_ROW)
NUM_CONSEQUENCE_ROWS = len(CONSEQUENCE_ROWS)
# The losses are normalised per kind of row, so a group of many quiet rows
# cannot hide a group of few loud ones.
CONSEQUENCE_GROUPS = ("public_entity", "field", "state_value")
CONSEQUENCE_NOISE_SIZE = 32
CONSEQUENCE_GROUP_IDS = np.array(
    [0] * NUM_PUBLIC_SLOTS + [1] * (len(PUBLIC_STATE_ROWS) - NUM_PUBLIC_SLOTS) + [2]
)


def public_row_alignment(
    order_now: jax.Array, order_next: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Where each of this step's 16 rows sits at the NEXT step, and whether it
    exists there. Public entity rows are ordered actives first, so a switch
    reorders them: they are followed by identity through PUBLIC_ORDER (-1 =
    unrevealed, never matched). Field rows and the value row are fixed slots.
    """
    same = (order_now[:, None] == order_next[None, :]) & (order_now[:, None] >= 0)
    fixed = jnp.arange(NUM_PUBLIC_SLOTS, NUM_CONSEQUENCE_ROWS)
    next_index = jnp.concatenate((jnp.argmax(same, axis=1), fixed))
    matched = jnp.concatenate((same.any(axis=1), jnp.ones(len(fixed), jnp.bool_)))
    return next_index, matched


class PairConsequence(nn.Module):
    """change[j] = out(sum_c q_c(conditioning_c) * k(row_j)): bilinear in
    (conditioning, row_j) per output channel. `out` is the single zero
    factor, over a live product, so it moves at step 1. `conditioning` maps
    a role ("source", "target", "state") to its row; the role names its
    projection."""

    rank: int

    @nn.compact
    def __call__(
        self, state_rows: jax.Array, conditioning: dict[str, jax.Array]
    ) -> jax.Array:
        dtype = state_rows.dtype
        query = 0.0
        for role, row in conditioning.items():
            query = query + nn.Dense(
                self.rank, use_bias=False, dtype=dtype, name=f"{role}_query"
            )(row)
        keys = nn.Dense(self.rank, use_bias=False, dtype=dtype, name="key")(state_rows)
        return nn.Dense(
            state_rows.shape[-1],
            kernel_init=nn.initializers.zeros_init(),
            use_bias=False,
            dtype=dtype,
            name="out",
        )(query[None] * keys)


class ConsequenceSampler(nn.Module):
    """One sample of the change: a small trunk over the 16 current rows, with
    the action, the state summary and the noise entering as one vector added
    to every token. `out_proj` is the single zero factor; `noise` must NOT be
    zero-init, or the two training samples never separate."""

    cfg: ConfigDict

    @nn.compact
    def __call__(
        self,
        state_rows: jax.Array,
        state_valid: jax.Array,
        conditioning: dict[str, jax.Array],
        noise: jax.Array,
    ) -> jax.Array:
        dtype = state_rows.dtype
        width = state_rows.shape[-1]
        vector = nn.Dense(width, use_bias=False, dtype=dtype, name="noise")(
            noise.astype(dtype)
        )
        for role, row in conditioning.items():
            vector = vector + nn.Dense(
                width, use_bias=False, dtype=dtype, name=f"{role}_conditioning"
            )(row)
        tokens = jnp.where(state_valid[:, None], state_rows + vector[None], 0)
        read_mask = np.ones((NUM_CONSEQUENCE_ROWS, NUM_CONSEQUENCE_ROWS), bool)
        mixed = Trunk(self.cfg.trunk, name="trunk")(tokens, state_valid, read_mask)
        return nn.Dense(
            width,
            kernel_init=nn.initializers.zeros_init(),
            use_bias=False,
            dtype=dtype,
            name="out_proj",
        )(mixed)


class ConsequenceModel(nn.Module):
    cfg: ConfigDict

    @nn.compact
    def observable(self, action_features: jax.Array) -> jax.Array:
        from rl.environment.consequence_labels import NUM_OBSERVABLE_LOGITS

        return nn.Dense(
            NUM_OBSERVABLE_LOGITS,
            kernel_init=nn.initializers.zeros_init(),
            dtype=action_features.dtype,
            name="observable_outcomes",
        )(action_features)

    def setup(self):
        self.pair = PairConsequence(self.cfg.rank, name="pair")
        self.state_only_pair = PairConsequence(self.cfg.rank, name="state_only_pair")
        self.sampler = ConsequenceSampler(self.cfg, name="sampler")

    def __call__(
        self, inputs: ConsequenceInputs, noise: jax.Array
    ) -> ConsequenceOutput:
        action = dict(source=inputs.source_row, target=inputs.target_row)
        return ConsequenceOutput(
            mean=self.pair(inputs.state_rows, action),
            # The control is a panel, never a force on the trunk.
            state_only_mean=self.state_only_pair(
                jax.lax.stop_gradient(inputs.state_rows),
                dict(state=jax.lax.stop_gradient(inputs.cls_row)),
            ),
            sample=self.sampler(
                inputs.state_rows,
                inputs.state_valid,
                dict(action, state=inputs.cls_row),
                noise,
            ),
        )
