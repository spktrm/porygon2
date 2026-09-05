"""The mixed-precision contract, pinned for VRAM auditing.

Two properties, and they are different things (conflating them wasted a
measurement on 2026-08-25):

  param_dtype  what parameters are STORED in. f32, always. Optimiser state
               and the EMA target track it, so this is most of the resident
               VRAM floor and it is not something to economise on.
  dtype        what the forward COMPUTES in. bf16, inherited from
               cfg.dtype, everywhere except the few places precision is
               paid for.

The failure mode this catches is silent and easy: a `nn.Dense`/`LayerNorm`/
`Embed` written without `dtype=` does not error — Flax promotes from
inputs and params, so a bf16 input meets an f32 param and the ACTIVATION
comes out f32. Every downstream layer then faithfully inherits the upcast
via `dtype=x.dtype`, so one omission at the top silently doubles the
activation memory of everything after it.

Uses `jax.eval_shape`, so this is abstract — no kernels, no GPU, no
checkpoint. It belongs in the fast suite precisely because the regression
it catches is invisible in any test that only checks values.
"""

import jax
import numpy as np
import pytest
from flax.traverse_util import flatten_dict

# Paths whose captured output is allowed to be f32, each with the reason.
# A NEW f32 activation fails the test and names itself; a path here that has
# become bf16 also fails, so the list cannot rot into a blanket exemption.
F32_ALLOWED = {
    "action_head/__call__": (
        "the action grid is cast f32 ONCE, at the readout's output, before "
        "the masked log-softmax. bf16 log_softmax normalisation holds only to "
        "~3e-3 and every term of the policy loss reads this array. The two "
        "entries this replaced were the macro/micro RMS gauges, which retired "
        "with the hierarchical head on 2026-08-29."
    ),
    "transition/mask_head/__call__": (
        "the same readout form instantiated a second time on the imagined "
        "rows (2026-09-05): its 295 cell logits are the next-action-mask BCE's "
        "input, cast f32 at the readout's output for the same reason."
    ),
}


def _abstract_forward():
    """(params, captured intermediates) without running a single kernel."""
    from rl.environment.utils import get_ex_player_step
    from rl.model.config import get_player_model_config
    from rl.model.heads import HeadParams
    from rl.model.player_model import get_player_model

    actor_input, actor_output = jax.tree.map(lambda x: x[:, 0], get_ex_player_step())
    net = get_player_model(get_player_model_config(9, train=True))
    params = jax.eval_shape(
        lambda: net.init(jax.random.PRNGKey(0), actor_input, actor_output, HeadParams())
    )
    _, state = jax.eval_shape(
        lambda p: net.apply(
            p,
            actor_input,
            actor_output,
            HeadParams(),
            capture_intermediates=True,
        ),
        params,
    )
    return params, state.get("intermediates", {})


@pytest.fixture(scope="module")
def abstract_forward():
    """Shared: the abstract trace is the only expensive part of this file."""
    return _abstract_forward()


def test_params_are_stored_in_f32(abstract_forward):
    params, _ = abstract_forward
    leaves = jax.tree.leaves(params)
    assert leaves, "captured no params — the probe is broken, not the model"
    offenders = {str(leaf.dtype) for leaf in leaves} - {"float32"}
    assert not offenders, f"params must be stored f32, found {offenders}"


def test_forward_computes_in_bf16_except_where_precision_is_paid_for(
    abstract_forward,
):
    _, intermediates = abstract_forward

    bf16, f32 = [], []
    for path, value in flatten_dict(intermediates).items():
        if not isinstance(value, tuple):
            value = (value,)
        for arr in value:
            if not hasattr(arr, "dtype"):
                continue
            name = "/".join(map(str, path))
            if arr.dtype == np.float32:
                f32.append(name)
            else:
                bf16.append(name)

    # Guards the whole test against capturing nothing and passing vacuously.
    # The bar was >50 while the model unpacked every entity into 10-11
    # attribute tokens through separately-sown modules; one row per entity
    # and one trunk sows far fewer, so it is sized to the model that exists.
    assert len(bf16) > 15, f"only {len(bf16)} bf16 activations — probe broken?"

    unexpected = sorted(set(f32) - set(F32_ALLOWED))
    assert not unexpected, (
        "these activations are f32 and should inherit bf16 from cfg.dtype — a "
        "layer written without dtype= promotes against its f32 params and every "
        "downstream layer inherits the upcast:\n  " + "\n  ".join(unexpected)
    )

    # The other direction: an entry that is no longer f32 must be removed, or
    # the allowlist silently becomes a blanket exemption for its path.
    stale = sorted(set(F32_ALLOWED) - set(f32))
    assert (
        not stale
    ), "allowlisted as f32 but no longer f32 — delete the entry:\n  " + "\n  ".join(
        stale
    )
