"""Fast, stub-driven contract tests for targets.compute_intrinsic_targets."""

import jax
import jax.numpy as jnp
import numpy as np

from rl.environment.interfaces import Batch, PlayerEnvOutput, PlayerTransition
from rl.online.config import Porygon2LearnerConfig
from rl.online.training.targets import compute_intrinsic_targets


def _batch(done):
    done = jnp.asarray(done, dtype=jnp.bool_)
    return Batch(
        player_transitions=PlayerTransition(env_output=PlayerEnvOutput(done=done))
    )


def _run(ens_logits, done, int_value=None, int_rms=1.0, **overrides):
    T, B, K, _ = ens_logits.shape
    cfg = Porygon2LearnerConfig(**overrides)
    return compute_intrinsic_targets(
        _batch(done),
        ens_value_logits=jnp.asarray(ens_logits, dtype=jnp.float32),
        int_value=jnp.zeros((T, B)) if int_value is None else jnp.asarray(int_value),
        isr=jnp.ones((T, B)),
        int_rms=jnp.asarray(int_rms, dtype=jnp.float32),
        config=cfg,
    )


def _spread_logits(T, B, K, scale, seed=0):
    rng = np.random.default_rng(seed)
    return scale * rng.standard_normal((T, B, K, 3)).astype(np.float32)


def test_reward_zero_on_terminal_pastdone_and_final_rows():
    T, B, K = 6, 2, 4
    logits = _spread_logits(T, B, K, 3.0)
    # Column 0: done at row 3 (rows 4, 5 past done). Column 1: never done.
    done = np.zeros((T, B), dtype=bool)
    done[3, 0] = True
    out = _run(logits, done)
    r = np.asarray(out.int_reward)
    assert np.isfinite(r).all()
    # Reward for a_t is the spread at s_{t+1}: the step INTO the terminal
    # row earns nothing (its value is the known reward), nor does anything
    # at or after done, nor the chunk's bootstrap-only final row.
    assert r[2, 0] == 0.0 and r[3, 0] == 0.0 and r[4, 0] == 0.0 and r[5, 0] == 0.0
    assert r[5, 1] == 0.0
    assert (r[:2, 0] > 0).all() and (r[:5, 1] > 0).all()
    # ens_std itself is the spread at s_t, masked past done.
    s = np.asarray(out.ens_std)
    assert (s[:4, 0] > 0).all() and (s[4:, 0] == 0).all()


def test_reward_is_the_next_state_spread():
    T, B, K = 4, 1, 5
    logits = _spread_logits(T, B, K, 2.0)
    out = _run(logits, np.zeros((T, B), dtype=bool), int_rms=1.0)
    r = np.asarray(out.int_reward)[:, 0]
    s = np.asarray(out.ens_std)[:, 0]
    np.testing.assert_allclose(r[:-1], s[1:], rtol=1e-5)


def test_rms_normalisation_is_scale_free():
    T, B, K = 5, 3, 4
    logits = _spread_logits(T, B, K, 1.5)
    done = np.zeros((T, B), dtype=bool)
    a = _run(logits, done, int_rms=0.25)
    b = _run(logits, done, int_rms=1.0)
    np.testing.assert_allclose(
        np.asarray(a.int_reward), 2.0 * np.asarray(b.int_reward), rtol=1e-5
    )
    # The RMS update moves toward this batch's mean-square at the configured rate.
    ms = (np.asarray(a.int_reward) * 0.5) ** 2  # back to raw units
    rows = np.ones((T, B))  # every non-done row counts, final row's 0 included
    expect = 0.25 + Porygon2LearnerConfig().player_int_rms_rate * (
        (ms * rows).sum() / rows.sum() - 0.25
    )
    np.testing.assert_allclose(float(a.int_rms_new), expect, rtol=1e-5)


def test_identical_heads_pay_nothing():
    T, B, K = 4, 2, 3
    one = np.random.default_rng(1).standard_normal((T, B, 1, 3)).astype(np.float32)
    logits = np.repeat(one, K, axis=2)
    out = _run(logits, np.zeros((T, B), dtype=bool))
    # float32 std of identical values is ~1e-8, not exactly 0.
    assert np.abs(np.asarray(out.int_reward)).max() < 1e-5
    assert np.abs(np.asarray(out.int_adv)).max() < 1e-5


def test_vtrace_targets_and_advantage_consistency():
    """With V_int = 0 and rho = 1, int_returns is the plain discounted
    lambda-return of r_int and int_adv = r_t + gamma * int_returns_{t+1}."""
    T, B, K = 6, 1, 4
    logits = _spread_logits(T, B, K, 2.0, seed=3)
    done = np.zeros((T, B), dtype=bool)
    cfg = Porygon2LearnerConfig()
    out = _run(logits, done)
    r = np.asarray(out.int_reward)[:, 0]
    g = np.asarray(out.int_returns)[:, 0]
    adv = np.asarray(out.int_adv)[:, 0]
    gamma, lam = cfg.player_int_gamma, cfg.player_lambda
    # v=0 everywhere: errors_t = r_t + gamma*lam*errors_{t+1}.
    expect = np.zeros(T)
    acc = 0.0
    for t in reversed(range(T)):
        acc = r[t] + gamma * lam * acc
        expect[t] = acc
    np.testing.assert_allclose(g, expect, rtol=1e-5, atol=1e-6)
    g_next = np.concatenate([g[1:], [0.0]])
    np.testing.assert_allclose(adv, r + gamma * g_next, rtol=1e-5, atol=1e-6)


def test_prior_twin_receives_no_gradient():
    from ml_collections import ConfigDict

    from rl.model.heads import EnsembleValueLogitHead

    cfg = ConfigDict()
    cfg.num_heads = 3
    cfg.prior_scale = 1.0
    cfg.mlp = ConfigDict()
    cfg.mlp.layer_sizes = (8, 9)
    head = EnsembleValueLogitHead(cfg)
    # A random input: MLP layer-norms its input first, so a constant row
    # would zero every activation and (vacuously) every gradient.
    x = jax.random.normal(jax.random.PRNGKey(1), (4, 6))
    params = head.init(jax.random.PRNGKey(0), x)

    def loss(p):
        return (head.apply(p, x) ** 2).sum()

    grads = jax.grad(loss)(params)
    flat = jax.tree_util.tree_leaves_with_path(grads)
    prior = [g for path, g in flat if "prior" in jax.tree_util.keystr(path)]
    other = [g for path, g in flat if "prior" not in jax.tree_util.keystr(path)]
    assert prior and other
    assert all(float(jnp.abs(g).max()) == 0.0 for g in prior)
    assert any(float(jnp.abs(g).max()) > 0.0 for g in other)
    assert head.apply(params, x).shape == (4, 3, 3)
