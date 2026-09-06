"""How many prior samples does depth-one search need, and is there
anything to find? (2026-09-06, Step 3's first read.)

For ~500 root states from self-play on a checkpoint, every legal cell's
imagined value is drawn `--samples` times from the transition PRIOR (the
live search draws 8). The draws are split: the second half is the
REFERENCE Q(a) = E_z[V(g(h, a, z))], the first half supplies the n-sample
estimates the live search would act on. Reported per root:

- the spread of Q across legal actions (what search could act on) against
  the spread of V across z for one action (what a sample is noisy by);
  their ratio at n=8 is the signal-to-noise of the live operator;
- how often the n-sample argmax is the reference argmax, Kendall's tau of
  the two rankings, and the value the reference model assigns to acting
  on the n-sample estimate against acting on pi -- the model's OWN
  account of the improvement, at each n;
- the root KL the live operator (temp 0.1) implies at each n, beside the
  0.015-0.04 the eval slot reads;
- the switch axis: Q on switch cells against move cells (mean over the
  modality, and best-vs-best), and the switch mass under pi and under the
  searched policy.

`--calibration` (Step 3b's pre-fix baseline) adds, per root that has a
real next request: the imagined CHANGE in value against the real change
(`value_delta_r2`, R^2 -- the copy predictor scores exactly 0), and the
value CE improvement over the copy baseline scaled so copy = 0 and the
real next state = 1 (`value_gain`). Both from the POSTERIOR decode (the
learner's read) and from the prior MODE (the rollout's). The offline
label is the game's Monte Carlo OUTCOME one-hot over CAT_VF_SUPPORT, not
the learner's v-trace two-hot -- more variance, same sign.

    PS_SERVICE_URI=ws://localhost:8081 env/bin/python \\
        rl/offline/search_samples_probe.py --ckpt ckpts/gen9/ckpt_01560000
"""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.12")

import argparse  # noqa: E402
import logging  # noqa: E402
from dataclasses import dataclass  # noqa: E402

import flax.linen as nn  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import kendalltau  # noqa: E402

from rl.environment.data import CAT_VF_SUPPORT, CELL_MODALITY_MASK  # noqa: E402
from rl.environment.protos.service_pb2 import ModalityEnum  # noqa: E402
from rl.model.config import get_player_model_config  # noqa: E402
from rl.model.constants import (  # noqa: E402
    CLS_ROW,
    MOVE_ROWS,
    POLICY_READABLE_ROWS,
    PRIVATE_ROWS,
    SEQUENCE_READ_MASK,
    TARGET_ROWS,
)
from rl.model.encoder import Encoder  # noqa: E402
from rl.model.player_model import get_player_model  # noqa: E402
from rl.model.transition import unimix_probs  # noqa: E402
from rl.offline import harness  # noqa: E402
from rl.offline.separation_probe import actor_input_of  # noqa: E402
from rl.offline.trunk_homogeneity import valid_steps  # noqa: E402
from rl.online.training.batching import stack_batch  # noqa: E402

SWITCH = ModalityEnum.MODALITY_ENUM__SWITCH
IS_SWITCH_CELL = np.asarray(CELL_MODALITY_MASK) == SWITCH
LIVE_TEMP = 0.1
SAMPLE_COUNTS = (1, 8, 32, 64)


def _encode(module, actor_input, actor_output):
    """One chunk -> the post-trunk policy-readable rows (T, 73, D) and
    their validity, exactly what the live search reads."""
    encoder = module.encoder
    env = actor_input.env
    *history_inputs, _ = encoder._history_inputs(
        env, actor_input.packed_history, actor_input.history
    )
    assemble = nn.vmap(
        Encoder._assemble_sequence,
        variable_axes={"params": None},
        split_rngs={"params": False},
        in_axes=0,
        out_axes=0,
    )
    sequence, row_valid, _, _ = assemble(encoder, env, *history_inputs)
    kept = encoder.kept_rows()
    read_mask = SEQUENCE_READ_MASK[np.ix_(kept, kept)]
    trunk_out = jax.vmap(lambda seq, ok: encoder.trunk(seq, ok, read_mask))(
        sequence, row_valid
    )
    return trunk_out[:, POLICY_READABLE_ROWS], row_valid[:, POLICY_READABLE_ROWS]


def _root_values(module, rows, row_valid, legal, rng, num_samples, max_cells):
    """`depth_one_expectimax`'s body with the per-sample values kept:
    values (N, C) over the first `max_cells` legal cells, plus pi over
    those cells, the prior's mode mass per cell and V at the root."""
    transition = module.transition
    cells = jnp.nonzero(legal, size=max_cells, fill_value=0)[0]
    cell_valid = jnp.arange(max_cells) < legal.sum()
    prior_logits = jax.vmap(transition.prior, (None, None, 0))(rows, row_valid, cells)
    code_probs = unimix_probs(prior_logits)
    samples = jax.random.categorical(
        rng,
        jnp.log(code_probs),
        axis=-1,
        shape=(num_samples, *code_probs.shape[:-1]),
    )
    code_one_hot = jax.nn.one_hot(samples, code_probs.shape[-1], dtype=jnp.float32)
    src_rows, tgt_rows = jax.vmap(transition.action_rows, (None, 0))(rows, cells)
    imagine_cells = jax.vmap(transition.imagine, (None, None, 0, 0, 0))
    imagined = jax.vmap(imagine_cells, (None, None, None, None, 0))(
        rows, row_valid, src_rows, tgt_rows, code_one_hot
    )
    values = module.v_head(imagined[:, :, CLS_ROW]).expectation
    base_logits = module._legal_logits(
        (rows[PRIVATE_ROWS], rows[MOVE_ROWS], rows[TARGET_ROWS]), legal, 1.0
    )
    log_pi = jax.nn.log_softmax(base_logits.astype(jnp.float32))[cells]
    return {
        "values": values.astype(jnp.float32),
        "cell_valid": cell_valid,
        "cells": cells,
        "log_pi": log_pi,
        "mode_mass": jnp.prod(code_probs.max(-1), axis=-1),
        "root_value": module.v_head(rows[CLS_ROW]).expectation.astype(jnp.float32),
    }


@dataclass
class Root:
    values: np.ndarray  # (N, k)
    log_pi: np.ndarray  # (k,)
    is_switch: np.ndarray  # (k,)
    mode_mass: np.ndarray  # (k,)
    root_value: float


def _calibration(
    module, rows, row_valid, action, next_rows, next_valid, rng, num_samples
):
    """V on the real root, the real next request, the posterior decode, the
    prior-MODE decode (the learner's `_prior` panels) and the prior
    EXPECTATION decode: `num_samples` codes drawn from the transition prior
    at the TAKEN action, each decoded with `imagine`, V averaged over z --
    the number search reads (`Q(a) = E_z[V(g(h, a, z))]`). `sample` is the
    first draw alone (one rollout branch); `expect_sigma_z` the std of V
    over the draws. Expectations and f32 logits, one transition."""
    transition = module.transition
    out = transition._step(rows, row_valid, action, next_rows, next_valid, None)
    prior_logits = transition.prior(rows, row_valid, action)
    code_probs = unimix_probs(prior_logits)
    samples = jax.random.categorical(
        rng, jnp.log(code_probs), axis=-1, shape=(num_samples, *code_probs.shape[:-1])
    )
    code_one_hot = jax.nn.one_hot(samples, code_probs.shape[-1], dtype=jnp.float32)
    src_row, tgt_row = transition.action_rows(rows, action)
    imagined = jax.vmap(transition.imagine, (None, None, None, None, 0))(
        rows, row_valid, src_row, tgt_row, code_one_hot
    )
    sampled = module.v_head(imagined[:, CLS_ROW])
    sampled_v = sampled.expectation.astype(jnp.float32)
    sampled_probs = jax.nn.softmax(sampled.logits.astype(jnp.float32), axis=-1)
    reads = {
        "root": rows[CLS_ROW],
        "real": next_rows[CLS_ROW],
        "post": out.pred[CLS_ROW],
        "prior": out.pred_prior[CLS_ROW],
    }
    values = {}
    for name, cls in reads.items():
        head = module.v_head(cls)
        values[f"{name}_v"] = head.expectation
        values[f"{name}_logits"] = head.logits
    values["expect_v"] = sampled_v.mean(0)
    values["expect_logits"] = jnp.log(sampled_probs.mean(0) + 1e-8)
    values["expect_sigma_z"] = sampled_v.std(0)
    values["sample_v"] = sampled_v[0]
    values["sample_logits"] = sampled.logits[0]
    return values


@dataclass
class Transition:
    values: dict[str, np.ndarray]
    outcome: float
    is_switch: bool


def _searched(log_pi, estimate, temp):
    logits = log_pi + estimate / temp
    logits = logits - logits.max()
    probs = np.exp(logits)
    return probs / probs.sum()


def _kl(probs, log_pi):
    keep = probs > 0
    return float(np.sum(probs[keep] * (np.log(probs[keep]) - log_pi[keep])))


def candidate_steps(chunks):
    """(chunk index, t) of every decision with >= 2 legal cells."""
    found = []
    for index, chunk in enumerate(chunks):
        env = chunk.player_transitions.env_output
        done = np.asarray(env.done)
        legal = np.asarray(env.action_mask, bool)
        ok = valid_steps(done) & ~done.astype(bool) & (legal.sum(-1) >= 2)
        for step in np.flatnonzero(ok):
            found.append((index, int(step)))
    return found


def collect(net, variables, chunks, roots, num_samples, samples_per_call, seed):
    encode = jax.jit(
        jax.vmap(
            lambda params, actor_input, actor_output: net.apply(
                params, actor_input, actor_output, method=_encode
            ),
            in_axes=(None, 1, 1),
            out_axes=1,
        )
    )
    root_values = jax.jit(
        lambda params, rows, row_valid, legal, rng: net.apply(
            params,
            rows,
            row_valid,
            legal,
            rng,
            samples_per_call,
            16,
            method=_root_values,
        )
    )
    dev_variables = jax.device_put(variables)
    rng = jax.random.PRNGKey(seed)
    by_chunk = {}
    for chunk_index, step in roots:
        by_chunk.setdefault(chunk_index, []).append(step)
    out = []
    for chunk_index, steps in sorted(by_chunk.items()):
        batch = stack_batch(chunks[chunk_index : chunk_index + 1])
        actor_input = actor_input_of(batch)
        actor_output = batch.player_transitions.agent_output.actor_output
        rows, row_valid = encode(dev_variables, actor_input, actor_output)
        legal_all = jnp.asarray(np.asarray(actor_input.env.action_mask, bool))
        for step in steps:
            draws = []
            for call in range(num_samples // samples_per_call):
                rng, call_rng = jax.random.split(rng)
                read = root_values(
                    dev_variables,
                    rows[step, 0],
                    row_valid[step, 0],
                    legal_all[step, 0],
                    call_rng,
                )
                draws.append(np.asarray(read["values"]))
            read = jax.tree.map(np.asarray, read)
            keep = read["cell_valid"]
            cells = read["cells"][keep]
            out.append(
                Root(
                    values=np.concatenate(draws, axis=0)[:, keep],
                    log_pi=read["log_pi"][keep],
                    is_switch=IS_SWITCH_CELL[cells],
                    mode_mass=read["mode_mass"][keep],
                    root_value=float(read["root_value"]),
                )
            )
        if len(out) % 100 < len(steps):
            print(f"  {len(out)} roots", flush=True)
    return out


def collect_calibration(net, variables, chunks, outcomes, roots, num_samples, seed):
    """One record per root whose next request is real: `outcomes[i]` is
    the terminal result of the side chunk i belongs to."""
    encode = jax.jit(
        jax.vmap(
            lambda params, actor_input, actor_output: net.apply(
                params, actor_input, actor_output, method=_encode
            ),
            in_axes=(None, 1, 1),
            out_axes=1,
        )
    )
    calibrate = jax.jit(
        lambda params, *args: net.apply(params, *args, num_samples, method=_calibration)
    )
    dev_variables = jax.device_put(variables)
    rng = jax.random.PRNGKey(seed)
    by_chunk = {}
    for chunk_index, step in roots:
        by_chunk.setdefault(chunk_index, []).append(step)
    out = []
    for chunk_index, steps in sorted(by_chunk.items()):
        chunk = chunks[chunk_index]
        done = np.asarray(chunk.player_transitions.env_output.done).astype(bool)
        usable = valid_steps(done)
        batch = stack_batch(chunks[chunk_index : chunk_index + 1])
        actor_input = actor_input_of(batch)
        actor_output = batch.player_transitions.agent_output.actor_output
        rows, row_valid = encode(dev_variables, actor_input, actor_output)
        actions = np.asarray(actor_output.action_head.action_index)
        for step in steps:
            if step + 1 >= usable.shape[0] or not usable[step + 1]:
                continue
            action = int(actions[step, 0])
            rng, draw = jax.random.split(rng)
            read = calibrate(
                dev_variables,
                rows[step, 0],
                row_valid[step, 0],
                jnp.asarray(action),
                rows[step + 1, 0],
                row_valid[step + 1, 0],
                draw,
            )
            out.append(
                Transition(
                    values=jax.tree.map(np.asarray, read),
                    outcome=outcomes[chunk_index],
                    is_switch=bool(IS_SWITCH_CELL[action]),
                )
            )
        if len(out) % 100 < len(steps):
            print(f"  {len(out)} transitions", flush=True)
    return out


def _r2(prediction, target):
    residual = float(np.sum((target - prediction) ** 2))
    total = float(np.sum((target - target.mean()) ** 2))
    return 1.0 - residual / (total + 1e-8)


def _cross_entropy(logits, label):
    shifted = logits - logits.max(-1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(-1, keepdims=True))
    return float(-np.mean(np.sum(label * log_probs, -1)))


def summarise_calibration(transitions: list[Transition]) -> dict[str, float]:
    """copy predictor = 0 on both reads, the real next state = 1."""
    if not transitions:
        return {}
    stack = {
        key: np.stack([t.values[key] for t in transitions])
        for key in transitions[0].values
    }
    support = np.asarray(CAT_VF_SUPPORT, np.float32)
    outcomes = np.asarray([t.outcome for t in transitions], np.float32)
    label = (outcomes[:, None] == support[None]).astype(np.float32)
    target_delta = stack["real_v"] - stack["root_v"]
    ce_copy = _cross_entropy(stack["root_logits"], label)
    ce_real = _cross_entropy(stack["real_logits"], label)
    # The CE against a one-hot OUTCOME is outlier-driven under a head
    # trained on two-hot targets (it never puts mass on the far bin), so
    # the squared error of the expectation against the outcome is read
    # beside it; the learner's panel uses its own two-hot label.
    mse_copy = float(np.mean((stack["root_v"] - outcomes) ** 2))
    mse_real = float(np.mean((stack["real_v"] - outcomes) ** 2))
    stats = {
        "n": len(transitions),
        "abs_real_delta_v": float(np.abs(target_delta).mean()),
        "sign_acc_root": float(np.mean(np.sign(stack["root_v"]) == outcomes)),
        "sign_acc_real": float(np.mean(np.sign(stack["real_v"]) == outcomes)),
        "ce_copy": ce_copy,
        "ce_real": ce_real,
        "mse_copy": mse_copy,
        "mse_real": mse_real,
        "expect_sigma_z": float(stack["expect_sigma_z"].mean()),
    }
    for name in ("post", "prior", "expect", "sample"):
        mse_imagined = float(np.mean((stack[f"{name}_v"] - outcomes) ** 2))
        stats[f"value_gain_mse_{name}"] = (mse_copy - mse_imagined) / max(
            mse_copy - mse_real, 1e-3
        )
        stats[f"value_delta_r2_{name}"] = _r2(
            stack[f"{name}_v"] - stack["root_v"], target_delta
        )
        stats[f"value_gap_{name}"] = float(
            np.abs(stack[f"{name}_v"] - stack["real_v"]).mean()
        )
        ce_imagined = _cross_entropy(stack[f"{name}_logits"], label)
        stats[f"ce_{name}"] = ce_imagined
        stats[f"value_gain_{name}"] = (ce_copy - ce_imagined) / max(
            ce_copy - ce_real, 1e-3
        )
    return stats


def print_calibration(title, stats):
    print(f"\n== calibration: {title} (n={stats.get('n', 0)}) ==")
    for name, value in stats.items():
        if name != "n":
            print(f"  {name:24s} {value:+.4f}")


def summarise(roots: list[Root]) -> dict[str, list[float]]:
    stats = {}

    def add(name, value):
        stats.setdefault(name, []).append(float(value))

    for root in roots:
        num_draws = root.values.shape[0]
        half = num_draws // 2
        reference = root.values[half:].mean(0)
        pi = np.exp(root.log_pi)
        pi = pi / pi.sum()
        value_under_pi = float(pi @ reference)
        best = int(np.argmax(reference))
        oracle_gain = reference[best] - value_under_pi
        add("k_legal", len(reference))
        add("spread_actions_std", reference.std())
        add("spread_actions_range", reference.max() - reference.min())
        add("sigma_z", root.values.std(0).mean())
        add("mode_mass", root.mode_mass.mean())
        add("oracle_gain", oracle_gain)
        add("best_minus_root_v", reference[best] - root.root_value)
        add("var_pi_q", float(pi @ (reference - value_under_pi) ** 2))
        add("switch_top1_ref", root.is_switch[best])
        add("switch_top1_pi", root.is_switch[int(np.argmax(pi))])
        add("switch_mass_pi", pi[root.is_switch].sum())
        if root.is_switch.any() and (~root.is_switch).any():
            add(
                "q_switch_minus_move",
                reference[root.is_switch].mean() - reference[~root.is_switch].mean(),
            )
            add(
                "q_best_switch_minus_best_move",
                reference[root.is_switch].max() - reference[~root.is_switch].max(),
            )
        for count in SAMPLE_COUNTS:
            if count > half:
                continue
            estimate = root.values[:count].mean(0)
            searched = _searched(root.log_pi, estimate, LIVE_TEMP)
            add(f"top1_agree@{count}", int(np.argmax(estimate)) == best)
            if len(reference) >= 3:
                tau = kendalltau(estimate, reference).statistic
                if np.isfinite(tau):
                    add(f"kendall_tau@{count}", tau)
            add(
                f"greedy_gain@{count}",
                reference[int(np.argmax(estimate))] - value_under_pi,
            )
            add(f"search_gain@{count}", float(searched @ reference) - value_under_pi)
            add(f"root_kl@{count}", _kl(searched, root.log_pi))
            add(f"switch_mass_search@{count}", searched[root.is_switch].sum())
            add(
                f"snr@{count}",
                reference.std() / (root.values.std(0).mean() / np.sqrt(count) + 1e-9),
            )
    return stats


def print_stats(title, stats):
    if not stats:
        print(f"\n== {title} (n=0 roots) ==")
        return
    print(f"\n== {title} (n={len(stats['k_legal'])} roots) ==")
    for name, values in stats.items():
        arr = np.asarray(values)
        print(
            f"  {name:28s} mean {arr.mean():+.4f}  median {np.median(arr):+.4f}"
            f"  p90 {np.quantile(arr, 0.9):+.4f}  (n={len(arr)})"
        )


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--games-pkl", default=None, help="reuse played games")
    parser.add_argument("--games", type=int, default=24)
    parser.add_argument("--pairs", type=int, default=4)
    parser.add_argument("--roots", type=int, default=500)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--samples-per-call", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--calibration",
        action="store_true",
        help="value_delta_r2 / value_gain of the posterior, prior-mode, "
        "prior-expectation and single-sample decodes",
    )
    parser.add_argument(
        "--calibration-samples",
        type=int,
        default=32,
        help="prior draws per transition for the expectation decode",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)

    variables = harness.load_params(args.ckpt)
    if args.games_pkl and os.path.exists(args.games_pkl):
        sides = harness.load(args.games_pkl)
    else:
        sides = harness.play_games(
            variables, args.games, pairs=args.pairs, tag="searchprobe", seed=args.seed
        )
        if args.games_pkl:
            harness.dump(sides, args.games_pkl)
    outcomes = [harness.outcome(side) for side in sides for _ in side]
    chunks = harness.flatten(sides)
    candidates = candidate_steps(chunks)
    picker = np.random.default_rng(args.seed)
    chosen = picker.choice(
        len(candidates), min(args.roots, len(candidates)), replace=False
    )
    roots = [candidates[index] for index in sorted(chosen)]
    print(
        f"params: {args.ckpt}; {len(sides)} sides, {len(chunks)} chunks, "
        f"{len(candidates)} decisions with >= 2 legal cells, {len(roots)} roots, "
        f"{args.samples} prior draws each (reference = last {args.samples // 2})",
        flush=True,
    )
    net = get_player_model(get_player_model_config(9, train=True))
    data = collect(
        net, variables, chunks, roots, args.samples, args.samples_per_call, args.seed
    )
    print_stats("all roots", summarise(data))
    both = [root for root in data if root.is_switch.any() and (~root.is_switch).any()]
    print_stats("roots offering both a move and a switch", summarise(both))
    moves_only = [root for root in data if not root.is_switch.any()]
    print_stats("moves only", summarise(moves_only))
    switches_only = [root for root in data if root.is_switch.all()]
    print_stats("switches only (force-switch / preview)", summarise(switches_only))
    if args.calibration:
        transitions = collect_calibration(
            net,
            variables,
            chunks,
            outcomes,
            roots,
            args.calibration_samples,
            args.seed,
        )
        print_calibration("all transitions", summarise_calibration(transitions))
        switch_taken = [t for t in transitions if t.is_switch]
        print_calibration("switch taken", summarise_calibration(switch_taken))
        move_taken = [t for t in transitions if not t.is_switch]
        print_calibration("move taken", summarise_calibration(move_taken))


if __name__ == "__main__":
    main()
