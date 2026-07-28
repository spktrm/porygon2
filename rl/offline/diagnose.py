"""One-shot diagnostic for the offline critic pipeline.

Runs the three checks that separate "wiring bug" from "slow convergence",
on a single batch from the real shards:

1. Feature variation: do the pooled latents differ across timesteps and
   across trajectories? (If not, state_at_requests/pooling is broken and
   the probe has nothing to read.)
2. Gradient flow: per-subtree gradient norms — a zero anywhere means a
   blocked path.
3. Overfit-one-batch: 300 steps on the SAME batch must drive loss toward
   ~0. If it does, the pipeline can learn and any plateau on the full
   dataset is a capacity/data/schedule question, not a bug.

Usage:
    python -m rl.offline.diagnose [--dataset-dir replays/shards]
"""

import argparse

import jax
import jax.numpy as jnp
import numpy as np
import optax

from rl.offline.config import get_offline_config
from rl.offline.dataset import (
    collate,
    iter_shard_payloads,
    list_shards,
    record_to_examples,
)
from rl.offline.model import get_offline_critic
from rl.offline.train import _metrics_from_logits, _value_mask


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument(
        "--generalize",
        action="store_true",
        help="Also run the mini train/held-out generalization probe",
    )
    parser.add_argument("--num-records", type=int, default=400)
    parser.add_argument("--generalize-steps", type=int, default=2000)
    args = parser.parse_args()

    config = get_offline_config()
    if args.dataset_dir:
        config = config.replace(dataset_dir=args.dataset_dir)
    config = config.replace(batch_size=args.batch_size)

    # --- Build one real batch ---
    examples = []
    for shard in list_shards(config):
        for payload in iter_shard_payloads(shard):
            examples.extend(record_to_examples(payload, config))
            if len(examples) >= config.batch_size:
                break
        if len(examples) >= config.batch_size:
            break
    examples = examples[: config.batch_size]
    assert examples, "no examples decoded from shards"
    batch = collate(examples, config)
    labels = np.asarray(batch.labels)
    print(
        f"batch: B={labels.shape[0]}, label counts "
        f"loss/tie/win = {labels.sum(axis=0).astype(int).tolist()}"
    )

    model = get_offline_critic(config.generation)
    apply_fn = jax.vmap(model.apply, in_axes=(None, 1), out_axes=1)
    ex_column = jax.tree.map(lambda x: jnp.asarray(x[:, 0]), batch.actor_input)
    params = model.init(jax.random.key(0), ex_column)

    # --- 1. Feature variation ---
    # Read the value-head expectation across time — if the input pathway is
    # broken (bad gather stamps, constant pooling), it is constant across
    # timesteps and/or trajectories even at init.
    out = apply_fn(params, batch.actor_input)
    expectation = np.asarray(out.expectation, dtype=np.float32)  # (T, B)
    mask = np.asarray(_value_mask(jnp.asarray(batch.actor_input.env.done)))
    per_traj_std = [
        float(np.std(expectation[mask[:, b].astype(bool), b]))
        for b in range(expectation.shape[1])
    ]
    cross_traj_std = float(np.std(expectation[0]))
    print(
        f"feature variation @init: within-trajectory std of Φ over time = "
        f"{np.mean(per_traj_std):.2e} (want > 0), "
        f"across-trajectory std at t=0 = {cross_traj_std:.2e} (want > 0)"
    )

    # --- 2. Gradient flow per subtree ---
    def loss_fn(params):
        value_head = apply_fn(params, batch.actor_input)
        m = _value_mask(batch.actor_input.env.done)
        return _metrics_from_logits(value_head.logits, batch.labels, m)["loss"]

    loss0, grads = jax.value_and_grad(loss_fn)(params)
    print(f"initial loss = {float(loss0):.4f} (untrained ~ln3=1.099)")
    flat = jax.tree_util.tree_flatten_with_path(grads)[0]
    subtree_sq: dict[str, float] = {}
    for path, leaf in flat:
        top = "/".join(str(getattr(k, "key", k)) for k in path[:3])
        subtree_sq[top] = subtree_sq.get(top, 0.0) + float(
            jnp.sum(jnp.square(leaf.astype(jnp.float32)))
        )
    print("gradient norms per subtree (zeros indicate a blocked path):")
    for name in sorted(subtree_sq):
        print(f"  {name}: {np.sqrt(subtree_sq[name]):.3e}")

    # --- 3. Overfit one batch ---
    tx = optax.adamw(args.learning_rate, b1=0.9)
    opt_state = tx.init(params)

    @jax.jit
    def step(params, opt_state):
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = tx.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    for i in range(1, args.steps + 1):
        params, opt_state, loss = step(params, opt_state)
        if i % 50 == 0 or i == 1:
            print(f"overfit step {i}: loss {float(loss):.4f}")
    final = float(loss)
    print(
        "VERDICT: "
        + (
            "pipeline can learn (loss collapsed on a fixed batch) — any "
            "full-run plateau is data/schedule, not wiring."
            if final < 0.3
            else "loss did NOT collapse on a fixed batch — genuine "
            "learning blocker in the model/optimization path."
        )
    )

    if args.generalize:
        generalization_probe(args, config, model, apply_fn)


def generalization_probe(args, config, model, apply_fn):
    """Miniature train/held-out experiment: reproduces (or rules out) the
    'memorizes but never generalizes' failure in minutes instead of a full
    run. Split is per record (= per replay), same as training."""
    print(f"\n--- generalization probe ({args.num_records} replays) ---")
    records = []
    for shard in list_shards(config):
        for payload in iter_shard_payloads(shard):
            examples = record_to_examples(payload, config)
            if examples:
                records.append(examples)
            if len(records) >= args.num_records:
                break
        if len(records) >= args.num_records:
            break
    # Full perspective pairs only, and pair-aware batches below — matching
    # rl/offline/dataset.py: unpaired batches let spurious in-batch
    # correlations drown the side-differenced signal.
    records = [r for r in records if len(r) == 2]
    split = int(0.8 * len(records))
    train_records = records[:split]
    train_pool = [e for r in train_records for e in r]
    eval_pool = [e for r in records[split:] for e in r]
    print(f"train trajectories={len(train_pool)} eval trajectories={len(eval_pool)}")

    params = model.init(
        jax.random.key(1),
        jax.tree.map(
            lambda x: jnp.asarray(x[:, 0]),
            collate(train_pool[: config.batch_size], config).actor_input,
        ),
    )
    tx = optax.adamw(args.learning_rate, b1=0.9)
    opt_state = tx.init(params)

    def batch_metrics(params, batch):
        value_head = apply_fn(params, batch.actor_input)
        m = _value_mask(batch.actor_input.env.done)
        return _metrics_from_logits(value_head.logits, batch.labels, m)

    @jax.jit
    def step(params, opt_state, batch):
        def loss_fn(p):
            return batch_metrics(p, batch)["loss"]

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = tx.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    eval_metrics_jit = jax.jit(batch_metrics)
    rng = np.random.default_rng(0)

    def evaluate_pool(pool):
        losses, accs, last_accs, weights = [], [], [], []
        for i in range(0, min(len(pool), 64), config.batch_size):
            chunk = pool[i : i + config.batch_size]
            if len(chunk) < config.batch_size:
                break
            m = jax.device_get(eval_metrics_jit(params, collate(chunk, config)))
            losses.append(m["loss"])
            accs.append(m["accuracy"])
            last_accs.append(m["accuracy_last_step"])
            weights.append(m["num_valid_steps"])
        return (
            np.average(losses, weights=weights),
            np.average(accs, weights=weights),
            np.mean(last_accs),
        )

    games_per_batch = max(1, config.batch_size // 2)
    for i in range(1, args.generalize_steps + 1):
        picks = rng.choice(len(train_records), size=games_per_batch, replace=False)
        batch = collate(
            [e for j in picks for e in train_records[j]][: config.batch_size],
            config,
        )
        params, opt_state, _ = step(params, opt_state, batch)
        if i % 200 == 0:
            tr = evaluate_pool(train_pool)
            ev = evaluate_pool(eval_pool)
            print(
                f"probe step {i}: train loss {tr[0]:.3f} acc {tr[1]:.3f} "
                f"last {tr[2]:.3f} | HELD-OUT loss {ev[0]:.3f} "
                f"acc {ev[1]:.3f} last {ev[2]:.3f}"
            )
    print(
        "Interpretation: held-out last-step acc > ~0.7 here means the "
        "pipeline generalizes and a flat full run is environmental (stale "
        "code/shards). Held-out stuck at 0.5 while train climbs means the "
        "side signal is memorizable but not being learned as a rule — an "
        "architecture problem, and the next escalation is an explicitly "
        "side-differenced (antisymmetric) readout."
    )


if __name__ == "__main__":
    main()