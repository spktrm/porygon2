"""One-shot diagnostic for the offline critic pipeline.

Runs the three checks that separate "wiring bug" from "slow convergence",
on a single batch from the real shards:

1. Feature variation: do the pooled latents differ across timesteps and
   across trajectories? (If not, state_at_requests/pooling is broken and
   the probe has nothing to read.)
2. Gradient flow: per-subtree gradient norms — a zero anywhere means a
   blocked path.
3. Overfit-one-batch: 300 steps on the SAME batch must drive the margin
   loss to its label-entropy floor (censored concession labels are soft,
   so the floor is > 0) and the survival loss toward ~0. If both collapse,
   the pipeline can learn and any plateau on the full dataset is a
   capacity/data/schedule question, not a bug.

Usage:
    python -m rl.offline.diagnose [--dataset-dir replays/shards]
"""

import argparse
import functools
import itertools

import jax
import jax.numpy as jnp
import numpy as np
import optax

from rl.offline.config import get_offline_config
from rl.offline.dataset import (
    NUM_SURVIVAL_BINS,
    OfflineDataset,
    collate,
    iter_shard_payloads,
    list_shards,
    record_to_examples,
)
from rl.offline.model import Porygon2OfflineCritic, get_offline_critic
from rl.offline.train import _metrics_from_logits, _survival_loss, _value_mask


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
    parser.add_argument(
        "--floor-batches",
        type=int,
        default=32,
        help="Holdout batches for the POOL-level constant survival floor "
        "(the single-batch floor has 8-game variance); 0 skips it",
    )
    parser.add_argument(
        "--survival-loss-weight",
        type=float,
        default=None,
        help="Override the aux weight (0 disables the survival head's "
        "gradient) — A/B whether the aux pathway amplifies memorization "
        "on small probe datasets",
    )
    args = parser.parse_args()

    config = get_offline_config()
    if args.dataset_dir:
        config = config.replace(dataset_dir=args.dataset_dir)
    config = config.replace(batch_size=args.batch_size)
    if args.survival_loss_weight is not None:
        config = config.replace(survival_loss_weight=args.survival_loss_weight)

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
        f"batch: B={labels.shape[0]}, margin-bin label mass (-6..+6) = "
        + " ".join(f"{v:.1f}" for v in labels.sum(axis=0))
    )

    model = get_offline_critic(config.generation)
    apply_fn = jax.vmap(
        functools.partial(model.apply, method=Porygon2OfflineCritic.with_aux),
        in_axes=(None, 1),
        out_axes=1,
    )
    ex_column = jax.tree.map(lambda x: jnp.asarray(x[:, 0]), batch.actor_input)
    params = model.init(
        jax.random.key(0), ex_column, method=Porygon2OfflineCritic.with_aux
    )

    # --- 1. Feature variation ---
    # Read the value-head expectation across time — if the input pathway is
    # broken (bad gather stamps, constant pooling), it is constant across
    # timesteps and/or trajectories even at init.
    out, _ = apply_fn(params, batch.actor_input)
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

    # --- 1b. Survival target sanity + constant-predictor floor ---
    # The aggregate survival loss is dominated by far-future/censored
    # cells that a constant prior fits; this section prints what the
    # targets look like and the loss a model that IGNORES the input can
    # reach. Training must beat that floor to be learning anything
    # conditional — compare full-run survival_loss against it before
    # calling the aux head stuck.
    targets = np.asarray(batch.survival_targets)
    weight = np.asarray(batch.survival_masks) * mask[..., None]
    supervised = weight > 0
    num_allowed = targets.sum(-1)
    exact = supervised & (num_allowed == 1)
    censored = supervised & (num_allowed > 1)
    total = max(int(supervised.sum()), 1)
    print(
        f"survival targets: {supervised.mean():.1%} of (step, slot) cells "
        f"supervised; exact {int(exact.sum()) / total:.1%}, "
        f"censored {int(censored.sum()) / total:.1%}"
    )
    hist = np.bincount(targets.argmax(-1)[exact], minlength=NUM_SURVIVAL_BINS)
    print(
        "exact-target bins (0 = far/never .. 15 = faints now): "
        + " ".join(str(int(c)) for c in hist)
    )
    tgt = jnp.asarray(targets)
    wgt = jnp.asarray(weight.astype(np.float32))

    def const_loss(logit_vec):
        log_probs = jax.nn.log_softmax(logit_vec)
        nll = -jax.scipy.special.logsumexp(jnp.where(tgt > 0, log_probs, -1e9), axis=-1)
        return (nll * wgt).sum() / wgt.sum().clip(min=1.0)

    const_logits = jnp.zeros(NUM_SURVIVAL_BINS)
    const_tx = optax.adam(0.1)
    const_state = const_tx.init(const_logits)

    @jax.jit
    def const_step(logit_vec, opt_state):
        grad = jax.grad(const_loss)(logit_vec)
        updates, opt_state = const_tx.update(grad, opt_state, logit_vec)
        return optax.apply_updates(logit_vec, updates), opt_state

    for _ in range(300):
        const_logits, const_state = const_step(const_logits, const_state)
    print(
        f"best-constant survival loss ≈ {float(const_loss(const_logits)):.4f} "
        f"(uniform = ln{NUM_SURVIVAL_BINS} ≈ {np.log(NUM_SURVIVAL_BINS):.4f})"
    )

    # --- 1c. Pool-level constant survival floor ---
    # The single-batch floor above rides on 8 games' exact/censored mix.
    # THIS is the number a full run's eval_survival_loss must beat to be
    # learning anything conditional: the best input-independent predictor
    # over many holdout batches. eval surv ≈ this floor = prior fitted,
    # conditional learning still to come (watch survival_loss_imminent);
    # meaningfully above = not even the prior — an optimization problem.
    if args.floor_batches > 0:
        pool_rows = []
        for pool_batch in itertools.islice(
            OfflineDataset(config).eval_batches(), args.floor_batches
        ):
            pool_targets = np.asarray(pool_batch.survival_targets)
            pool_weight = (
                np.asarray(pool_batch.survival_masks)
                * np.asarray(_value_mask(jnp.asarray(pool_batch.actor_input.env.done)))[
                    ..., None
                ]
            )
            pool_rows.append(pool_targets[pool_weight > 0])
        pool = jnp.asarray(np.concatenate(pool_rows))

        def pool_const_loss(logit_vec):
            log_probs = jax.nn.log_softmax(logit_vec)
            nll = -jax.scipy.special.logsumexp(
                jnp.where(pool > 0, log_probs, -1e9), axis=-1
            )
            return nll.mean()

        pool_logits = jnp.zeros(NUM_SURVIVAL_BINS)
        pool_state = const_tx.init(pool_logits)

        @jax.jit
        def pool_step(logit_vec, opt_state):
            grad = jax.grad(pool_const_loss)(logit_vec)
            updates, opt_state = const_tx.update(grad, opt_state, logit_vec)
            return optax.apply_updates(logit_vec, updates), opt_state

        for _ in range(300):
            pool_logits, pool_state = pool_step(pool_logits, pool_state)
        num_allowed_pool = np.asarray(pool.sum(axis=-1))
        pool_uniform = float(
            np.log(NUM_SURVIVAL_BINS) - np.mean(np.log(num_allowed_pool))
        )
        print(
            f"POOL constant survival floor ≈ "
            f"{float(pool_const_loss(pool_logits)):.4f} over "
            f"{pool.shape[0]} supervised cells / {args.floor_batches} "
            f"holdout batches (uniform predictor on this pool ≈ "
            f"{pool_uniform:.4f})"
        )

    # Margin overfit floor: censored (concession) labels are soft, so CE
    # against them bottoms out at the label entropy, not 0 — a perfectly
    # overfit model reproduces each label distribution exactly.
    label_entropy = -np.sum(
        labels * np.log(np.clip(labels, 1e-9, 1.0)), axis=-1
    )  # (B,)
    margin_floor = float(
        np.average(label_entropy, weights=np.maximum(mask.sum(axis=0), 1))
    )
    print(
        f"margin overfit floor ≈ {margin_floor:.4f} "
        "(mean censored-label entropy; 0 only if every label is one-hot)"
    )

    # --- 2. Gradient flow per subtree ---
    def loss_fn(params):
        value_head, survival_logits = apply_fn(params, batch.actor_input)
        m = _value_mask(batch.actor_input.env.done)
        metrics = _metrics_from_logits(value_head.logits, batch.labels, m)
        metrics.update(
            _survival_loss(
                survival_logits,
                batch.survival_targets,
                batch.survival_masks,
                m,
            )
        )
        total = metrics["loss"] + config.survival_loss_weight * metrics["survival_loss"]
        return total, metrics

    (loss0, metrics0), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    print(
        f"initial margin loss = {float(metrics0['loss']):.4f} | "
        f"survival loss = {float(metrics0['survival_loss']):.4f} "
        f"(zeros should sit near the uniform values above; a zero "
        f"survival_head gradient below means the aux path is disconnected)"
    )
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
        (_, step_metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state = tx.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, step_metrics

    for i in range(1, args.steps + 1):
        params, opt_state, metrics = step(params, opt_state)
        if i % 50 == 0 or i == 1:
            print(
                f"overfit step {i}: margin {float(metrics['loss']):.4f} | "
                f"survival {float(metrics['survival_loss']):.4f}"
            )
    final = float(metrics["loss"])
    final_surv = float(metrics["survival_loss"])
    print(
        "VERDICT (margin): "
        + (
            f"pipeline can learn (loss collapsed to the label-entropy "
            f"floor ≈ {margin_floor:.3f}) — any full-run plateau is "
            "data/schedule, not wiring."
            if final < margin_floor + 0.1
            else "loss did NOT collapse to the label-entropy floor "
            f"(≈ {margin_floor:.3f}) on a fixed batch — genuine learning "
            "blocker in the model/optimization path."
        )
    )
    if config.survival_loss_weight == 0:
        print(
            "VERDICT (survival): skipped — aux disabled "
            "(--survival-loss-weight 0), so no gradient reaches the head."
        )
    else:
        print(
            "VERDICT (survival): "
            + (
                "aux head can fit its targets (collapsed well below the "
                "best-constant floor)."
                if final_surv < 0.5
                else "survival loss did NOT collapse on a fixed batch — the "
                "aux pathway (targets/mask/head) is broken, not just slow."
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
        method=Porygon2OfflineCritic.with_aux,
    )
    tx = optax.adamw(args.learning_rate, b1=0.9)
    opt_state = tx.init(params)

    def batch_metrics(params, batch):
        value_head, survival_logits = apply_fn(params, batch.actor_input)
        m = _value_mask(batch.actor_input.env.done)
        metrics = _metrics_from_logits(value_head.logits, batch.labels, m)
        metrics.update(
            _survival_loss(
                survival_logits,
                batch.survival_targets,
                batch.survival_masks,
                m,
            )
        )
        return metrics

    @jax.jit
    def step(params, opt_state, batch):
        def loss_fn(p):
            metrics = batch_metrics(p, batch)
            return (
                metrics["loss"] + config.survival_loss_weight * metrics["survival_loss"]
            )

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = tx.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    eval_metrics_jit = jax.jit(batch_metrics)
    rng = np.random.default_rng(0)

    def evaluate_pool(pool):
        losses, accs, last_accs, survs, weights = [], [], [], [], []
        for i in range(0, min(len(pool), 64), config.batch_size):
            chunk = pool[i : i + config.batch_size]
            if len(chunk) < config.batch_size:
                break
            m = jax.device_get(eval_metrics_jit(params, collate(chunk, config)))
            losses.append(m["loss"])
            accs.append(m["accuracy"])
            last_accs.append(m["accuracy_last_step"])
            survs.append(m["survival_loss"])
            weights.append(m["num_valid_steps"])
        return (
            np.average(losses, weights=weights),
            np.average(accs, weights=weights),
            np.mean(last_accs),
            np.average(survs, weights=weights),
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
                f"last {tr[2]:.3f} surv {tr[3]:.3f} | HELD-OUT loss {ev[0]:.3f} "
                f"acc {ev[1]:.3f} last {ev[2]:.3f} surv {ev[3]:.3f}"
            )
    print(
        "Interpretation: held-out last-step acc > ~0.7 here means the "
        "pipeline generalizes even at probe scale. Held-out at chance "
        "while train climbs is memorization — expected at a few hundred "
        "games (two of the recipe's four layers, large data and tuned "
        "regularization, are absent here), so before concluding anything "
        "rerun with --num-records 2000+, and A/B --survival-loss-weight 0 "
        "to see whether the aux head amplifies it. The full run's holdout "
        "evals are the real referee."
    )


if __name__ == "__main__":
    main()
