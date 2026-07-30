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

--martingale runs a separate mode on a TRAINED artifact: the announced-state
calibration audit that gates dice-excised PBRS. If Φ_ann really is
E[Φ(t+1) | history, announced actions], the dice term Φ(t) − Φ_ann(t) is a
zero-mean residual conditional on ANY function of the announcement — so its
mean inside each Φ_ann value bin (a reliability curve) must be ~0, per game
phase. (The pooled, unconditioned mean is exactly zero by mirror
antisymmetry over paired perspectives, so only conditional slices test
anything.) Systematic per-bin residuals = a reward bias dice-excised
shaping would inject; fix the critic before flipping the learner flag.
Also reports the decision share of per-turn swing — the quantified version
of the anticipation question (a purely reactive Φ puts ~everything in the
dice term).

Usage:
    python -m rl.offline.diagnose [--dataset-dir replays/shards]
    python -m rl.offline.diagnose --martingale [--ckpt ...] \
        [--martingale-batches 64]
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
    parser.add_argument(
        "--martingale",
        action="store_true",
        help="Run the announced-state (Φ_ann) calibration audit on a "
        "trained artifact instead of the fresh-init wiring checks",
    )
    parser.add_argument(
        "--ckpt",
        action="append",
        default=None,
        help="artifact dir for --martingale (repeat for an ensemble); "
        "default: latest under ckpts/offline/{format_id}*/",
    )
    parser.add_argument(
        "--martingale-batches",
        type=int,
        default=64,
        help="holdout batches for the --martingale audit",
    )
    args = parser.parse_args()

    config = get_offline_config()
    if args.dataset_dir:
        config = config.replace(dataset_dir=args.dataset_dir)
    config = config.replace(batch_size=args.batch_size)
    if args.survival_loss_weight is not None:
        config = config.replace(survival_loss_weight=args.survival_loss_weight)

    if args.martingale:
        martingale_audit(args, config)
        return

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


def _turn_announcement_fired(
    history_field: np.ndarray,
    edge_cache: np.ndarray,
    request_counts: np.ndarray,
) -> np.ndarray:
    """(T-1, B) bool: does the turn leading into state t (the history steps
    stamped with state t's request count) contain at least one announcement
    edge? Numpy ground truth mirroring mask_outcome_features — independent
    of the model, so it can arbitrate whether a zero decision term means
    "no announcement existed" or "the announcement never reached Φ_ann".

    history_field: (H, B, F); edge_cache: (P, B, F); request_counts: (T, B).
    """
    from rl.environment.protos.enums_pb2 import BattlemajorargsEnum
    from rl.environment.protos.features_pb2 import EntityEdgeFeature, FieldFeature
    from rl.model.history_encoder import (
        _OUTCOME_MAJOR_ARGS,
        _RELEVANT_ENTITY_FEATURES,
    )

    num_steps, num_examples = request_counts.shape
    fired = np.zeros((num_steps - 1, num_examples), dtype=bool)
    for b in range(num_examples):
        rows = history_field[:, b]
        valid = rows[:, FieldFeature.FIELD_FEATURE__VALID] > 0
        rc = rows[:, FieldFeature.FIELD_FEATURE__REQUEST_COUNT]
        relevant = rows[:, _RELEVANT_ENTITY_FEATURES]  # (H, K)
        edge_ok = (
            np.arange(relevant.shape[1])[None]
            < rows[:, FieldFeature.FIELD_FEATURE__NUM_RELEVANT][:, None]
        ) & valid[:, None]
        major = edge_cache[:, b, EntityEdgeFeature.ENTITY_EDGE_FEATURE__MAJOR_ARG]
        row_is_announcement = (
            major != BattlemajorargsEnum.BATTLEMAJORARGS_ENUM___UNSPECIFIED
        ) & ~np.isin(major, _OUTCOME_MAJOR_ARGS)
        step_announced = (
            np.take(row_is_announcement, np.clip(relevant, 0, major.size - 1))
            & edge_ok
        ).any(axis=1)  # (H,)
        # match[t-1, h]: history step h belongs to the turn entering state t.
        match = rc[None, :] == request_counts[1:, b][:, None]
        fired[:, b] = (match & (step_announced & valid)[None, :]).any(axis=1)
    return fired


def martingale_audit(args, config):
    """Announced-state calibration audit on a trained artifact — the gate
    for dice-excised PBRS.

    For each holdout step t (with a same-trajectory predecessor), using the
    ensemble-MEAN potential the learner would consume:

        dice(t)     = Φ(t) − Φ_ann(t)      (what chance was worth)
        decision(t) = Φ_ann(t) − Φ(t−1)    (what the revealed choices were worth)

    If Φ_ann = E[Φ(t) | history, announcement], dice is a zero-mean
    residual conditional on any function of the announcement. The pooled
    mean is exactly zero by mirror antisymmetry (each game contributes both
    perspectives with negated terms) — printed only as a pairing sanity
    check. The informative test is the reliability curve: mean dice inside
    each Φ_ann value bin, overall and per game-phase tercile. A bin residual
    several stderrs from zero is a systematic bias that dice-excised
    shaping (γ·Φ_ann(t+1) − Φ(t)) would inject into the reward.

    Both learner readouts are audited: "margin" (expected margin / 6) and
    "win" (P(win) − P(loss)). The decision share Σ|decision| / Σ(|decision|
    + |dice|) quantifies anticipation: a reactive Φ that only updates after
    outcomes resolve pushes swings into the dice term even on quiet turns.
    """
    from rl.environment.protos.features_pb2 import InfoFeature
    from rl.offline.artifact import load_critic_params
    from rl.offline.dataset import MAX_MARGIN
    from rl.offline.visualise import discover_ckpts

    ckpt_paths = args.ckpt or discover_ckpts(config.format_id)
    print(f"checkpoints: {ckpt_paths}")
    params = load_critic_params(ckpt_paths)
    num_members = jax.tree.leaves(params)[0].shape[0]
    model = get_offline_critic(
        config.generation,
        rating_conditioning="rating_embed" in params.get("params", {}),
    )
    apply_fn = jax.jit(
        jax.vmap(
            functools.partial(model.apply, method=Porygon2OfflineCritic.announced),
            in_axes=(None, 1),
            out_axes=1,
        )
    )

    def readouts(head) -> dict[str, np.ndarray]:
        probs = np.exp(np.asarray(head.log_probs, dtype=np.float32))  # (T, B, 13)
        return {
            "margin": np.asarray(head.expectation, dtype=np.float32),
            "win": probs[..., MAX_MARGIN + 1 :].sum(-1) - probs[..., :MAX_MARGIN].sum(-1),
        }

    rows: dict[str, dict[str, list]] = {
        name: {"dice": [], "decision": [], "ann": [], "phase": []}
        for name in ("margin", "win")
    }
    fired_rows: list[np.ndarray] = []
    num_batches = 0
    for batch in itertools.islice(
        OfflineDataset(config).eval_batches(), args.martingale_batches
    ):
        phi_sum = {"margin": 0.0, "win": 0.0}
        ann_sum = {"margin": 0.0, "win": 0.0}
        for k in range(num_members):
            member_params = jax.tree.map(lambda x: x[k], params)  # noqa: B023
            head, ann_head = jax.device_get(apply_fn(member_params, batch.actor_input))
            for name, value in readouts(head).items():
                phi_sum[name] = phi_sum[name] + value
            for name, value in readouts(ann_head).items():
                ann_sum[name] = ann_sum[name] + value

        mask = np.asarray(_value_mask(jnp.asarray(batch.actor_input.env.done)))
        pair = (mask[1:] * mask[:-1]) > 0  # (T-1, B): t and t−1 both valid
        fired_rows.append(
            _turn_announcement_fired(
                np.asarray(batch.actor_input.history.field),
                np.asarray(batch.actor_input.packed_history.edge_cache),
                np.asarray(
                    batch.actor_input.env.info[
                        ..., InfoFeature.INFO_FEATURE__REQUEST_COUNT
                    ]
                ),
            )[pair]
        )
        n_valid = mask.sum(axis=0)  # (B,)
        phase = (
            np.arange(1, mask.shape[0])[:, None] / np.maximum(n_valid[None] - 1, 1)
        )[pair]
        for name in rows:
            phi = phi_sum[name] / num_members  # ensemble mean, learner-facing
            ann = ann_sum[name] / num_members
            rows[name]["dice"].append((phi[1:] - ann[1:])[pair])
            rows[name]["decision"].append((ann[1:] - phi[:-1])[pair])
            rows[name]["ann"].append(ann[1:][pair])
            rows[name]["phase"].append(phase)
        num_batches += 1

    assert num_batches, "no holdout batches decoded"
    bin_edges = np.linspace(-1.0, 1.0, 11)
    for name in ("margin", "win"):
        dice = np.concatenate(rows[name]["dice"])
        decision = np.concatenate(rows[name]["decision"])
        ann = np.concatenate(rows[name]["ann"])
        phase = np.concatenate(rows[name]["phase"])
        n = dice.size
        print(
            f"\n=== {name} readout — {n} turn pairs / {num_batches} holdout "
            "batches ==="
        )
        print(
            f"pooled dice mean {dice.mean():+.2e} (≈0 by antisymmetry — a "
            "pairing sanity check, not a calibration result) | dice std "
            f"{dice.std():.4f} | decision std {decision.std():.4f}"
        )
        share = np.abs(decision).sum() / max(
            np.abs(decision).sum() + np.abs(dice).sum(), 1e-9
        )
        print(
            f"decision share of swing magnitude = {share:.1%} "
            "(low = reactive Φ: swings land in the dice term even where "
            "choices, not chance, moved the game)"
        )

        # Announced-movement stats: does Φ_ann USE the announcement at all?
        # |decision| = |Φ_ann(t) − Φ(t−1)| is the announced movement;
        # |decision + dice| = |Φ(t) − Φ(t−1)| the realised movement. `fired`
        # is the numpy ground truth for "the turn had ≥1 announcement edge",
        # so the cross-tabs separate three failure stories the decision
        # share alone can't: no announcement existed / announcement never
        # reached the state (wiring) / reached it but the model ignores it.
        fired = np.concatenate(fired_rows)
        ann_moved = np.abs(decision) > 1e-5
        realised_move = np.abs(decision + dice)
        print(f"announcement coverage: {fired.mean():.1%} of turn pairs fired")
        if fired.any():
            movement_ratio = np.abs(decision)[fired].mean() / max(
                realised_move[fired].mean(), 1e-9
            )
            print(
                f"announced movement on fired turns: mean|Φ_ann(t)−Φ(t−1)| = "
                f"{np.abs(decision)[fired].mean():.4f} vs realised "
                f"mean|Φ(t)−Φ(t−1)| = {realised_move[fired].mean():.4f} "
                f"(ratio {movement_ratio:.2f} — ~0 means Φ_ann ignores the "
                "announcement; near 1 means it carries the realised update)"
            )
            dead = float((fired & ~ann_moved).mean())
            print(
                f"dead announcements (fired but Φ_ann didn't move): {dead:.1%} "
                "— persistent nonzero at scale means announcements reach the "
                "state but the readout is insensitive to them"
            )
        ghost = float((~fired & ann_moved).mean())
        print(
            f"ghost movement (no announcement yet Φ_ann ≠ Φ(t−1)): {ghost:.1%} "
            "— should be ~0; nonzero means the announced pre-state gather "
            "sees history steps between the two states, i.e. request-count "
            "stamping is misaligned with exported states"
        )
        print(
            f"{'Φ_ann bin':>14} {'count':>7} {'mean dice':>10} {'stderr':>8} "
            f"{'z':>6}  flag"
        )
        worst_z, ece, total = 0.0, 0.0, max(n, 1)
        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            sel = (ann >= lo) & (ann < hi if hi < 1.0 else ann <= hi)
            count = int(sel.sum())
            if count < 20:
                continue
            mean = float(dice[sel].mean())
            stderr = float(dice[sel].std() / np.sqrt(count))
            z = mean / max(stderr, 1e-9)
            worst_z = max(worst_z, abs(z))
            ece += (count / total) * abs(mean)
            flag = "  <-- systematic" if abs(z) > 3 and abs(mean) > 0.01 else ""
            print(
                f"[{lo:+.1f}, {hi:+.1f}) {count:>7} {mean:>+10.4f} "
                f"{stderr:>8.4f} {z:>+6.1f}{flag}"
            )
        print(f"weighted |bin residual| (ECE-style) = {ece:.4f}")
        for label, sel in (
            ("early (t/T < 1/3)", phase < 1 / 3),
            ("mid   (1/3..2/3)", (phase >= 1 / 3) & (phase < 2 / 3)),
            ("late  (t/T ≥ 2/3)", phase >= 2 / 3),
        ):
            if not sel.any():
                continue
            phase_ece, phase_total = 0.0, max(int(sel.sum()), 1)
            for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
                bin_sel = sel & (ann >= lo) & (ann < hi if hi < 1.0 else ann <= hi)
                count = int(bin_sel.sum())
                if count < 20:
                    continue
                phase_ece += (count / phase_total) * abs(float(dice[bin_sel].mean()))
            print(f"  {label}: n={int(sel.sum())}, binned |residual| = {phase_ece:.4f}")
        verdict_ok = worst_z <= 3 or ece < 0.01
        print(
            "VERDICT: "
            + (
                "no systematic conditional dice residual at this sample size "
                "— Φ_ann is consistent with E[Φ | announcement] under this "
                "readout, and dice-excised shaping would not inject "
                "measurable bias."
                if verdict_ok
                else "flagged bins show a conditional dice residual — Φ_ann "
                "disagrees with the realised Φ's conditional mean there, and "
                "dice-excised shaping would inject that bias into the "
                "reward. Retrain/calibrate the critic (announced loss "
                "weight, more data) before enabling the learner flag."
            )
        )


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