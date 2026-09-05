"""Is the mean dynamics head sitting between branches? (2026-09-05)

Step 1 of the stochastic-transition plan. `dynamics_delta_head` predicts
the conditional MEAN of each target row's next-step change, and between
two of my requests the transition branches over things I never observe
as choices: the opponent's decision, the rolls, the reveals. A mean over
discrete branches is on no branch -- a mon 60% fainted -- and the value
of that state is not the expected value of the branches. Two reads,
both on the EMA params (`target_params`, what the actors play) over
self-play chunks:

  1. RESIDUAL BIMODALITY on hp-moved public rows: the residual
     `delta - pred` projected onto the hp subspace of the public row
     (`dynamics_hp_basis`), scored along its top principal direction and
     fitted as one vs two Gaussians; `far_frac` is the share of rows on
     which the residual is at least half the delta itself -- the mean is
     nowhere near where the row actually went. The faint split is the
     positive control: a mon that fainted and one that survived a hit are
     two known branches, and the mean should sit between their residuals.
  2. THE VALUE GAP: the 21 predicted pre-trunk rows are written into the
     REAL t+1 sequence (each row j at t lands on its `dynamics_alignment`
     row at t+1, with that row's group/row bias), the frozen trunk and
     value head run over it, and `|V(sub) - V(real t+1)|` is what the
     mean's state is worth against the state that happened. The copy
     predictor (delta 0) and `|V(t+1) - V(t)|` bracket it from above.

Both split by what the transition spans (`transition_edges`) and whether
an opponent token was revealed (`transition_reveals`), the learner's own
step-1 panels.

    PS_SERVICE_URI=ws://localhost:8081 env/bin/python \\
        rl/offline/transition_probe.py --ckpt ckpts/gen9/ckpt_01220000 --games 60
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
from sklearn.mixture import GaussianMixture  # noqa: E402

from rl.environment.protos.features_pb2 import EntityPublicNodeFeature  # noqa: E402
from rl.model.config import get_player_model_config  # noqa: E402
from rl.model.constants import (  # noqa: E402
    ALLY_TARGET_ROWS,
    CLS_ROW,
    DYNAMICS_GROUP_SLICES,
    DYNAMICS_TARGET_ROWS,
    ENEMY_TARGET_ROWS,
    HISTORY_ENTITY_ROWS,
    MY_ACTIVE_PUBLIC_ROWS,
    OPP_ACTIVE_PUBLIC_ROWS,
    PUBLIC_ROWS,
    SEQUENCE_GROUP_IDS,
    SEQUENCE_READ_MASK,
    SEQUENCE_SLICES,
    TARGET_ROWS,
    SequenceGroup,
)
from rl.model.encoder import Encoder  # noqa: E402
from rl.model.player_model import dynamics_alignment, get_player_model  # noqa: E402
from rl.offline import harness  # noqa: E402
from rl.offline.separation_probe import actor_input_of  # noqa: E402
from rl.offline.trunk_homogeneity import valid_steps  # noqa: E402
from rl.online.training.batching import stack_batch  # noqa: E402
from rl.online.training.train_step import (  # noqa: E402
    TRANSITION_LONG_EDGES,
    TRANSITION_SHORT_EDGES,
    dynamics_hp_basis,
    transition_edges,
    transition_reveals,
)

logger = logging.getLogger(__name__)

HP_COLUMN = EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__HP_RATIO
FAINTED_COLUMN = EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__FAINTED
PUBLIC = DYNAMICS_GROUP_SLICES["public"]


def _probe_forward(module, actor_input, actor_output):
    """One chunk, (T, ...): the value on the real sequence, on the t+1
    sequence with the mean head's rows substituted, and with the copy
    predictor's; the prediction, the pre-trunk target rows and the t -> t+1
    alignment. EMA params throughout: the prediction is what the trained
    head says, the value what the critic says of the state it describes."""
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
    sequence, row_valid, _, target_rows = assemble(encoder, env, *history_inputs)
    kept = encoder.kept_rows()
    read_mask = SEQUENCE_READ_MASK[np.ix_(kept, kept)]
    # The history rows at t+1 (entity diaries, field memory) already encode
    # the real transition, so a value read over the full t+1 sequence sees
    # most of the outcome whatever the 21 rows say. The history-blind read
    # masks them out of every variant alike: what the 21 rows carry on
    # their own. (The INFO row's request kind at t+1 still leaks a
    # force-switch; the plan's mask head is what would predict it.)
    history_blind = np.ones(len(kept), dtype=bool)
    history_blind[HISTORY_ENTITY_ROWS] = False
    history_blind[SEQUENCE_SLICES[SequenceGroup.HISTORY_FIELD]] = False
    history_blind = jnp.asarray(history_blind)
    # The four entity-derived target rows are built by ADDING the active
    # mons' pre-bias public rows (encoder: target_slot_embeddings +
    # public_rows[active] + side bias), so a substituted public row carries
    # its change onto its target row too.
    ally_targets = TARGET_ROWS.start + jnp.asarray(ALLY_TARGET_ROWS)
    enemy_targets = TARGET_ROWS.start + jnp.asarray(ENEMY_TARGET_ROWS)
    my_actives = PUBLIC_ROWS.start + jnp.asarray(MY_ACTIVE_PUBLIC_ROWS)
    opp_actives = PUBLIC_ROWS.start + jnp.asarray(OPP_ACTIVE_PUBLIC_ROWS)

    def value_of(rows, valid):
        trunk_out = jax.vmap(lambda seq, ok: encoder.trunk(seq, ok, read_mask))(
            rows, valid
        )
        return trunk_out, module.v_head(trunk_out[:, CLS_ROW]).expectation

    def blind_value_of(rows, valid):
        _, blind = value_of(rows, valid & history_blind)
        return blind

    trunk_out, value = value_of(sequence, row_valid)
    value_blind = blind_value_of(sequence, row_valid)
    pred = jax.vmap(module._forward_dynamics_head)(
        trunk_out, actor_output.action_head.action_index
    )
    matched, next_index = jax.vmap(dynamics_alignment)(
        jax.tree.map(lambda leaf: leaf[:-1], env),
        jax.tree.map(lambda leaf: leaf[1:], env),
    )
    # Row j's predicted t+1 content lands on its aligned row of the t+1
    # sequence, with THAT row's additive identity (assembled after the
    # target rows are taken; encoder._assemble_sequence).
    dest = jnp.asarray(DYNAMICS_TARGET_ROWS)[next_index]
    dtype = sequence.dtype
    bias = (
        encoder.sequence_group_bias.astype(dtype)[jnp.asarray(SEQUENCE_GROUP_IDS)[dest]]
        + encoder.sequence_row_bias.astype(dtype)[dest]
    )
    real_next = sequence[1:]
    next_valid = row_valid[1:]

    def substituted_value(content):
        current = jnp.take_along_axis(real_next, dest[..., None], axis=1)
        rows = jnp.where(matched[..., None], content.astype(dtype) + bias, current)
        substituted = jax.vmap(lambda seq, at, values: seq.at[at].set(values))(
            real_next, dest, rows
        )
        public_change = substituted - real_next
        substituted = substituted.at[:, ally_targets].add(public_change[:, my_actives])
        substituted = substituted.at[:, enemy_targets].add(
            public_change[:, opp_actives]
        )
        substituted = jnp.where(next_valid[..., None], substituted, 0)
        _, sub_value = value_of(substituted, next_valid)
        return sub_value, blind_value_of(substituted, next_valid)

    target_f32 = target_rows.astype(jnp.float32)
    value_mean, value_mean_blind = substituted_value(
        target_f32[:-1] + pred[:-1].astype(jnp.float32)
    )
    value_copy, value_copy_blind = substituted_value(target_f32[:-1])
    return dict(
        value=value,
        value_blind=value_blind,
        value_mean=value_mean,
        value_mean_blind=value_mean_blind,
        value_copy=value_copy,
        value_copy_blind=value_copy_blind,
        pred=pred[:-1].astype(jnp.float32),
        target=target_f32,
        matched=matched,
        next_index=next_index,
    )


@dataclass
class Collected:
    """Per transition (n,): the three values and the splits; per hp-moved
    public row (m, r): residual and delta in the hp subspace, with the
    row's faint flag."""

    value_now: np.ndarray
    value_next: np.ndarray
    value_mean: np.ndarray
    value_copy: np.ndarray
    value_next_blind: np.ndarray
    value_mean_blind: np.ndarray
    value_copy_blind: np.ndarray
    hp_moved: np.ndarray
    fainted: np.ndarray
    edges: np.ndarray
    reveal: np.ndarray
    residual_hp: np.ndarray
    delta_hp: np.ndarray
    row_fainted: np.ndarray


def collect(net, variables, chunks, batch_size: int) -> Collected:
    apply = jax.jit(
        jax.vmap(
            lambda params, actor_input, actor_output: net.apply(
                params, actor_input, actor_output, method=_probe_forward
            ),
            in_axes=(None, 1, 1),
            out_axes=1,
        )
    )
    splits = jax.jit(
        lambda env, history_field, matched, next_index: (
            transition_edges(env, history_field),
            transition_reveals(env, matched, next_index),
        )
    )
    dev_variables = jax.device_put(variables)
    hp_basis = np.asarray(dynamics_hp_basis(variables))
    parts = {name: [] for name in Collected.__dataclass_fields__}
    for start in range(0, len(chunks), batch_size):
        batch = stack_batch(chunks[start : start + batch_size])
        actor_input = actor_input_of(batch)
        actor_output = batch.player_transitions.agent_output.actor_output
        out = apply(dev_variables, actor_input, actor_output)
        env = actor_input.env
        edges, reveal = splits(
            env, batch.player_history.field, out["matched"], out["next_index"]
        )
        out = jax.tree.map(np.asarray, out)
        done = np.asarray(env.done)
        valid = valid_steps(done)
        # Transition t -> t+1: an action was taken at t, t+1 is a real state.
        step_ok = valid[1:] & ~done[:-1]
        public_matched = out["matched"][..., PUBLIC] & step_ok[..., None]
        hp = np.asarray(env.public_team[..., HP_COLUMN])
        fainted = np.asarray(env.public_team[..., FAINTED_COLUMN])
        public_next = out["next_index"][..., PUBLIC]
        hp_next = np.take_along_axis(hp[1:], public_next, axis=2)
        fainted_next = np.take_along_axis(fainted[1:], public_next, axis=2)
        row_moved = public_matched & (hp[:-1] != hp_next)
        row_fainted = row_moved & (fainted[:-1] == 0) & (fainted_next == 1)

        target = out["target"]
        target_next = np.take_along_axis(
            target[1:], out["next_index"][..., None], axis=2
        )
        delta = target_next - target[:-1]
        residual = delta - out["pred"]
        delta_hp = delta[..., PUBLIC, :] @ hp_basis
        residual_hp = residual[..., PUBLIC, :] @ hp_basis

        parts["value_now"].append(out["value"][:-1][step_ok])
        parts["value_next"].append(out["value"][1:][step_ok])
        parts["value_mean"].append(out["value_mean"][step_ok])
        parts["value_copy"].append(out["value_copy"][step_ok])
        parts["value_next_blind"].append(out["value_blind"][1:][step_ok])
        parts["value_mean_blind"].append(out["value_mean_blind"][step_ok])
        parts["value_copy_blind"].append(out["value_copy_blind"][step_ok])
        parts["hp_moved"].append(row_moved.any(-1)[step_ok])
        parts["fainted"].append(row_fainted.any(-1)[step_ok])
        parts["edges"].append(np.asarray(edges)[step_ok])
        parts["reveal"].append(np.asarray(reveal)[step_ok])
        parts["residual_hp"].append(residual_hp[row_moved])
        parts["delta_hp"].append(delta_hp[row_moved])
        parts["row_fainted"].append(row_fainted[row_moved])
    return Collected(**{name: np.concatenate(rows) for name, rows in parts.items()})


def _gap_line(name, gap, mask):
    if mask.sum() == 0:
        return f"  {name:<14} n=0"
    values = gap[mask]
    return (
        f"  {name:<14} n={mask.sum():<6} mean {values.mean():.4f}"
        f"  p50 {np.median(values):.4f}  p90 {np.percentile(values, 90):.4f}"
    )


def print_value_gap(data: Collected) -> None:
    gap_mean = np.abs(data.value_mean - data.value_next)
    gap_copy = np.abs(data.value_copy - data.value_next)
    gap_now = np.abs(data.value_now - data.value_next)
    everything = np.ones_like(data.hp_moved)
    splits = [
        ("all", everything),
        ("hp_moved", data.hp_moved),
        ("faint", data.fainted),
        ("no_hp_move", ~data.hp_moved),
        ("short", data.edges <= TRANSITION_SHORT_EDGES),
        ("long", data.edges >= TRANSITION_LONG_EDGES),
        ("reveal", data.reveal),
        ("no_reveal", ~data.reveal),
    ]
    print(
        f"\nvalue gap |V(x) - V(real t+1)|, CAT_VF_SUPPORT units; n={len(gap_mean)}"
        f" transitions, hp_moved {data.hp_moved.mean():.3f}, faint"
        f" {data.fainted.mean():.3f}, reveal {data.reveal.mean():.3f},"
        f" edges mean {data.edges.mean():.2f} p90 {np.percentile(data.edges, 90):.0f}"
    )
    gap_mean_blind = np.abs(data.value_mean_blind - data.value_next_blind)
    gap_copy_blind = np.abs(data.value_copy_blind - data.value_next_blind)
    tables = (
        ("MEAN HEAD", gap_mean),
        ("copy", gap_copy),
        ("V(t)", gap_now),
        ("MEAN HEAD, history rows masked", gap_mean_blind),
        ("copy, history rows masked", gap_copy_blind),
    )
    for label, gap in tables:
        print(f"{label}:")
        for name, mask in splits:
            print(_gap_line(name, gap, mask))
    signed = data.value_mean - data.value_next
    print(
        f"mean head signed bias (V(sub) - V(real)): all {signed.mean():+.4f},"
        f" hp_moved {signed[data.hp_moved].mean():+.4f}"
    )


def _ashman(mean_a, var_a, mean_b, var_b) -> float:
    return float(np.sqrt(2) * abs(mean_a - mean_b) / np.sqrt(var_a + var_b))


def print_bimodality(data: Collected, seed: int) -> None:
    residual = data.residual_hp
    delta = data.delta_hp
    n = len(residual)
    print(
        f"\nresidual on hp-moved public rows: n={n}, fainted {data.row_fainted.mean():.3f}"
    )
    if n < 50:
        print("  too few rows")
        return
    res_norm = np.linalg.norm(residual, axis=-1)
    delta_norm = np.linalg.norm(delta, axis=-1)
    gain = 1.0 - np.square(res_norm).sum() / np.square(delta_norm).sum()
    far = res_norm >= 0.5 * delta_norm
    print(
        f"  hp-subspace gain {gain:.3f}; far_frac (|residual| >= 0.5 |delta|)"
        f" {far.mean():.3f} (fainted rows {far[data.row_fainted].mean():.3f},"
        f" survived {far[~data.row_fainted].mean():.3f})"
    )
    centred = residual - residual.mean(0)
    _, _, components = np.linalg.svd(centred, full_matrices=False)
    score = centred @ components[0]
    scores = score[:, None]
    fits = [GaussianMixture(k, n_init=3, random_state=seed).fit(scores) for k in (1, 2)]
    bic = [fit.bic(scores) for fit in fits]
    two = fits[1]
    means = two.means_.ravel()
    variances = two.covariances_.ravel()
    weights = two.weights_
    print(
        f"  top hp direction: BIC 1-comp {bic[0]:.0f} vs 2-comp {bic[1]:.0f}"
        f" (delta {bic[0] - bic[1]:+.0f}, positive favours two);"
        f" 2-comp means {means[0]:+.3f}/{means[1]:+.3f}, weights"
        f" {weights[0]:.2f}/{weights[1]:.2f}, Ashman D"
        f" {_ashman(means[0], variances[0], means[1], variances[1]):.2f}"
        f" (>2 = separated); minority weight {weights.min():.2f}"
    )
    fainted = data.row_fainted
    if fainted.sum() >= 10 and (~fainted).sum() >= 10:
        a, b = score[fainted], score[~fainted]
        print(
            f"  faint control: score mean fainted {a.mean():+.3f} (sd {a.std():.3f})"
            f" vs survived {b.mean():+.3f} (sd {b.std():.3f}), Ashman D"
            f" {_ashman(a.mean(), a.var(), b.mean(), b.var()):.2f};"
            f" residual norm fainted {res_norm[fainted].mean():.3f} vs survived"
            f" {res_norm[~fainted].mean():.3f}"
        )


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--games-pkl", default=None, help="reuse played games")
    parser.add_argument("--games", type=int, default=60)
    parser.add_argument("--pairs", type=int, default=4)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)

    variables = harness.load_params(args.ckpt)
    if args.games_pkl and os.path.exists(args.games_pkl):
        sides = harness.load(args.games_pkl)
    else:
        sides = harness.play_games(
            variables, args.games, pairs=args.pairs, tag="transition", seed=args.seed
        )
        if args.games_pkl:
            harness.dump(sides, args.games_pkl)
    chunks = harness.flatten(sides)
    print(f"params: {args.ckpt}; {len(sides)} sides, {len(chunks)} chunks", flush=True)
    net = get_player_model(get_player_model_config(9, train=True))
    data = collect(net, variables, chunks, args.batch)
    print_value_gap(data)
    print_bimodality(data, args.seed)


if __name__ == "__main__":
    main()
