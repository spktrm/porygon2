"""Executed-event PROBE (stochastic-transition plan Step 4b, widened
2026-09-06): what does the chance code carry about the opponent's
EXECUTED action, and where along the prior's path is that information
lost?

For every t -> t+1 transition of self-play games on a checkpoint (played
through the second service), the LABEL is the opponent's first executed
decision event on the history edges the transition spans -- the window
steps whose FIELD_FEATURE__REQUEST_COUNT equals the request count at t+1
(the learner's `transition_edges` join) -- one of NONE / MOVE / SWITCH /
DRAG / CANT, plus the MOVE_TOKEN on MOVE rows. Executed is not chosen
(the log shows the sim's resolution of a choice under chance); the label
is read by a stop-gradient readout and NEVER by the model.

The same label is predicted from several INPUTS, each frozen:
  post        the posterior's code probabilities (2 x 16)
  prior       the prior's code probabilities
  prior_feat  the prior net's EXACT input (RowRead(h_t) ; src ; tgt)
  delta_read  the posterior's own evidence, RowRead(h_{t+1} - h_t)
  pooled      masked-mean h_t + the src/tgt action rows
  rows_pca    the full 73 x 256 post-trunk rows, PCA'd per fold
with a linear readout (logistic regression) and a small MLP, held out by
SIDE (the mirrored side of the same game is in-sample: a weak leak,
recorded). Reported as accuracy above the majority marginal -- 0 is
"knows nothing", higher is better -- and the pre-registered branches:
(a) predictable from the rows but not from the prior's z -> the prior's
READ is the deficit; (b) not predictable from the rows even nonlinearly
-> the frozen policy representation lacks opponent intent; (c) not
predictable from the posterior's z either -> the code is chance /
reveal dominated on this clock.

    PS_SERVICE_URI=ws://localhost:8081 env/bin/python -m rl.offline.event_probe \
        --ckpt ckpts/gen9/ckpt_XXXXXXXX --games 60
"""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.12")

import argparse  # noqa: E402
import logging  # noqa: E402
from collections import Counter  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
from sklearn.neural_network import MLPClassifier  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from rl.environment.protos.enums_pb2 import BattlemajorargsEnum  # noqa: E402
from rl.environment.protos.features_pb2 import (  # noqa: E402
    EntityEdgeFeature,
    EntityPublicNodeFeature,
    FieldFeature,
    InfoFeature,
)
from rl.model.categoricals import unimix_probs  # noqa: E402
from rl.model.config import get_player_model_config  # noqa: E402
from rl.model.history_encoder import (  # noqa: E402
    SIDE_MINE,
    relevant_edges,
    source_rows,
)
from rl.model.player_model import get_player_model  # noqa: E402
from rl.offline import harness  # noqa: E402
from rl.offline.harness import encode_policy_rows  # noqa: E402
from rl.offline.separation_probe import actor_input_of  # noqa: E402
from rl.offline.trunk_homogeneity import valid_steps  # noqa: E402
from rl.online.training.batching import stack_batch  # noqa: E402

EVENT_NAMES = ("NONE", "MOVE", "SWITCH", "DRAG", "CANT")
EVENT_OF_MAJOR = {
    BattlemajorargsEnum.BATTLEMAJORARGS_ENUM__MOVE: 1,
    BattlemajorargsEnum.BATTLEMAJORARGS_ENUM__SWITCH: 2,
    BattlemajorargsEnum.BATTLEMAJORARGS_ENUM__DRAG: 3,
    BattlemajorargsEnum.BATTLEMAJORARGS_ENUM__CANT: 4,
}
SIDE_OPP = 1 - SIDE_MINE
MIN_TOKEN_COUNT = 5
INPUT_NAMES = ("post", "prior", "prior_feat", "delta_read", "pooled", "rows_pca")


def _transition_reads(module, rows, row_valid, actions):
    """Per t -> t+1 pair: the prior's and posterior's code probabilities
    and every intermediate the readouts compare (the prior net's exact
    input, the posterior's delta read, the pooled rows)."""
    transition = module.transition

    def one(row, ok, action, next_row, next_ok):
        src_row, tgt_row = transition.action_rows(row, action)
        features = transition.prior_features(row, ok, src_row, tgt_row)
        prior_logits = transition.code_logits(transition.prior_read_net, features)
        delta_read = transition.row_read(next_row - row, ok & next_ok)
        post_logits = transition.code_logits(
            transition.posterior_read_net,
            jnp.concatenate((features, delta_read), axis=-1),
        )
        weights = ok.astype(jnp.float32)[:, None]
        pooled = (row.astype(jnp.float32) * weights).sum(0) / jnp.maximum(
            weights.sum(), 1.0
        )
        return {
            "post": unimix_probs(post_logits).reshape(-1),
            "prior": unimix_probs(prior_logits).reshape(-1),
            "prior_feat": features.astype(jnp.float32),
            "delta_read": delta_read.astype(jnp.float32),
            "pooled": jnp.concatenate(
                (pooled, src_row.astype(jnp.float32), tgt_row.astype(jnp.float32))
            ),
            "rows": jnp.where(ok[:, None], row, 0).astype(jnp.float32).reshape(-1),
        }

    return jax.vmap(one)(
        rows[:-1], row_valid[:-1], actions[:-1], rows[1:], row_valid[1:]
    )


def _first_event(edges, edge_mask, source, sides, side):
    """The first decision event by `side` over the spanned steps, in
    step order, with its move token: (event class, token)."""
    for step_edges, step_mask, step_source, step_sides in zip(
        edges, edge_mask, source, sides
    ):
        for edge, live, is_source, edge_side in zip(
            step_edges, step_mask, step_source, step_sides
        ):
            if not (live and is_source and edge_side == side):
                continue
            event = EVENT_OF_MAJOR.get(
                int(edge[EntityEdgeFeature.ENTITY_EDGE_FEATURE__MAJOR_ARG])
            )
            if event is None:
                continue
            token = 0
            if event == 1:
                token = int(edge[EntityEdgeFeature.ENTITY_EDGE_FEATURE__MOVE_TOKEN])
            return event, token
    return 0, 0


def chunk_labels(chunk):
    """Per transition t -> t+1 of one chunk: opponent event, opponent move
    token, my own event (the readout's positive control -- my action is a
    known input), and the spanned edge count."""
    field = np.asarray(chunk.player_history.field)
    public = np.asarray(chunk.player_packed_history.public_cache)
    edge_cache = np.asarray(chunk.player_packed_history.edge_cache)
    info = np.asarray(chunk.player_transitions.env_output.info)
    relevant, edge_mask = (
        np.asarray(array) for array in relevant_edges(jnp.asarray(field))
    )
    major = edge_cache[relevant, EntityEdgeFeature.ENTITY_EDGE_FEATURE__MAJOR_ARG]
    source = np.asarray(source_rows(jnp.asarray(major), jnp.asarray(edge_mask)))
    sides = public[relevant, EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__SIDE]
    field_valid = field[:, FieldFeature.FIELD_FEATURE__VALID] > 0
    field_requests = field[:, FieldFeature.FIELD_FEATURE__REQUEST_COUNT]
    requests = info[:, InfoFeature.INFO_FEATURE__REQUEST_COUNT]
    labels = []
    for step in range(info.shape[0] - 1):
        spanned = np.flatnonzero(field_valid & (field_requests == requests[step + 1]))
        step_edges = edge_cache[relevant[spanned]]
        opp_event, opp_token = _first_event(
            step_edges, edge_mask[spanned], source[spanned], sides[spanned], SIDE_OPP
        )
        my_event, _ = _first_event(
            step_edges, edge_mask[spanned], source[spanned], sides[spanned], SIDE_MINE
        )
        labels.append((opp_event, opp_token, my_event, int((edge_mask[spanned]).sum())))
    return np.asarray(labels, dtype=np.int64)


def collect(net, variables, sides):
    encode = jax.jit(
        jax.vmap(
            lambda params, actor_input, actor_output: net.apply(
                params, actor_input, actor_output, method=encode_policy_rows
            ),
            in_axes=(None, 1, 1),
            out_axes=1,
        )
    )
    reads = jax.jit(
        lambda params, rows, row_valid, actions: net.apply(
            params, rows, row_valid, actions, method=_transition_reads
        )
    )
    dev_variables = jax.device_put(variables)
    inputs = {
        name: []
        for name in ("post", "prior", "prior_feat", "delta_read", "pooled", "rows")
    }
    labels, groups = [], []
    for side_index, side in enumerate(sides):
        for chunk in side:
            done = np.asarray(chunk.player_transitions.env_output.done).astype(bool)
            usable = valid_steps(done)
            pair_valid = usable[:-1] & usable[1:]
            if not pair_valid.any():
                continue
            batch = stack_batch([chunk])
            actor_output = batch.player_transitions.agent_output.actor_output
            rows, row_valid = encode(dev_variables, actor_input_of(batch), actor_output)
            actions = jnp.asarray(actor_output.action_head.action_index)[:, 0]
            out = reads(dev_variables, rows[:, 0], row_valid[:, 0], actions)
            out = jax.tree_util.tree_map(np.asarray, out)
            for name in inputs:
                inputs[name].append(out[name][pair_valid])
            labels.append(chunk_labels(chunk)[pair_valid])
            groups.append(np.full(int(pair_valid.sum()), side_index))
    inputs = {name: np.concatenate(parts) for name, parts in inputs.items()}
    return inputs, np.concatenate(labels), np.concatenate(groups)


def _linear():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000))


def _mlp(seed):
    return make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(256,),
            alpha=1e-3,
            early_stopping=True,
            max_iter=300,
            random_state=seed,
        ),
    )


def held_out_accuracy(features, target, groups, readout, pca_dims=0, folds=5):
    """Mean held-out accuracy over GroupKFold; PCA (if any) is fit on
    the train fold only."""
    correct = 0
    for train, test in GroupKFold(n_splits=folds).split(features, target, groups):
        train_x, test_x = features[train], features[test]
        if pca_dims:
            pca = PCA(pca_dims, svd_solver="randomized", random_state=0)
            train_x = pca.fit_transform(train_x)
            test_x = pca.transform(test_x)
        model = readout()
        model.fit(train_x, target[train])
        correct += int((model.predict(test_x) == target[test]).sum())
    return correct / target.shape[0]


def report(name, inputs, target, groups, seed, rows_mask=None):
    if rows_mask is not None:
        inputs = {key: value[rows_mask] for key, value in inputs.items()}
        target = target[rows_mask]
        groups = groups[rows_mask]
    counts = Counter(target.tolist())
    majority = max(counts.values()) / target.shape[0]
    print(
        f"\n== {name}: n={target.shape[0]}, classes={len(counts)}, "
        f"majority={majority:.3f}",
        flush=True,
    )
    print(
        f"{'input':>12} {'dims':>6} {'linear':>8} {'mlp':>8}   (accuracy above the majority marginal)"
    )
    for input_name in INPUT_NAMES:
        if input_name == "rows_pca":
            pca_dims = 256
            features = inputs["rows"]
        else:
            pca_dims = 0
            features = inputs[input_name]
        linear = held_out_accuracy(features, target, groups, _linear, pca_dims)
        mlp = held_out_accuracy(features, target, groups, lambda: _mlp(seed), pca_dims)
        print(
            f"{input_name:>12} {features.shape[1]:>6} {linear - majority:>+8.3f} "
            f"{mlp - majority:>+8.3f}",
            flush=True,
        )


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--games-pkl", default=None, help="reuse played games")
    parser.add_argument("--games", type=int, default=60)
    parser.add_argument("--pairs", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)

    variables = harness.load_params(args.ckpt)
    if args.games_pkl and os.path.exists(args.games_pkl):
        sides = harness.load(args.games_pkl)
    else:
        sides = harness.play_games(
            variables, args.games, pairs=args.pairs, tag="eventprobe", seed=args.seed
        )
        if args.games_pkl:
            harness.dump(sides, args.games_pkl)
    net = get_player_model(get_player_model_config(9, train=True))
    inputs, labels, groups = collect(net, variables, sides)
    opp_event, opp_token, my_event, edges = labels.T
    print(
        f"params: {args.ckpt}; {len(sides)} sides, {labels.shape[0]} transitions, "
        f"edges mean {edges.mean():.2f} p90 {np.percentile(edges, 90):.0f}",
        flush=True,
    )
    print(
        "opp event marginal: "
        + ", ".join(
            f"{EVENT_NAMES[event]} {np.mean(opp_event == event):.3f}"
            for event in range(len(EVENT_NAMES))
        )
    )
    print(
        "my event marginal:  "
        + ", ".join(
            f"{EVENT_NAMES[event]} {np.mean(my_event == event):.3f}"
            for event in range(len(EVENT_NAMES))
        ),
        flush=True,
    )
    report("opponent executed event", inputs, opp_event, groups, args.seed)
    report("my own executed event (control)", inputs, my_event, groups, args.seed)
    is_move = opp_event == 1
    token_counts = Counter(opp_token[is_move].tolist())
    frequent = {
        token for token, count in token_counts.items() if count >= MIN_TOKEN_COUNT
    }
    token_target = np.where(np.isin(opp_token, list(frequent)), opp_token, -1)
    print(
        f"\nopp move token: {len(token_counts)} distinct on MOVE rows, "
        f"{len(frequent)} with >= {MIN_TOKEN_COUNT} rows (the rest -> OTHER)"
    )
    report(
        "opponent move token (MOVE rows only)",
        inputs,
        token_target,
        groups,
        args.seed,
        rows_mask=is_move,
    )


if __name__ == "__main__":
    main()
