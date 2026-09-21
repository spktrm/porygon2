"""Fixed-label consequence probe: can what an action then CAUSED be read
linearly from the rows the policy scores that action with?

Inputs are post-trunk rows at the decision: the taken cell's source and
target readout rows (the pair `FlatActionReadout` scores), against the CLS
row alone as the state-only control. Labels come from the event log between
that request and the next one, and no training loss touches them -- so
unlike a latent prediction loss, the target does not move when the trunk
does, and the read is comparable across checkpoints. The probes are refit
from scratch at every checkpoint, on a split BY GAME, with balanced class
weights -- a sleep CANT or a status is rare, and an unweighted probe that
predicts the majority reads the same as rows that carry nothing. Balanced
accuracy is the headline.

The retention probes beside it decode fixed raw features (hp, status,
fainted) from the public entity rows: a trunk that made its rows easier to
predict by discarding content shows up here, which no energy or loss panel
can see.

The value probes regress the game's outcome on the mean of the 15 public
state rows (12 entity + 3 field) and on the CLS row: the measured prior for
a value reader restricted to those rows.

    env/bin/python -m rl.probes.consequence_probe \\
        --root runtime/tactical-cohort-20260920 --checkpoint ckpts/gen9/ckpt_N \\
        --out runtime/tactical-cohort-20260920/consequence-ckpt_N.json
"""

import argparse
import json
import logging
import os
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, log_loss, r2_score

from rl.environment.event_labels import SIDE_MINE, EventKind, step_events
from rl.environment.protos.features_pb2 import EntityPublicNodeFeature, InfoFeature
from rl.environment.utils import acted_rows
from rl.model.constants import (
    _BANK_MOVE_OFFSET,
    CELL_BANK_SRC,
    CELL_BANK_TGT,
    CLS_ROW,
    FIELD_ROWS,
    MOVE_ROWS,
    NUM_PUBLIC_SLOTS,
    OPP_ACTIVE_PUBLIC_ROWS,
    PRIVATE_ROWS,
    PUBLIC_ROWS,
    TARGET_ROWS,
)
from rl.model.player_model import get_player_model
from rl.offline import harness
from rl.online.artifact import player_model_config_for
from rl.online.config import get_learner_config
from rl.online.training.batching import stack_batch
from rl.probes.separation_probe import actor_input_of

logger = logging.getLogger(__name__)

_REQUEST_COUNT = InfoFeature.INFO_FEATURE__REQUEST_COUNT
_ORDER = slice(
    InfoFeature.INFO_FEATURE__PUBLIC_ORDER_0,
    InfoFeature.INFO_FEATURE__PUBLIC_ORDER_0 + NUM_PUBLIC_SLOTS,
)
_HP = EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__HP_RATIO
_STATUS = EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__STATUS
_FAINTED = EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__FAINTED
RETENTION = ("hp_quartile", "status", "fainted")
_OPP_ROW = int(OPP_ACTIVE_PUBLIC_ROWS[0])
_NUM_MOVE_SLOTS = MOVE_ROWS.stop - MOVE_ROWS.start
LABELS = ("own_move_executed", "opp_active_hp_moved", "any_faint", "moved_first")
PROPENSITY_EDGES = (0.0, 0.05, 0.2, 0.5, 1.0 + 1e-9)


def interval_labels(events, interval, env, time_index, column):
    """The transition's labels, NaN where one is undefined for it."""
    mine = events.actor_side[interval] == SIDE_MINE
    kind = events.kind[interval]
    own_attempt = mine & ((kind == EventKind.MOVE) | (kind == EventKind.CANT))
    their_attempt = ~mine & ((kind == EventKind.MOVE) | (kind == EventKind.CANT))

    executed = np.nan
    if own_attempt.any():
        executed = float((mine & (kind == EventKind.MOVE)).any())

    moved_first = np.nan
    if own_attempt.any() and their_attempt.any():
        moved_first = float(own_attempt.argmax() < their_attempt.argmax())

    hp_moved = np.nan
    order_now = env.info[time_index, column, _ORDER]
    order_next = env.info[time_index + 1, column, _ORDER]
    identity = order_now[_OPP_ROW]
    matches = np.flatnonzero(order_next == identity)
    if identity >= 0 and len(matches) == 1:
        hp_now = env.public_team[time_index, column, _OPP_ROW, _HP]
        hp_next = env.public_team[time_index + 1, column, matches[0], _HP]
        hp_moved = float(hp_now != hp_next)

    return dict(
        own_move_executed=executed,
        opp_active_hp_moved=hp_moved,
        any_faint=float((kind == EventKind.FAINT).any()),
        moved_first=moved_first,
    )


def collect(arguments):
    root = Path(arguments.root)
    sides = harness.load(str(root / "games.pkl"))
    chunks = []
    game_of_chunk = []
    for game_id, side in enumerate(sides):
        chunks.extend(side)
        game_of_chunk.extend([game_id] * len(side))

    net = get_player_model(player_model_config_for(get_learner_config()))
    variables = jax.device_put(harness.load_params(arguments.checkpoint))
    read_rows = jax.jit(
        jax.vmap(
            lambda params, inputs, outputs: net.apply(
                params, inputs, outputs, method=harness.encode_policy_rows
            ),
            in_axes=(None, 1, 1),
            out_axes=1,
        )
    )

    outcomes = [harness.outcome(side) for side in sides]
    records = {
        name: []
        for name in (
            "src",
            "tgt",
            "cls",
            "pooled",
            "outcome",
            "game",
            "propensity",
            "move",
        )
    }
    records.update({name: [] for name in LABELS})
    entity = {name: [] for name in ("row", "game", "hp", "status", "fainted")}
    rng = np.random.default_rng(arguments.seed)
    for start in range(0, len(chunks), arguments.batch):
        group = chunks[start : start + arguments.batch]
        if len(group) < arguments.batch:
            # One compiled shape: the short final batch is padded with its
            # own last chunk and only the real columns are read.
            group = group + [group[-1]] * (arguments.batch - len(group))
        batch = stack_batch(group)
        transitions = batch.player_transitions
        env = jax.tree.map(np.asarray, transitions.env_output)
        rows, row_valid = read_rows(
            variables, actor_input_of(batch), transitions.agent_output.actor_output
        )
        rows = np.asarray(rows, np.float32)
        row_valid = np.asarray(row_valid)
        head = transitions.agent_output.actor_output.action_head
        taken = np.asarray(head.action_index)
        propensity = np.exp(np.asarray(head.log_prob, np.float32))
        acted = acted_rows(env.done)
        field = np.asarray(batch.player_history.field)
        packed = jax.tree.map(np.asarray, batch.player_packed_history)
        for column in range(min(arguments.batch, len(chunks) - start)):
            events = step_events(
                field[:, column],
                packed.edge_cache[:, column],
                packed.public_cache[:, column],
                packed.revealed_cache[:, column],
            )
            for time_index in np.flatnonzero(acted[:, column]):
                next_count = env.info[time_index + 1, column, _REQUEST_COUNT]
                interval = events.valid & (events.request_count == next_count)
                if not interval.any():
                    continue
                sequence = rows[time_index, column]
                bank = np.concatenate(
                    (sequence[PRIVATE_ROWS], sequence[MOVE_ROWS], sequence[TARGET_ROWS])
                )
                cell = int(taken[time_index, column])
                source = int(CELL_BANK_SRC[cell])
                records["src"].append(bank[source])
                records["tgt"].append(bank[int(CELL_BANK_TGT[cell])])
                records["cls"].append(sequence[CLS_ROW])
                state_rows = np.concatenate(
                    (sequence[PUBLIC_ROWS], sequence[FIELD_ROWS])
                )
                state_valid = np.concatenate(
                    (
                        row_valid[time_index, column, PUBLIC_ROWS],
                        row_valid[time_index, column, FIELD_ROWS],
                    )
                )
                records["pooled"].append(state_rows[state_valid].mean(axis=0))
                records["outcome"].append(outcomes[game_of_chunk[start + column]])
                records["game"].append(game_of_chunk[start + column])
                records["propensity"].append(propensity[time_index, column])
                records["move"].append(
                    0 <= source - _BANK_MOVE_OFFSET < _NUM_MOVE_SLOTS
                )
                labels = interval_labels(events, interval, env, time_index, column)
                for name in LABELS:
                    records[name].append(labels[name])
                live = np.flatnonzero(row_valid[time_index, column, PUBLIC_ROWS])
                drawn = rng.choice(
                    live, min(arguments.rows_per_state, len(live)), replace=False
                )
                for slot in drawn:
                    features = env.public_team[time_index, column, slot]
                    entity["row"].append(sequence[PUBLIC_ROWS.start + slot])
                    entity["game"].append(game_of_chunk[start + column])
                    entity["hp"].append(features[_HP])
                    entity["status"].append(features[_STATUS])
                    entity["fainted"].append(features[_FAINTED])
        if (start // arguments.batch) % 100 == 0:
            logger.info("read %d/%d chunks", start, len(chunks))
    return (
        {name: np.asarray(values) for name, values in records.items()},
        {name: np.asarray(values) for name, values in entity.items()},
    )


def feature_sets(records):
    action = np.concatenate((records["src"], records["tgt"]), axis=1)
    return dict(
        action=action,
        state=records["cls"],
        action_and_state=np.concatenate((action, records["cls"]), axis=1),
    )


def fit_label(features, labels, train, test, propensity):
    """One label: every feature set on the same game-level split."""
    result = dict(
        train=int(train.sum()),
        test=int(test.sum()),
        positive_rate=float(labels[test].mean()),
        majority_accuracy=float(max(labels[test].mean(), 1 - labels[test].mean())),
    )
    prior = np.full(test.sum(), labels[train].mean()).clip(1e-6, 1 - 1e-6)
    result["prior_log_loss"] = float(log_loss(labels[test], prior, labels=[0, 1]))
    for name, values in features.items():
        probe = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")
        probe.fit(values[train], labels[train])
        predicted = probe.predict(values[test])
        probability = probe.predict_proba(values[test])[:, 1]
        result[f"{name}_accuracy"] = float((predicted == labels[test]).mean())
        result[f"{name}_balanced_accuracy"] = float(
            balanced_accuracy_score(labels[test], predicted)
        )
        result[f"{name}_log_loss"] = float(
            log_loss(labels[test], probability, labels=[0, 1])
        )
        if name == "action":
            for low, high in zip(PROPENSITY_EDGES, PROPENSITY_EDGES[1:]):
                chosen = (propensity[test] >= low) & (propensity[test] < high)
                key = f"action_accuracy_propensity_{low:g}_{min(high, 1.0):g}"
                result[f"{key}_states"] = int(chosen.sum())
                if chosen.any():
                    result[key] = float(
                        (predicted[chosen] == labels[test][chosen]).mean()
                    )
    return result


def fit_retention(entity, held_out):
    """Multi-class decode of fixed raw features from a public entity row, on
    the consequence probes' own game split."""
    is_test = np.isin(entity["game"], held_out)
    # Fixed-width bins of the hp fraction (0 = fainted ... 4 = above three
    # quarters): quantile edges collapse because most rows sit at full hp.
    fraction = entity["hp"] / max(float(entity["hp"].max()), 1.0)
    targets = dict(
        hp_quartile=np.ceil(fraction * 4).astype(int),
        status=entity["status"].astype(int),
        fainted=entity["fainted"].astype(int),
    )
    result = dict(rows=len(is_test))
    for name in RETENTION:
        labels = targets[name]
        if len(np.unique(labels[~is_test])) < 2:
            result[name] = dict(skipped="one class only")
            continue
        probe = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")
        probe.fit(entity["row"][~is_test], labels[~is_test])
        predicted = probe.predict(entity["row"][is_test])
        counts = np.bincount(labels[is_test])
        result[name] = dict(
            accuracy=float((predicted == labels[is_test]).mean()),
            balanced_accuracy=float(
                balanced_accuracy_score(labels[is_test], predicted)
            ),
            majority_accuracy=float(counts.max() / counts.sum()),
            classes=int(len(np.unique(labels))),
        )
    return result


def fit(records, entity, arguments):
    rng = np.random.default_rng(arguments.seed)
    games = np.unique(records["game"])
    held_out = rng.permutation(games)[
        : int(round(len(games) * arguments.test_fraction))
    ]
    is_test = np.isin(records["game"], held_out)
    features = feature_sets(records)
    result = dict(
        transitions=len(records["game"]),
        games=len(games),
        test_games=sorted(int(game) for game in held_out),
    )
    for name in LABELS:
        defined = ~np.isnan(records[name])
        if name == "own_move_executed":
            defined &= records["move"]
        labels = records[name][defined].astype(int)
        train = ~is_test[defined]
        test = is_test[defined]
        if len(np.unique(labels[train])) < 2 or len(np.unique(labels[test])) < 2:
            result[name] = dict(skipped="one class only", states=int(defined.sum()))
            continue
        selected = {key: values[defined] for key, values in features.items()}
        result[name] = fit_label(
            selected, labels, train, test, records["propensity"][defined]
        )
    result["retention"] = fit_retention(entity, held_out)
    result["value"] = {}
    for name in ("pooled", "cls"):
        probe = Ridge(alpha=1.0)
        probe.fit(records[name][~is_test], records["outcome"][~is_test])
        result["value"][f"{name}_r2"] = float(
            r2_score(records["outcome"][is_test], probe.predict(records[name][is_test]))
        )
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--test-fraction", type=float, default=0.3)
    parser.add_argument("--rows-per-state", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    records, entity = collect(arguments)
    result = dict(checkpoint=arguments.checkpoint, **fit(records, entity, arguments))
    if arguments.out:
        Path(arguments.out).parent.mkdir(parents=True, exist_ok=True)
        Path(arguments.out).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
