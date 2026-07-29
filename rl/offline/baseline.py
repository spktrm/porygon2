"""Model-free reference evaluation for the offline critic.

Computes a hand-crafted win predictor — side-differenced faint count + mean
HP, read directly from the raw cache features at each decision point (the
rule that scores 6/6 on terminal states) — over held-out games, stratified
by game phase. Optionally evaluates a trained artifact on the same states.

This bounds the task from below with zero learned parameters: a healthy
trained critic must match or beat the hand rule in every phase bucket. A
model below the hand rule late-game indicates a broken learned pathway,
not a hard task.

Usage:
    python -m rl.offline.baseline [--artifact ckpts/offline/.../ckpt_best]
        [--num-records 2000] [--split holdout|train]
"""

import argparse

import jax
import numpy as np

from constants import MAX_RATIO_TOKEN
from rl.environment.protos.features_pb2 import (
    EntityEdgeFeature,
    EntityPublicNodeFeature,
    FieldFeature,
    InfoFeature,
)
from rl.model.history_encoder import NUM_PUBLIC_SLOTS
from rl.offline.artifact import load_critic_params, make_potential_apply
from rl.offline.config import get_offline_config
from rl.offline.dataset import (
    OfflineExample,
    _is_holdout,
    collate,
    iter_shard_payloads,
    list_shards,
    record_to_examples,
)

_RELEVANT = [
    FieldFeature.FIELD_FEATURE__RELEVANT_ENTITY_IDX0,
    FieldFeature.FIELD_FEATURE__RELEVANT_ENTITY_IDX1,
    FieldFeature.FIELD_FEATURE__RELEVANT_ENTITY_IDX2,
    FieldFeature.FIELD_FEATURE__RELEVANT_ENTITY_IDX3,
]

NUM_PHASE_BUCKETS = 4


def hand_scores(example: OfflineExample) -> np.ndarray:
    """(T,) side-differenced hand score per decision point; > 0 predicts
    win. Replays the shared history caches forward, tracking each slot's
    latest snapshot as of each request — the same prefix logic as
    state_at_requests, in numpy."""
    env = example.actor_input.env
    packed = example.actor_input.packed_history
    field = np.asarray(example.actor_input.history.field)

    request_counts = np.asarray(env.info)[:, InfoFeature.INFO_FEATURE__REQUEST_COUNT]
    stamps = field[:, FieldFeature.FIELD_FEATURE__REQUEST_COUNT]
    valid_rows = field[:, FieldFeature.FIELD_FEATURE__VALID] == 1
    relevant = field[:, _RELEVANT]
    num_relevant = field[:, FieldFeature.FIELD_FEATURE__NUM_RELEVANT]

    edge_cache = np.asarray(packed.edge_cache)
    public_cache = np.asarray(packed.public_cache)
    slot_of_row = edge_cache[:, EntityEdgeFeature.ENTITY_EDGE_FEATURE__ENTITY_IDX]
    side_of_row = public_cache[
        :, EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__SIDE
    ]
    hp_of_row = (
        public_cache[:, EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__HP_RATIO]
        / MAX_RATIO_TOKEN
    )
    fainted_of_row = public_cache[
        :, EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__FAINTED
    ]

    T = np.asarray(env.done).shape[0]
    H = field.shape[0]
    slot_last: dict[int, int] = {}
    scores = np.zeros(T, dtype=np.float32)
    h = 0
    for t in range(T):
        while h < H and valid_rows[h] and stamps[h] <= request_counts[t]:
            for k in range(int(num_relevant[h])):
                row = int(relevant[h, k])
                slot = int(slot_of_row[row])
                if 0 <= slot < NUM_PUBLIC_SLOTS:
                    slot_last[slot] = row
            h += 1
        my_hp, opp_hp, my_faints, opp_faints, my_n, opp_n = 0.0, 0.0, 0, 0, 0, 0
        for row in slot_last.values():
            if side_of_row[row] == 1:
                my_hp += hp_of_row[row]
                my_faints += fainted_of_row[row]
                my_n += 1
            else:
                opp_hp += hp_of_row[row]
                opp_faints += fainted_of_row[row]
                opp_n += 1
        hp_diff = my_hp / max(my_n, 1) - opp_hp / max(opp_n, 1)
        scores[t] = (opp_faints - my_faints) + 4.0 * hp_diff
    return scores


class PhaseAccuracy:
    def __init__(self, name: str):
        self.name = name
        self.correct = np.zeros(NUM_PHASE_BUCKETS, dtype=np.int64)
        self.total = np.zeros(NUM_PHASE_BUCKETS, dtype=np.int64)
        self.last_correct = 0
        self.last_total = 0

    def add(self, predict_win: np.ndarray, label_win: bool):
        t_valid = predict_win.shape[0]
        buckets = np.minimum(
            (np.arange(t_valid) * NUM_PHASE_BUCKETS) // max(t_valid, 1),
            NUM_PHASE_BUCKETS - 1,
        )
        correct = predict_win == label_win
        for b in range(NUM_PHASE_BUCKETS):
            sel = buckets == b
            self.correct[b] += correct[sel].sum()
            self.total[b] += sel.sum()
        self.last_correct += int(correct[-1])
        self.last_total += 1

    def report(self):
        cells = " | ".join(
            f"Q{b + 1} {self.correct[b] / max(self.total[b], 1):.3f}"
            for b in range(NUM_PHASE_BUCKETS)
        )
        overall = self.correct.sum() / max(self.total.sum(), 1)
        last = self.last_correct / max(self.last_total, 1)
        print(
            f"{self.name:<10} {cells} | overall {overall:.3f} | "
            f"last-step {last:.3f}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=str, default=None)
    parser.add_argument("--artifact", type=str, default=None)
    parser.add_argument("--num-records", type=int, default=2000)
    parser.add_argument("--split", choices=["holdout", "train"], default="holdout")
    args = parser.parse_args()

    config = get_offline_config()
    if args.dataset_dir:
        config = config.replace(dataset_dir=args.dataset_dir)
    holdout = args.split == "holdout"

    examples: list[OfflineExample] = []
    for shard in list_shards(config):
        for index, payload in enumerate(iter_shard_payloads(shard)):
            if _is_holdout(shard, index, config.holdout_modulus) != holdout:
                continue
            examples.extend(record_to_examples(payload, config))
            if len(examples) >= 2 * args.num_records:
                break
        if len(examples) >= 2 * args.num_records:
            break
    print(f"{len(examples)} trajectories from the {args.split} split")

    potential_fn = None
    if args.artifact:
        params = load_critic_params(args.artifact)
        potential_fn = make_potential_apply(config.generation)
        print(f"loaded artifact {args.artifact}")

    hand = PhaseAccuracy("hand-rule")
    model = PhaseAccuracy("model")
    for start in range(0, len(examples) - config.batch_size + 1, config.batch_size):
        chunk = examples[start : start + config.batch_size]
        lengths = [np.asarray(e.actor_input.env.done).shape[0] for e in chunk]
        phis = None
        if potential_fn is not None:
            batch = collate(chunk, config)
            phis = np.asarray(
                jax.device_get(potential_fn(params, batch.actor_input))
            )  # (T_padded, B)
        for i, example in enumerate(chunk):
            label_win = bool(np.asarray(example.label)[2] == 1)
            t_valid = lengths[i]
            hand.add(hand_scores(example)[:t_valid] > 0, label_win)
            if phis is not None:
                model.add(phis[:t_valid, i] > 0, label_win)

    print(f"\naccuracy by game phase (quartiles of each game's length):")
    hand.report()
    if potential_fn is not None:
        model.report()
        print(
            "\nA working critic matches or beats the hand rule in every "
            "bucket; below it (especially Q4/last-step) means the learned "
            "pathway is losing information the inputs provably contain."
        )


if __name__ == "__main__":
    main()
