"""Collect research-training self-play and freeze unilateral interval features.

No existing evaluation archive can be imported as training. The collection
manifest establishes eligibility, paired game identities and the held-out split.
"""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("WANDB_MODE", "disabled")

import argparse
import hashlib
import json
import logging
from pathlib import Path

import jax
import numpy as np

from rl.model.categoricals import unimix_probs
from rl.model.config import get_player_model_config
from rl.model.player_model import get_player_model
from rl.offline import harness
from rl.offline.harness import encode_policy_rows
from rl.offline.interval_data import game_split, iter_intervals
from rl.offline.separation_probe import actor_input_of
from rl.online.training.batching import stack_batch

logger = logging.getLogger(__name__)


def _features(module, actor_input, actor_output):
    rows, valid = encode_policy_rows(module, actor_input, actor_output)
    probabilities = jax.vmap(module.transition.action_logits)(
        rows, actor_output.action_head.action_index
    )
    return rows, valid, unimix_probs(probabilities)


def export_features(checkpoint, sides, game_ids, output):
    network = get_player_model(get_player_model_config(9, train=True))
    params = harness.load_params(checkpoint)
    encode = jax.jit(
        jax.vmap(
            lambda variables, inputs, outputs: network.apply(
                variables, inputs, outputs, method=_features
            ),
            in_axes=(None, 1, 1),
            out_axes=1,
        )
    )
    records = {
        name: []
        for name in (
            "rows",
            "next_rows",
            "valid",
            "action_probs",
            "game",
            "heldout",
            "kind",
            "terminal",
        )
    }
    seen = set()
    for side_index, side in enumerate(sides):
        game_id = game_ids[side_index]
        for chunk in side:
            intervals = list(iter_intervals(chunk))
            if not intervals:
                continue
            batch = stack_batch([chunk])
            rows, valid, probabilities = encode(
                params,
                actor_input_of(batch),
                batch.player_transitions.agent_output.actor_output,
            )
            rows = np.asarray(rows[:, 0], dtype=np.float32)
            valid = np.asarray(valid[:, 0], dtype=bool)
            probabilities = np.asarray(probabilities[:, 0])
            for interval in intervals:
                identity = (side_index, interval.game_step)
                if identity in seen:
                    raise ValueError("Duplicate interval from overlapping chunks")
                seen.add(identity)
                source = interval.source_row
                successor = interval.successor_row
                records["rows"].append(rows[source])
                records["next_rows"].append(rows[successor])
                records["valid"].append(valid[source] | valid[successor])
                records["action_probs"].append(probabilities[source])
                records["game"].append(game_id)
                records["heldout"].append(game_split(game_id) == "heldout")
                records["kind"].append(interval.request_type)
                records["terminal"].append(interval.terminal)
        if (side_index + 1) % 16 == 0:
            logger.info("encoded %d/%d sides", side_index + 1, len(sides))
    arrays = {name: np.asarray(values) for name, values in records.items()}
    np.savez_compressed(output, **arrays)
    print(
        json.dumps(
            {"intervals": len(arrays["rows"]), "heldout": int(arrays["heldout"].sum())}
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--directory", required=True)
    parser.add_argument("--collect-games", type=int, default=0)
    parser.add_argument("--seed", type=int, default=817)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    if not any(device.platform == "gpu" for device in jax.devices()):
        raise RuntimeError("Frozen feature export requires GPU")
    directory = Path(args.directory)
    directory.mkdir(parents=True, exist_ok=True)
    archive = directory / "selfplay.pkl"
    manifest_path = directory / "collection.json"
    if args.collect_games:
        if archive.exists() or manifest_path.exists():
            raise FileExistsError("Refusing to overwrite a collection")
        sides = harness.play_games(
            harness.load_params(args.checkpoint),
            args.collect_games,
            pairs=2,
            tag=f"interval-training-{args.seed}",
            seed=args.seed,
            opponent="self",
            temperature=1.0,
            device="gpu",
            deadline_s=1800,
        )
        if len(sides) != 2 * args.collect_games:
            raise RuntimeError("Incomplete self-play collection")
        harness.dump(sides, str(archive))
        game_ids = [
            f"interval-{args.seed}-stored-{index // 2}" for index in range(len(sides))
        ]
        manifest = {
            "purpose": "research_training",
            "opponent": "self",
            "is_eval": False,
            "checkpoint": args.checkpoint,
            "seed": args.seed,
            "game_ids": game_ids,
            "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
            "split": "game_split(seed=0, heldout_fraction=0.2)",
            "format": "gen9randombattle",
            "temperature": 1.0,
            "identity": "stored completion order; adjacent sides share a game",
        }
        manifest_path.write_text(json.dumps(manifest, indent=2))
    else:
        manifest = json.loads(manifest_path.read_text())
        if manifest["purpose"] != "research_training" or manifest["is_eval"]:
            raise ValueError("Evaluation trajectories cannot supply training features")
        if (
            manifest["archive_sha256"]
            != hashlib.sha256(archive.read_bytes()).hexdigest()
        ):
            raise ValueError("Collection archive does not match its manifest")
        sides = harness.load(str(archive))
        game_ids = manifest["game_ids"]
    if len(game_ids) != len(sides):
        raise ValueError("Each saved perspective must have a declared game identity")
    export_features(args.checkpoint, sides, game_ids, directory / "features.npz")
    (directory / "features.json").write_text(
        json.dumps(
            {
                "checkpoint": args.checkpoint,
                "collection": str(manifest_path),
                "scope": "policy-readable frozen features; both-absent rows unscored",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
