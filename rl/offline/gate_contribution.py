"""Residual-gate contribution read (architecture audit F1/F2, 2026-08-24).

Every RoundBlock residual write is `stream += g * f(x)` with g a zeros-init
scalar gate. The checkpoint shows every zeros-init gate at 1e-3..1e-1 after
74.6k steps while every ones-init gate trained — but the gates multiply
UN-normalised block outputs, so a small g is not proof of a small
contribution. This measures, per gate per round, on real self-play chunks:

    contribution = rms(g_r . f_r(x)) / rms(stream after round r)

over valid rows and in-game steps, plus for every stream the fraction of
its final content that varies with the state at all:

    state_dependence = rms(stream - mean_over_states(stream))
                       / rms(mean_over_states(stream))

(the value rungs start as learned constant queries; if the read gates are
inert their final content is that constant plus noise, and V is a function
of ~nothing).

Usage (second service on 8081, no learner live):
    PS_SERVICE_URI=ws://localhost:8081 env/bin/python -m rl.offline.gate_contribution \\
        ckpts/gen9/ckpt_00074597 --games 16 --chunks 48 [--games-pkl path]
"""

import os

os.environ["COLLECT_INTERMEDIATES"] = "1"  # before any rl.model import

import argparse  # noqa: E402
import logging  # noqa: E402
import sys  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from rl.environment.interfaces import PlayerActorInput  # noqa: E402
from rl.model.config import get_player_model_config  # noqa: E402
from rl.model.heads import HeadParams  # noqa: E402
from rl.model.player_model import get_player_model  # noqa: E402
from rl.offline.harness import load_params  # noqa: E402
from rl.offline.harness import dump, flatten, load, play_games
from rl.online.artifact import merge_params  # noqa: E402
from rl.online.training.batching import stack_batch  # noqa: E402

logger = logging.getLogger("gate_contribution")

# (block intermediate, gate param pattern, stream index, row-group source)
# Row groups: "state" = one group over all state rows; "action" = the
# move/switch/target concat (sizes sown as action_part_sizes); "value_all"
# / "value_private" = the two halves of the shared value_read output.
READS = [
    ("state_global_attn", "state_global_gate", 0, "state"),
    ("opp_cross_attn", "opp_read_gate", 1, "opp"),
    ("action_global_attn", "{}_global_gate", 2, "action"),
    ("state_to_action", "state_to_{}_gate", 2, "action"),
    ("action_to_state", "action_to_state_gate", 0, "state"),
    ("value_read", "value_all_read_gate", 3, "value_all"),
    ("value_read", "value_private_read_gate", 4, "value_private"),
    ("history_to_value_public", "value_public_read_gate", 5, "value_public"),
    ("state_ffw", "state_ffw_gate", 0, "state"),
    ("action_ffw", "{}_ffw_gate", 2, "action"),
    ("opp_ffw", "opp_ffw_gate", 1, "opp"),
    ("value_all_ffw", "value_all_ffw_gate", 3, "value_all"),
    ("value_private_ffw", "value_private_ffw_gate", 4, "value_private"),
    ("value_public_ffw", "value_public_ffw_gate", 5, "value_public"),
]
STREAM_NAMES = ["state", "opp", "action", "value_all", "value_private", "value_public"]
ACTION_GROUPS = ["move", "switch", "target"]


def _first(x):
    # capture_intermediates stores a tuple of call outputs; take the first.
    return x[0] if isinstance(x, (tuple, list)) else x


def _rms(x, mask=None):
    x = np.asarray(x, dtype=np.float64)
    if mask is None:
        return float(np.sqrt(np.mean(x**2)))
    m = np.broadcast_to(np.asarray(mask, bool)[..., None], x.shape)
    return float(np.sqrt(np.sum((x**2) * m) / max(m.sum(), 1)))


def collect(params, chunks, batch, num_rounds):
    net = get_player_model(get_player_model_config(9, train=True))

    def filt(mdl, method):
        return method == "__call__" and "round_trunk" in mdl.path

    def one(p, actor_input, actor_output):
        _, mods = net.apply(
            p,
            actor_input,
            actor_output,
            HeadParams(),
            capture_intermediates=filt,
            mutable=["intermediates"],
        )
        return mods["intermediates"]

    apply = jax.jit(jax.vmap(one, in_axes=(None, 1, 1), out_axes=0))
    # Checkpoints predating a head change lack its leaves (here the Step-3a
    # micro_local routes): overlay the checkpoint on a fresh init, exactly
    # as the learner's resume and the overfit probe do.
    b0 = stack_batch(chunks[:1])
    pt0 = b0.player_transitions
    fresh = jax.jit(net.init)(
        jax.random.key(0),
        jax.tree.map(
            lambda x: x[:, 0],
            PlayerActorInput(
                env=pt0.env_output,
                packed_history=b0.player_packed_history,
                history=b0.player_history,
            ),
        ),
        jax.tree.map(lambda x: x[:, 0], pt0.agent_output.actor_output),
        HeadParams(),
    )
    if params is None:  # --which fresh: read the init itself
        params, kept_fresh = fresh, ["<all: fresh init>"]
    else:
        params, kept_fresh = merge_params(fresh, params)
    logger.info("fresh-init leaves (missing from ckpt): %s", kept_fresh)
    dev_params = jax.device_put(params)
    trunk_params = params["params"]["encoder"]["round_trunk"]
    gates = {
        k: np.asarray(v, dtype=np.float64).reshape(num_rounds)
        for k, v in trunk_params.items()
        if k.endswith("_gate")
    }

    acc = {}  # name -> list of per-batch dicts
    stream_final = {n: [] for n in STREAM_NAMES}
    stream_first = {n: [] for n in STREAM_NAMES}
    for i in range(0, len(chunks), batch):
        b = stack_batch(chunks[i : i + batch])
        pt = b.player_transitions
        actor_input = PlayerActorInput(
            env=pt.env_output,
            packed_history=b.player_packed_history,
            history=b.player_history,
        )
        inter = apply(dev_params, actor_input, pt.agent_output.actor_output)
        inter = jax.tree.map(np.asarray, inter)
        trunk = inter["encoder"]["round_trunk"]
        if i == 0:
            for k, v in trunk.items():
                leaf = jax.tree.leaves(v)[0]
                logger.info("intermediate %-26s %s", k, getattr(leaf, "shape", None))
        done = np.asarray(pt.env_output.done)  # (T, B)
        valid_t = (np.cumsum(done, axis=0) - done) == 0  # (T, B)
        valid_bt = valid_t.T  # (B, T)

        streams = _first(trunk["__call__"])[0]  # tuple of 6, each (B, T, R, rows, d)
        sizes = np.asarray(jax.tree.leaves(trunk["action_part_sizes"])[0]).reshape(-1)[
            -3:
        ]
        assert streams[0].shape[2] == num_rounds, streams[0].shape
        bounds = np.concatenate([[0], np.cumsum(sizes)])
        groups = {
            "state": [("state", slice(None))],
            "opp": [("opp", slice(None))],
            "action": [
                (g, slice(int(bounds[j]), int(bounds[j + 1])))
                for j, g in enumerate(ACTION_GROUPS)
            ],
            "value_all": [("value_all", slice(None))],
            "value_private": [("value_private", slice(None))],
            "value_public": [("value_public", slice(None))],
        }
        for name, s in zip(STREAM_NAMES, streams):
            stream_final[name].append((s[:, :, -1], valid_bt))
            stream_first[name].append((s[:, :, 0], valid_bt))

        for block, gate_pat, si, grp in READS:
            f = _first(trunk[block]["__call__"])  # (B, T, R, rows, d)
            if block == "value_read":
                n_value = streams[3].shape[3]
                f = f[..., :n_value, :] if grp == "value_all" else f[..., n_value:, :]
            stream = streams[si]
            for gname, rows in groups[grp]:
                gate_name = gate_pat.format(gname)
                g = gates[gate_name]
                for r in range(num_rounds):
                    fr = f[:, :, r, rows]
                    sr = stream[:, :, r, rows]
                    row_valid = np.linalg.norm(sr, axis=-1) > 0  # hard-zeroed rows
                    m = row_valid & valid_bt[..., None]
                    key = (block, gate_name)
                    acc.setdefault(key, []).append(
                        dict(
                            r=r,
                            g=float(g[r]),
                            f_rms=_rms(fr, m),
                            gf_rms=_rms(g[r] * fr, m),
                            s_rms=_rms(sr, m),
                        )
                    )
        logger.info("batch %d/%d", i // batch + 1, (len(chunks) + batch - 1) // batch)
    return acc, stream_first, stream_final, gates


def summarise(acc, stream_first, stream_final, gates, num_rounds):
    lines = []
    lines.append(
        "gate contribution rms(g.f)/rms(stream), per round  [gate | f_rms | ratio]"
    )
    for (block, gate_name), recs in acc.items():
        per_r = []
        for r in range(num_rounds):
            rr = [x for x in recs if x["r"] == r]
            g = rr[0]["g"]
            f_rms = np.mean([x["f_rms"] for x in rr])
            ratio = np.mean([x["gf_rms"] for x in rr]) / max(
                np.mean([x["s_rms"] for x in rr]), 1e-12
            )
            per_r.append(f"{g:+.4f}|{f_rms:6.2f}|{ratio:.4f}")
        lines.append(f"  {block:24s} {gate_name:26s} " + "  ".join(per_r))

    lines.append("")
    lines.append(
        "stream state-dependence rms(x - mean_states x)/rms(mean_states x): after round 0 -> after last round; rms(stream)"
    )
    for name in STREAM_NAMES:
        out = []
        for store in (stream_first, stream_final):
            xs = [x for x, _ in store[name]]
            ms = [m for _, m in store[name]]
            x = np.concatenate(xs, axis=0)  # (N, T, rows, d)
            m = np.concatenate(ms, axis=0)  # (N, T)
            row_valid = np.linalg.norm(x, axis=-1) > 0
            mm = row_valid & m[..., None]
            w = mm.astype(np.float64)[..., None]
            mean = (x * w).sum(axis=(0, 1)) / np.maximum(
                w.sum(axis=(0, 1)), 1
            )  # (rows, d)
            dev = x - mean
            out.append(
                (
                    _rms(dev, mm)
                    / max(_rms(np.broadcast_to(mean, x.shape), mm), 1e-12),
                    _rms(x, mm),
                )
            )
        lines.append(
            f"  {name:14s} dep {out[0][0]:.4f} -> {out[1][0]:.4f}   rms {out[0][1]:.3f} -> {out[1][1]:.3f}"
        )
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--games", type=int, default=16)
    ap.add_argument("--pairs", type=int, default=4)
    ap.add_argument("--chunks", type=int, default=48)
    ap.add_argument(
        "--batch", type=int, default=3
    )  # != num_rounds, keeps axes unambiguous
    ap.add_argument("--games-pkl", default=None)
    ap.add_argument("--which", default="target_params")
    a = ap.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stdout
    )

    # --which fresh: measure the INIT (no checkpoint leaves) -- the sanity
    # read for an init change; needs --games-pkl since there is no policy
    # worth playing.
    params = None if a.which == "fresh" else load_params(a.ckpt, a.which)
    if a.games_pkl and os.path.exists(a.games_pkl):
        sides = load(a.games_pkl)
    else:
        assert params is not None, "--which fresh needs --games-pkl"
        sides = play_games(params, n_games=a.games, pairs=a.pairs, tag="gates")
        if a.games_pkl:
            dump(sides, a.games_pkl)
    chunks = flatten(sides)[: a.chunks]
    logger.info("%d chunks", len(chunks))
    num_rounds = int(get_player_model_config(9, train=True).encoder.num_rounds)
    acc, first, final, gates = collect(params, chunks, a.batch, num_rounds)
    print(summarise(acc, first, final, gates, num_rounds))


if __name__ == "__main__":
    main()
