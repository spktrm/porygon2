"""
Visualise attention scores for each transformer step in the player encoder.

Produces per-head attention heatmaps and an entropy summary bar chart that
makes it easy to spot attention bottlenecks (low entropy = concentrated
attention on few keys).

Transformer steps covered
-------------------------
local_timestep_decoder  – field tokens attend to relevant entity/edge tokens
input_decoder           – latent queries attend to current game-state tokens
history_decoder         – latent queries attend to past timestep embeddings
state_transformer (kv)  – self-attention on the latent KV sequence
state_transformer (q)   – self- then cross-attention on the output-state tokens

Usage
-----
# Random params (good for checking shapes)
python -m scripts.viz_attention

# With a checkpoint
python -m scripts.viz_attention --ckpt ./ckpts/gen3/ckpt_0001.pkl

# Custom output directory and trajectory step
python -m scripts.viz_attention --ckpt ./ckpts/gen3/ckpt_0001.pkl \\
    --output ./my_viz --traj-step 2 --generation 3
"""

from __future__ import annotations

import argparse
import os
from typing import NamedTuple

import cloudpickle as pickle
import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

from rl.environment.protos.features_pb2 import FieldFeature, InfoFeature
from rl.environment.utils import get_ex_player_step
from rl.model.config import get_player_model_config
from rl.model.encoder import Encoder

# ---------------------------------------------------------------------------
# Token-sequence label helpers
# ---------------------------------------------------------------------------


def _labels(n: int, prefix: str) -> list[str]:
    return [f"{prefix}{i}" for i in range(n)]


def _input_kv_labels(T: int) -> list[str]:
    """Best-effort labels for the input_decoder key sequence."""
    base = (
        [f"priv_{i}" for i in range(6)]
        + [f"pub_{i}" for i in range(12)]
        + ["prev_src", "prev_tgt"]
        + ["field", "my_side", "opp_side"]
    )
    if T <= len(base):
        return base[:T]
    return base + [f"extra_{i}" for i in range(T - len(base))]


def _seq_labels(component: str, T: int, role: str) -> list[str]:
    """Return human-readable labels for a sequence of length T."""
    if component == "input_decoder":
        return _labels(T, "latent_") if role == "q" else _input_kv_labels(T)
    if component == "history_decoder":
        return _labels(T, "latent_") if role == "q" else [f"hist_{i}" for i in range(T)]
    if component == "local_timestep_decoder":
        return (
            ["field", "my_side", "opp_side"][:T]
            if role == "q"
            else [f"entity_{i}" for i in range(T)]
        )
    if component in ("state_transformer_kv_enc", "state_transformer_q_self"):
        return [f"tok_{i}" for i in range(T)]
    if component == "state_transformer_q_cross":
        return _labels(T, "latent_") if role == "k" else [f"tok_{i}" for i in range(T)]
    return [f"pos_{i}" for i in range(T)]


# ---------------------------------------------------------------------------
# Flax-intermediate extraction
# ---------------------------------------------------------------------------


def _walk(tree: dict, target_key: str) -> list[tuple[str, object]]:
    """Recursively collect all values whose dict key matches target_key."""
    out: list[tuple[str, object]] = []

    def _rec(d: dict, path: str) -> None:
        for k, v in d.items():
            p = f"{path}/{k}" if path else k
            if k == target_key:
                out.append((p, v))
            elif isinstance(v, dict):
                _rec(v, p)

    _rec(tree, "")
    return out


def _unwrap_sow(v: object) -> np.ndarray:
    """sow() aggregates values into a tuple; unwrap and convert."""
    if isinstance(v, tuple):
        v = v[0]
    return np.array(v, dtype=np.float32)


class AttentionRecord(NamedTuple):
    component: str  # e.g. "input_decoder"
    layer: int
    head_role: str  # "enc", "q_self", or "q_cross" for state_transformer
    attn_probs: np.ndarray  # [H, T_q, T_k]
    attn_entropy: np.ndarray  # [H, T_q]


def _intermediates_to_records(
    intermediates: dict, ordered_labels: list[tuple[str, int, str]]
) -> list[AttentionRecord]:
    """
    Pair the attention tensors in `intermediates` with human-readable labels.

    `ordered_labels` is a list of (component, layer, head_role) triples in the
    exact order that MultiHeadAttention is instantiated inside the forward pass.
    """
    weights = [_unwrap_sow(v) for _, v in _walk(intermediates, "attn_weights")]
    entropies = [_unwrap_sow(v) for _, v in _walk(intermediates, "attn_entropy")]

    records = []
    for i, (comp, layer, role) in enumerate(ordered_labels):
        if i >= len(weights):
            break
        records.append(
            AttentionRecord(
                component=comp,
                layer=layer,
                head_role=role,
                attn_probs=weights[i],
                attn_entropy=entropies[i],
            )
        )
    return records


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

_CMAP = "viridis"
_COLORS = plt.cm.tab10.colors  # type: ignore[attr-defined]


def plot_heatmap(rec: AttentionRecord, output_dir: str) -> str:
    """One PNG per AttentionRecord: one subplot per head."""
    A = rec.attn_probs  # (H, T_q, T_k)
    E = rec.attn_entropy  # (H, T_q)
    H, T_q, T_k = A.shape

    q_labels = _seq_labels(rec.component, T_q, "q")
    k_labels = _seq_labels(rec.component, T_k, "k")

    fig, axes = plt.subplots(
        1, H, figsize=(5 * H, max(5, T_q * 0.4 + 2)), squeeze=False
    )
    title = f"{rec.component}  layer={rec.layer}"
    if rec.head_role:
        title += f"  ({rec.head_role})"
    fig.suptitle(title, fontsize=12)

    for h, ax in enumerate(axes[0]):
        head_A = A[h]  # (T_q, T_k)
        mean_e = float(np.nan_to_num(E[h]).mean())

        im = ax.imshow(
            head_A,
            aspect="auto",
            vmin=0.0,
            vmax=float(head_A.max()) + 1e-9,
            cmap=_CMAP,
        )
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(f"Head {h}\nentropy={mean_e:.3f}", fontsize=9)

        font_q = max(5, 8 - T_q // 10)
        font_k = max(5, 8 - T_k // 10)

        if T_q <= 64:
            ax.set_yticks(range(T_q))
            ax.set_yticklabels(q_labels[:T_q], fontsize=font_q)
        if T_k <= 64:
            ax.set_xticks(range(T_k))
            ax.set_xticklabels(k_labels[:T_k], rotation=90, fontsize=font_k)

        ax.set_xlabel("Keys (attended to)", fontsize=8)
        ax.set_ylabel("Queries", fontsize=8)

    plt.tight_layout()
    safe = f"{rec.component}_l{rec.layer}"
    if rec.head_role:
        safe += f"_{rec.head_role}"
    fname = os.path.join(output_dir, f"attn_{safe}.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fname


def plot_entropy_summary(records: list[AttentionRecord], output_dir: str) -> str:
    """
    Bar chart of mean per-head attention entropy across all components.

    Low entropy means attention is concentrated on few keys — a potential
    information bottleneck.  The x-axis groups bars by component/layer/head.
    """
    bar_labels: list[str] = []
    bar_values: list[float] = []
    bar_colors: list[object] = []

    component_names = list(dict.fromkeys(r.component for r in records))
    color_map = {c: _COLORS[i % len(_COLORS)] for i, c in enumerate(component_names)}

    for rec in records:
        E = np.nan_to_num(rec.attn_entropy)  # (H, T_q)
        for h in range(E.shape[0]):
            mean_e = float(E[h].mean())
            label = f"{rec.component}\nL{rec.layer} H{h}"
            if rec.head_role:
                label = f"{rec.component}\nL{rec.layer} {rec.head_role} H{h}"
            bar_labels.append(label)
            bar_values.append(mean_e)
            bar_colors.append(color_map[rec.component])

    fig, ax = plt.subplots(figsize=(max(10, len(bar_labels) * 0.7), 6))
    x = range(len(bar_labels))
    ax.bar(x, bar_values, color=bar_colors, alpha=0.85)
    ax.set_xticks(list(x))
    ax.set_xticklabels(bar_labels, rotation=45, ha="right", fontsize=6)
    ax.set_ylim(0, 1.05)
    ax.axhline(
        0.5,
        color="red",
        linestyle="--",
        linewidth=0.9,
        label="0.5 threshold",
    )
    ax.set_ylabel("Mean normalized entropy  (0 = peaked,  1 = uniform)")
    ax.set_title(
        "Attention entropy per head / layer\n"
        "Low entropy  →  concentrated attention  →  potential bottleneck"
    )

    legend_handles = [Patch(facecolor=color_map[c], label=c) for c in component_names]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=7, ncol=2)

    plt.tight_layout()
    fname = os.path.join(output_dir, "attn_entropy_summary.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fname


# ---------------------------------------------------------------------------
# Ordered MHA labels for _batched_forward and _embed_local_timestep
# ---------------------------------------------------------------------------


def _build_batched_labels(cfg) -> list[tuple[str, int, str]]:
    """
    Return labels in the exact order that MultiHeadAttention modules are
    created inside _batched_forward (which calls input_decoder, history_decoder,
    then state_transformer).

    state_transformer internal order (two separate loops):
      for l in range(num_layers): encoder_layer(l)  → 1 MHA each
      for l in range(num_layers): decoder_layer(l)  → 2 MHA each (self, cross)
    """
    labels = []

    n_input = cfg.encoder.input_decoder.num_layers
    for l in range(n_input):
        labels.append(("input_decoder", l, ""))

    n_hist = cfg.encoder.history_decoder.num_layers
    for l in range(n_hist):
        labels.append(("history_decoder", l, ""))

    n_st = cfg.encoder.state_transformer.num_layers
    for l in range(n_st):
        labels.append(("state_transformer_kv_enc", l, "enc"))
    for l in range(n_st):
        labels.append(("state_transformer_q_self", l, "q_self"))
        labels.append(("state_transformer_q_cross", l, "q_cross"))

    return labels


def _build_local_labels(cfg) -> list[tuple[str, int, str]]:
    n = cfg.encoder.local_timestep_decoder.num_layers
    return [("local_timestep_decoder", l, "") for l in range(n)]


# ---------------------------------------------------------------------------
# Forward-pass helpers
# ---------------------------------------------------------------------------


def _run(encoder: Encoder, enc_vars: dict, method_fn, *args):
    """Apply encoder method, return (output, intermediates)."""
    out, state = encoder.apply(
        enc_vars,
        *args,
        method=method_fn,
        mutable=["intermediates"],
    )
    return out, state.get("intermediates", {})


def _capture_local_timestep(
    encoder: Encoder,
    enc_vars: dict,
    packed_history,
    history,
    cfg,
) -> list[AttentionRecord]:
    """Capture attention from local_timestep_decoder for one history step."""
    # Build entity and edge embedding caches (vmapped, but we only need the cache)
    entity_cache = encoder.apply(
        enc_vars,
        packed_history,
        method=lambda self, p: jax.vmap(self._embed_public_entity)(
            p.public, p.revealed
        )[0],
    )
    edge_cache = encoder.apply(
        enc_vars,
        packed_history,
        method=lambda self, p: jax.vmap(self._embed_edge)(p.edges)[0],
    )

    # Pick the most recent valid history timestep
    valid_flags = np.array(
        encoder.apply(
            enc_vars,
            history,
            packed_history,
            method=lambda self, h, p: self._embed_global_timestep(h, p)[1],
        )
    )
    valid_indices = np.where(valid_flags)[0]
    h_idx = int(valid_indices[-1]) if len(valid_indices) > 0 else 0

    single_hist = jax.tree.map(lambda x: x[h_idx], history)

    _, intermediates = _run(
        encoder,
        enc_vars,
        lambda self, h, e, eg: self._embed_local_timestep(h, e, eg),
        single_hist,
        entity_cache,
        edge_cache,
    )
    labels = _build_local_labels(cfg)
    return _intermediates_to_records(intermediates, labels)


def _capture_batched_forward(
    encoder: Encoder,
    enc_vars: dict,
    env_step_single,
    packed_history,
    history,
    cfg,
) -> list[AttentionRecord]:
    """Capture attention from input_decoder, history_decoder, state_transformer."""
    timestep_embeddings, history_valid_mask, history_request_count = encoder.apply(
        enc_vars,
        history,
        packed_history,
        method=lambda self, h, p: self._embed_global_timestep(h, p),
    )

    info = env_step_single.info
    request_count = info[InfoFeature.INFO_FEATURE__REQUEST_COUNT]
    history_step_count = info[InfoFeature.INFO_FEATURE__HISTORY_STEP_COUNT]

    timestep_mask = request_count >= jnp.where(
        history_valid_mask,
        history_request_count,
        jnp.iinfo(jnp.int32).max,
    )
    current_position = jnp.expand_dims(request_count, axis=-1)
    current_history_step = jnp.expand_dims(history_step_count, axis=-1)

    _, intermediates = _run(
        encoder,
        enc_vars,
        lambda self, *a: self._batched_forward(*a),
        env_step_single,
        timestep_mask,
        current_position,
        current_history_step,
        timestep_embeddings,
        history_request_count,
        env_step_single.private_team,
        history_valid_mask,
    )
    labels = _build_batched_labels(cfg)
    return _intermediates_to_records(intermediates, labels)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualise player-encoder attention scores."
    )
    parser.add_argument(
        "--ckpt",
        default=None,
        help="Path to a cloudpickle checkpoint file. "
        "If omitted, random parameters are used.",
    )
    parser.add_argument(
        "--generation",
        type=int,
        default=9,
        help="Game generation to use (1 or 9, default: 9).",
    )
    parser.add_argument(
        "--output",
        default="attention_viz",
        help="Output directory for PNG files (default: attention_viz/).",
    )
    parser.add_argument(
        "--traj-step",
        type=int,
        default=0,
        help="Which trajectory timestep to analyse (default: 0).",
    )
    args = parser.parse_args()

    # ── Build model ──────────────────────────────────────────────────────────
    print("Building encoder …")
    cfg = get_player_model_config(generation=args.generation, train=False)
    encoder = Encoder(cfg.encoder)

    # ── Load example data ────────────────────────────────────────────────────
    print("Loading example input …")
    actor_input, _ = get_ex_player_step()

    traj_step = args.traj_step

    # env is (T, 1, ...) after get_ex_player_step; take single step, single batch
    env_step_single = jax.device_put(
        jax.tree.map(lambda x: x[traj_step, 0], actor_input.env)
    )
    # packed_history / history are (N, 1, ...) – just squeeze the batch dim
    packed_history = jax.device_put(
        jax.tree.map(lambda x: x[:, 0], actor_input.packed_history)
    )
    history = jax.device_put(jax.tree.map(lambda x: x[:, 0], actor_input.history))

    # ── Load / init params ───────────────────────────────────────────────────
    if args.ckpt:
        print(f"Loading checkpoint from {args.ckpt} …")
        with open(args.ckpt, "rb") as f:
            ckpt = pickle.load(f)
        # Support both flat and nested checkpoint formats
        player_state = ckpt.get("player_state", ckpt)
        params = player_state.get("params", player_state)
        enc_params = params["params"]["encoder"]
    else:
        print("No checkpoint supplied — using random parameters.")
        # env_step_single has no T-dim; encoder.init expects a T=1 leading dim
        env_for_init = jax.tree.map(lambda x: x[None], env_step_single)
        key = jax.random.key(42)
        init_vars = encoder.init(key, env_for_init, packed_history, history)
        enc_params = init_vars["params"]

    enc_vars = {"params": enc_params}

    # ── Capture attention ────────────────────────────────────────────────────
    print("Capturing local_timestep_decoder attention …")
    local_records = _capture_local_timestep(
        encoder, enc_vars, packed_history, history, cfg
    )

    print("Capturing input_decoder / history_decoder / state_transformer attention …")
    batched_records = _capture_batched_forward(
        encoder, enc_vars, env_step_single, packed_history, history, cfg
    )

    all_records = local_records + batched_records
    if not all_records:
        print(
            "ERROR: no attention tensors were captured.\n"
            "Make sure rl/model/modules.py MultiHeadAttention.sow() calls are present."
        )
        return

    # ── Plot ─────────────────────────────────────────────────────────────────
    os.makedirs(args.output, exist_ok=True)
    print(f"\nPlotting {len(all_records)} attention map(s) → '{args.output}/'")

    for rec in all_records:
        fpath = plot_heatmap(rec, args.output)
        role_str = f" ({rec.head_role})" if rec.head_role else ""
        print(
            f"  {rec.component} L{rec.layer}{role_str}  "
            f"shape={rec.attn_probs.shape}  →  {fpath}"
        )

    summary = plot_entropy_summary(all_records, args.output)
    print(f"\nEntropy summary → {summary}")

    print(
        "\nDone!  Look at attn_entropy_summary.png first — bars close to 0 "
        "indicate heads with highly concentrated (bottlenecked) attention."
    )


if __name__ == "__main__":
    main()
