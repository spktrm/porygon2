"""Static attention probe over a checkpoint — outside wandb.

Loads a saved player checkpoint, runs ONE forward pass on the bundled
example step with COLLECT_INTERMEDIATES=1 so MultiHeadAttention sows its
weights, then attributes each attention module's mass to the encoder's
substreams (state = [state] -- the latent array since 2026-08-21, one
group; action = [move, switch, target]).

Since the latent read, the state rows are learned latents rather than
per-entity slots, so the old headline ("when a SWITCH row reads the state
stream, how much lands on the opponent's public entities?") is answered
one level up: read the `latent_input_read/read` attention, whose KEYS are
the 186 raw tokens (public 120 | private 48 | field 3 | prev 2 | history
13), to see which token groups each latent attends. Run with the learner
stopped — it wants the GPU.

    COLLECT_INTERMEDIATES=1 env/bin/python scripts/attn_probe.py \
        [--ckpt ckpts/gen9/ckpt_00040000] [--out runtime/attn_probe.html]
"""

import argparse
import os
import pickle

import numpy as np

STATE_PARTS = ["state"]
# Key layout of the latent read's attention (tokens, in concat order).
READ_TOKEN_PARTS = [
    ("public", 12 * 10),
    ("private", 6 * 8),
    ("field", 3),
    ("prev_action", 2),
    ("history", 13),
]
ACTION_PARTS = ["move", "switch", "target"]


def _latest_ckpt(root="ckpts/gen9"):
    cands = [
        os.path.join(root, d)
        for d in os.listdir(root)
        if d.startswith("ckpt_") and os.path.isdir(os.path.join(root, d))
    ]
    return max(cands, key=lambda p: int(p.rsplit("_", 1)[1]))


def collect(tree, path="", out=None):
    """Flatten a Flax intermediates tree to {path: array}."""
    out = {} if out is None else out
    if isinstance(tree, dict):
        for k, v in tree.items():
            collect(v, f"{path}/{k}", out)
    elif isinstance(tree, (list, tuple)):
        for i, v in enumerate(tree):
            collect(v, path if len(tree) == 1 else f"{path}[{i}]", out)
    elif hasattr(tree, "shape"):
        out[path] = np.asarray(tree)
    return out


def bounds(sizes):
    """[(name, lo, hi)] from part sizes."""
    edges, acc = [], 0
    for n, s in zip(sizes[0], sizes[1]):
        edges.append((n, acc, acc + int(s)))
        acc += int(s)
    return edges


def mass_by_part(w, key_edges, query_edges=None):
    """w: (..., heads, q, k) -> per (query-part, key-part) mean mass.

    Attention rows already sum to 1 over keys, so summing within a key
    part gives that part's share directly. Averaged over every leading
    axis (rounds, time) and heads.
    """
    rows = {}
    q_groups = query_edges or [("all", 0, w.shape[-2])]
    for qn, qlo, qhi in q_groups:
        sub = w[..., qlo:qhi, :].reshape(-1, w.shape[-1])
        # Masked query rows are all-zero (attn_probs is zeroed under the
        # mask), and the valid fraction differs per substream — averaging
        # over them would dilute each part by a different factor and make
        # the parts incomparable. Keep only rows that actually attended.
        keep = sub.sum(axis=-1) > 0.5
        if not keep.any():
            rows[qn] = {kn: 0.0 for kn, _, _ in key_edges}
            continue
        m = sub[keep].mean(axis=0)
        rows[qn] = {kn: float(m[klo:khi].sum()) for kn, klo, khi in key_edges}
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--out", default="runtime/attn_probe.html")
    args = ap.parse_args()
    ckpt = args.ckpt or _latest_ckpt()

    os.environ.setdefault("COLLECT_INTERMEDIATES", "1")
    import jax

    from rl.environment.utils import get_ex_player_step
    from rl.model.config import get_player_model_config
    from rl.model.heads import HeadParams
    from rl.model.player_model import get_player_model

    net = get_player_model(get_player_model_config(generation=9, train=True))
    ai, ao = jax.tree.map(lambda x: x[:, 0], get_ex_player_step())
    params = pickle.loads(open(os.path.join(ckpt, "player/params"), "rb").read())

    # jit the apply: RoundBlock is wrapped in nn.remat, so in an eager
    # apply the sown values escape as tracers (they are remat residuals)
    # and np.asarray on them raises TracerArrayConversionError. Under jit
    # they come back as concrete outputs.
    @jax.jit
    def _run(p):
        _, mut = net.apply(p, ai, ao, HeadParams(), mutable=["intermediates"])
        return mut["intermediates"]

    inter = collect(jax.device_get(_run(params)))
    print(f"# ckpt={ckpt}")
    print(f"# intermediates captured: {len(inter)}")
    for k, v in sorted(inter.items()):
        print(f"#   {k:72s} {tuple(v.shape)}")

    sp = next((v for k, v in inter.items() if "state_part_sizes" in k), None)
    apz = next((v for k, v in inter.items() if "action_part_sizes" in k), None)
    if sp is None or apz is None:
        print("!! part sizes not sown — is COLLECT_INTERMEDIATES=1 set?")
        return
    sp = np.asarray(sp).reshape(-1, len(STATE_PARTS))[0]
    apz = np.asarray(apz).reshape(-1, len(ACTION_PARTS))[0]
    s_edges = bounds((STATE_PARTS, sp))
    a_edges = bounds((ACTION_PARTS, apz))
    print(f"\n# state parts {list(zip(STATE_PARTS, sp))}")
    print(f"# action parts {list(zip(ACTION_PARTS, apz))}")

    sections = []
    for path, w in sorted(inter.items()):
        if "attn_weights" not in path:
            continue
        name = path.split("/")[-2]
        n_q, n_k = w.shape[-2], w.shape[-1]
        q_edges = a_edges if n_q == sum(apz) else (s_edges if n_q == sum(sp) else None)
        k_edges = a_edges if n_k == sum(apz) else (s_edges if n_k == sum(sp) else None)
        if k_edges is None:
            continue
        rows = mass_by_part(w, k_edges, q_edges)
        sections.append((name, tuple(w.shape), k_edges, rows))
        print(f"\n## {name}  shape={tuple(w.shape)}  (q={n_q}, k={n_k})")
        hdr = "".join(f"{kn:>13s}" for kn, _, _ in k_edges)
        print(f"{'query part':<14}{hdr}")
        for qn, d in rows.items():
            print(f"{qn:<14}" + "".join(f"{d[kn]:>13.4f}" for kn, _, _ in k_edges))

    # ---- HTML ----
    def bar(v):
        pct = max(0.0, min(1.0, v)) * 100
        return (
            f'<div class="bar"><span style="width:{pct:.1f}%"></span>'
            f"<em>{v:.3f}</em></div>"
        )

    html = [
        "<title>Attention Probe</title>",
        "<style>",
        ":root{--bg:#fff;--fg:#111;--mut:#666;--line:#e3e3e3;--acc:#3563b3;",
        "--card:#fafafa}",
        '@media (prefers-color-scheme:dark){:root:not([data-theme="light"])',
        "{--bg:#14161a;--fg:#e8e8e8;--mut:#9aa;--line:#2a2e36;--acc:#7aa2e3;",
        "--card:#1b1e24}}",
        ':root[data-theme="dark"]{--bg:#14161a;--fg:#e8e8e8;--mut:#9aa;',
        "--line:#2a2e36;--acc:#7aa2e3;--card:#1b1e24}",
        "body{background:var(--bg);color:var(--fg);margin:0;padding:2rem;",
        "font:15px/1.5 ui-sans-serif,system-ui,-apple-system,sans-serif;",
        "max-width:64rem;margin-inline:auto}",
        "h1{font-size:1.5rem;margin:0 0 .25rem}",
        "h2{font-size:1.05rem;margin:2rem 0 .5rem;font-family:ui-monospace,",
        "monospace;color:var(--acc)}",
        "p.sub{color:var(--mut);margin:0 0 2rem}",
        ".wrap{overflow-x:auto;border:1px solid var(--line);border-radius:8px;",
        "background:var(--card)}",
        "table{border-collapse:collapse;width:100%;min-width:38rem}",
        "th,td{padding:.5rem .75rem;text-align:left;border-bottom:1px solid ",
        "var(--line);font-size:.9rem}",
        "th{font-weight:600;color:var(--mut);font-size:.78rem;",
        "text-transform:uppercase;letter-spacing:.04em}",
        "tr:last-child td{border-bottom:none}",
        "td:first-child{font-family:ui-monospace,monospace}",
        ".bar{position:relative;background:color-mix(in srgb,var(--acc) 14%,",
        "transparent);border-radius:3px;height:1.35rem;min-width:5.5rem}",
        ".bar span{position:absolute;inset:0 auto 0 0;background:var(--acc);",
        "border-radius:3px;opacity:.75}",
        ".bar em{position:relative;font-style:normal;font-variant-numeric:",
        "tabular-nums;padding-left:.4rem;font-size:.82rem;line-height:1.35rem}",
        "code{font-family:ui-monospace,monospace;font-size:.85rem}",
        "</style>",
        "<h1>Attention probe</h1>",
        f'<p class="sub">Checkpoint <code>{ckpt}</code> · one forward pass on '
        "the bundled example step · rows are shares of each query substream\u2019s attention, summing to 1 across key "
        "substreams, averaged over rounds, time and heads.</p>",
    ]
    for name, shape, k_edges, rows in sections:
        html.append(f"<h2>{name} <span style='color:var(--mut)'>{shape}</span></h2>")
        html.append('<div class="wrap"><table><thead><tr><th>query</th>')
        html += [f"<th>{kn}</th>" for kn, _, _ in k_edges]
        html.append("</tr></thead><tbody>")
        for qn, d in rows.items():
            html.append(f"<tr><td>{qn}</td>")
            html += [f"<td>{bar(d[kn])}</td>" for kn, _, _ in k_edges]
            html.append("</tr>")
        html.append("</tbody></table></div>")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    open(args.out, "w").write("\n".join(html))
    print(f"\n# wrote {args.out}")


if __name__ == "__main__":
    main()
