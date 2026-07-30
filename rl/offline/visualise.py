"""Visualize the offline critic's potential Φ over a single battle log.

Produces a standalone HTML page: the official Showdown replay player
(rendered from the raw log via play.pokemonshowdown.com's replay-embed.js)
side by side with the inferred score — per-turn Φ (expected final alive-mon
margin in [-1, 1]) with per-ensemble-member traces, mean ± std band, the
optional uncertainty-gated Φ, a mirror-antisymmetry check from the opposite
perspective, and the full 13-bin margin distribution at the selected turn.
Checkpoints trained with the survival aux head additionally get a per-mon
faint-risk heatmap (E[discount^steps-to-faint] per revealed mon per turn) —
the timing signal the margin probe alone can't show. Clicking any chart
seeks the replay; the chart cursors follow playback.

The replay is encoded through the SAME exporter path as training shards
(service/src/scripts/exportReplay.ts -> encodePerspective), so the states
Φ is evaluated on here are exactly the states it would see in training.

Usage:
    python -m rl.offline.visualise <replay> [<replay> ...] \
        [--ckpt ckpts/offline/gen9randombattle/ckpt_00050000 [--ckpt ...]] \
        [--uncertainty-scale 2.0] [--output-dir viz] [--limit N]

Each <replay> is a local replay JSON (replays/data/...), a replay id
(gen9randombattle-2654504071), a replay.pokemonshowdown.com URL (ids and
URLs are fetched), or a directory of replay JSONs. Pages land in
--output-dir (default viz/) as {replay_id}.phi.html; batches also get an
index.html linking every page. --output overrides the path for a single
replay. Checkpoints load and compile once for the whole batch. With no
--ckpt, the latest checkpoint under ckpts/offline/{format_id}*/ is used
(all ensemble member dirs, if present). Requires node + a compiled
service/ (npx tsc is run automatically if the exporter is missing/stale).
"""

import argparse
import functools
import json
import os
import re
import subprocess
import tempfile
import urllib.request

import jax
import numpy as np

from rl.environment.data import ITOS
from rl.environment.protos.features_pb2 import (
    EntityEdgeFeature,
    EntityPublicNodeFeature,
    EntityRevealedNodeFeature,
    InfoFeature,
)
from rl.offline.artifact import load_critic_params
from rl.offline.config import get_offline_config
from rl.offline.dataset import (
    MAX_MARGIN,
    NUM_SLOTS,
    NUM_SURVIVAL_BINS,
    _final_margin,
    collate,
    iter_shard_payloads,
    record_to_examples,
)
from rl.offline.model import Porygon2OfflineCritic, get_offline_critic

REPLAY_URL = "https://replay.pokemonshowdown.com/{replay_id}.json"
USER_AGENT = "porygon2-replay-downloader (https://github.com/spktrm/porygon2)"
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SERVICE_DIR = os.path.join(REPO_ROOT, "service")


def resolve_replay(spec: str, tmpdir: str) -> tuple[dict, str]:
    """Returns (replay dict, path to its JSON). Local paths are used as-is;
    bare ids and replay.pokemonshowdown.com URLs are fetched."""
    if os.path.exists(spec):
        with open(spec) as f:
            return json.load(f), os.path.abspath(spec)
    replay_id = re.sub(r"^https?://replay\.pokemonshowdown\.com/", "", spec.strip())
    replay_id = re.sub(r"\.(json|log|html)$", "", replay_id)
    url = REPLAY_URL.format(replay_id=replay_id)
    print(f"fetching {url}")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request) as response:
        replay = json.load(response)
    path = os.path.join(tmpdir, f"{replay_id}.json")
    with open(path, "w") as f:
        json.dump(replay, f)
    return replay, path


def _exporter_is_stale(exporter: str) -> bool:
    """True if the compiled exporter is missing or older than any TypeScript
    source (src/ or the generated protos/)."""
    if not os.path.exists(exporter):
        return True
    built = os.path.getmtime(exporter)
    for root in ("src", "protos"):
        for dirpath, _, files in os.walk(os.path.join(SERVICE_DIR, root)):
            for name in files:
                if name.endswith(".ts"):
                    if os.path.getmtime(os.path.join(dirpath, name)) > built:
                        return True
    return False


def export_record(replay_json_path: str, tmpdir: str) -> tuple[bytes, dict]:
    """Encodes one replay through the shard exporter; returns the
    EnvironmentBatch payload and the exporter's stats ({perspectives, states})."""
    exporter = os.path.join(SERVICE_DIR, "dist", "scripts", "exportReplay.js")
    if _exporter_is_stale(exporter):
        print("compiling service/ (dist exporter missing or older than src) ...")
        # --noEmitOnError: a failed compile must not leave a half-updated
        # dist/ that a later run would mistake for current.
        subprocess.run(["npx", "tsc", "--noEmitOnError"], cwd=SERVICE_DIR, check=True)
    out_bin = os.path.join(tmpdir, "record.bin")
    result = subprocess.run(
        ["node", exporter, replay_json_path, out_bin],
        cwd=SERVICE_DIR,  # data.ts resolves ../constants and ../data from CWD
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise SystemExit(f"replay exporter failed:\n{result.stderr.strip()}")
    stats = json.loads(result.stdout.strip().splitlines()[-1])
    payloads = list(iter_shard_payloads(out_bin))
    assert len(payloads) == 1, f"expected 1 record, got {len(payloads)}"
    return payloads[0], stats


def discover_ckpts(format_id: str) -> list[str]:
    """Latest checkpoint in each ckpts/offline/{format_id}*/ dir — all
    ensemble members when present, else the single-model dir."""
    root = os.path.join(REPO_ROOT, "ckpts", "offline")
    candidates = []
    if os.path.isdir(root):
        for name in sorted(os.listdir(root)):
            if name == format_id or name.startswith(f"{format_id}-ens"):
                ckpt_dir = os.path.join(root, name)
                steps = sorted(d for d in os.listdir(ckpt_dir) if d.startswith("ckpt_"))
                if steps:
                    candidates.append(os.path.join(ckpt_dir, steps[-1]))
    ensembles = [
        c for c in candidates if "-ens" in os.path.basename(os.path.dirname(c))
    ]
    if ensembles:
        return ensembles
    if not candidates:
        raise FileNotFoundError(
            f"No offline checkpoints under {root}/{format_id}* — pass --ckpt."
        )
    return candidates


def _format_generation(format_id: str) -> int:
    """gen{N}{tier} -> N. The model config and embedding tables are per
    generation; the tier only routes shards and artifacts."""
    match = re.match(r"gen(\d+)", format_id)
    if not match:
        raise ValueError(f"cannot parse a generation from format {format_id!r}")
    return int(match.group(1))


class CriticRunner:
    """Loads offline critic checkpoint(s) and compiles the apply once, then
    scores any number of exported records. Used here for one record, and by
    the causality check (rl/offline/causality.py) which scores many
    truncated variants of one replay — checkpoint loading and jit
    compilation must not repeat per variant.

    ``format_id`` selects the generation-specific model config/embeddings;
    the checkpoints must be for the same format."""

    def __init__(self, ckpt_paths: list[str], format_id: str = "gen9randombattle"):
        self.format_id = format_id
        self.config = get_offline_config().replace(
            generation=_format_generation(format_id)
        )
        self.ckpt_paths = list(ckpt_paths)
        self.params = load_critic_params(ckpt_paths)  # leading ensemble axis K
        self.num_members = jax.tree.leaves(self.params)[0].shape[0]
        # Feature detection from the params themselves, so checkpoints from
        # any era of the architecture load: the survival aux head and Elo
        # conditioning each leave their own subtree. (Mixed ensembles can't
        # reach here — load_critic_params' stacking rejects mismatched
        # trees.)
        subtrees = self.params.get("params", {})
        self.has_survival = "survival_head" in subtrees
        self.rating_conditioned = "rating_embed" in subtrees
        model = get_offline_critic(
            self.config.generation, rating_conditioning=self.rating_conditioned
        )
        apply = (
            functools.partial(model.apply, method=Porygon2OfflineCritic.with_aux)
            if self.has_survival
            else model.apply
        )
        self._apply_fn = jax.jit(jax.vmap(apply, in_axes=(None, 1), out_axes=1))

    def run(self, payload: bytes) -> tuple[dict, list]:
        """Runs every ensemble member over both perspectives. Returns
        (per-trajectory outputs, examples). Outputs: phi (K, T, B), probs
        (K, T, B, 13), valid-step mask (T, B); with the survival aux head,
        also per-mon faint risk E[y] in [0, 1] as survival (K, T, B, 12)
        plus its (T, B, 12) revealed-and-alive mask (else None)."""
        # No config: raw view — keep clamped-forfeit games and exact one-hot
        # labels so every replay stays inspectable regardless of the
        # training forfeit policy.
        examples = record_to_examples(payload)
        if not examples:
            raise ValueError("replay produced no usable trajectories")
        max_t = max(e.actor_input.env.done.shape[0] for e in examples)
        if max_t > self.config.max_trajectory_length:
            print(
                f"WARNING: game has {max_t} states, truncated to "
                f"{self.config.max_trajectory_length} "
                "(config.max_trajectory_length)"
            )
        batch = collate(examples, self.config)

        # Risk readout: expected y = E[discount^(steps to next faint)] off
        # the aux head's bins — 0 = safe/never, 1 = faints immediately.
        bin_centers = (np.arange(NUM_SURVIVAL_BINS) + 0.5) / NUM_SURVIVAL_BINS
        phis, probs, risks = [], [], []
        for k in range(self.num_members):
            member_params = jax.tree.map(lambda x: x[k], self.params)  # noqa: B023
            out = jax.device_get(self._apply_fn(member_params, batch.actor_input))
            if self.has_survival:
                out, aux = out
                survival_logits = np.asarray(aux.survival, dtype=np.float32)
                survival_probs = np.exp(
                    survival_logits - survival_logits.max(axis=-1, keepdims=True)
                )
                survival_probs /= survival_probs.sum(axis=-1, keepdims=True)
                risks.append(survival_probs @ bin_centers)  # (T, B, 12)
            phis.append(np.asarray(out.expectation, dtype=np.float32))
            probs.append(np.exp(np.asarray(out.log_probs, dtype=np.float32)))

        done = np.asarray(batch.actor_input.env.done).astype(np.int32)
        mask = (np.cumsum(done, axis=0) - done) == 0  # (T, B)
        return {
            "phi": np.stack(phis),
            "probs": np.stack(probs),
            "mask": mask,
            "survival": np.stack(risks) if risks else None,
            "survival_mask": (np.asarray(batch.survival_masks) if risks else None),
        }, examples


def expand_replay_specs(specs: list[str], limit: int | None) -> list[str]:
    """Directories expand to their sorted *.json contents; everything else
    (paths, ids, URLs) passes through. ``limit`` caps the total."""
    expanded: list[str] = []
    for spec in specs:
        if os.path.isdir(spec):
            expanded.extend(
                sorted(
                    os.path.join(spec, name)
                    for name in os.listdir(spec)
                    if name.endswith(".json")
                )
            )
        else:
            expanded.append(spec)
    return expanded[:limit] if limit else expanded


def _slot_identities(packed_history) -> tuple[list[int], list[str | None]]:
    """Per stable history slot, (relative side, species name) from the
    terminal caches. Side: 1 = anchor's mon, 0 = opponent's, -1 = slot
    never revealed (mirrors Encoder.history_slot_sides, in numpy). Species
    is decoded from the slot's LAST cache row via the shared token table —
    forme changes therefore show the final forme — as a Showdown id
    ("greattusk"); None when unmapped."""
    public = np.asarray(packed_history.public_cache)
    edges = np.asarray(packed_history.edge_cache)
    revealed = np.asarray(packed_history.revealed_cache)
    sides = np.full(NUM_SLOTS, -1, dtype=np.int64)
    names: list[str | None] = [None] * NUM_SLOTS
    slots = edges[:, EntityEdgeFeature.ENTITY_EDGE_FEATURE__ENTITY_IDX]
    species_itos = ITOS["species"]
    for row in np.nonzero(public.any(axis=1))[0]:
        slot = int(slots[row])
        if not 0 <= slot < NUM_SLOTS:
            continue
        sides[slot] = int(
            public[row, EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__SIDE]
        )
        token = int(
            revealed[
                row,
                EntityRevealedNodeFeature.ENTITY_REVEALED_NODE_FEATURE__SPECIES,
            ]
        )
        name = species_itos.get(token, "")
        # Special tokens (_UNSPECIFIED/_NULL/_PAD/_UNK) are not names.
        if name and not name.startswith("_"):
            names[slot] = name
    return sides.tolist(), names


def _active_slots(env, num_steps: int) -> np.ndarray:
    """(num_steps, 12) 0/1: which stable slots are on the field at each
    step — the per-step public team's ACTIVE bits scattered back to slots
    via PUBLIC_ORDER, the same alignment the training targets use."""
    info = np.asarray(env.info)[:num_steps]
    public_team = np.asarray(env.public_team)[:num_steps]
    order = info[
        :,
        InfoFeature.INFO_FEATURE__PUBLIC_ORDER_0 : InfoFeature.INFO_FEATURE__PUBLIC_ORDER_11
        + 1,
    ].astype(np.int64)
    valid = (order >= 0) & (order < NUM_SLOTS)
    row_active = (
        public_team[:, :, EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__ACTIVE]
        > 0
    )
    active = np.zeros((num_steps, NUM_SLOTS), dtype=int)
    step_idx = np.arange(num_steps)
    for i in range(NUM_SLOTS):
        rows = valid[:, i]
        active[step_idx[rows], order[rows, i]] = row_active[rows, i]
    return active


def build_payload(
    replay: dict,
    stats: dict,
    outputs: dict,
    examples: list,
    ckpt_paths: list[str],
    uncertainty_scale: float,
) -> dict:
    perspectives = stats["perspectives"]  # example index -> playerIndex
    anchor = perspectives.index(0) if 0 in perspectives else 0
    anchor_player = perspectives[anchor]

    def series(example_idx: int):
        valid = int(outputs["mask"][:, example_idx].sum())
        phi = outputs["phi"][:, :valid, example_idx]  # (K, n)
        return valid, phi

    n, phi = series(anchor)
    mean = phi.mean(axis=0)
    std = phi.std(axis=0)
    gated = mean * np.exp(-uncertainty_scale * std) if uncertainty_scale > 0 else None

    mirror = None
    if len(perspectives) == 2:
        other = 1 - anchor
        n_other, phi_other = series(other)
        m = min(n, n_other)
        mirror = (-phi_other.mean(axis=0)[:m]).tolist()

    survival = survival_mask = slot_sides = slot_names = active = None
    if outputs.get("survival") is not None:
        # Ensemble-mean per-mon faint risk from the anchor's perspective,
        # masked to revealed-and-alive (mask 0 renders as neutral cells).
        survival = outputs["survival"][:, :n, anchor].mean(axis=0)  # (n, 12)
        survival_mask = outputs["survival_mask"][:n, anchor]  # (n, 12)
        slot_sides, slot_names = _slot_identities(
            examples[anchor].actor_input.packed_history
        )
        active = _active_slots(examples[anchor].actor_input.env, n)

    member_probs = outputs["probs"][:, :n, anchor]  # (K, n, 13)
    probs = member_probs.mean(axis=0)  # (n, 13)
    # Win readout of the same head: P(win) − P(loss), the signed sign-mass
    # of the margin bins — what potential_readout="win" feeds the learner.
    win_phi = (
        member_probs[..., MAX_MARGIN + 1 :].sum(-1)
        - member_probs[..., :MAX_MARGIN].sum(-1)
    ).mean(axis=0)
    final_reward = np.asarray(examples[anchor].actor_input.env.win_reward)[-1]
    actual_margin, ending, _ = _final_margin(examples[anchor].actor_input, final_reward)
    # Reader-facing labels — the page is a product surface, not a debug
    # view; ending semantics (censoring etc.) live in dataset.py.
    ending_labels = {
        "played_out": "played to the end",
        "conceded": "opponent forfeited",
        "clamped": "ended by forfeit",
        "tie": "tie",
    }

    return {
        "replayId": replay.get("id", ""),
        "format": replay.get("format", replay.get("formatid", "")),
        "rating": replay.get("rating"),
        "players": replay.get("players", ["p1", "p2"]),
        "anchorPlayer": anchor_player,  # playerIndex whose view Φ is plotted
        "numSteps": n,
        "members": np.round(phi, 4).tolist(),
        "mean": np.round(mean, 4).tolist(),
        "winMean": np.round(win_phi, 4).tolist(),
        "std": np.round(std, 4).tolist(),
        "gated": np.round(gated, 4).tolist() if gated is not None else None,
        "mirror": [round(v, 4) for v in mirror] if mirror is not None else None,
        "probs": np.round(probs, 4).tolist(),
        "survival": np.round(survival, 3).tolist() if survival is not None else None,
        "survivalMask": (
            survival_mask.astype(int).tolist() if survival_mask is not None else None
        ),
        "slotSides": slot_sides,
        "slotNames": slot_names,
        "active": active.tolist() if active is not None else None,
        "actualMargin": actual_margin,
        "endingLabel": ending_labels[ending],
        "maxMargin": MAX_MARGIN,
        "uncertaintyScale": uncertainty_scale,
        "ckpts": [os.path.relpath(p, REPO_ROOT) for p in ckpt_paths],
    }


def render_page(
    replay: dict,
    stats: dict,
    payload: bytes,
    runner: CriticRunner,
    uncertainty_scale: float = 0.0,
) -> tuple[str, dict]:
    """Scores one exported replay and renders the standalone HTML page.
    Returns (html, index-payload dict). Pure — no filesystem writes; also
    the entry point for the serverless app (serve/modal_app.py)."""
    outputs, examples = runner.run(payload)
    data = build_payload(
        replay, stats, outputs, examples, runner.ckpt_paths, uncertainty_scale
    )
    # The page bootstrap un-escapes \/ back to / (Showdown replay-file
    # convention), so </script> in the log (and the embedded JSON) can't
    # terminate the enclosing script tag.
    html = (
        HTML_TEMPLATE.replace("__PHI_TITLE__", f"Replay review — {data['replayId']}")
        .replace("__PHI_DATA__", json.dumps(data).replace("</", "<\\/"))
        .replace("__PHI_LOG__", replay["log"].replace("</", "<\\/"))
    )
    return html, data


def render_replay(
    replay: dict, stats: dict, payload: bytes, runner: CriticRunner, args
) -> tuple[str, dict]:
    """CLI wrapper: renders and writes the page. Returns
    (output path, payload dict for the batch index)."""
    html, data = render_page(replay, stats, payload, runner, args.uncertainty_scale)
    output = args.output or os.path.join(
        args.output_dir, f"{data['replayId'] or 'replay'}.phi.html"
    )
    with open(output, "w") as f:
        f.write(html)
    print(f"wrote {output} ({len(html) // 1024} KB)")
    return output, data


def write_index(output_dir: str, entries: list[tuple[str, dict]]) -> str:
    rows = "".join(
        '<tr><td><a href="{file}">{rid}</a></td><td>{players}</td>'
        "<td>{rating}</td><td>{margin:+d}</td><td>{ending}</td></tr>".format(
            file=os.path.basename(path),
            rid=data["replayId"],
            players=" vs ".join(data["players"]),
            rating=data["rating"] or "—",
            margin=data["actualMargin"],
            ending=data["endingLabel"],
        )
        for path, data in entries
    )
    html = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        f"<title>Replay reviews ({len(entries)})</title>"
        "<style>body{font-family:system-ui,sans-serif;margin:24px;}"
        "table{border-collapse:collapse;}td,th{padding:6px 14px;"
        "border-bottom:1px solid #ddd;text-align:left;font-size:14px;}"
        "th{color:#666;}</style></head><body>"
        f"<h1>Replay reviews ({len(entries)})</h1><table>"
        "<tr><th>replay</th><th>players</th><th>rating</th>"
        "<th>final lead</th><th>ending</th></tr>"
        f"{rows}</table></body></html>"
    )
    index_path = os.path.join(output_dir, "index.html")
    with open(index_path, "w") as f:
        f.write(html)
    return index_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "replays",
        nargs="+",
        help="replay JSON paths, replay ids, URLs, or directories of " "replay JSONs",
    )
    parser.add_argument(
        "--ckpt",
        action="append",
        default=None,
        help="offline critic checkpoint dir (repeat for an ensemble); "
        "default: latest under ckpts/offline/{format_id}*/",
    )
    parser.add_argument("--uncertainty-scale", type=float, default=0.0)
    parser.add_argument(
        "--output", default=None, help="output HTML path (single replay only)"
    )
    parser.add_argument(
        "--output-dir", default="viz", help="directory for generated pages"
    )
    parser.add_argument("--limit", type=int, default=None, help="max replays to render")
    args = parser.parse_args()

    specs = expand_replay_specs(args.replays, args.limit)
    if not specs:
        parser.error("no replays found")
    if args.output and len(specs) > 1:
        parser.error("--output is for a single replay; use --output-dir for batches")
    if not args.output:
        os.makedirs(args.output_dir, exist_ok=True)

    runners: dict[str, CriticRunner] = {}
    entries: list[tuple[str, dict]] = []
    for spec in specs:
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                replay, replay_json_path = resolve_replay(spec, tmpdir)
                payload, stats = export_record(replay_json_path, tmpdir)
            format_id = replay.get("formatid", "gen9randombattle")
            if format_id not in runners:
                if args.ckpt and runners:
                    raise ValueError(
                        f"--ckpt pins one format's checkpoints, but the batch "
                        f"mixes {sorted(runners)} with {format_id}"
                    )
                # Checkpoints load and jit-compile once per format; a batch
                # spanning formats gets one runner each.
                ckpt_paths = args.ckpt or discover_ckpts(format_id)
                print(f"{format_id} checkpoints: {ckpt_paths}")
                runners[format_id] = CriticRunner(ckpt_paths, format_id)
            entries.append(
                render_replay(replay, stats, payload, runners[format_id], args)
            )
        except (SystemExit, ValueError, OSError) as err:
            # One undecided/corrupt replay must not kill the batch.
            print(f"SKIPPED {spec}: {err}")

    if len(entries) > 1:
        index_path = write_index(args.output_dir, entries)
        print(f"\n{len(entries)}/{len(specs)} replays rendered — index: {index_path}")
    elif entries:
        print("open it in a browser")


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8" />
<title>__PHI_TITLE__</title>
<style>
  .viz-root {
    color-scheme: light;
    --surface-1: #fcfcfb;
    --text-primary: #0b0b0b;
    --text-secondary: #52514e;
    --muted: #898781;
    --grid: #e1e0d9;
    --baseline: #c3c2b7;
    --series-1: #2a78d6;   /* win chance (headline series) */
    --series-2: #eb6834;   /* uncertainty-gated forecast (details) */
    --series-3: #1baf7a;   /* material forecast (details) */
    --pos-pole: #2a78d6;   /* diverging: anchor ahead */
    --neg-pole: #e34948;   /* diverging: opponent ahead */
    --mid: #f0efec;
    --border: rgba(11, 11, 11, 0.1);
    font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
    color: var(--text-primary);
    background: var(--surface-1);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px 20px;
    flex: 1 1 480px;
    min-width: 420px;
    max-width: 720px;
    align-self: flex-start;
  }
  body.dark .viz-root {
    color-scheme: dark;
    --surface-1: #1a1a19;
    --text-primary: #ffffff;
    --text-secondary: #c3c2b7;
    --muted: #898781;
    --grid: #2c2c2a;
    --baseline: #383835;
    --series-1: #3987e5;
    --series-2: #d95926;
    --series-3: #199e70;
    --pos-pole: #3987e5;
    --neg-pole: #e66767;
    --mid: #383835;
    --border: rgba(255, 255, 255, 0.1);
  }
  .viz-layout {
    display: flex;
    flex-wrap: wrap;
    gap: 20px;
    align-items: flex-start;
    padding: 12px;
  }
  .viz-layout > .wrapper.replay-wrapper {
    flex: 0 1 980px;
    position: relative;
    margin: 0;
  }
  .viz-root h1 { font-size: 16px; margin: 0 0 2px; font-weight: 600; }
  .viz-root .sub { font-size: 12px; color: var(--text-secondary); margin: 0 0 12px; }
  .viz-root .sub code { font-size: 11px; }
  .viz-root svg { display: block; width: 100%; height: auto; }
  .viz-root .legend {
    display: flex; flex-wrap: wrap; gap: 12px;
    font-size: 11px; color: var(--text-secondary); margin: 6px 0 14px;
  }
  .viz-root .legend .item { display: flex; align-items: center; gap: 5px; }
  .viz-root .axis-label { font-size: 10px; fill: var(--muted); }
  .viz-root .tick-label { font-size: 10px; fill: var(--muted); font-variant-numeric: tabular-nums; }
  .viz-root .side-label { font-size: 10px; fill: var(--text-secondary); font-weight: 600; }
  .viz-root .chart-title { font-size: 12px; color: var(--text-secondary); margin: 10px 0 4px; }
  .viz-root .key-turns { margin-top: 4px; }
  .viz-root .key-turns .row {
    display: flex; align-items: center; gap: 9px;
    padding: 4px 7px; border-radius: 5px; cursor: pointer; font-size: 12px;
  }
  .viz-root .key-turns .row:hover {
    background: color-mix(in oklab, var(--text-primary) 7%, transparent);
  }
  .viz-root .key-turns .badge {
    display: inline-block; min-width: 22px; text-align: center;
    font-weight: 700; font-size: 11px; border-radius: 10px;
    padding: 1px 6px; color: #fff;
  }
  .viz-root .key-turns .delta { font-variant-numeric: tabular-nums; }
  .viz-root .note { color: var(--muted); font-size: 11px; margin: 8px 0 0; }
  /* Model internals (ensemble traces, mirror check, checkpoints) hide
     behind the details toggle — the default page speaks player language. */
  .viz-root:not(.show-adv) .adv { display: none; }
  .viz-root .adv-toggle {
    margin: 2px 0 10px; padding: 3px 10px; font-size: 11px;
    color: var(--text-secondary); background: none; cursor: pointer;
    border: 1px solid var(--border); border-radius: 999px;
  }
  .viz-root .adv-toggle:hover { border-color: var(--muted); }
  #phi-tooltip {
    position: fixed; pointer-events: none; display: none; z-index: 10;
    background: var(--surface-1); color: var(--text-primary);
    border: 1px solid var(--border); border-radius: 6px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
    padding: 6px 9px; font-size: 11px; line-height: 1.5;
    font-variant-numeric: tabular-nums; white-space: nowrap;
  }
  #phi-tooltip .k { color: var(--text-secondary); }
</style>
</head>
<body>
<div class="viz-layout">
  <div class="wrapper replay-wrapper">
    <div class="battle"></div>
    <div class="battle-log"></div>
    <div class="replay-controls"></div>
    <div class="replay-controls-2"></div>
  </div>
  <div class="viz-root">
    <h1 id="viz-title"></h1>
    <p class="sub" id="viz-sub"></p>
    <button class="adv-toggle" id="adv-toggle">Show model details</button>
    <svg id="phi-chart" viewBox="0 0 640 300"></svg>
    <div class="legend" id="phi-legend"></div>
    <div class="chart-title" id="delta-title"></div>
    <svg id="delta-chart" viewBox="0 0 640 110"></svg>
    <div class="chart-title" id="surv-title"></div>
    <svg id="surv-chart" viewBox="0 0 640 0"></svg>
    <div class="chart-title" id="margin-title"></div>
    <svg id="margin-chart" viewBox="0 0 640 130"></svg>
    <div class="chart-title" id="key-title"></div>
    <div class="key-turns" id="key-turns"></div>
    <p class="note" id="key-note"></p>
    <div id="phi-tooltip"></div>
  </div>
</div>
<script type="text/plain" class="battle-log-data">__PHI_LOG__</script>
<script>window.PHI_DATA = __PHI_DATA__;</script>
<script>
/* Battle player bootstrap — the load list comes from Showdown's MIT
   replay-embed.js, but scripts are injected with async=false so they
   execute in order: the embed's own injector runs them in arbitrary
   completion order, which intermittently breaks graphics.js (its top-level
   throws, leaving BattleBackdrops undefined and Battle unconstructable). */
(function () {
  "use strict";
  window.exports = window;
  var HOST = "https://play.pokemonshowdown.com/";
  [
    "style/font-awesome.css?",
    "style/battle.css?a7",
    "style/replay.css?a7",
    "style/utilichart.css?a7",
  ].forEach(function (href) {
    var link = document.createElement("link");
    link.rel = "stylesheet";
    link.href = HOST + href;
    document.head.appendChild(link);
  });
  var scripts = [
    "js/lib/ps-polyfill.js",
    "config/config.js?a7",
    "js/lib/jquery-1.11.0.min.js",
    "js/lib/html-sanitizer-minified.js",
    "js/battle-sound.js",
    "js/battledata.js?a7",
    "data/pokedex-mini.js?a7",
    "data/pokedex-mini-bw.js?a7",
    "data/graphics.js?a7",
    "data/pokedex.js?a7",
    "data/moves.js?a7",
    "data/abilities.js?a7",
    "data/items.js?a7",
    "data/teambuilder-tables.js?a7",
    "js/battle-tooltips.js?a7",
    "js/battle.js?a7",
  ];
  var remaining = scripts.length;
  var failed = false;
  scripts.forEach(function (src) {
    var script = document.createElement("script");
    script.src = HOST + src;
    script.async = false;
    script.onload = function () {
      if (--remaining === 0 && !failed) initBattle();
    };
    script.onerror = function () {
      if (failed) return;
      failed = true;
      document.querySelector(".battle").innerHTML =
        '<div style="padding:40px;font:12px system-ui;color:#555">' +
        "battle renderer unavailable (needs network access to " +
        "play.pokemonshowdown.com) — the Φ chart still works</div>";
    };
    document.head.appendChild(script);
  });

  if (window.matchMedia) {
    if (window.matchMedia("(prefers-color-scheme: dark)").matches) {
      document.body.className = "dark";
    }
    window
      .matchMedia("(prefers-color-scheme: dark)")
      .addEventListener("change", function (event) {
        document.body.className = event.matches ? "dark" : "";
      });
  }

  var muted = true;
  function initBattle() {
    var logText = (
      document.querySelector("script.battle-log-data").textContent || ""
    ).replace(/\\\//g, "/");
    var battle = new Battle({
      id: (window.PHI_DATA.replayId || "").toLowerCase(),
      $frame: $(".battle"),
      $logFrame: $(".battle-log"),
      log: logText.split("\n"),
      isReplay: true,
      paused: true,
      autoresize: true,
    });
    window.battle = battle;
    battle.setMute(muted);

    var controls = document.querySelector(".replay-controls");
    function render() {
      controls.innerHTML =
        (battle.paused
          ? '<button data-action="play"><i class="fa fa-play"></i> Play</button>'
          : '<button data-action="pause"><i class="fa fa-pause"></i> Pause</button>') +
        '<button data-action="reset"' +
        (battle.started ? "" : " disabled") +
        '><i class="fa fa-undo"></i> Reset</button> ' +
        '<button data-action="rewind"><i class="fa fa-step-backward"></i> Last turn</button>' +
        '<button data-action="ff"><i class="fa fa-step-forward"></i> Next turn</button> ' +
        '<button data-action="ffend"><i class="fa fa-fast-forward"></i> End</button> ' +
        '<button data-action="switch"><i class="fa fa-random"></i> Switch sides</button> ' +
        '<button data-action="sound"><i class="fa fa-volume-' +
        (muted ? "off" : "up") +
        '"></i> Sound</button> ' +
        '<button data-action="theme"><i class="fa fa-adjust"></i> Theme</button>';
    }
    controls.addEventListener("click", function (evt) {
      var button = evt.target.closest("button");
      if (!button) return;
      switch (button.getAttribute("data-action")) {
        case "play": battle.play(); break;
        case "pause": battle.pause(); break;
        case "reset": battle.reset(); break;
        case "rewind": battle.seekBy(-1); break;
        case "ff": battle.seekBy(1); break;
        case "ffend": battle.seekTurn(Infinity); break;
        case "switch": battle.switchViewpoint(); break;
        case "sound":
          muted = !muted;
          battle.setMute(muted);
          break;
        case "theme": document.body.classList.toggle("dark"); break;
      }
      render();
    });
    battle.subscribe(function () { render(); });
    render();
  }
})();
</script>
<script>
(function () {
  "use strict";
  var D = window.PHI_DATA;
  var SVG = "http://www.w3.org/2000/svg";
  var root = document.querySelector(".viz-root");
  var N = D.numSteps;                 // steps: turns 1..N-1 plus terminal
  var K = D.members.length;
  var anchorName = D.players[D.anchorPlayer] || "p1";
  var oppName = D.players[1 - D.anchorPlayer] || "p2";

  document.getElementById("viz-title").textContent =
    D.players.join(" vs ") + " — " + D.format +
    (D.rating ? " (rating " + D.rating + ")" : "");
  var winnerName =
    D.actualMargin > 0 ? anchorName : D.actualMargin < 0 ? oppName : null;
  var resultText = winnerName
    ? "<b>" + esc(winnerName) + "</b> won by " + Math.abs(D.actualMargin) +
      " Pokémon (" + esc(D.endingLabel) + ")"
    : "the game ended in a tie";
  document.getElementById("viz-sub").innerHTML =
    resultText + " · charts read from <b>" + esc(anchorName) +
    "</b>'s side — up is good for them · ←/→ step turns" +
    '<span class="adv"><br />' + K + " ensemble member" + (K > 1 ? "s" : "") +
    (D.uncertaintyScale > 0 ? " · gate scale " + D.uncertaintyScale : "") +
    " · <code>" + D.ckpts.map(esc).join("</code>, <code>") + "</code></span>";

  function esc(s) {
    return String(s).replace(/[&<>"]/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c];
    });
  }
  function fmtSigned(v) { return (v > 0 ? "+" : "") + v; }
  // Advantage formatting: internal values live in [-1, 1] (P(win) −
  // P(loss) from the anchor's side). Shown 0-centred, chess.com style:
  // +100% = anchor certain to win, 0 = even, −100% = opponent certain.
  function pct(v) {
    return (v < 0 ? "−" : v > 0 ? "+" : "") +
      Math.abs(v * 100).toFixed(0) + "%";
  }
  function el(name, attrs, parent) {
    var node = document.createElementNS(SVG, name);
    for (var key in attrs) node.setAttribute(key, attrs[key]);
    if (parent) parent.appendChild(node);
    return node;
  }
  function stepLabel(i) { return i === N - 1 ? "end" : "turn " + (i + 1); }

  // --- Φ timeline -------------------------------------------------------
  var W = 640, H = 300;
  // r fits the right-hand mons axis; shared across lanes so columns align.
  var PAD = { l: 40, r: 40, t: 18, b: 30 };
  var chart = document.getElementById("phi-chart");
  var xOf = function (i) {
    return PAD.l + (N === 1 ? 0 : (i / (N - 1)) * (W - PAD.l - PAD.r));
  };
  var yOf = function (v) {
    return PAD.t + ((1 - v) / 2) * (H - PAD.t - PAD.b);
  };
  function linePath(values) {
    var d = "";
    for (var i = 0; i < values.length; i++) {
      d += (i ? "L" : "M") + xOf(i).toFixed(1) + " " + yOf(values[i]).toFixed(1);
    }
    return d;
  }

  // Grid + axes. Two y scales share the [-1, 1] plot space: advantage
  // (±100%, left, blue) and the material forecast (±maxMargin mons,
  // right, green). Inline fill styles: presentation attributes lose to
  // the .tick-label class rule.
  [-1, -0.5, 0, 0.5, 1].forEach(function (v) {
    el("line", {
      x1: PAD.l, x2: W - PAD.r, y1: yOf(v), y2: yOf(v),
      stroke: v === 0 ? "var(--baseline)" : "var(--grid)",
      "stroke-width": v === 0 ? 1.5 : 1,
    }, chart);
    var left = el("text", {
      x: PAD.l - 6, y: yOf(v) + 3, "text-anchor": "end", "class": "tick-label",
      style: "fill: var(--series-1)",
    }, chart);
    left.textContent = (v > 0 ? "+" : "") + (v * 100) + "%";
    var right = el("text", {
      x: W - PAD.r + 6, y: yOf(v) + 3, "text-anchor": "start",
      "class": "tick-label", style: "fill: var(--series-3)",
    }, chart);
    right.textContent = (v > 0 ? "+" : "") + (v * D.maxMargin);
  });
  var xTickStep = Math.max(1, Math.round((N - 1) / 8 / 5) * 5) || 1;
  for (var t = xTickStep; t < N - 1; t += xTickStep) {
    var tick = el("text", {
      x: xOf(t - 1), y: H - PAD.b + 14, "text-anchor": "middle", "class": "tick-label",
    }, chart);
    tick.textContent = t;
  }
  var endTick = el("text", {
    x: xOf(N - 1), y: H - PAD.b + 14, "text-anchor": "middle", "class": "tick-label",
  }, chart);
  endTick.textContent = "end";
  var xAxisLabel = el("text", {
    x: (PAD.l + W - PAD.r) / 2, y: H - 4, "text-anchor": "middle", "class": "axis-label",
  }, chart);
  xAxisLabel.textContent = "turn";
  var topSide = el("text", { x: PAD.l + 6, y: PAD.t + 12, "class": "side-label" }, chart);
  topSide.textContent = anchorName + " winning";
  var bottomSide = el("text", {
    x: PAD.l + 6, y: H - PAD.b - 6, "class": "side-label",
  }, chart);
  bottomSide.textContent = oppName + " winning";

  // Deep model internals carry class "adv" (hidden until the details
  // toggle); the ensemble spread stays default-visible — disagreement is
  // honest information for any reader.
  // ±std band around the material forecast.
  if (K > 1) {
    var band = "";
    for (var i = 0; i < N; i++) {
      band += (i ? "L" : "M") + xOf(i).toFixed(1) + " " +
        yOf(Math.min(1, D.mean[i] + D.std[i])).toFixed(1);
    }
    for (var j = N - 1; j >= 0; j--) {
      band += "L" + xOf(j).toFixed(1) + " " +
        yOf(Math.max(-1, D.mean[j] - D.std[j])).toFixed(1);
    }
    el("path", { d: band + "Z", fill: "var(--series-3)", opacity: 0.12 }, chart);
  }
  // Ensemble members: hairlines, identity carried by the legend not hue.
  D.members.forEach(function (member) {
    el("path", {
      d: linePath(member), fill: "none",
      stroke: "var(--muted)", "stroke-width": 1, opacity: 0.45,
    }, chart);
  });
  // Mirror check: −Φ from the other player's trajectory (should overlap).
  if (D.mirror) {
    el("path", {
      d: linePath(D.mirror), fill: "none", stroke: "var(--text-secondary)",
      "stroke-width": 1.5, "stroke-dasharray": "2 4", opacity: 0.8,
      "class": "adv",
    }, chart);
  }
  if (D.gated) {
    el("path", {
      d: linePath(D.gated), fill: "none", stroke: "var(--series-2)",
      "stroke-width": 2, "stroke-dasharray": "6 4", "class": "adv",
    }, chart);
  }
  // Material forecast (expected final mons ahead): keeps grading decided
  // positions where the advantage line saturates.
  el("path", {
    d: linePath(D.mean), fill: "none", stroke: "var(--series-3)",
    "stroke-width": 2, "stroke-linejoin": "round",
  }, chart);
  // The headline series: win chance, chess.com-eval style.
  el("path", {
    d: linePath(D.winMean), fill: "none", stroke: "var(--series-1)",
    "stroke-width": 2.5, "stroke-linejoin": "round",
  }, chart);
  // Final result at the terminal step (win/loss/tie pole).
  el("circle", {
    cx: xOf(N - 1),
    cy: yOf(D.actualMargin > 0 ? 1 : D.actualMargin < 0 ? -1 : 0),
    r: 4.5,
    fill: "var(--series-1)", stroke: "var(--surface-1)", "stroke-width": 2,
  }, chart);

  var hoverLine = el("line", {
    y1: PAD.t, y2: H - PAD.b, stroke: "var(--text-secondary)",
    "stroke-width": 1, visibility: "hidden",
  }, chart);
  var playLine = el("line", {
    y1: PAD.t, y2: H - PAD.b, stroke: "var(--series-1)",
    "stroke-width": 1.5, opacity: 0.9, visibility: "hidden",
  }, chart);
  var hoverDot = el("circle", {
    r: 3.5, fill: "var(--series-1)", stroke: "var(--surface-1)",
    "stroke-width": 1.5, visibility: "hidden",
  }, chart);

  // Legend. Fourth field marks details-only entries.
  var legendItems = [
    ["Advantage (left axis)", "var(--series-1)", "solid", false],
    ["Mons ahead (right axis)", "var(--series-3)", "solid", false],
    K > 1 ? ["Model spread", "var(--series-3)", "band", false] : null,
    ["Final result", "var(--series-1)", "dot", false],
    D.gated ? ["Gated forecast", "var(--series-2)", "dashed", true] : null,
    D.mirror ? ["Mirror check", "var(--text-secondary)", "dotted", true] : null,
  ].filter(Boolean);
  var legendEl = document.getElementById("phi-legend");
  legendItems.forEach(function (item) {
    var div = document.createElement("div");
    div.className = "item" + (item[3] ? " adv" : "");
    var swatchSvg =
      '<svg width="18" height="10" viewBox="0 0 18 10">' +
      (item[2] === "band"
        ? '<rect x="0" y="1" width="18" height="8" fill="' + item[1] + '" opacity="0.35"/>'
        : item[2] === "dot"
          ? '<circle cx="9" cy="5" r="3.5" fill="' + item[1] + '"/>'
          : '<line x1="0" x2="18" y1="5" y2="5" stroke="' + item[1] +
            '" stroke-width="2"' +
            (item[2] === "dashed" ? ' stroke-dasharray="5 3"' : "") +
            (item[2] === "dotted" ? ' stroke-dasharray="2 3"' : "") + "/>") +
      "</svg>";
    div.innerHTML = swatchSvg + "<span>" + esc(item[0]) + "</span>";
    legendEl.appendChild(div);
  });

  // --- Momentum lane ------------------------------------------------------
  // Δ_j = win-chance(step j+1) − win-chance(step j) covers exactly the
  // events of turn j+1 (states sit at |turn| boundaries). Win-chance (not
  // material) deltas: decided-game conversion turns stay quiet, like a
  // chess eval graph. Includes chance — a big bar can be a crit.
  var deltaChart = document.getElementById("delta-chart");
  var deltas = [];
  for (var di = 0; di + 1 < N; di++) {
    deltas.push(D.winMean[di + 1] - D.winMean[di]);
  }
  if (deltas.length) {
    var DH = 110;
    var DPAD = { t: 16, b: 18 };
    document.getElementById("delta-title").innerHTML =
      "Momentum — <b>" + esc(anchorName) +
      "</b> gained (blue) or lost (red) ground each turn · includes " +
      "luck as well as skill · click a bar to watch that turn";
    var maxAbs = Math.max(1e-6, Math.max.apply(null, deltas.map(Math.abs)));
    var zeroY = DPAD.t + (DH - DPAD.t - DPAD.b) / 2;
    var scaleY = (DH - DPAD.t - DPAD.b) / 2 / maxAbs;
    el("line", {
      x1: PAD.l, x2: W - PAD.r, y1: zeroY, y2: zeroY,
      stroke: "var(--baseline)", "stroke-width": 1,
    }, deltaChart);
    var bestJ = deltas.indexOf(Math.max.apply(null, deltas));
    var worstJ = deltas.indexOf(Math.min.apply(null, deltas));
    var deltaBars = deltas.map(function (d, j) {
      var mid = (xOf(j) + xOf(j + 1)) / 2;
      var barW = Math.max(1.5, xOf(1) - xOf(0) - 2);
      var h = Math.max(Math.abs(d) * scaleY, 0.5);
      return el("rect", {
        x: mid - barW / 2,
        y: d >= 0 ? zeroY - h : zeroY,
        width: barW, height: h, rx: 1.5,
        fill: d >= 0 ? "var(--pos-pole)" : "var(--neg-pole)",
        opacity: 0.85,
      }, deltaChart);
    });
    [[bestJ, "▲", -4], [worstJ, "▼", 12]].forEach(function (mark) {
      var j = mark[0];
      var tipY = deltas[j] >= 0 ? zeroY - Math.abs(deltas[j]) * scaleY : zeroY + Math.abs(deltas[j]) * scaleY;
      var label = el("text", {
        x: (xOf(j) + xOf(j + 1)) / 2, y: tipY + mark[2],
        "text-anchor": "middle", "class": "side-label",
      }, deltaChart);
      label.textContent = mark[1] + " turn " + (j + 1);
    });

    function deltaAt(clientX) {
      var rect = deltaChart.getBoundingClientRect();
      var frac = (clientX - rect.left) / rect.width * W;
      var j = Math.round((frac - PAD.l) / (W - PAD.l - PAD.r) * (N - 1) - 0.5);
      return Math.max(0, Math.min(deltas.length - 1, j));
    }
    var deltaHover = null;
    deltaChart.style.cursor = "pointer";
    deltaChart.addEventListener("mousemove", function (evt) {
      var j = deltaAt(evt.clientX);
      if (deltaHover !== null && deltaHover !== j) {
        deltaBars[deltaHover].setAttribute("opacity", 0.85);
      }
      deltaHover = j;
      deltaBars[j].setAttribute("opacity", 1);
      var d = deltas[j];
      tooltip.innerHTML =
        "<b>turn " + (j + 1) + "</b><br />" +
        '<span class="k">swing</span> ' + (d >= 0 ? "+" : "−") +
        Math.abs(d * 100).toFixed(0) + "%<br />" +
        '<span class="k">advantage</span> ' + pct(D.winMean[j]) +
        " → " + pct(D.winMean[j + 1]);
      tooltip.style.display = "block";
      tooltip.style.left = (evt.clientX + 14) + "px";
      tooltip.style.top = (evt.clientY - 10) + "px";
    });
    deltaChart.addEventListener("mouseleave", function () {
      if (deltaHover !== null) deltaBars[deltaHover].setAttribute("opacity", 0.85);
      deltaHover = null;
      tooltip.style.display = "none";
    });
    deltaChart.addEventListener("click", function (evt) {
      // Seek to the start of the swing's turn so playing shows its events.
      if (window.battle) window.battle.seekTurn(deltaAt(evt.clientX) + 1);
    });
  }

  // --- Per-mon faint risk heatmap (survival aux head) --------------------
  // Present only for checkpoints trained with the aux head. Cell value is
  // E[y] = E[discount^(steps to next faint)]: 1 = faints now, 0 = safe or
  // never. Rows are stable revelation-order slots, anchor's mons first;
  // neutral cells are fainted or not-yet-revealed mons (loss-masked in
  // training, so the head's output there is meaningless).
  var survChart = document.getElementById("surv-chart");
  var survPlayLine = null;
  var survXOf = null;
  var survRowOrder = [];
  var SPAD = { t: 6, b: 18 };
  var SRH = 13;
  // Species names need a wider label gutter than the shared PAD.l, so the
  // heatmap has its own x mapping — cells, cursor, hover and click all use
  // it, keeping the lane internally consistent (columns sit slightly left
  // of the charts above; the synced cursor is remapped, not shared).
  var SXL = 96;
  if (D.survival) {
    for (var sm = 0; sm < D.slotSides.length; sm++) {
      if (D.slotSides[sm] === 1) survRowOrder.push(sm);
    }
    var survMyRows = survRowOrder.length;
    for (var so = 0; so < D.slotSides.length; so++) {
      if (D.slotSides[so] === 0) survRowOrder.push(so);
    }
    var survRows = survRowOrder.length;
    // Rows are laid out under per-side group headers — random battles
    // often have the SAME species on both teams, so ownership must be
    // structural, not just label colour.
    var SGH = 15;
    var survRowY = function (r) {
      return SPAD.t + SGH + r * SRH + (r >= survMyRows ? SGH : 0);
    };
    var SVH = survRowY(survRows - 1) + SRH + SPAD.b;
    survChart.setAttribute("viewBox", "0 0 " + W + " " + SVH);
    survXOf = function (i) {
      return SXL + (N === 1 ? 0 : (i / (N - 1)) * (W - SXL - PAD.r));
    };
    // Cell edges sit at midpoints between steps, clamped to the lane —
    // centered cells overflowed half a cell into the label gutter on
    // short games and covered the species names.
    var survLeft = function (i) {
      return i === 0 ? SXL : (survXOf(i - 1) + survXOf(i)) / 2;
    };
    var survRight = function (i) {
      return i === N - 1 ? W - PAD.r : (survXOf(i) + survXOf(i + 1)) / 2;
    };
    document.getElementById("surv-title").innerHTML =
      "Faint watch — how much danger each Pokémon is in · " +
      "outlined = on the field · grey = fainted or not yet seen";
    var survName = function (slot) {
      var name = D.slotNames && D.slotNames[slot];
      if (!name) return "#" + (slot + 1);
      return name.charAt(0).toUpperCase() + name.slice(1);
    };
    var survRowLabel = function (r) {
      var slot = survRowOrder[r];
      return survName(slot) +
        " (" + (D.slotSides[slot] === 1 ? anchorName : oppName) + ")";
    };
    if (survMyRows > 0) {
      var myHeader = el("text", {
        x: SXL, y: survRowY(0) - 4, "class": "side-label",
        style: "fill: var(--pos-pole)",
      }, survChart);
      myHeader.textContent = anchorName + "'s team";
    }
    if (survRows > survMyRows) {
      var oppHeader = el("text", {
        x: SXL, y: survRowY(survMyRows) - 4, "class": "side-label",
        style: "fill: var(--neg-pole)",
      }, survChart);
      oppHeader.textContent = oppName + "'s team";
    }
    survRowOrder.forEach(function (slot, r) {
      var rowY = survRowY(r);
      var rowLabel = el("text", {
        x: SXL - 6, y: rowY + SRH - 3, "text-anchor": "end",
        "class": "tick-label",
        style: "fill: " +
          (D.slotSides[slot] === 1 ? "var(--pos-pole)" : "var(--neg-pole)"),
      }, survChart);
      var labelText = survName(slot);
      rowLabel.textContent =
        labelText.length > 13 ? labelText.slice(0, 12) + "…" : labelText;
      var firstSeen = -1;
      for (var f = 0; f < N; f++) {
        if (D.survivalMask[f][slot] > 0) { firstSeen = f; break; }
      }
      for (var i = 0; i < N; i++) {
        var on = D.survivalMask[i][slot] > 0;
        // Not yet revealed: leave blank — grey blocks tiling the lane
        // before each mon's first appearance read as clutter.
        if (!on && (firstSeen < 0 || i < firstSeen)) continue;
        var left = survLeft(i);
        el("rect", {
          x: left + 0.5, y: rowY + 1,
          width: Math.max(1, survRight(i) - left - 1), height: SRH - 2,
          rx: 1.5,
          fill: on
            ? "color-mix(in oklab, var(--neg-pole) " +
              Math.round(D.survival[i][slot] * 100) + "%, var(--surface-1))"
            : "var(--mid)",
          opacity: on ? 1 : 0.25,
        }, survChart);
      }
      // On-field indicator: one soft ring per continuous stint on the
      // field — a hard box on every active cell tiled the lane with
      // black/white rectangles and read as noise.
      if (D.active) {
        var runStart = -1;
        for (var t = 0; t <= N; t++) {
          var act = t < N && D.active[t][slot] > 0;
          if (act && runStart < 0) runStart = t;
          if (!act && runStart >= 0) {
            el("rect", {
              x: survLeft(runStart) + 0.5, y: rowY + 0.5,
              width: Math.max(2, survRight(t - 1) - survLeft(runStart) - 1),
              height: SRH - 1, rx: 3, fill: "none",
              stroke:
                "color-mix(in oklab, var(--text-primary) 45%, transparent)",
              "stroke-width": 1,
            }, survChart);
            runStart = -1;
          }
        }
      }
    });
    survPlayLine = el("line", {
      y1: SPAD.t, y2: SVH - SPAD.b, stroke: "var(--series-1)",
      "stroke-width": 1.5, opacity: 0.9, visibility: "hidden",
    }, survChart);

    var survAt = function (evt) {
      var rect = survChart.getBoundingClientRect();
      var fracX = (evt.clientX - rect.left) / rect.width * W;
      var i = Math.round((fracX - SXL) / (W - SXL - PAD.r) * (N - 1));
      var fracY = (evt.clientY - rect.top) / rect.height * SVH;
      var yRel = fracY - SPAD.t - SGH;
      if (yRel >= survMyRows * SRH) yRel -= SGH;
      var r = Math.floor(yRel / SRH);
      return {
        step: Math.max(0, Math.min(N - 1, i)),
        row: Math.max(0, Math.min(survRows - 1, r)),
      };
    };
    survChart.style.cursor = "pointer";
    survChart.addEventListener("mousemove", function (evt) {
      var at = survAt(evt);
      var slot = survRowOrder[at.row];
      var on = D.survivalMask[at.step][slot] > 0;
      var isAct = D.active && D.active[at.step][slot] > 0;
      tooltip.innerHTML =
        "<b>" + esc(stepLabel(at.step)) + " · " + esc(survRowLabel(at.row)) +
        "</b>" + (isAct ? " · on the field" : "") + "<br />" +
        (on
          ? '<span class="k">danger</span> ' +
            (D.survival[at.step][slot] * 100).toFixed(0) + "%"
          : '<span class="k">fainted or not yet seen</span>');
      tooltip.style.display = "block";
      tooltip.style.left = (evt.clientX + 14) + "px";
      tooltip.style.top = (evt.clientY - 10) + "px";
    });
    survChart.addEventListener("mouseleave", function () {
      tooltip.style.display = "none";
    });
    survChart.addEventListener("click", function (evt) {
      var at = survAt(evt);
      if (window.battle) {
        window.battle.seekTurn(at.step === N - 1 ? Infinity : at.step + 1);
      }
    });
  }

  // --- Margin distribution ---------------------------------------------
  var MW = 640, MH = 130;
  var MPAD = { l: 40, r: 14, t: 12, b: 24 };
  var marginChart = document.getElementById("margin-chart");
  var marginTitle = document.getElementById("margin-title");
  var bins = 2 * D.maxMargin + 1;
  var slotW = (MW - MPAD.l - MPAD.r) / bins;
  var barNodes = [], barLabels = [];
  el("line", {
    x1: MPAD.l, x2: MW - MPAD.r, y1: MH - MPAD.b, y2: MH - MPAD.b,
    stroke: "var(--baseline)", "stroke-width": 1,
  }, marginChart);
  for (var b = 0; b < bins; b++) {
    var m = b - D.maxMargin;
    // NOT named pct: a `var` here shares the IIFE's function scope and
    // would clobber the pct() formatter (it did — broke every tooltip).
    var mixPct = Math.round((Math.abs(m) / D.maxMargin) * 65) + 35;
    var fill = m === 0
      ? "var(--mid)"
      : "color-mix(in oklab, var(--" + (m > 0 ? "pos" : "neg") +
        "-pole) " + mixPct + "%, var(--surface-1))";
    barNodes.push(el("rect", {
      x: MPAD.l + b * slotW + 1, width: slotW - 2,
      y: MH - MPAD.b, height: 0, rx: 3, fill: fill,
      stroke: m === 0 ? "var(--baseline)" : "none",
    }, marginChart));
    barLabels.push(el("text", {
      x: MPAD.l + b * slotW + slotW / 2, "text-anchor": "middle",
      "class": "tick-label", visibility: "hidden",
    }, marginChart));
    var binLabel = el("text", {
      x: MPAD.l + b * slotW + slotW / 2, y: MH - MPAD.b + 13,
      "text-anchor": "middle", "class": "tick-label",
    }, marginChart);
    binLabel.textContent = fmtSigned(m);
    if (m === D.actualMargin) {
      var star = el("text", {
        x: MPAD.l + b * slotW + slotW / 2, y: MH - 1,
        "text-anchor": "middle", "class": "side-label",
      }, marginChart);
      star.textContent = "actual";
    }
  }
  var expectLine = el("line", {
    y1: MPAD.t - 4, y2: MH - MPAD.b, stroke: "var(--text-primary)",
    "stroke-width": 1.5,
  }, marginChart);

  function renderMargin(step) {
    var probs = D.probs[step];
    var top = Math.max.apply(null, probs);
    var phi = D.mean[step];
    var lead = phi * D.maxMargin;
    marginTitle.innerHTML =
      "Predicted final score — <b>" + esc(stepLabel(step)) + "</b> · " +
      (Math.abs(lead) < 0.25
        ? "too close to call"
        : "<b>" + esc(lead > 0 ? anchorName : oppName) +
          "</b> to finish ~" + Math.abs(lead).toFixed(1) + " Pokémon ahead");
    for (var b = 0; b < bins; b++) {
      var h = (probs[b] / Math.max(top, 1e-6)) * (MH - MPAD.t - MPAD.b);
      barNodes[b].setAttribute("y", MH - MPAD.b - h);
      barNodes[b].setAttribute("height", Math.max(h, 0.5));
      // Selective labels: only the modal bin gets a number.
      if (probs[b] === top && top > 0) {
        barLabels[b].setAttribute("y", MH - MPAD.b - h - 4);
        barLabels[b].setAttribute("visibility", "visible");
        barLabels[b].textContent = (probs[b] * 100).toFixed(0) + "%";
      } else {
        barLabels[b].setAttribute("visibility", "hidden");
      }
    }
    var ex = MPAD.l + ((phi * D.maxMargin + D.maxMargin + 0.5) / bins) *
      (MW - MPAD.l - MPAD.r);
    expectLine.setAttribute("x1", ex);
    expectLine.setAttribute("x2", ex);
  }

  // --- Chance-event tags --------------------------------------------------
  // A state-value model cannot split a swing into skill vs luck (it only
  // sees states), but the protocol log explicitly marks DISCRETE chance
  // events — the one luck signal a public replay carries. Parse them per
  // turn so key moments stop crediting players for crits and full paras.
  // Damage-roll and speed-tie luck are not recorded and stay invisible.
  // Display-side attribution only; never a training signal.
  var chanceEvents = {};
  (function () {
    var logText = (
      document.querySelector("script.battle-log-data").textContent || ""
    ).replace(/\\\//g, "/");
    var turn = 0;
    var sawDamage = false; // within the current move block (secondary procs)
    var add = function (label) {
      if (!turn) return;
      var list = (chanceEvents[turn] = chanceEvents[turn] || []);
      if (list.indexOf(label) < 0) list.push(label);
    };
    logText.split("\n").forEach(function (line) {
      if (!line.startsWith("|")) return;
      var parts = line.split("|"); // ["", cmd, ...args]
      var cmd = parts[1];
      if (cmd === "turn") {
        turn = parseInt(parts[2], 10) || 0;
      } else if (cmd === "move") {
        sawDamage = false;
      } else if (cmd === "-damage") {
        if (line.indexOf("[from] confusion") >= 0) add("confusion self-hit");
        else sawDamage = true;
      } else if (cmd === "-crit") {
        add("critical hit");
      } else if (cmd === "-miss") {
        add("miss");
      } else if (cmd === "-ohko") {
        add("OHKO");
      } else if (cmd === "cant") {
        if (parts[3] === "par") add("full paralysis");
        else if (parts[3] === "slp") add("stayed asleep");
        else if (parts[3] === "frz") add("frozen solid");
        else if (parts[3] === "flinch") add("flinch");
      } else if (cmd === "-status") {
        // Status from a damaging move's secondary (30% para etc.) or a
        // contact ability is chance; a status MOVE (Thunder Wave) is not.
        if (line.indexOf("[from] ability:") >= 0) add("ability proc");
        else if (sawDamage) add("secondary effect");
      }
    });
  })();

  // A faint the model already priced (high danger at the turn's start) is
  // anticipated doom resolving, not someone's blunder.
  function faintSeenComing(step) {
    if (!D.survival || step < 1) return false;
    for (var slot = 0; slot < D.slotSides.length; slot++) {
      if (
        D.survivalMask[step - 1][slot] > 0 &&
        D.survivalMask[step][slot] === 0 &&
        D.survival[step - 1][slot] >= 0.6
      ) {
        return true;
      }
    }
    return false;
  }

  // --- Key moments: chess.com-style swing annotations ---------------------
  // Swing over turn s's events = win-chance[s] − win-chance[s−1] (state
  // s−1 is the start of turn s). Attribution: a Pokemon turn contains
  // BOTH players' moves plus the dice — logged chance events and
  // model-anticipated faints soften the glyph; the rest is left to the
  // viewer. Thresholds in advantage units of [-1, 1]: 0.30 = a 30-point
  // swing on the 0-centred ±100% scale.
  var SWING_MAJOR = 0.30, SWING_MINOR = 0.15;
  var swings = [];
  for (var s = 1; s < N; s++) {
    var d = D.winMean[s] - D.winMean[s - 1];
    var tier = Math.abs(d) >= SWING_MAJOR ? 2 : Math.abs(d) >= SWING_MINOR ? 1 : 0;
    if (tier) {
      swings.push({
        step: s,
        delta: d,
        tier: tier,
        luck: chanceEvents[s] || [],
        expected: faintSeenComing(s),
      });
    }
  }
  function swingColor(sw) {
    return sw.delta > 0 ? "var(--pos-pole)" : "var(--neg-pole)";
  }
  function swingGlyph(sw) {
    if (sw.luck.length) return "🎲";
    if (sw.expected) return "…";
    if (sw.delta > 0) return sw.tier === 2 ? "!!" : "!";
    return sw.tier === 2 ? "??" : "?";
  }
  function swingLabel(sw) {
    return sw.step === N - 1 ? "final turn" : "turn " + sw.step;
  }
  swings.forEach(function (sw) {
    el("circle", {
      cx: xOf(sw.step), cy: yOf(D.winMean[sw.step]), r: sw.tier === 2 ? 5 : 3.5,
      fill: swingColor(sw), stroke: "var(--surface-1)", "stroke-width": 1.5,
    }, chart);
    var glyph = el("text", {
      x: xOf(sw.step),
      y: yOf(D.winMean[sw.step]) + (sw.delta > 0 ? -9 : 16),
      "text-anchor": "middle", "font-size": "11", "font-weight": "700",
      fill: swingColor(sw),
    }, chart);
    glyph.textContent = swingGlyph(sw);
  });

  var keyTurns = swings
    .slice()
    .sort(function (a, b) { return Math.abs(b.delta) - Math.abs(a.delta); })
    .slice(0, 8);
  if (keyTurns.length) {
    document.getElementById("key-title").textContent = "Key moments";
    document.getElementById("key-note").textContent =
      "🎲 = the log records a chance event that turn (crit, miss, full " +
      "paralysis, ...) — credit the dice, not the player. … = a faint " +
      "the model already saw coming. Damage rolls and speed ties aren't " +
      "recorded in replays, so some luck is untaggable. Everything else " +
      "was both players' decisions — click a moment and judge.";
    var keyEl = document.getElementById("key-turns");
    keyTurns.forEach(function (sw) {
      var row = document.createElement("div");
      row.className = "row";
      var toward = sw.delta > 0 ? anchorName : oppName;
      var cause = sw.luck.length
        ? ' <span style="color:var(--muted)">· ' +
          sw.luck.map(esc).join(", ") + "</span>"
        : sw.expected
          ? ' <span style="color:var(--muted)">· loss was already likely</span>'
          : "";
      var disagree = K > 1 && D.std[sw.step] > 0.12
        ? ' <span class="adv" style="color:var(--muted)">· models disagree</span>'
        : "";
      row.innerHTML =
        '<span class="badge" style="background:' + swingColor(sw) + '">' +
        swingGlyph(sw) + "</span><b>" + esc(swingLabel(sw)) + "</b>" +
        '<span class="delta">' + (sw.delta > 0 ? "+" : "−") +
        Math.abs(sw.delta * 100).toFixed(0) + "% swing</span>" +
        "<span>toward " + esc(toward) + "</span>" + cause + disagree;
      row.addEventListener("click", function () {
        var battle = window.battle;
        if (battle) battle.seekTurn(Math.min(sw.step, N - 1));
      });
      keyEl.appendChild(row);
    });
  }

  // --- Interaction: hover, click-to-seek, playback sync ------------------
  var tooltip = document.getElementById("phi-tooltip");
  var hoverStep = null;
  var playStep = 0;

  function stepAt(clientX) {
    var rect = chart.getBoundingClientRect();
    var frac = (clientX - rect.left) / rect.width * W;
    var i = Math.round((frac - PAD.l) / (W - PAD.l - PAD.r) * (N - 1));
    return Math.max(0, Math.min(N - 1, i));
  }
  function showHover(step, evt) {
    hoverStep = step;
    var x = xOf(step);
    hoverLine.setAttribute("x1", x);
    hoverLine.setAttribute("x2", x);
    hoverLine.setAttribute("visibility", "visible");
    hoverDot.setAttribute("cx", x);
    hoverDot.setAttribute("cy", yOf(D.winMean[step]));
    hoverDot.setAttribute("visibility", "visible");
    var lead = D.mean[step] * D.maxMargin;
    var rows =
      "<b>" + esc(stepLabel(step)) + "</b><br />" +
      '<span class="k">advantage</span> ' + pct(D.winMean[step]) +
      '<br /><span class="k">forecast</span> ' +
      (Math.abs(lead) < 0.25
        ? "even"
        : esc(lead > 0 ? anchorName : oppName) + " +" +
          Math.abs(lead).toFixed(1) + " mons");
    if (step > 0) {
      var turnDelta = D.winMean[step] - D.winMean[step - 1];
      rows += '<br /><span class="k">swing turn ' + step + "</span> " +
        (turnDelta >= 0 ? "+" : "−") +
        Math.abs(turnDelta * 100).toFixed(0) + "%";
    }
    if (root.classList.contains("show-adv")) {
      if (K > 1) {
        rows += '<br /><span class="k">± spread</span> ' +
          D.std[step].toFixed(3);
      }
      if (D.gated) {
        rows += '<br /><span class="k">gated Φ</span> ' +
          D.gated[step].toFixed(3);
      }
      if (D.mirror && step < D.mirror.length) {
        rows += '<br /><span class="k">mirror Δ</span> ' +
          Math.abs(D.mean[step] - D.mirror[step]).toFixed(4);
      }
    }
    tooltip.innerHTML = rows;
    tooltip.style.display = "block";
    tooltip.style.left = (evt.clientX + 14) + "px";
    tooltip.style.top = (evt.clientY - 10) + "px";
    renderMargin(step);
  }
  chart.addEventListener("mousemove", function (evt) {
    showHover(stepAt(evt.clientX), evt);
  });
  chart.addEventListener("mouseleave", function () {
    hoverStep = null;
    hoverLine.setAttribute("visibility", "hidden");
    hoverDot.setAttribute("visibility", "hidden");
    tooltip.style.display = "none";
    renderMargin(playStep);
  });
  chart.style.cursor = "pointer";
  chart.addEventListener("click", function (evt) {
    var step = stepAt(evt.clientX);
    var battle = window.battle;
    if (battle) battle.seekTurn(step === N - 1 ? Infinity : step + 1);
  });
  marginChart.addEventListener("mousemove", function (evt) {
    // The bars follow whichever step the timeline shows; hovering them
    // pins the tooltip's step too.
    if (hoverStep !== null) showHover(hoverStep, evt);
  });

  // Playback sync: state i is emitted at the |turn|i+1 boundary, so while
  // turn t animates the freshest state is i = t-1; the terminal state only
  // applies once the queue has fully played out.
  setInterval(function () {
    var battle = window.battle;
    if (!battle) return;
    var step = battle.atQueueEnd
      ? N - 1
      : Math.max(0, Math.min(N - 2, (battle.turn || 0) - 1));
    if (step !== playStep) {
      playStep = step;
      if (hoverStep === null) renderMargin(playStep);
    }
    var x = xOf(playStep);
    playLine.setAttribute("x1", x);
    playLine.setAttribute("x2", x);
    playLine.setAttribute("visibility", "visible");
    if (survPlayLine) {
      var sx = survXOf(playStep);
      survPlayLine.setAttribute("x1", sx);
      survPlayLine.setAttribute("x2", sx);
      survPlayLine.setAttribute("visibility", "visible");
    }
  }, 250);

  var advBtn = document.getElementById("adv-toggle");
  advBtn.addEventListener("click", function () {
    var on = root.classList.toggle("show-adv");
    advBtn.textContent = on ? "Hide model details" : "Show model details";
  });

  // Arrow keys step the replay turn by turn (same as the player's
  // last/next buttons); the playback-sync loop moves every chart cursor.
  document.addEventListener("keydown", function (evt) {
    var battle = window.battle;
    if (!battle) return;
    if (evt.key === "ArrowLeft") {
      battle.seekBy(-1);
      evt.preventDefault();
    } else if (evt.key === "ArrowRight") {
      battle.seekBy(1);
      evt.preventDefault();
    }
  });

  renderMargin(0);
})();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()