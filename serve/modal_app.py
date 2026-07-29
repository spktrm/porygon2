"""Serverless Φ replay visualiser on Modal.

Anyone pastes a replay.pokemonshowdown.com URL or id; the app fetches the
replay, encodes it through the real shard exporter (node subprocess), runs
the offline critic ensemble on CPU, and serves the standalone visualisation
page (rl/offline/visualise.py's template — battle player + Φ chart + swing
annotations, plus the per-mon faint-risk heatmap when the baked-in
checkpoints were trained with the survival aux head; CriticRunner detects
that from the params, so no flag is needed here). Pages are cached by
replay id (replays are immutable), so each replay is computed once, ever.

Deploy:
    pip install modal && modal setup     # once
    modal deploy serve/modal_app.py

Serve locally for development:
    modal serve serve/modal_app.py

Cost/latency profile:
- CPU-only inference: one forward pass over one trajectory per request —
  no GPU needed or billed.
- The critic ensemble loads once per container (@modal.enter), and the
  container stays warm for SCALEDOWN_WINDOW seconds between requests;
  scale-to-zero otherwise.
- JAX's persistent compilation cache lives on a Modal volume, so XLA
  compiles for each (time, history) shape bucket happen once across all
  containers/deploys, not per cold start.
- Rendered pages go in a Modal Dict keyed by (model version, replay id);
  repeat views skip everything.

The image bakes in: this repo (exporter compiled via tsc at build time),
the offline checkpoints under ckpts/offline/, node, and the python deps.
Redeploy after training a new critic to pick up new checkpoints — the
cache key is a hash of the renderer source + checkpoint bytes, so any
redeploy that could change a page's content invalidates it (names alone
wouldn't: retrains overwrite ckpt_best in place).
"""

import glob
import os
import re

import modal

REPO_REMOTE = "/root/porygon2"
UNCERTAINTY_SCALE = 0.0  # gate display in the page; 0 = raw ensemble mean
SCALEDOWN_WINDOW = 300  # seconds a container stays warm after last request
MAX_LOG_BYTES = 2_000_000  # refuse absurd inputs before spending compute

REPLAY_ID_RE = re.compile(r"[a-z0-9]+-[0-9]+(-[a-z0-9]+)?")

# Served formats are whatever the deploying checkout has critics for:
# every ckpts/offline/{format_id}[-ens{k}]/ckpt_best dir gets baked in
# (best artifacts only — not periodic saves; discover_ckpts picks the
# latest ckpt_* per member dir, and with ckpt_best as the sole entry
# that's what it finds). Train a critic for a new format
# (rl/offline/train.py --generation N --smogon-format X) and redeploy.


def _discover_ckpt_dirs() -> list[str]:
    """Repo-relative ckpt_best dirs. Module-level code runs BOTH at deploy
    time (CWD = the checkout) and inside the container (repo baked at
    REPO_REMOTE, CWD elsewhere) — resolve against whichever root exists so
    both contexts derive the same list."""
    for root in (".", REPO_REMOTE):
        dirs = sorted(
            os.path.relpath(path, root)
            for path in glob.glob(
                os.path.join(root, "ckpts", "offline", "*", "ckpt_best")
            )
        )
        if dirs:
            return dirs
    return []


CKPT_DIRS = _discover_ckpt_dirs()
if not CKPT_DIRS:
    raise RuntimeError(
        "no ckpts/offline/*/ckpt_best found — train an offline critic "
        "before deploying (rl/offline/train.py)"
    )
SUPPORTED_FORMATS = sorted(
    {os.path.basename(os.path.dirname(d)).split("-ens")[0] for d in CKPT_DIRS}
)
# Embedding tables are per generation; bake one set per served generation.
SERVED_GENERATIONS = sorted(
    {int(re.match(r"gen(\d+)", f).group(1)) for f in SUPPORTED_FORMATS}
)

# Keep these in sync with requirements.txt — protobuf in particular must be
# able to load the generated rl/environment/protos/*_pb2.py modules.
PY_DEPS = [
    "jax[cpu]",
    "flax",
    "optax",
    "chex",
    "ml_collections",
    "jaxtyping",
    "numpy",
    # The generated rl/environment/protos/*_pb2.py import
    # google.protobuf.runtime_version, which exists only in protobuf
    # >= 5.27; protobuf also validates gencode-vs-runtime versions, so if
    # this still complains at import, pin to exactly the version in the
    # training env (`pip show protobuf`).
    "protobuf>=5.27",
    "cloudpickle",
    "fastapi[standard]",
]

image = (
    modal.Image.debian_slim(python_version="3.11")
    # Debian's apt nodejs is ancient (no `??` support — the compiled
    # exporter targets ES2020); install a current LTS from NodeSource.
    .apt_install("curl", "ca-certificates")
    .run_commands(
        "curl -fsSL https://deb.nodesource.com/setup_22.x | bash -"
        " && apt-get install -y nodejs",
    )
    .pip_install(*PY_DEPS)
    # Additive, not repo-wholesale: exactly what the app touches at
    # runtime, so a live training run writing elsewhere in the repo can
    # never race the build (Modal aborts if an uploaded file mutates), and
    # the image stays small.
    #
    #  rl/                 — python source: offline pipeline, model, protos
    #  data/data/data.json — read by both python (rl/environment/data.py,
    #                        CWD-relative) and the exporter's data.ts
    #                        (../data/data relative to CWD=service/)
    #  gen9 embedding npys — the five files add_pretrained_embedding
    #                        actually loads (added individually below;
    #                        the gen9 JSONs are pipeline inputs, not
    #                        runtime deps). Deploy from a checkout that
    #                        HAS them: add_local_file fails fast if not,
    #                        and the Visualizer double-checks at startup
    #                        because data.py silently zero-fills missing
    #                        embeddings.
    #  constants/          — constants/data.json, for data.ts
    #  service/            — exporter source; npm ci + tsc at image build
    .add_local_dir(
        "rl",
        remote_path=f"{REPO_REMOTE}/rl",
        copy=True,
        ignore=["**/__pycache__", "**/.DS_Store"],
    )
    .add_local_file(
        "data/data/data.json",
        remote_path=f"{REPO_REMOTE}/data/data/data.json",
        copy=True,
    )
)

# The pretrained-embedding tables (rl/environment/data.py,
# add_pretrained_embedding) — the only .npy files the critic reads; one
# set per served generation.
EMBEDDING_NPYS = ["species", "abilities", "items", "moves", "learnset"]
for _gen in SERVED_GENERATIONS:
    for _name in EMBEDDING_NPYS:
        image = image.add_local_file(
            f"data/data/gen{_gen}/{_name}.npy",
            remote_path=f"{REPO_REMOTE}/data/data/gen{_gen}/{_name}.npy",
            copy=True,
        )

for _ckpt in CKPT_DIRS:
    image = image.add_local_dir(
        _ckpt, remote_path=f"{REPO_REMOTE}/{_ckpt}", copy=True
    )

image = (
    image.add_local_dir(
        "constants", remote_path=f"{REPO_REMOTE}/constants", copy=True
    )
    .add_local_dir(
        "service",
        remote_path=f"{REPO_REMOTE}/service",
        copy=True,
        ignore=["**/node_modules", "**/dist"],
    )
    # Compile the exporter at image-build time; dist/ is then newer than
    # src/, so visualise.py's staleness check never re-runs tsc at request
    # time.
    .run_commands(
        f"cd {REPO_REMOTE}/service && npm ci && npx tsc",
    )
    .env({"PYTHONPATH": REPO_REMOTE})
)

app = modal.App("porygon2-phi-viz", image=image)

# XLA compilation cache shared across containers and deploys: each new
# (time, history) shape bucket compiles once, ever.
jax_cache = modal.Volume.from_name("porygon2-jax-cache", create_if_missing=True)
# Rendered pages, keyed by (checkpoint version, replay id).
page_cache = modal.Dict.from_name("porygon2-phi-pages", create_if_missing=True)
# Reserved key holding the landing page's recently-viewed list.
RECENT_KEY = "__recent__"
MAX_RECENT = 24

LANDING = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Game review for Pokémon Showdown</title>
<style>
 body {{ font-family: system-ui, sans-serif; max-width: 640px;
        margin: 12vh auto; padding: 0 20px; }}
 form {{ display: flex; gap: 8px; }}
 input {{ flex: 1; font-size: 16px; padding: 10px 12px;
         box-sizing: border-box; }}
 button {{ font-size: 16px; padding: 10px 18px; cursor: pointer; }}
 p {{ color: #555; }}
 h2 {{ font-size: 13px; color: #888; margin: 32px 0 10px;
      text-transform: uppercase; letter-spacing: 0.05em; }}
 .grid {{ display: grid; gap: 10px;
         grid-template-columns: repeat(auto-fill, minmax(190px, 1fr)); }}
 .card {{ border: 1px solid #e3e3e0; border-radius: 8px; padding: 10px 12px;
         text-decoration: none; color: inherit; display: block; }}
 .card:hover {{ border-color: #999; }}
 .card .vs {{ font-weight: 600; font-size: 13px; margin-bottom: 4px;
             overflow: hidden; text-overflow: ellipsis;
             white-space: nowrap; }}
 .card .meta {{ font-size: 12px; color: #777; }}
 .chip {{ display: inline-block; border-radius: 9px; padding: 0 7px;
         font-weight: 700; color: #fff; font-size: 11px; }}
</style></head><body>
<h1>Game review for Pokémon Showdown</h1>
<p>Paste a <a href="https://replay.pokemonshowdown.com">replay</a> link and
watch your battle next to a turn-by-turn review: win chances as the game
swings, the key moments that decided it, the predicted final score, and
which Pokémon were in danger. Works with {formats} replays.</p>
<form id="f">
<input id="q" placeholder="https://replay.pokemonshowdown.com/{example_format}-2654504071"
 autofocus />
<button type="submit">Go</button>
</form>
<p id="err" style="color:#b33"></p>
__RECENT__
<script>
document.getElementById("f").addEventListener("submit", function (e) {{
  e.preventDefault();
  var v = document.getElementById("q").value.trim()
    .replace(/^https?:\\/\\/replay\\.pokemonshowdown\\.com\\//, "")
    .replace(/\\.(json|log|html)$/, "");
  if (/^[a-z0-9]+-[0-9]+(-[a-z0-9]+)?$/.test(v)) location.href = "/r/" + v;
  else document.getElementById("err").textContent = "That doesn't look like a replay id.";
}});
</script></body></html>""".format(
    formats=", ".join(SUPPORTED_FORMATS), example_format=SUPPORTED_FORMATS[0]
)


@app.cls(
    cpu=4.0,
    memory=8192,
    volumes={"/jax_cache": jax_cache},
    scaledown_window=SCALEDOWN_WINDOW,
)
@modal.concurrent(max_inputs=4)
class Visualizer:
    @modal.enter()
    def load(self):
        import os

        os.chdir(REPO_REMOTE)
        import jax

        jax.config.update("jax_compilation_cache_dir", "/jax_cache")

        # data.py's add_pretrained_embedding silently substitutes
        # ZeroEmbedding when an .npy is missing — the critic would then
        # produce garbage Φ without crashing. Refuse to serve that.
        from rl.environment.data import ONEHOT_ENCODERS
        from rl.model.modules import ZeroEmbedding

        zeroed = []
        for gen in SERVED_GENERATIONS:
            if gen not in ONEHOT_ENCODERS:
                raise RuntimeError(
                    f"generation {gen} has a checkpoint but no model/"
                    "embedding support (rl/environment/data.py "
                    "VALID_GENERATIONS)"
                )
            zeroed += [
                f"gen{gen}/{name}"
                for name, enc in ONEHOT_ENCODERS[gen].items()
                if isinstance(enc, ZeroEmbedding)
            ]
        if zeroed:
            raise RuntimeError(
                f"pretrained embeddings missing from the image: {zeroed} — "
                "deploy from a checkout with data/data/gen*/{*}.npy present"
            )

        import hashlib

        from rl.offline import visualise

        # One runner (checkpoint ensemble + jit) per served format.
        self.runners = {
            fmt: visualise.CriticRunner(visualise.discover_ckpts(fmt), fmt)
            for fmt in SUPPORTED_FORMATS
        }
        # Cache pages per (renderer, model) CONTENT, not names: retrains
        # overwrite ckpt_best in place and template changes ship under the
        # same paths, so name-based keys never invalidate. Hashing the
        # renderer source + checkpoint bytes makes any redeploy that could
        # change a page's pixels a cache miss; unchanged redeploys keep
        # serving cached pages. (Hashing the params files costs a few
        # seconds once per container start — noise next to jax warmup.)
        digest = hashlib.sha256()
        with open(visualise.__file__, "rb") as f:
            digest.update(f.read())
        for fmt in SUPPORTED_FORMATS:
            for path in self.runners[fmt].ckpt_paths:
                for root, _, files in sorted(os.walk(path)):
                    for name in sorted(files):
                        digest.update(name.encode())
                        with open(os.path.join(root, name), "rb") as f:
                            digest.update(f.read())
        self.model_version = digest.hexdigest()[:16]
        for fmt, runner in self.runners.items():
            print(
                f"{fmt}: {len(runner.ckpt_paths)} member(s) — "
                + "|".join("/".join(p.split("/")[-2:]) for p in runner.ckpt_paths)
            )
        print(f"page version {self.model_version}")

    def _bump_recent(self, replay_id: str, data: dict | None):
        """Maintains the landing page's recently-viewed list (most recent
        first, deduped, capped). Cosmetic: last-writer-wins under container
        concurrency, and never fails a request."""
        try:
            recent = list(page_cache.get(RECENT_KEY) or [])
            entry = next((r for r in recent if r["id"] == replay_id), None)
            if data is not None:
                entry = dict(
                    id=replay_id,
                    players=data["players"],
                    rating=data["rating"],
                    margin=data["actualMargin"],
                    # First clause only ("played out", "conceded", ...).
                    ending=data["endingLabel"].split(" — ")[0],
                )
            if entry is None:
                return
            recent = [r for r in recent if r["id"] != replay_id]
            recent.insert(0, entry)
            page_cache[RECENT_KEY] = recent[:MAX_RECENT]
        except Exception as err:  # noqa: BLE001 — strictly cosmetic
            print(f"recent-list update failed: {err}")

    def _render(self, replay_id: str) -> str:
        import tempfile

        from rl.offline.visualise import (
            export_record,
            render_page,
            resolve_replay,
        )

        key = f"{self.model_version}:{replay_id}"
        cached = page_cache.get(key)
        if cached is not None:
            self._bump_recent(replay_id, None)
            return cached

        with tempfile.TemporaryDirectory() as tmpdir:
            replay, replay_json_path = resolve_replay(replay_id, tmpdir)
            runner = self.runners.get(replay.get("formatid"))
            if runner is None:
                raise ValueError(
                    f"this deployment scores {', '.join(sorted(self.runners))} "
                    f"replays (got {replay.get('formatid')!r})"
                )
            if len(replay.get("log", "")) > MAX_LOG_BYTES:
                raise ValueError("replay log too large")
            payload, stats = export_record(replay_json_path, tmpdir)

        html, data = render_page(
            replay, stats, payload, runner, UNCERTAINTY_SCALE
        )
        page_cache[key] = html
        self._bump_recent(replay_id, data)
        return html

    @modal.asgi_app()
    def web(self):
        import urllib.error

        from fastapi import FastAPI
        from fastapi.responses import HTMLResponse

        api = FastAPI()

        @api.get("/", response_class=HTMLResponse)
        def landing():
            import html as html_lib

            cards = ""
            for r in page_cache.get(RECENT_KEY) or []:
                margin = int(r.get("margin", 0))
                chip_color = (
                    "#2a78d6" if margin > 0
                    else "#e34948" if margin < 0
                    else "#898781"
                )
                meta = f'<span class="chip" style="background:{chip_color}">{margin:+d}</span> '
                meta += html_lib.escape(str(r.get("ending", "")))
                if r.get("rating"):
                    meta += f" · {int(r['rating'])}"
                cards += (
                    f'<a class="card" href="/r/{r["id"]}">'
                    f'<div class="vs">{html_lib.escape(" vs ".join(r.get("players", [])))}</div>'
                    f'<div class="meta">{meta}</div></a>'
                )
            section = (
                f'<h2>Recently viewed</h2><div class="grid">{cards}</div>'
                if cards
                else ""
            )
            return HTMLResponse(
                LANDING.replace("__RECENT__", section),
                # No validators are emitted, so without this browsers
                # heuristically cache — and keep showing pre-redeploy pages.
                headers={"Cache-Control": "no-store"},
            )

        @api.get("/r/{replay_id}", response_class=HTMLResponse)
        def replay(replay_id: str):
            if not REPLAY_ID_RE.fullmatch(replay_id):
                return HTMLResponse("invalid replay id", status_code=400)
            try:
                # Server-side page cache is the real cache; the URL's
                # content changes across redeploys, so the browser must
                # always revalidate here.
                return HTMLResponse(
                    self._render(replay_id),
                    headers={"Cache-Control": "no-store"},
                )
            except urllib.error.HTTPError as err:
                return HTMLResponse(
                    f"could not fetch that replay from showdown ({err.code})",
                    status_code=404,
                )
            except (ValueError, SystemExit) as err:
                # SystemExit: exporter refused the replay (e.g. no decided
                # outcome) — a user-input problem, not a server fault.
                return HTMLResponse(str(err), status_code=400)

        return api