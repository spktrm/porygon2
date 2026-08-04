"""Serverless Φ replay visualiser on Modal.

Anyone pastes a replay.pokemonshowdown.com URL or id; the app fetches the
replay, encodes it through the real shard exporter (node subprocess), runs
the offline critic ensemble on CPU, and serves the standalone visualisation
page (rl/offline/visualise.py's template — battle player + Φ chart + swing
annotations, plus the per-mon faint-risk heatmap when the baked-in
checkpoints were trained with the survival aux head; CriticRunner detects
that from the params, so no flag is needed here). Pages are cached by
replay id (replays are immutable), so each replay is computed once, ever.

The landing page is a full-page, infinite-scroll feed of the latest public
replays for each served format, proxied live from Showdown's search API
(/latest/{format}.json below). Listing is free — a replay is only fetched,
exported and scored when someone actually opens it.

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
  repeat views skip everything. Entries from older page versions are
  pruned at container start, so the Dict never accumulates dead pages.

The image bakes in: this repo (exporter compiled via tsc at build time),
the offline checkpoints under ckpts/offline/, node, and the python deps.
Redeploy after training a new critic to pick up new checkpoints — the
cache key is a hash of the renderer source + checkpoint bytes, so any
redeploy that could change a page's content invalidates it (names alone
wouldn't: retrains overwrite ckpt_best in place).
"""

import json
import os
import re

import modal

REPO_REMOTE = "/root/porygon2"
UNCERTAINTY_SCALE = 0.0  # gate display in the page; 0 = raw ensemble mean
SCALEDOWN_WINDOW = 300  # seconds a container stays warm after last request
MAX_LOG_BYTES = 2_000_000  # refuse absurd inputs before spending compute

REPLAY_ID_RE = re.compile(r"[a-z0-9]+-[0-9]+(-[a-z0-9]+)?")

# Served formats are whatever the deploying checkout has critics for:
# each ckpts/offline/{format_id}[-ens{k}]/ dir contributes one artifact —
# ckpt_best when present, else the latest periodic save — so any ensemble
# size works, including a single un-suffixed model trained without
# --ensemble. Train a critic for a new format
# (rl/offline/train.py --generation N --smogon-format X) and redeploy.


def _discover_ckpt_dirs() -> list[str]:
    """Repo-relative artifact dirs to bake, mirroring
    rl/offline/visualise.discover_ckpts so the image holds exactly what
    CriticRunner will load: per member dir the lexically-last ckpt_*
    ('ckpt_best' sorts after 'ckpt_{step:08}', so best wins when present),
    and when a format has both -ens{k} member dirs and a plain
    single-model dir, only the ensemble (discover_ckpts ignores the plain
    dir in that case, so baking it would be dead weight). Module-level
    code runs BOTH at deploy time (CWD = the checkout) and inside the
    container (repo baked at REPO_REMOTE, CWD elsewhere) — resolve against
    whichever root exists so both contexts derive the same list."""
    for root in (".", REPO_REMOTE):
        offline_root = os.path.join(root, "ckpts", "offline")
        if not os.path.isdir(offline_root):
            continue
        singles: dict[str, list[str]] = {}
        ensembles: dict[str, list[str]] = {}
        for name in sorted(os.listdir(offline_root)):
            member_dir = os.path.join(offline_root, name)
            if not os.path.isdir(member_dir):
                continue
            steps = sorted(
                d for d in os.listdir(member_dir) if d.startswith("ckpt_")
            )
            if not steps:
                continue
            fmt = name.split("-ens")[0]
            group = ensembles if "-ens" in name else singles
            group.setdefault(fmt, []).append(
                os.path.relpath(os.path.join(member_dir, steps[-1]), root)
            )
        dirs = sorted(
            path
            for fmt in set(singles) | set(ensembles)
            for path in ensembles.get(fmt) or singles.get(fmt, [])
        )
        if dirs:
            return dirs
    return []


CKPT_DIRS = _discover_ckpt_dirs()
if not CKPT_DIRS:
    raise RuntimeError(
        "no ckpts/offline/*/ckpt_* artifacts found — train an offline "
        "critic before deploying (rl/offline/train.py)"
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
        ignore=["**/__pycache__", "**/.DS_Store", "**/*.log"],
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

LANDING = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Game review for Pokémon Showdown</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
 body { font-family: system-ui, sans-serif; margin: 0; }
 header { max-width: 720px; margin: 0 auto; padding: 48px 20px 0; }
 main { max-width: 1400px; margin: 0 auto; padding: 8px 20px 48px; }
 form { display: flex; gap: 8px; }
 input { flex: 1; font-size: 16px; padding: 10px 12px;
         box-sizing: border-box; }
 button { font-size: 16px; padding: 10px 18px; cursor: pointer; }
 p { color: #555; }
 h2 { font-size: 13px; color: #888; margin: 32px 0 10px;
      text-transform: uppercase; letter-spacing: 0.05em; }
 .tabs { display: flex; gap: 6px; flex-wrap: wrap; margin: 0 0 12px; }
 .tab { font-size: 13px; padding: 5px 14px; cursor: pointer;
        border: 1px solid #ccc; border-radius: 15px; background: #fff; }
 .tab.active { background: #222; border-color: #222; color: #fff; }
 .grid { display: grid; gap: 10px;
         grid-template-columns: repeat(auto-fill, minmax(210px, 1fr)); }
 .card { border: 1px solid #e3e3e0; border-radius: 8px; padding: 10px 12px;
         text-decoration: none; color: inherit; display: block; }
 .card:hover { border-color: #999; }
 .card .vs { font-weight: 600; font-size: 13px; margin-bottom: 4px;
             overflow: hidden; text-overflow: ellipsis;
             white-space: nowrap; }
 .card .meta { font-size: 12px; color: #777; }
 #status { text-align: center; color: #888; font-size: 14px;
           padding: 24px 0; }
</style></head><body>
<header>
<h1>Game review for Pokémon Showdown</h1>
<p>Paste a <a href="https://replay.pokemonshowdown.com">replay</a> link —
or pick any recent battle below — and watch it next to a turn-by-turn
review: win chances as the game swings, the key moments that decided it,
the predicted final score, and which Pokémon were in danger. Works with
__FORMATS__ replays.</p>
<form id="f">
<input id="q" placeholder="https://replay.pokemonshowdown.com/__EXAMPLE__-2654504071"
 autofocus />
<button type="submit">Go</button>
</form>
<p id="err" style="color:#b33"></p>
</header>
<main>
<h2>Latest replays</h2>
<div class="tabs" id="tabs"></div>
<div class="grid" id="grid"></div>
<div id="status">Loading&hellip;</div>
</main>
<script>
var FORMATS = __FORMATS_JSON__;

document.getElementById("f").addEventListener("submit", function (e) {
  e.preventDefault();
  var v = document.getElementById("q").value.trim()
    .replace(/^https?:\\/\\/replay\\.pokemonshowdown\\.com\\//, "")
    .replace(/\\.(json|log|html)$/, "");
  if (/^[a-z0-9]+-[0-9]+(-[a-z0-9]+)?$/.test(v)) location.href = "/r/" + v;
  else document.getElementById("err").textContent = "That doesn't look like a replay id.";
});

// Infinite-scroll feed of the latest replays, one Showdown search page
// (51 rows) at a time via /latest/{format}.json. Nothing is scored here:
// each card is a plain link, and the model only runs when it's opened.
var grid = document.getElementById("grid");
var statusEl = document.getElementById("status");
var tabsEl = document.getElementById("tabs");
// Replaced wholesale on tab switch, so in-flight responses for the old
// format recognise themselves as stale and drop.
var state = {format: FORMATS[0], before: null, busy: false, done: false,
             seen: {}};

function timeAgo(ts) {
  var s = Math.max(0, Date.now() / 1000 - ts);
  if (s < 90) return "just now";
  if (s < 5400) return Math.round(s / 60) + "m ago";
  if (s < 129600) return Math.round(s / 3600) + "h ago";
  return Math.round(s / 86400) + "d ago";
}

function addCard(r) {
  var a = document.createElement("a");
  a.className = "card";
  a.href = "/r/" + encodeURIComponent(r.id);
  var vs = document.createElement("div");
  vs.className = "vs";
  vs.textContent = (r.players || []).join(" vs ");
  var meta = document.createElement("div");
  meta.className = "meta";
  meta.textContent = (r.rating ? Math.round(r.rating) + " \\u00b7 " : "") +
    timeAgo(r.uploadtime);
  a.appendChild(vs);
  a.appendChild(meta);
  grid.appendChild(a);
}

function loadMore() {
  if (state.busy || state.done) return;
  state.busy = true;
  statusEl.textContent = "Loading\\u2026";
  var requested = state;
  var url = "/latest/" + state.format + ".json" +
    (state.before ? "?before=" + state.before : "");
  fetch(url).then(function (resp) {
    if (!resp.ok) throw new Error("HTTP " + resp.status);
    return resp.json();
  }).then(function (rows) {
    if (requested !== state) return;
    state.busy = false;
    if (!rows.length) {
      state.done = true;
      statusEl.textContent = "No more replays.";
      return;
    }
    state.before = rows[rows.length - 1].uploadtime;
    rows.forEach(function (r) {
      if (state.seen[r.id]) return;
      state.seen[r.id] = true;
      addCard(r);
    });
    statusEl.textContent = "";
    // A short page can leave the sentinel visible without a new
    // intersection event — keep filling until the viewport overflows.
    if (statusEl.getBoundingClientRect().top < innerHeight + 1200) loadMore();
  }).catch(function () {
    if (requested !== state) return;
    state.busy = false;
    statusEl.textContent =
      "Couldn't reach Showdown \\u2014 scroll to retry.";
  });
}

if (FORMATS.length > 1) FORMATS.forEach(function (fmt) {
  var b = document.createElement("button");
  b.type = "button";
  b.className = "tab" + (fmt === state.format ? " active" : "");
  b.textContent = fmt;
  b.addEventListener("click", function () {
    if (state.format === fmt) return;
    state = {format: fmt, before: null, busy: false, done: false, seen: {}};
    grid.textContent = "";
    Array.prototype.forEach.call(tabsEl.children, function (t) {
      t.classList.toggle("active", t === b);
    });
    loadMore();
  });
  tabsEl.appendChild(b);
});

new IntersectionObserver(function (entries) {
  if (entries.some(function (e) { return e.isIntersecting; })) loadMore();
}, {rootMargin: "1200px"}).observe(statusEl);
</script></body></html>"""
LANDING = (
    LANDING.replace("__FORMATS_JSON__", json.dumps(SUPPORTED_FORMATS))
    .replace("__FORMATS__", ", ".join(SUPPORTED_FORMATS))
    .replace("__EXAMPLE__", SUPPORTED_FORMATS[0])
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

        # Only keys for THIS page version are ever served again — anything
        # else (older versions, the retired recently-viewed list) is dead
        # weight in the Dict; drop it. Cosmetic and racy on purpose:
        # during a rolling deploy an outgoing container may re-add its own
        # entries, which the next new-version container start prunes.
        try:
            prefix = f"{self.model_version}:"
            stale = [k for k in page_cache.keys() if not k.startswith(prefix)]
            for key in stale:
                page_cache.pop(key)
            if stale:
                print(f"pruned {len(stale)} stale cached page(s)")
        except Exception as err:  # noqa: BLE001 — cache hygiene only
            print(f"cache prune failed: {err}")

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

        html, _ = render_page(
            replay, stats, payload, runner, UNCERTAINTY_SCALE
        )
        page_cache[key] = html
        return html

    @modal.asgi_app()
    def web(self):
        import urllib.error
        import urllib.request

        from fastapi import FastAPI
        from fastapi.responses import HTMLResponse, JSONResponse

        from rl.offline.visualise import USER_AGENT

        api = FastAPI()

        @api.get("/", response_class=HTMLResponse)
        def landing():
            return HTMLResponse(
                LANDING,
                # No validators are emitted, so without this browsers
                # heuristically cache — and keep showing pre-redeploy pages.
                headers={"Cache-Control": "no-store"},
            )

        @api.get("/latest/{format_id}.json")
        def latest(format_id: str, before: int | None = None):
            """One page of showdown's replay search (newest first, `before`
            pages back by uploadtime) for the landing feed. Proxied because
            the browser can't call showdown cross-origin, and trimmed to
            the fields the page renders. Nothing here is scored — the
            model only runs when a listed replay is opened."""
            if format_id not in SUPPORTED_FORMATS:
                return JSONResponse(
                    {"error": "unsupported format"}, status_code=404
                )
            url = (
                "https://replay.pokemonshowdown.com/search.json"
                f"?format={format_id}"
            )
            if before is not None:
                url += f"&before={before}"
            request = urllib.request.Request(
                url, headers={"User-Agent": USER_AGENT}
            )
            try:
                with urllib.request.urlopen(request, timeout=10) as response:
                    rows = json.load(response)
            except (urllib.error.URLError, ValueError) as err:
                print(f"showdown search failed: {err}")
                return JSONResponse(
                    {"error": "showdown search unavailable"}, status_code=502
                )
            return JSONResponse(
                [
                    dict(
                        id=r["id"],
                        players=r.get("players") or [],
                        rating=r.get("rating"),
                        uploadtime=r.get("uploadtime"),
                    )
                    for r in rows
                    # Private replays 404 without their password suffix, so
                    # a card would dead-end; drop them (and anything whose
                    # id /r/ would reject anyway).
                    if not r.get("private")
                    and REPLAY_ID_RE.fullmatch(r.get("id") or "")
                ],
                # Fresh enough for a feed; spares showdown identical
                # queries when several people land at once.
                headers={"Cache-Control": "public, max-age=30"},
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