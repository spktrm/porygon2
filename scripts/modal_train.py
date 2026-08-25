"""GPU training on Modal: offline critic + main RL agent.

Two entrypoints on one app (shared image and volumes):

- train_offline — rl/offline/train.py on replay shards. Shards live on the
  `porygon2-replays` volume (upload them first, below); artifacts land in
  ckpts/offline/ on the `porygon2-ckpts` volume in the exact layout the RL
  learner consumes (offline_critic_ckpt_path).
- train_rl — rl/online/main.py plus the node game service in the same container
  (compiled at image-build time, launched on ws://localhost:8080 before the
  learner starts). Checkpoints go to ckpts/gen{N}/ on the same volume, so a
  relaunch resumes from the newest checkpoint automatically — relevant
  because Modal caps a single function call at 24 h.

One-time setup:
    pip install modal && modal setup
    modal secret create wandb-secret WANDB_API_KEY=<your key>

Refresh replays Modal-side (no local upload — scrape + export shards on
the box): `modal run --detach scripts/modal_train.py::refresh_replays`.
Or upload existing local shards (replays/shards/{format_id}/ -> volume):
    modal run scripts/modal_train.py::upload_replays

Train the offline critic (args after --cli go verbatim to rl.offline.train):
    modal run --detach scripts/modal_train.py::train_offline \
        --cli "--ensemble --num-steps 10000"

Train the RL agent (self-contained — the RL learner has consumed nothing
from the offline critic since PBRS shaping was retired, Aug 2026):
    modal run --detach scripts/modal_train.py::train_rl

Fetch artifacts back:
    modal volume get porygon2-ckpts offline/ ckpts/offline/
    modal volume get porygon2-ckpts gen9/ ckpts/gen9/

GPU type is chosen at deploy/run time: PORYGON2_GPU=H100 modal run ...
(default A100). --detach keeps a run alive after you close the terminal;
watch it with `modal app logs porygon2-train`.

The image bakes in: this repo's python source (rl/, constants/), the full
data/data/ tree (data.json + per-generation pretrained-embedding .npy
tables), and the compiled node service. rl/environment/ex.bin (required at
import time by rl/environment/data.py) is regenerated at image-build time
via the service's ex.ts, so a fresh checkout deploys cleanly. The
pretrained embeddings CANNOT be regenerated in-image — deploy from a
checkout where data/data/gen{N}/*.npy exist (run `make datas` /
the embeddings pipeline first); both functions fail fast otherwise, since
rl/environment/data.py silently zero-fills missing tables and the model
would train on garbage.
"""

import glob
import os
import shlex
import socket
import subprocess
import sys
import threading
import time

import modal

REPO_REMOTE = "/root/porygon2"
GPU_TYPE = os.environ.get("PORYGON2_GPU", "A100")
# Modal's hard ceiling for one function call; long RL runs relaunch and
# resume from the newest checkpoint on the volume.
TRAIN_TIMEOUT = 24 * 60 * 60
# Background-commit the ckpts volume so a crash/timeout loses at most this
# much progress beyond the last saved checkpoint.
COMMIT_INTERVAL = 10 * 60

EMBEDDING_NPYS = ("species", "abilities", "items", "moves", "learnset")

# Versions pinned to requirements.txt — protobuf in particular must match
# the generated rl/environment/protos/*_pb2.py gencode.
PY_DEPS = [
    "jax[cuda13]==0.9.1",
    "flax==0.12.4",
    "optax==0.2.6",
    "chex==0.1.91",
    "ml_collections==1.1.0",
    "jaxtyping==0.3.9",
    "numpy==2.3.5",
    "protobuf==6.33.5",
    "wandb==0.25.0",
    "python-dotenv==1.2.2",
    "websockets==16.0",
    "cloudpickle==3.1.2",
    "tqdm==4.67.3",
    "plotly",
    # replay scraper (refresh_replays)
    "aiohttp",
]

image = (
    modal.Image.debian_slim(python_version="3.11")
    # Debian's apt nodejs is ancient; the service targets modern JS.
    .apt_install("curl", "ca-certificates")
    .run_commands(
        "curl -fsSL https://deb.nodesource.com/setup_22.x | bash -"
        " && apt-get install -y nodejs",
    )
    .pip_install(*PY_DEPS)
    .add_local_dir(
        "rl",
        remote_path=f"{REPO_REMOTE}/rl",
        copy=True,
        ignore=["**/__pycache__", "**/.DS_Store", "**/*.log"],
    )
    # Whole data tree (unlike the serve app's minimal picks): training
    # touches the embedding .npy tables AND the service's runtime data
    # (randombattle sets etc.), and shaving ~100 MB off an image that
    # builds once isn't worth the fragility.
    .add_local_dir(
        "data/data",
        remote_path=f"{REPO_REMOTE}/data/data",
        copy=True,
        ignore=["**/.DS_Store"],
    )
    .add_local_dir("constants", remote_path=f"{REPO_REMOTE}/constants", copy=True)
    # The scraper script is baked at the repo root (not under replays/,
    # which the porygon2-replays volume mount shadows at runtime). Its
    # ROOT_DIR is CWD-relative ("replays/data/"), so run from REPO_REMOTE
    # and it writes straight onto the volume.
    .add_local_file("replays/main.py", f"{REPO_REMOTE}/replays_main.py", copy=True)
    .add_local_dir(
        "service",
        remote_path=f"{REPO_REMOTE}/service",
        copy=True,
        ignore=["**/node_modules", "**/dist", "**/.DS_Store"],
    )
    # Compile the service once at build time, then regenerate
    # rl/environment/ex.bin (ex.ts writes it into ../../../rl/environment
    # relative to service/src/tests) — data.py opens it at import, and it's
    # gitignored so a fresh checkout won't have it.
    .run_commands(
        f"cd {REPO_REMOTE}/service && npm ci && npx tsc"
        " && npx ts-node src/tests/ex.ts",
    )
    .env(
        {
            "PYTHONPATH": REPO_REMOTE,
            # Shared XLA compile cache (volume below): each shape bucket
            # compiles once across containers and relaunches.
            "JAX_COMPILATION_CACHE_DIR": "/jax_cache",
        }
    )
)

app = modal.App("porygon2-train", image=image)

ckpts_volume = modal.Volume.from_name("porygon2-ckpts", create_if_missing=True)
replays_volume = modal.Volume.from_name("porygon2-replays", create_if_missing=True)
jax_cache = modal.Volume.from_name("porygon2-jax-cache", create_if_missing=True)

VOLUMES = {
    f"{REPO_REMOTE}/ckpts": ckpts_volume,
    f"{REPO_REMOTE}/replays": replays_volume,
    "/jax_cache": jax_cache,
}
wandb_secret = modal.Secret.from_name("wandb-secret")


def _require_embeddings(generation: int) -> None:
    """data.py zero-fills missing pretrained-embedding tables without
    crashing; refuse to train on that."""
    missing = [
        name
        for name in EMBEDDING_NPYS
        if not os.path.exists(f"{REPO_REMOTE}/data/data/gen{generation}/{name}.npy")
    ]
    if missing:
        raise RuntimeError(
            f"pretrained embeddings missing from the image for "
            f"gen{generation}: {missing} — deploy from a checkout with "
            f"data/data/gen{generation}/*.npy present (make datas / "
            "embeddings pipeline)"
        )


def _start_commit_loop() -> None:
    """Daemon thread persisting ckpts-volume writes periodically. Modal
    also commits on clean function exit; this bounds losses on crash or
    the 24 h timeout."""

    def loop():
        while True:
            time.sleep(COMMIT_INTERVAL)
            try:
                ckpts_volume.commit()
            except Exception as err:  # noqa: BLE001 — never kill training
                print(f"ckpts volume commit failed: {err}")

    threading.Thread(target=loop, daemon=True, name="ckpt-commit").start()


@app.function(
    cpu=8.0,
    timeout=4 * 60 * 60,
    volumes=VOLUMES,
)
def refresh_replays(
    format_id: str = "gen9randombattle",
    scrape_cli: str = "--min-rating 1500 --limit 20000",
    export_cli: str = "",
):
    """Scrapes fresh replays and exports shards entirely on Modal — no
    local upload. Raw JSONs land in replays/data/{format_id} and shards in
    replays/shards/{format_id} on the porygon2-replays volume, which
    train_offline reads directly. The scraper skips already-downloaded
    ids, so rerunning tops the corpus up with new games; the exporter
    rewrites the shard set from the full raw corpus each time.

        modal run --detach scripts/modal_train.py::refresh_replays
        modal run --detach scripts/modal_train.py::refresh_replays \
            --scrape-cli "--min-rating 1000 --limit 50000"
    """
    os.chdir(REPO_REMOTE)
    subprocess.run(
        [sys.executable, "replays_main.py", format_id, *shlex.split(scrape_cli)],
        check=True,
    )
    # Persist raw logs before the export pass so a crash there costs
    # nothing already downloaded.
    replays_volume.commit()
    subprocess.run(
        ["node", "dist/scripts/offline.js", format_id, *shlex.split(export_cli)],
        cwd=os.path.join(REPO_REMOTE, "service"),
        check=True,
    )
    replays_volume.commit()


@app.function(
    gpu=GPU_TYPE,
    cpu=8.0,
    memory=32768,
    timeout=TRAIN_TIMEOUT,
    volumes=VOLUMES,
    secrets=[wandb_secret],
)
def train_offline(cli: str = ""):
    """Runs `python -m rl.offline.train <cli>` on the GPU.

    Shards are read from the replays volume ({dataset_dir}/{format_id}/,
    default replays/shards) and artifacts written to the ckpts volume
    (ckpts/offline/...). Pass anything rl.offline.train accepts, e.g.
    --cli "--ensemble --num-steps 10000" or
    --cli "--ensemble-index 2 --resume-from ..."."""
    os.chdir(REPO_REMOTE)
    args = shlex.split(cli)

    generation = 9  # rl/offline/config.py default
    if "--generation" in args:
        generation = int(args[args.index("--generation") + 1])
    _require_embeddings(generation)

    smogon_format = "randombattle"
    if "--smogon-format" in args:
        smogon_format = args[args.index("--smogon-format") + 1]
    dataset_dir = "replays/shards"
    if "--dataset-dir" in args:
        dataset_dir = args[args.index("--dataset-dir") + 1]
    shard_dir = os.path.join(dataset_dir, f"gen{generation}{smogon_format}")
    if not glob.glob(os.path.join(shard_dir, "*")):
        raise RuntimeError(
            f"no replay shards under {shard_dir} on the porygon2-replays "
            "volume — run `modal run scripts/modal_train.py::upload_replays` "
            "first"
        )

    _start_commit_loop()
    subprocess.run(
        [sys.executable, "-m", "rl.offline.train", *args],
        cwd=REPO_REMOTE,
        check=True,
    )
    ckpts_volume.commit()


@app.function(
    gpu=GPU_TYPE,
    cpu=16.0,
    memory=65536,
    timeout=TRAIN_TIMEOUT,
    volumes=VOLUMES,
    secrets=[wandb_secret],
)
def train_rl(debug: bool = False, load_state_mode: str = "checkpoint"):
    """Starts the node game service, waits for ws://localhost:8080, then
    runs rl/main.py. Resumes from the newest ckpts/gen{N}/ checkpoint on
    the volume (load_state_mode="params" merges params only — use for the
    first launch after an architecture change)."""
    os.chdir(REPO_REMOTE)

    # The learner config decides generation/format — read it in a
    # CPU-pinned side process so the parent never touches the GPU the
    # training subprocess needs whole.
    probe_env = dict(os.environ, JAX_PLATFORMS="cpu")
    generation = int(
        subprocess.check_output(
            [
                sys.executable,
                "-c",
                "from rl.learner.config import get_learner_config;"
                "print(get_learner_config().generation)",
            ],
            cwd=REPO_REMOTE,
            env=probe_env,
        )
        .strip()
        .decode()
    )
    _require_embeddings(generation)

    service = subprocess.Popen(
        ["node", "dist/server/index.js"],
        cwd=f"{REPO_REMOTE}/service",
    )
    try:
        deadline = time.time() + 180
        while True:
            if service.poll() is not None:
                raise RuntimeError(
                    f"game service exited early (code {service.returncode})"
                )
            try:
                socket.create_connection(("127.0.0.1", 8080), timeout=2).close()
                break
            except OSError:
                if time.time() > deadline:
                    raise RuntimeError("game service never opened port 8080") from None
                time.sleep(1)
        print("game service up on :8080")

        _start_commit_loop()
        train_env = dict(os.environ, LOAD_STATE_MODE=load_state_mode)
        cmd = [sys.executable, "-m", "rl.online.main"] + (["--debug"] if debug else [])
        subprocess.run(cmd, cwd=REPO_REMOTE, env=train_env, check=True)
    finally:
        service.terminate()
        try:
            service.wait(timeout=10)
        except subprocess.TimeoutExpired:
            service.kill()
        ckpts_volume.commit()


@app.local_entrypoint()
def upload_replays(local_dir: str = "replays/shards"):
    """Pushes local replay shards to the porygon2-replays volume, preserving
    the {format_id}/ layout the offline dataset expects."""
    if not os.path.isdir(local_dir):
        raise SystemExit(f"{local_dir} is not a directory")
    with replays_volume.batch_upload(force=True) as batch:
        batch.put_directory(local_dir, "/shards")
    print(f"uploaded {local_dir} -> porygon2-replays:/shards")
