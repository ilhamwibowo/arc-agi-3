"""Build the Kaggle submission notebook: notebooks/<name>/<name>.ipynb + kernel-metadata.json.

The notebook is self-contained: our source tree (src/arc3) is embedded as a base64 tar.gz and
unpacked at run time, so no extra Kaggle dataset is needed for our code. Serving uses public
infrastructure only (Pennyroyal SGLang wheelhouse, W4A16 checkpoint, MTP drafter); see docs/kaggle.md.

    .venv/bin/python -m arc3.notebook [--name arc3-submission] [--offline-subset 3]
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import tarfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src" / "arc3"
KAGGLE_USER = "wibowoilham"
COMPETITION = "arc-prize-2026-arc-agi-3"
DATASETS = ["dfranzen/pennyroyal-v253"]
MODELS = ["dfranzen/intel-qwen3.8-flash-next-w4a16-autoround/Transformers/default/1",
          "dfranzen/albucino-qwen3-8-flash-next-drafter/Transformers/default/1"]
# Same image as the public Milestone-2 notebook: Python 3.12, which the cp312 wheels require.
DOCKER_IMAGE = ("gcr.io/kaggle-private-byod/python@sha256:"
                "57e612b484cf3df5026ee4dcc3cb176974b22b2bc0937fb1e16132a8be4cb13c")


def source_tar_b64(src: Path = SRC, arcroot: str = "arc3") -> tuple[str, str]:
    """Deterministic tar.gz of a directory (no caches); returns (base64, sha256 of the bytes)."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz", compresslevel=9) as tar:
        for path in sorted(src.rglob("*")):
            if "__pycache__" in path.parts or path.suffix in (".pyc", ".pyo") or not path.is_file():
                continue
            info = tar.gettarinfo(str(path), arcname=str(Path(arcroot) / path.relative_to(src)))
            info.mtime, info.uid, info.gid, info.uname, info.gname = 0, 0, 0, "", ""
            with path.open("rb") as f:
                tar.addfile(info, f)
    data = buf.getvalue()
    return base64.b64encode(data).decode(), hashlib.sha256(data).hexdigest()


# ---- cell sources (kept as plain text; each must run top to bottom in the Kaggle kernel) ----
CELL_SETUP = '''\
# 0. Mode, budget, configuration
import os, sys, time, threading, traceback
NOTEBOOK_START = time.time()
# Kaggle's 9 h limit counts from CONTAINER start, and in reruns the first cell runs ~40 min after
# that (Tufa, 2nd place, forum 753843, Oct 29: "the notebook itself starts only ~40 minutes after
# the container starts ... these 40 minutes count against submission"). /proc/uptime is the
# container's age, so the wall is anchored to it; the runner still ends games 15 min before it.
def _container_start():
    try:
        with open("/proc/uptime") as f:
            return time.time() - float(f.read().split()[0])
    except Exception:
        return NOTEBOOK_START
CONTAINER_START = _container_start()
HARD_WALL = CONTAINER_START + 9 * 3600 - 300
print(f"container age at notebook start: {NOTEBOOK_START - CONTAINER_START:.0f}s; "
      f"wall in {(HARD_WALL - NOTEBOOK_START) / 60:.0f} min")
TRUE_SUBMISSION = os.environ.get("KAGGLE_IS_COMPETITION_RERUN", "").strip().lower() in {"1", "true"}
OFFLINE_SUBSET = __OFFLINE_SUBSET__        # offline only: None=all, int=first n, or list of id prefixes
PER_GAME_CAP_SECONDS = __PER_GAME_CAP__
AGENT_OVERRIDES = __AGENT_OVERRIDES__      # {game-id prefix: {AgentConfig field: value}} for A/B runs
# Save-run shortcut: skip the model server and play nothing (only unpack, install, write the dummy
# parquet). Never applies in the competition rerun. Used when GPU quota is nearly gone.
SMOKE_NO_SERVER = __SMOKE_NO_SERVER__ and not TRUE_SUBMISSION
# Experiments (never in the rerun): LOCAL_GATEWAY plays through a local copy of the competition
# gateway in competition mode (the exact code path of a rerun, with logs); EXTRA_ENVS swaps the
# game set for one bundled in the notebook (e.g. community games as a hidden-like validation set).
LOCAL_GATEWAY = __LOCAL_GATEWAY__ and not TRUE_SUBMISSION
EXTRA_ENVS = __EXTRA_ENVS__          # None, or the bundled environments dir name
MAX_ACTIVE = 10
PORT = 30000
WORKING = "/kaggle/working"
ARC3_ROOT = WORKING                        # package unpacked to /kaggle/working/arc3
LOG_DIR = f"{WORKING}/runs/kaggle"
SERVE_LOG = f"{WORKING}/serve.log"
COMPETITION_DIR = "/kaggle/input/competitions/arc-prize-2026-arc-agi-3"
os.environ["ONLY_RESET_LEVELS"] = "true"   # RESET keeps the level (must be set before arc_agi import)
os.environ["MPLBACKEND"] = "Agg"
os.environ.setdefault("RECORDINGS_DIR", f"{WORKING}/recordings")
print(f"TRUE_SUBMISSION={TRUE_SUBMISSION} python={sys.version.split()[0]} build=__BUILD_ID__")
if sys.version_info[:2] != (3, 12):
    raise RuntimeError(f"arc3 needs Python 3.12 (cp312 wheels); kernel has {sys.version}. "
                       "Check docker_image in kernel-metadata.json.")
'''

CELL_SOURCE = '''\
# 1. Unpack our source tree (embedded base64 tar.gz of src/arc3)
import base64, hashlib, io, shutil, sys, tarfile
from pathlib import Path
ARC3_TAR_B64 = "__TAR_B64__"
ARC3_TAR_SHA256 = "__TAR_SHA__"
_data = base64.b64decode(ARC3_TAR_B64)
if hashlib.sha256(_data).hexdigest() != ARC3_TAR_SHA256:
    raise RuntimeError("embedded arc3 source archive is corrupt (sha256 mismatch)")
shutil.rmtree(Path(ARC3_ROOT) / "arc3", ignore_errors=True)
with tarfile.open(fileobj=io.BytesIO(_data), mode="r:gz") as _tar:
    _tar.extractall(ARC3_ROOT, filter="data")
if ARC3_ROOT not in sys.path:
    sys.path.insert(0, ARC3_ROOT)
print("arc3 source:", sorted(p.name for p in (Path(ARC3_ROOT) / "arc3").iterdir()))
ENVS_TAR_B64 = "__ENVS_TAR_B64__"
ENV_DIR = f"{COMPETITION_DIR}/environment_files"
if EXTRA_ENVS and ENVS_TAR_B64:
    ENV_DIR = f"{WORKING}/envs/{EXTRA_ENVS}"
    shutil.rmtree(ENV_DIR, ignore_errors=True)
    with tarfile.open(fileobj=io.BytesIO(base64.b64decode(ENVS_TAR_B64)), mode="r:gz") as _tar:
        _tar.extractall(f"{WORKING}/envs", filter="data")
    print("bundled environments:", EXTRA_ENVS, len(list(Path(ENV_DIR).iterdir())), "games")
'''

CELL_PIP = '''\
# 2. Install arc-agi offline from the competition wheelhouse
import subprocess
_wheels = f"{COMPETITION_DIR}/arc_agi_3_wheels"
if not Path(_wheels).is_dir():
    raise RuntimeError(f"competition wheelhouse missing at {_wheels}; is the competition attached?")
_r = subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "--no-index",
                     "--no-warn-conflicts", "--disable-pip-version-check",
                     "--find-links", _wheels, "arc-agi"], capture_output=True, text=True)
if _r.returncode:
    raise RuntimeError("offline pip install of arc-agi failed:\\n" + _r.stdout[-3000:] + _r.stderr[-3000:])
import arc_agi, arcengine, numpy
print("arc_agi", getattr(arc_agi, "__version__", "?"), "numpy", numpy.__version__)
'''

CELL_PREFETCH = '''\
# 3. Warm the page cache for the model shards in the background (wheelhouse: in the server thread)
from arc3 import kaggle_serve as ks
PREFETCH_THREAD = ks.prefetch_in_background(ks.MODEL_DIR, ks.DRAFT_MODEL_DIR, delay=60, threads=3)
'''

CELL_SERVE = '''\
# 4. Launch SGLang in the background (install + start); games start right away (keeper RESETs)
SERVER = {"proc": None, "error": None}
SERVER_DEAD = threading.Event()
STOPPING = threading.Event()      # set by the last cell: no relaunch after the final stop

RELAUNCH_MEMFRAC = [None, {"MEMFRAC": 0.94}, {"MEMFRAC": 0.92}]   # attempt 1 stock, then smaller pools

def _launch():
    # sub-001 (2026-10-28) scored 0.00 after ~35 min: a server that dies at boot must not end the run.
    # Retry the whole install+launch up to 3 times (the keeper holds the scorecard meanwhile); only a
    # third failure aborts. A server that dies after being ready is relaunched the same way.
    attempt = 0
    while True:
        try:
            ks.prefetch(ks.WHEELHOUSE_DIR, threads=16)    # the offline install reads all of it
            SERVER["proc"] = ks.launch_sglang(SERVE_LOG, port=PORT, overrides=RELAUNCH_MEMFRAC[attempt])
            ready_at = None
            while True:          # watchdog: a dead server is relaunched instead of idling for hours
                if STOPPING.is_set():
                    return
                if SERVER["proc"].poll() is not None:
                    raise RuntimeError(f"SGLang exited with code {SERVER['proc'].returncode}")
                if ready_at is None and ks.healthy(PORT):
                    ready_at = time.time()
                    ks.say(f"SGLang READY {ready_at - NOTEBOOK_START:.0f}s after notebook start (attempt {attempt + 1})")
                time.sleep(10)
        except Exception:
            SERVER["error"] = traceback.format_exc()
            ks.say(f"!!! SGLANG FAILED (attempt {attempt + 1}) !!!\\n" + SERVER["error"]
                   + "\\n--- serve.log tail ---\\n" + ks.log_tail(SERVE_LOG))
            ks.stop_server(SERVER.get("proc"))
            attempt += 1
            if attempt >= len(RELAUNCH_MEMFRAC) or time.time() > HARD_WALL - 3600:
                SERVER_DEAD.set()
                return
            SERVER["error"] = None
            time.sleep(30)

SERVER_THREAD = threading.Thread(target=_launch, name="sglang", daemon=True)
if SMOKE_NO_SERVER:
    print("SMOKE_NO_SERVER: model server not started (save run only)")
else:
    SERVER_THREAD.start()
'''

CELL_MONITOR = '''\
# 5. System monitor (RAM / VRAM / disk + scheduler state every 10 minutes)
from arc3.scheduler import PriorityGate
GATE = PriorityGate(max_active=MAX_ACTIVE)
MONITOR = ks.start_monitor(600, extra=lambda: {"healthy": ks.healthy(PORT), **GATE.snapshot()})
'''

CELL_RUN = '''\
# 6. Play: competition gateway (rerun) or bundled environment files (offline)
from arc3 import kaggle_main
RUN_ERROR, RUN_SUMMARY = None, None
GATEWAY = None
try:
    if SMOKE_NO_SERVER:
        raise SystemExit  # caught below: no play in a no-server save run
    if LOCAL_GATEWAY:
        # The toolkit's own competition gateway, in-process, over ENV_DIR: the rerun's code path
        # (HTTP client, scorecard, keeper, retries) with logs we can read.
        GATEWAY = kaggle_main.start_local_gateway(ENV_DIR, f"{WORKING}/gateway-rec", port=8001)
        os.environ["ARC_BASE_URL"] = "http://127.0.0.1:8001/"
        os.environ["ARC_API_KEY"] = "test-key-123"
    RUN_SUMMARY = kaggle_main.main(
        competition=TRUE_SUBMISSION or LOCAL_GATEWAY, run_end_epoch=HARD_WALL, log_dir=LOG_DIR,
        llm_url=f"http://127.0.0.1:{PORT}/v1", model_name=ks.SERVED_MODEL_NAME,
        model_ready=lambda: ks.healthy(PORT), offline_subset=OFFLINE_SUBSET,
        per_game_cap_seconds=PER_GAME_CAP_SECONDS, gate=GATE, abort=SERVER_DEAD,
        agent_overrides=AGENT_OVERRIDES, env_dir=ENV_DIR)
except SystemExit:
    print("SMOKE_NO_SERVER: play skipped", flush=True)
except BaseException:
    RUN_ERROR = traceback.format_exc()
    print("!!! RUN FAILED !!!\\n" + RUN_ERROR, flush=True)
finally:
    if GATEWAY is not None:
        GATEWAY.shutdown()
'''

CELL_SUBMISSION = '''\
# 7. Offline runs still need a submission.parquet; in a rerun the gateway produces the real one
if not TRUE_SUBMISSION:
    kaggle_main.write_dummy_submission(f"{WORKING}/submission.parquet")
'''

CELL_TEARDOWN = '''\
# 8. Always stop the server, then surface failures
STOPPING.set()
ks.stop_server(SERVER.get("proc"))
print(ks.system_line())
if SERVER.get("error") or RUN_ERROR:
    print("--- serve.log tail ---\\n" + ks.log_tail(SERVE_LOG, 20000))
if RUN_SUMMARY is not None:
    print({k: RUN_SUMMARY[k] for k in ("games", "ok", "errors", "hung", "levels_completed", "aborted")})
problem = RUN_ERROR or SERVER.get("error")
if problem and not TRUE_SUBMISSION:
    # In a rerun we never raise here: whatever was played is already on the gateway scorecard.
    raise RuntimeError("arc3 run failed; see the log above")
'''


def _code(src: str) -> dict:
    return {"cell_type": "code", "metadata": {"trusted": True}, "outputs": [],
            "execution_count": None, "source": src}


def _markdown(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src}


def build_notebook(offline_subset=3, per_game_cap_seconds: float = 9 * 3600,
                   agent_overrides: dict | None = None, smoke_no_server: bool = False,
                   local_gateway: bool = False, extra_envs: Path | None = None) -> dict:
    tar_b64, tar_sha = source_tar_b64()
    envs_b64, envs_name = "", None
    if extra_envs is not None:
        envs_name = Path(extra_envs).name
        envs_b64, _ = source_tar_b64(Path(extra_envs), arcroot=envs_name)
    subs = {"__OFFLINE_SUBSET__": repr(offline_subset), "__PER_GAME_CAP__": repr(per_game_cap_seconds),
            "__AGENT_OVERRIDES__": repr(agent_overrides or {}),
            "__SMOKE_NO_SERVER__": repr(bool(smoke_no_server)),
            "__LOCAL_GATEWAY__": repr(bool(local_gateway)),
            "__EXTRA_ENVS__": repr(envs_name), "__ENVS_TAR_B64__": envs_b64,
            "__BUILD_ID__": tar_sha[:12], "__TAR_B64__": tar_b64, "__TAR_SHA__": tar_sha}

    def fill(src: str) -> str:
        for key, value in subs.items():
            src = src.replace(key, value)
        return src

    cells = [_markdown("# arc3: ARC-AGI-3 agent\n\nGenerated by `src/arc3/notebook.py`; "
                       "edit the source and rebuild, never this notebook.")]
    cells += [_code(fill(s)) for s in (CELL_SETUP, CELL_SOURCE, CELL_PIP, CELL_PREFETCH, CELL_SERVE,
                                       CELL_MONITOR, CELL_RUN, CELL_SUBMISSION, CELL_TEARDOWN)]
    return {
        "metadata": {
            "kernelspec": {"language": "python", "display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python", "mimetype": "text/x-python",
                              "file_extension": ".py", "pygments_lexer": "ipython3"},
            "kaggle": {"accelerator": "nvidiaRtxPro6000", "isInternetEnabled": False,
                       "isGpuEnabled": True, "language": "python", "sourceType": "notebook"},
        },
        "nbformat": 4, "nbformat_minor": 4, "cells": cells,
    }


def kernel_metadata(name: str) -> dict:
    return {
        "id": f"{KAGGLE_USER}/{name}", "title": name, "code_file": f"{name}.ipynb",
        "language": "python", "kernel_type": "notebook", "is_private": True,
        "enable_gpu": True, "enable_tpu": False, "enable_internet": False, "keywords": [],
        "dataset_sources": DATASETS, "kernel_sources": [], "competition_sources": [COMPETITION],
        "model_sources": MODELS, "docker_image": DOCKER_IMAGE, "machine_shape": "NvidiaRtxPro6000",
    }


def build(name: str = "arc3-submission", out_root: Path = REPO / "notebooks", **kwargs) -> Path:
    out = Path(out_root) / name
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.ipynb").write_text(json.dumps(build_notebook(**kwargs), indent=1) + "\n")
    (out / "kernel-metadata.json").write_text(json.dumps(kernel_metadata(name), indent=2) + "\n")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--name", default="arc3-submission")
    ap.add_argument("--offline-subset", default="3",
                    help="offline game subset: 'all', a count, or comma-separated id prefixes")
    ap.add_argument("--per-game-cap-hours", type=float, default=9.0)
    ap.add_argument("--agent-overrides", default="{}",
                    help='JSON {"prefix": {"send_images": false}} applied to matching games')
    ap.add_argument("--smoke-no-server", action="store_true",
                    help="save run skips the model server and plays nothing (competition rerun unaffected)")
    ap.add_argument("--local-gateway", action="store_true",
                    help="experiment: play via an in-process competition gateway (rerun code path)")
    ap.add_argument("--extra-envs", default=None,
                    help="experiment: bundle this environments dir and play it instead of the public games")
    args = ap.parse_args()
    subset = args.offline_subset
    subset = None if subset == "all" else int(subset) if subset.isdigit() else subset.split(",")
    out = build(args.name, offline_subset=subset, per_game_cap_seconds=args.per_game_cap_hours * 3600,
                agent_overrides=json.loads(args.agent_overrides), smoke_no_server=args.smoke_no_server,
                local_gateway=args.local_gateway,
                extra_envs=Path(args.extra_envs) if args.extra_envs else None)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
