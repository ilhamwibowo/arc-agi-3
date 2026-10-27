"""Kaggle serving helpers: checkpoint prefetch, SGLang (Pennyroyal) launch, health, system monitor.

Stdlib only: this runs in the notebook kernel before anything else is installed. The launch
procedure (offline uv venv from the Pennyroyal wheelhouse, bundled CUDA 13 toolkit, draft-model
config view, server flags) follows the public Milestone-2 notebook's serving cell; see docs/kaggle.md.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import shlex
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

WHEELHOUSE_DIR = "/kaggle/input/datasets/dfranzen/pennyroyal-v253"
MODEL_DIR = "/kaggle/input/models/dfranzen/intel-qwen3.8-flash-next-w4a16-autoround/transformers/default/1"
DRAFT_MODEL_DIR = "/kaggle/input/models/dfranzen/albucino-qwen3-8-flash-next-drafter/transformers/default/1"
SERVED_MODEL_NAME = "flashnext"
PORT = 30000
PREFIX = "/tmp/sgl-pennyroyal"

CFG = dict(CTX=139264, MEMFRAC=0.96, MAXREQ=12, MAMBA_CACHE=60, CHUNK=8192, MAX_PREFILL=16384,
           KVDTYPE="fp8_e4m3", SSM_DTYPE="bfloat16", BACKEND="flashinfer", SPEC_STEPS=3,
           GRAPH_BS=(1, 2, 4, 7, 8, 9, 10))
# Integrity pins of the FR-Spec hot-token map and the tokenizer it was built for.
TOKEN_MAP_SHA = "becfa41d394b86c26c632bea8f3c6ea64bbb76d7b238d8673c06afae21269f25"
TOKENIZER_SHA = "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523"
DRAFT_EXPERTS = r"re:^mtp\.layers\.0\.mlp\.experts\.[0-9]+\.(gate_proj|up_proj|down_proj)$"
DRAFT_IGNORE = r"re:^(?!mtp\.layers\.0\.mlp\.experts(?:\.|$)).*"


def say(msg: str) -> None:
    print(f"[arc3 {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run(cmd, env=None) -> subprocess.CompletedProcess:
    cmd = [str(c) for c in cmd]
    say("$ " + shlex.join(cmd))
    res = subprocess.run(cmd, text=True, capture_output=True, env=env)
    if res.returncode:
        raise RuntimeError(f"command failed ({res.returncode}): {shlex.join(cmd)}\n"
                           f"{res.stdout[-4000:]}\n{res.stderr[-4000:]}")
    return res


# ---- prefetch: read files once so the page cache is warm when SGLang loads them ----
def prefetch(*roots: str, threads: int = 8, chunk: int = 32 << 20) -> None:
    jobs = []
    for root in roots:
        for dirpath, _, files in os.walk(root):
            for name in files:
                path = os.path.join(dirpath, name)
                size = os.path.getsize(path)
                jobs += [(path, off) for off in range(0, max(size, 1), chunk)]
    t0 = time.time()

    def read(job):
        path, off = job
        try:
            with open(path, "rb") as f:
                f.seek(off)
                f.read(chunk)
        except OSError:
            pass                        # prefetch is an optimisation only
    with ThreadPoolExecutor(threads) as pool:
        list(pool.map(read, sorted(jobs)))
    say(f"prefetch {roots} done: {len(jobs)} chunks in {time.time() - t0:.0f}s")


def prefetch_in_background(*roots: str, delay: float = 0.0, threads: int = 3) -> threading.Thread:
    def go():
        time.sleep(delay)
        prefetch(*roots, threads=threads)
    t = threading.Thread(target=go, name="prefetch", daemon=True)
    t.start()
    return t


# ---- install ----
def _find_one(root: str, name: str, directory: bool = False) -> Path:
    root_p = Path(root)
    direct = root_p / name
    if (direct.is_dir() if directory else direct.is_file()):
        return direct
    hits = sorted(p for p in root_p.rglob(name) if (p.is_dir() if directory else p.is_file()))
    if len(hits) != 1:
        raise RuntimeError(f"expected exactly one {name} under {root}, found {hits}")
    return hits[0]


def install_sglang(wheelhouse: str = WHEELHOUSE_DIR, prefix: str = PREFIX) -> Path:
    """Offline install of the Pennyroyal SGLang fork into its own venv; returns the venv dir."""
    venv = Path(prefix) / "venv"
    marker = Path(prefix) / "installed.txt"
    if marker.is_file() and (venv / "bin/sglang").exists():
        return venv
    wheels = _find_one(wheelhouse, "wheels", directory=True)
    lock = _find_one(wheelhouse, "requirements.lock")
    sgl = sorted(wheels.glob("sglang-*.whl"))
    uvw = sorted(wheels.glob("uv-*.whl"))
    if len(sgl) != 1 or len(uvw) != 1:
        raise RuntimeError(f"wheelhouse {wheels} must hold one sglang and one uv wheel: {sgl} {uvw}")
    uv = Path(prefix) / "bin/uv"
    uv.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(uvw[0]) as z:
        member = [n for n in z.namelist() if Path(n).name == "uv" and not n.endswith("/")]
        if len(member) != 1:
            raise RuntimeError("cannot find the uv binary inside its wheel")
        uv.write_bytes(z.read(member[0]))
    uv.chmod(0o755)
    env = dict(os.environ, UV_OFFLINE="1", UV_PYTHON_DOWNLOADS="never", PYTHONNOUSERSITE="1")
    env.pop("PYTHONPATH", None)
    if not (venv / "bin/python").exists():
        run([uv, "venv", "--python", sys.executable, venv], env=env)
    pip = [uv, "pip", "install", "--python", venv / "bin/python", "--no-index", "--find-links", wheels]
    run(pip + ["-r", lock], env=env)
    run(pip + ["--reinstall", "--no-deps", sgl[0]], env=env)
    marker.write_text("ok")
    return venv


def _cuda_env(venv: Path, prefix: str) -> dict:
    """Environment for the server: bundled CUDA 13 toolkit, linker symlinks, offline caches."""
    cu = sorted(venv.glob("lib/python*/site-packages/nvidia/cu13"))
    if not cu or not (cu[0] / "bin/nvcc").is_file():
        raise RuntimeError("bundled CUDA 13 nvcc missing from the Pennyroyal venv")
    cuda = cu[0]
    lib = cuda / "lib"
    if not (cuda / "lib64").exists():
        (cuda / "lib64").symlink_to("lib")
    for so in lib.glob("*.so.*"):
        link = lib / re.sub(r"\.so\..*$", ".so", so.name)
        if not link.exists() and not link.is_symlink():
            link.symlink_to(so.name)
    if not (lib / "libcuda.so").exists():
        drivers = [p for d in ("/usr/local/nvidia/lib64", "/usr/local/nvidia/lib",
                               "/usr/lib/x86_64-linux-gnu", "/usr/lib64")
                   for p in sorted(Path(d).glob("libcuda.so*")) if p.is_file()]
        if not drivers:
            raise RuntimeError("GPU driver libcuda.so not found (is a GPU attached?)")
        (lib / "libcuda.so").symlink_to(drivers[0].resolve())
    cxx = next((c for c in ("g++-15", "g++-14", "g++-13", "g++-12", "g++-11", "g++") if shutil.which(c)), None)
    if not cxx or not shutil.which(cxx.replace("g++", "gcc")):
        raise RuntimeError("host C/C++ compiler not found")
    cache = f"{prefix}/cache"
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.update({
        "PYTHONNOUSERSITE": "1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "CUDA_HOME": str(cuda), "CUDACXX": f"{cuda}/bin/nvcc",
        "PATH": f"{cuda}/bin:{venv}/bin:" + env.get("PATH", ""),
        "LD_LIBRARY_PATH": f"{cuda}/lib64:{cuda}/lib:" + env.get("LD_LIBRARY_PATH", ""),
        "CC": cxx.replace("g++", "gcc"), "CXX": cxx, "CUDAHOSTCXX": cxx, "TORCH_CUDA_ARCH_LIST": "12.0",
        "HF_HOME": f"{cache}/hf", "XDG_CACHE_HOME": cache, "TRITON_CACHE_DIR": f"{cache}/triton",
        "TORCHINDUCTOR_CACHE_DIR": f"{cache}/inductor", "FLASHINFER_WORKSPACE_BASE": f"{cache}/flashinfer",
        "SGLANG_CACHE_DIR": f"{cache}/sglang", "SGLANG_JIT_CACHE_DIR": f"{cache}/sglang/jit",
        "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN": "1", "OMP_NUM_THREADS": "8",
        "TOKENIZERS_PARALLELISM": "false", "MAX_JOBS": "8",
        "SGLANG_ENABLE_SM120_LOWM_BF16_GEMM": "1", "SGLANG_MAMBA_CONV_DTYPE": "bfloat16",
        "SGLANG_MM_PREPROCESS_DEVICE": "cpu",
    })
    return env


def _draft_view(target: Path, draft_root: Path, prefix: str) -> Path:
    """Symlinked copy of the MTP draft checkpoint with quantization target rules SGLang accepts."""
    found = []
    for cfg_path in sorted(draft_root.rglob("config.json")):
        cfg = json.loads(cfg_path.read_text())
        q = cfg.get("quantization_config", {})
        if q.get("quant_method") == "compressed-tensors" and "mtp_routed_experts" in q.get("config_groups", {}):
            found.append((cfg_path.parent, cfg))
    if len(found) != 1:
        raise RuntimeError(f"expected one INT4 MTP draft checkpoint under {draft_root}, found {len(found)}")
    src, cfg = found[0]
    cfg = copy.deepcopy(cfg)
    q = cfg["quantization_config"]
    q["config_groups"]["mtp_routed_experts"]["targets"] = [DRAFT_EXPERTS]
    q["ignore"] = [DRAFT_IGNORE]
    view = Path(prefix) / "draft-view"
    shutil.rmtree(view, ignore_errors=True)
    view.mkdir(parents=True)
    (view / "config.json").write_text(json.dumps(cfg, indent=2))
    index = json.loads((src / "model.safetensors.index.json").read_text())
    names = set(index["weight_map"].values()) | {"model.safetensors.index.json"}
    for name in names:
        (view / name).symlink_to((src / name).resolve())
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
                 "preprocessor_config.json", "generation_config.json"):
        if (target / name).is_file():
            (view / name).symlink_to((target / name).resolve())
    return view


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def server_args(venv: Path, model_dir: str, draft_view: Path, token_map: Path,
                port: int = PORT) -> list[str]:
    c = CFG
    args = [str(venv / "bin/sglang"), "serve", "--model-path", model_dir, "--load-format", "safetensors",
            "--model-loader-extra-config", '{"enable_multithread_load":false}',
            "--served-model-name", SERVED_MODEL_NAME, "--host", "127.0.0.1", "--port", str(port),
            "--tensor-parallel-size", "1", "--dtype", "bfloat16", "--quantization", "auto-round",
            "--kv-cache-dtype", c["KVDTYPE"], "--mem-fraction-static", str(c["MEMFRAC"]),
            "--context-length", str(c["CTX"]), "--page-size", "64",
            "--max-running-requests", str(c["MAXREQ"]), "--chunked-prefill-size", str(c["CHUNK"]),
            "--max-prefill-tokens", str(c["MAX_PREFILL"]),
            "--cuda-graph-max-bs-decode", str(max(c["GRAPH_BS"])),
            "--cuda-graph-bs-decode", *map(str, c["GRAPH_BS"]),
            "--mamba-ssm-dtype", c["SSM_DTYPE"], "--max-mamba-cache-size", str(c["MAMBA_CACHE"]),
            "--mamba-radix-cache-strategy", "extra_buffer", "--mamba-track-interval", "64",
            "--mamba-backend", c["BACKEND"], "--linear-attn-decode-backend", c["BACKEND"],
            "--linear-attn-prefill-backend", c["BACKEND"],
            "--moe-runner-backend", "auto", "--ple-offload-embedding", "--trust-remote-code",
            "--mm-feature-transport", "cpu", "--image-processor-backend", "pil",
            "--reasoning-parser", "qwen3", "--tool-call-parser", "qwen3_coder",
            "--default-chat-template-kwargs", '{"preserve_thinking":true}',
            "--watchdog-timeout", "1800", "--schedule-policy", "lpm", "--warmups", "structured_output",
            "--enable-cache-report", "--enable-metrics",
            "--weight-loader-prefetch-checkpoints", "--weight-loader-drop-cache-after-load",
            "--gdn-mtp-cache-mode", "none",
            "--speculative-algorithm", "NEXTN", "--speculative-num-steps", str(c["SPEC_STEPS"]),
            "--speculative-eagle-topk", "1", "--speculative-num-draft-tokens", str(c["SPEC_STEPS"] + 1),
            "--speculative-draft-model-path", str(draft_view),
            "--speculative-draft-model-quantization", "compressed-tensors",
            "--speculative-moe-runner-backend", "auto", "--speculative-draft-kv-cache-dtype", c["KVDTYPE"],
            "--speculative-accept-threshold-single", "1.0", "--speculative-accept-threshold-acc", "1.0",
            "--speculative-token-map", str(token_map)]
    template = Path(model_dir) / "chat_template.jinja"
    if template.is_file():
        args += ["--chat-template", str(template)]
    return args


def _unlimit_address_space() -> None:
    """Runs in the SGLang child before exec. kaggle_main caps the agent process's RLIMIT_AS as a
    runaway-snippet backstop and the child would inherit it; CUDA needs an unbounded virtual
    address space (run #6 died at context creation with 'CUDA driver error: out of memory')."""
    try:
        import resource
        _soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        resource.setrlimit(resource.RLIMIT_AS, (hard, hard))
    except Exception:
        pass


def launch_sglang(log_path: str, port: int = PORT, model_dir: str = MODEL_DIR,
                  draft_dir: str = DRAFT_MODEL_DIR, wheelhouse: str = WHEELHOUSE_DIR,
                  prefix: str = PREFIX) -> subprocess.Popen:
    """Install (offline) and start the server detached; returns immediately. Raises loudly on any
    missing input so a broken mount is visible in the notebook log rather than as a hang."""
    for label, path in (("model", model_dir), ("draft", draft_dir), ("wheelhouse", wheelhouse)):
        if not Path(path).is_dir():
            raise RuntimeError(f"{label} directory missing: {path} (check kernel-metadata sources)")
    with socket.socket() as probe:
        if probe.connect_ex(("127.0.0.1", port)) == 0:
            raise RuntimeError(f"port {port} already in use")
    target = Path(model_dir)
    if json.loads((target / "config.json").read_text()).get("quantization_config", {}).get("quant_method") != "auto-round":
        raise RuntimeError(f"{model_dir} is not the AutoRound W4A16 checkpoint")
    venv = install_sglang(wheelhouse, prefix)
    env = _cuda_env(venv, prefix)
    token_map = _find_one(wheelhouse, "hot_tokens_64k.pt")
    if _sha256(token_map) != TOKEN_MAP_SHA or _sha256(target / "tokenizer.json") != TOKENIZER_SHA:
        raise RuntimeError("FR-Spec token map or tokenizer hash mismatch with the pinned bundle")
    args = server_args(venv, model_dir, _draft_view(target, Path(draft_dir), prefix), token_map, port)
    say("launching SGLang: " + shlex.join(args))
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "ab", buffering=0) as logf:
        proc = subprocess.Popen(args, env=env, stdout=logf, stderr=subprocess.STDOUT,
                                start_new_session=True, preexec_fn=_unlimit_address_space)
    say(f"SGLang pid {proc.pid}, log {log_path}")
    return proc


def healthy(port: int = PORT, timeout: float = 3.0) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def stop_server(proc) -> None:
    """Terminate the server's whole process group; always safe to call."""
    if proc is None or proc.poll() is not None:
        return
    import signal
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=60)
    except Exception:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            pass
    say("SGLang stopped")


def log_tail(log_path: str, nbytes: int = 8000) -> str:
    try:
        with open(log_path, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - nbytes))
            return f.read().decode(errors="replace")
    except OSError:
        return ""


# ---- system monitor ----
def system_line(disk: str = "/kaggle/working") -> str:
    parts = []
    try:
        info = dict(l.split(":", 1) for l in Path("/proc/meminfo").read_text().splitlines())
        total = int(info["MemTotal"].split()[0]) / 2**20
        avail = int(info["MemAvailable"].split()[0]) / 2**20
        parts.append(f"RAM {total - avail:.1f}/{total:.1f}G")
    except Exception:
        parts.append("RAM ?")
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
        used, total, util = [x.strip() for x in out.stdout.strip().split(",")[:3]]
        parts.append(f"VRAM {int(used) / 1024:.1f}/{int(total) / 1024:.1f}G util {util}%")
    except Exception:
        parts.append("VRAM ?")
    try:
        du = shutil.disk_usage(disk if Path(disk).exists() else "/")
        parts.append(f"disk {du.used / 2**30:.0f}/{du.total / 2**30:.0f}G")
    except Exception:
        parts.append("disk ?")
    return " | ".join(parts)


def start_monitor(interval: float = 600.0, extra=None) -> threading.Thread:
    """Print one system line every `interval` seconds; `extra()` may add e.g. gate state."""
    def loop():
        while True:
            line = system_line()
            if extra is not None:
                try:
                    line += " | " + str(extra())
                except Exception as exc:
                    line += f" | extra failed: {exc}"
            say(line)
            time.sleep(interval)
    t = threading.Thread(target=loop, name="sysmon", daemon=True)
    t.start()
    return t
