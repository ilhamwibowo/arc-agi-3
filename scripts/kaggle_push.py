"""Kaggle kernel wrapper: push / status / output / pull for notebooks/<name>, with local records.

Every action is recorded under runs/kaggle/<name>/:
  pushes.jsonl                      one line per push (time, version, metadata, notebook sha256)
  <version>/kernel-metadata.json    metadata as pushed
  <version>/<name>.ipynb            notebook as pushed
  <version>/status.txt              last status
  <version>/output/                 downloaded logs + output files
  <version>/pulled/                 notebook as Kaggle has it (pull)

    .venv/bin/python scripts/kaggle_push.py push arc3-submission [--no-run] [--timeout S]
    .venv/bin/python scripts/kaggle_push.py status arc3-submission
    .venv/bin/python scripts/kaggle_push.py output arc3-submission [--version N]
    .venv/bin/python scripts/kaggle_push.py pull arc3-submission [--version N]

Pushing starts a GPU run (quota!) unless --no-run; it asks for confirmation unless --yes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
KAGGLE = ["uv", "tool", "run", "--from", "kaggle==2.2.5", "kaggle"]


def kaggle(*args: str) -> str:
    cmd = KAGGLE + list(args)
    print("$", " ".join(cmd), flush=True)
    res = subprocess.run(cmd, capture_output=True, text=True)
    out = res.stdout + res.stderr
    print(out.strip())
    if res.returncode:
        sys.exit(f"kaggle command failed ({res.returncode})")
    return out


def notebook_dir(name: str) -> Path:
    d = REPO / "notebooks" / name
    if not (d / "kernel-metadata.json").is_file():
        sys.exit(f"{d}/kernel-metadata.json missing; build it first (python -m arc3.notebook --name {name})")
    return d


def record_dir(name: str, version: str) -> Path:
    d = REPO / "runs" / "kaggle" / name / str(version)
    d.mkdir(parents=True, exist_ok=True)
    return d


def kernel_id(name: str) -> str:
    return json.loads((notebook_dir(name) / "kernel-metadata.json").read_text())["id"]


def last_version(name: str) -> str:
    log = REPO / "runs" / "kaggle" / name / "pushes.jsonl"
    if not log.is_file():
        sys.exit(f"no recorded pushes for {name}; pass --version")
    return json.loads(log.read_text().strip().splitlines()[-1])["version"]


def cmd_push(args) -> None:
    src = notebook_dir(args.name)
    meta = json.loads((src / "kernel-metadata.json").read_text())
    nb = src / meta["code_file"]
    if not args.no_run and not args.yes:
        if input(f"push {meta['id']} and start a {meta.get('machine_shape')} run? [y/N] ").lower() != "y":
            sys.exit("aborted")
    extra = ["--no-run"] if args.no_run else []
    if args.timeout:
        extra += ["--timeout", str(args.timeout)]
    out = kaggle("kernels", "push", "-p", str(src), *extra)
    m = re.search(r"version\s+(\d+)", out, re.IGNORECASE)
    version = m.group(1) if m else time.strftime("unknown-%Y%m%d-%H%M%S")
    dest = record_dir(args.name, version)
    shutil.copy2(src / "kernel-metadata.json", dest / "kernel-metadata.json")
    shutil.copy2(nb, dest / nb.name)
    entry = {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "version": version, "no_run": args.no_run,
             "notebook_sha256": hashlib.sha256(nb.read_bytes()).hexdigest(), "metadata": meta}
    with (REPO / "runs" / "kaggle" / args.name / "pushes.jsonl").open("a") as f:
        f.write(json.dumps(entry) + "\n")
    print(f"recorded version {version} in {dest}")


def cmd_status(args) -> None:
    out = kaggle("kernels", "status", kernel_id(args.name))
    version = args.version or last_version(args.name)
    (record_dir(args.name, version) / "status.txt").write_text(time.strftime("%c") + "\n" + out)


def cmd_output(args) -> None:
    version = args.version or last_version(args.name)
    dest = record_dir(args.name, version) / "output"
    dest.mkdir(exist_ok=True)
    ref = kernel_id(args.name) + (f"/{version}" if version.isdigit() else "")
    kaggle("kernels", "output", ref, "-p", str(dest), "-o")
    print(f"output in {dest}")


def cmd_pull(args) -> None:
    version = args.version or last_version(args.name)
    dest = record_dir(args.name, version) / "pulled"
    dest.mkdir(exist_ok=True)
    kaggle("kernels", "pull", kernel_id(args.name), "-p", str(dest), "-m")
    print(f"pulled into {dest}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Kaggle kernel push/status/output/pull with records")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("push", cmd_push), ("status", cmd_status), ("output", cmd_output), ("pull", cmd_pull)):
        p = sub.add_parser(name)
        p.add_argument("name", help="notebook folder under notebooks/")
        p.set_defaults(fn=fn)
        if name == "push":
            p.add_argument("--no-run", action="store_true", help="quick save, no execution")
            p.add_argument("--timeout", type=int, default=0, help="kernel run limit in seconds")
            p.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
        else:
            p.add_argument("--version", default="", help="kernel version (default: last pushed)")
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
