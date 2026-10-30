"""Pick a random, reproducible subset of community games as a hidden-like validation set.

  .venv/bin/python scripts/pick_games.py SRC_ENV_DIR OUT_DIR --n 50 --seed 1

Skips the tutorial stems (ez01-ez04). Copies <game>/ directories verbatim (code + metadata)."""
import argparse
import random
import shutil
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("src")
p.add_argument("out")
p.add_argument("--n", type=int, default=50)
p.add_argument("--seed", type=int, default=1)
a = p.parse_args()
src, out = Path(a.src), Path(a.out)
games = sorted(d.name for d in src.iterdir() if d.is_dir() and not d.name.startswith("ez"))
rng = random.Random(a.seed)
chosen = sorted(rng.sample(games, a.n))
out.mkdir(parents=True, exist_ok=True)
for g in chosen:
    shutil.copytree(src / g, out / g, dirs_exist_ok=True)
print(len(chosen), "games ->", out)
print(" ".join(chosen))
