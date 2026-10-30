"""Run the real offline runner (scheduler, slots, keeper) against public games with a hosted
OpenAI-compatible model, e.g. Ollama Cloud. For agent-logic A/B tests when Kaggle GPU quota is gone.

  .venv/bin/python scripts/play_cloud.py --games vc33 ls20 --hours 1 --concurrency 4 \
      --model glm-5.3-flash --out runs/cloud/v5-a

Key: --key-file (default ~/.config/ollama/cloud.key). Overrides: --set release_at_trim=True ...
(AgentConfig fields, applied to every game). Writes runs/cloud/<out>/<game>/{trace.jsonl,summary.json}
and run_summary.json. The model never sees game ids or hidden state (same code path as Kaggle).
"""
from __future__ import annotations

import argparse
import ast
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arc3 import kaggle_main  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--games", nargs="+", required=True, help="game id prefixes")
    p.add_argument("--hours", type=float, default=1.0, help="per-game cap and run length")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--model", default="glm-5.3-flash")
    p.add_argument("--base-url", default="https://ollama.com/v1")
    p.add_argument("--key-file", default=os.path.expanduser("~/.config/ollama/cloud.key"))
    p.add_argument("--out", required=True)
    p.add_argument("--set", nargs="*", default=[], help="AgentConfig overrides key=value")
    a = p.parse_args()

    os.environ["ARC3_LLM_API_KEY"] = Path(a.key_file).read_text().strip()
    overrides = {}
    for kv in a.set:
        k, v = kv.split("=", 1)
        try:
            overrides[k] = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            overrides[k] = v
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    # Run window = the per-game cap plus the runner's 15-min margin, so the cap, not the wall, ends games.
    total = a.hours * 3600 + 15 * 60 + 60
    summary = kaggle_main.main(
        competition=False, run_end_epoch=time.time() + total, log_dir=str(out),
        llm_url=a.base_url, model_name=a.model, offline_subset=a.games,
        per_game_cap_seconds=a.hours * 3600, max_active=a.concurrency,
        model_ready=lambda: True,      # hosted model: no boot; without this the keeper waits forever
        env_dir=str(ROOT / "data" / "environments"),
        agent_overrides={"": overrides} if overrides else None)
    print({k: summary[k] for k in ("games", "ok", "errors", "hung", "levels_completed")})


if __name__ == "__main__":
    main()
