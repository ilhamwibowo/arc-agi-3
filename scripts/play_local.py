"""Play public games locally with a chosen model backend.

  .venv/bin/python scripts/play_local.py --games ls20 ft09 --backend ollama --model qwen3.5:4b
  .venv/bin/python scripts/play_local.py --games ls20 --backend http --base-url http://HOST:30000/v1 --model qwen
  .venv/bin/python scripts/play_local.py --games ls20 --backend fake --script tests/scripts/ls20.json

Writes runs/local/<run>/<game>/{trace.jsonl,summary.json}. No hidden state reaches the model.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arc3 import percept  # noqa: E402
from arc3.agent import Agent, AgentConfig  # noqa: E402
from arc3.env import Env, list_public_games  # noqa: E402
from arc3.llm import FakeLLM, LLMClient  # noqa: E402
from arc3.sandbox import PythonTool  # noqa: E402
from arc3.session import GameSession  # noqa: E402
from arc3.types import GameConfig  # noqa: E402


def make_llm(a):
    if a.backend == "fake":
        steps = json.loads(Path(a.script).read_text())
        return FakeLLM([FakeLLM.code(s) if isinstance(s, str) else FakeLLM.text(s["text"])
                        for s in steps])
    base = a.base_url or ("http://127.0.0.1:11434/v1" if a.backend == "ollama" else None)
    return LLMClient(base_url=base, model=a.model, timeout=a.timeout)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--games", nargs="+", required=True)
    p.add_argument("--backend", choices=["fake", "ollama", "http"], default="ollama")
    p.add_argument("--model", default="qwen3.5:4b")
    p.add_argument("--base-url")
    p.add_argument("--script")
    p.add_argument("--seconds", type=float, default=600)
    p.add_argument("--max-actions", type=int, default=200)
    p.add_argument("--timeout", type=float, default=300)
    p.add_argument("--context", type=int, default=24000)
    p.add_argument("--out", default=None)
    a = p.parse_args()

    envs = ROOT / "data" / "environments"
    out = Path(a.out or ROOT / "runs" / "local" / time.strftime("%Y%m%d-%H%M%S"))
    out.mkdir(parents=True, exist_ok=True)
    cfg = AgentConfig(context_tokens=a.context, output_tokens=min(8000, a.context // 3),
                      drain_tokens=a.context // 2)
    for prefix in a.games:
        gid = next(g for g in list_public_games(envs) if g.startswith(prefix))
        gdir = out / prefix
        gdir.mkdir(exist_ok=True)
        env = Env.offline(gid, envs, seed=0, recordings_dir=gdir / "toolkit")
        session = GameSession(env, GameConfig(game_id=gid, max_actions=a.max_actions,
                                              max_seconds=a.seconds),
                              trace_path=gdir / "trace.jsonl")
        agent = Agent(make_llm(a), cfg, tool_factory=PythonTool,
                      render_png=percept.render_png, render_diff_png=percept.render_diff_png)
        t0 = time.time()
        try:
            stats = agent.play_game(session, deadline=t0 + a.seconds)
            summary = stats.__dict__ if hasattr(stats, "__dict__") else dict(stats)
        except Exception as e:  # keep going with the next game
            summary = {"error": repr(e)}
        summary.update(levels=session.current.levels_completed, win_levels=session.current.win_levels,
                       state=session.current.state.value, actions=session.actions,
                       resets=session.resets, seconds=round(time.time() - t0, 1))
        (gdir / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
        print(prefix, json.dumps(summary, default=str), flush=True)
        session.close()


if __name__ == "__main__":
    main()
