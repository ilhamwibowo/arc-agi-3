"""Notebook entry point for the game run: pick games, build env + agents, call runner.run_games.

Two modes:
- competition (KAGGLE_IS_COMPETITION_RERUN): live gateway at ARC_BASE_URL, one scorecard, all games
  listed by the gateway. Scoring comes from the gateway; we write nothing for submission.
- offline (interactive Save & Run): the competition's bundled environment_files, optional subset.

`build_components` is the single integration point with arc3.env and arc3.agent.
"""
from __future__ import annotations

import os
import time
import urllib.request
from pathlib import Path
from typing import Optional

from arc3.runner import run_games
from arc3.scheduler import PriorityGate

COMPETITION_DIR = "/kaggle/input/competitions/arc-prize-2026-arc-agi-3"
OFFLINE_ENV_DIR = f"{COMPETITION_DIR}/environment_files"
DEFAULT_GATEWAY = "http://gateway:8001/"
DEFAULT_API_KEY = "test-key-123"


def say(msg: str) -> None:
    print(f"[arc3 {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def wait_for_gateway(base_url: str, timeout: float = 600.0) -> None:
    """Poll <base>/api/games until it answers (<500) or fail loudly."""
    url = base_url.rstrip("/") + "/api/games"
    end = time.time() + timeout
    last = ""
    while time.time() < end:
        try:
            with urllib.request.urlopen(url, timeout=10) as r:
                if r.status < 500:
                    say(f"gateway up: {url}")
                    return
        except Exception as exc:
            last = repr(exc)
        time.sleep(5)
    raise RuntimeError(f"competition gateway {url} not ready after {timeout:.0f}s: {last}")


def list_game_ids(mode: str, base_url: str = "", env_dir: str = OFFLINE_ENV_DIR,
                  api_key: str = "") -> list[str]:
    import arc_agi
    if mode == "competition":
        arcade = arc_agi.Arcade(arc_api_key=api_key, arc_base_url=base_url,
                                operation_mode=arc_agi.OperationMode.COMPETITION, environments_dir="")
    else:
        if not Path(env_dir).is_dir():
            raise RuntimeError(f"offline environment dir missing: {env_dir}")
        arcade = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir=env_dir)
    ids = [e.game_id for e in arcade.available_environments]
    if not ids:
        raise RuntimeError(f"{mode} arcade exposed zero games")
    return ids


def select_subset(game_ids: list[str], subset) -> list[str]:
    """`subset`: None/[] = all, int = first n, list of id prefixes = matching games."""
    if not subset:
        return list(game_ids)
    if isinstance(subset, int):
        return list(game_ids)[:subset]
    chosen = [g for g in game_ids if any(g.startswith(p) for p in subset)]
    if not chosen:
        raise RuntimeError(f"no games match subset {subset}; available: {game_ids}")
    return chosen


def build_components(mode: str, base_url: str, env_dir: str, api_key: str, llm_url: str,
                     model_name: str, gate: PriorityGate, log_dir: Path):
    """Return (arena, agent_factory). Integration point with arc3.env / arc3.agent."""
    from arc3.agent import Agent
    from arc3.env import Arena
    arena = Arena(mode=mode, base_url=base_url, environments_dir=env_dir, api_key=api_key)

    def agent_factory(game_id, progress):
        return Agent(game_id=game_id, progress=progress, gate=gate, llm_url=llm_url,
                     model=model_name, log_dir=log_dir / game_id)
    return arena, agent_factory


def main(competition: bool, run_end_epoch: float, log_dir: str, llm_url: str, model_name: str,
         model_ready=None, offline_subset=None, per_game_cap_seconds: float = 9 * 3600,
         max_active: int = 10, env_dir: str = OFFLINE_ENV_DIR, gate: Optional[PriorityGate] = None,
         abort=None):
    mode = "competition" if competition else "offline"
    base_url = os.environ.setdefault("ARC_BASE_URL", DEFAULT_GATEWAY) if competition else ""
    api_key = os.environ.setdefault("ARC_API_KEY", DEFAULT_API_KEY) if competition else ""
    if competition:
        wait_for_gateway(base_url)
    game_ids = list_game_ids(mode, base_url, env_dir, api_key)
    if not competition:
        game_ids = select_subset(game_ids, offline_subset)
    say(f"{mode}: {len(game_ids)} games: {' '.join(game_ids)}")
    gate = gate or PriorityGate(max_active=max_active)
    log_path = Path(log_dir)
    arena, agent_factory = build_components(mode, base_url, env_dir, api_key, llm_url, model_name,
                                            gate, log_path)
    # Competition mode: one scorecard opened by the runner around all games.
    summary = run_games(game_ids, None, agent_factory, gate,
                        total_seconds=run_end_epoch - time.time(),
                        per_game_cap_seconds=per_game_cap_seconds, log_dir=log_path,
                        arena=arena, model_ready=model_ready, abort=abort)
    say(f"run done: {summary['ok']} ok, {summary['errors']} errors, hung {summary['hung']}, "
        f"levels {summary['levels_completed']}")
    return summary


def write_dummy_submission(path: str = "/kaggle/working/submission.parquet") -> None:
    """Offline (non-rerun) only: the official starter's structural placeholder row."""
    import pandas as pd
    pd.DataFrame([["1_0", "1", True, 1]],
                 columns=["row_id", "game_id", "end_of_game", "score"]).to_parquet(path, index=False)
    say(f"wrote placeholder {path}")
