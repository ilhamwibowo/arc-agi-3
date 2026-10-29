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


def wait_for_gateway(base_url: str, timeout: float = 2400.0) -> None:
    """Poll <base>/api/games until it answers (<500) or fail loudly. 40 min: the Oct 2026 platform
    incident had gateway DNS failing for >10 min while weights loaded; a run that gives up scores 0."""
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
                     model_name: str, gate: PriorityGate, log_dir: Path,
                     agent_overrides: Optional[dict] = None):
    """Return (arena, make_session, agent_factory): the integration point with env/session/agent."""
    from arc3.agent import Agent, AgentConfig
    from arc3.env import Arena
    from arc3.llm import LLMClient
    from arc3.session import GameSession
    from arc3.types import GameConfig
    arena = Arena(mode=mode, base_url=base_url or None, environments_dir=env_dir or None,
                  api_key=api_key, recordings_dir=log_dir / "toolkit")

    def make_session(game_id, env):
        (log_dir / game_id).mkdir(parents=True, exist_ok=True)
        return GameSession(env, GameConfig(game_id=game_id), trace_path=log_dir / game_id / "trace.jsonl")

    def agent_factory(game_id, progress):
        # One client (one HTTP session) per game thread. Overrides keyed by id prefix allow A/B
        # runs (e.g. images on/off) inside one notebook.
        fields = {}
        for prefix, values in (agent_overrides or {}).items():
            if game_id.startswith(prefix):
                fields.update(values)
        say(f"{game_id}: agent overrides {fields}" if fields else f"{game_id}: default agent config")
        return Agent(LLMClient(base_url=llm_url, model=model_name), AgentConfig(**fields),
                     gate=gate, progress=progress)
    return arena, make_session, agent_factory


def cap_address_space(gb: Optional[float] = None) -> Optional[float]:
    """Backstop for the sandbox memory guard: cap this (agent) process's virtual address space so a
    runaway snippet gets MemoryError instead of the host OOM killer taking down the model server.
    The server is a child process launched after this and would inherit the cap, so
    kaggle_serve.launch_sglang resets RLIMIT_AS in the child before exec. Linux only;
    ARC3_AS_CAP_GB overrides the default of 40 GB, 0 disables. Returns the cap applied in GB."""
    import sys
    if not sys.platform.startswith("linux"):
        return None
    if gb is None:
        gb = float(os.environ.get("ARC3_AS_CAP_GB", "40"))
    if gb <= 0:
        return None
    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        cap = int(gb * (1 << 30))
        if hard != resource.RLIM_INFINITY:
            cap = min(cap, hard)
        resource.setrlimit(resource.RLIMIT_AS, (cap, hard))
        say(f"agent process address space capped at {cap / (1 << 30):.0f} GB")
        return cap / (1 << 30)
    except (ImportError, ValueError, OSError) as e:
        say(f"address-space cap not applied: {type(e).__name__}: {e}")
        return None


def main(competition: bool, run_end_epoch: float, log_dir: str, llm_url: str, model_name: str,
         model_ready=None, offline_subset=None, per_game_cap_seconds: float = 9 * 3600,
         max_active: int = 10, env_dir: str = OFFLINE_ENV_DIR, gate: Optional[PriorityGate] = None,
         abort=None, agent_overrides: Optional[dict] = None):
    mode = "competition" if competition else "offline"
    cap_address_space()
    base_url = os.environ.setdefault("ARC_BASE_URL", DEFAULT_GATEWAY) if competition else ""
    api_key = os.environ.setdefault("ARC_API_KEY", DEFAULT_API_KEY) if competition else ""
    if competition:
        # Wait as long as the run allows minus an hour of play: a run that gives up scores 0.
        wait_for_gateway(base_url, timeout=max(600.0, run_end_epoch - time.time() - 3600))
    elif not Path(env_dir).is_dir():
        raise RuntimeError(f"offline environment dir missing: {env_dir}")
    gate = gate or PriorityGate(max_active=max_active)
    log_path = Path(log_dir)
    arena, make_session, agent_factory = build_components(
        mode, base_url, "" if competition else env_dir, api_key, llm_url, model_name, gate, log_path,
        agent_overrides=agent_overrides)
    game_ids = arena.game_ids()
    if not game_ids:
        raise RuntimeError(f"{mode} arcade exposed zero games")
    if competition:
        # Offline smoke settings (subset, short per-game cap) must never bound a real run:
        # sub-001/sub-002 ended after ~33 min because a 180 s cap leaked into competition mode.
        if per_game_cap_seconds < 4 * 3600:
            say(f"competition: ignoring per_game_cap_seconds={per_game_cap_seconds} (offline smoke value)")
            per_game_cap_seconds = 9 * 3600
    else:
        game_ids = select_subset(game_ids, offline_subset)
    say(f"{mode}: {len(game_ids)} games: {' '.join(game_ids)}")
    # Competition mode: one scorecard opened by the runner around all games.
    summary = run_games(game_ids, None, agent_factory, gate,
                        total_seconds=run_end_epoch - time.time(),
                        per_game_cap_seconds=per_game_cap_seconds, log_dir=log_path,
                        arena=arena, model_ready=model_ready, abort=abort,
                        make_session=make_session)
    say(f"run done: {summary['ok']} ok, {summary['errors']} errors, hung {summary['hung']}, "
        f"levels {summary['levels_completed']}")
    return summary


def write_dummy_submission(path: str = "/kaggle/working/submission.parquet") -> None:
    """Offline (non-rerun) only: the official starter's structural placeholder row."""
    import pandas as pd
    pd.DataFrame([["1_0", "1", True, 1]],
                 columns=["row_id", "game_id", "end_of_game", "score"]).to_parquet(path, index=False)
    say(f"wrote placeholder {path}")
