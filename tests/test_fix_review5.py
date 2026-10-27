"""Fixes from docs/trace-review-run5.md: RESET inside a batch, wm argument handling, opener
evidence lines, cheaper cuts, helper retention with frozen snapshots, slot handover."""
from pathlib import Path

import numpy as np

from arc3 import prompts
from arc3.env import Env, list_public_games
from arc3.sandbox import PythonTool
from arc3.session import GameSession
from arc3.types import GameConfig

ENVS = Path(__file__).resolve().parents[1] / "data" / "environments"


def _session(prefix, tmp_path, **cfg):
    gid = next(g for g in list_public_games(ENVS) if g.startswith(prefix))
    env = Env.offline(gid, ENVS, seed=0, recordings_dir=tmp_path / "toolkit")
    return GameSession(env, GameConfig(game_id=gid, **cfg))


# ------------------------------------------------------------------ 1. RESET inside a batch

def test_reset_then_replay_in_one_call(tmp_path):
    s = _session("ls20", tmp_path)
    tool = PythonTool(s)
    tool.run("action(['UP', 'UP', 'LEFT'])")
    # The run #5 pattern: restart and replay a prefix in the same call.
    r = tool.run("res = action(['RESET', 'UP', 'UP'])\n"
                 "print(res['executed'], res['skipped'], res['stop_reason'], "
                 "res['per_action'][0]['action'])")
    assert r.error is None, r.error
    assert r.stdout.strip() == "3 0 None RESET"
    assert [a.action.name for a in r.actions] == ["RESET", "UP", "UP"]
    assert "actions may follow RESET in the same `action([...])` call" in prompts.build_system_prompt()
    assert "RESET" not in prompts.build_system_prompt(expose_reset=False).split("## Actions")[1].split("##")[0]
