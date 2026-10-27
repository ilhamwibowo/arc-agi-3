"""End-to-end on a real public game: Env -> GameSession -> PythonTool, no model."""
import json
from pathlib import Path

import numpy as np

from arc3.env import Env, list_public_games
from arc3.sandbox import PythonTool
from arc3.session import GameSession
from arc3.types import Action, GameConfig, GameState, Observation

ENVS = Path(__file__).resolve().parents[1] / "data" / "environments"


def _game(prefix: str) -> str:
    return next(g for g in list_public_games(ENVS) if g.startswith(prefix))


def test_session_step_and_tool(tmp_path):
    gid = _game("ls20")
    env = Env.offline(gid, ENVS, seed=0, recordings_dir=tmp_path / "toolkit")
    s = GameSession(env, GameConfig(game_id=gid), trace_path=tmp_path / "trace.jsonl")
    assert s.level == 1 and s.current.win_levels > 0
    assert "RESET" not in s.valid_actions
    r = s.step(Action("UP"))
    assert r.executed and s.actions == 1 and len(s.history) == 1
    tool = PythonTool(s)
    out = tool.run("print(current_frame.shape, len(transitions), valid_actions)\n"
                   "res = action(['LEFT', 'RIGHT'])\nprint(res['executed'], res['stop_reason'])")
    assert out.error is None, out.error
    assert "(64, 64) 1" in out.stdout
    assert s.actions >= 2
    lines = (tmp_path / "trace.jsonl").read_text().splitlines()
    assert lines[0].startswith('{"event": "start"') and len(lines) >= 3
    assert gid not in out.stdout
    s.close()


def test_auto_reset_noop_when_alive(tmp_path):
    gid = _game("ft09")
    env = Env.offline(gid, ENVS, seed=0, recordings_dir=tmp_path / "toolkit")
    s = GameSession(env, GameConfig(game_id=gid))
    assert s.auto_reset_if_needed() is None


# ------------------------------------------------------------------ silent level reset

def test_ls20_budget_exhaustion_is_reported_as_level_reset(tmp_path):
    # ls20's move bar empties after 42 moves; the 43rd silently puts the level back to its start.
    gid = _game("ls20")
    env = Env.offline(gid, ENVS, seed=0, recordings_dir=tmp_path / "toolkit")
    s = GameSession(env, GameConfig(game_id=gid), trace_path=tmp_path / "trace.jsonl")
    flags = [s.step(Action("UP" if i % 2 else "LEFT")).level_reset for i in range(44)]
    assert flags.index(True) == 42 and sum(flags) == 1
    res = s.history[42].result
    assert res.message.startswith("LEVEL RESET") and not res.game_over
    events = [json.loads(l) for l in (tmp_path / "trace.jsonl").read_text().splitlines()]
    assert [e["level_reset"] for e in events if e["event"] == "action"][42] is True
    s.close()


class _Env:
    """Scripted env: returns the given grids in order (8x8 board, hud_border=1)."""

    def __init__(self, grids):
        self.grids = list(grids)
        self.observation = self._obs(self.grids.pop(0))

    @staticmethod
    def _obs(g):
        return Observation(g, [g], GameState.NOT_FINISHED, 0, 2, ["LEFT", "RIGHT"], 0)

    def step(self, action):
        return self._obs(self.grids.pop(0))


def _board(player_col, bar):
    g = np.zeros((8, 8), np.uint8)
    g[0, :bar] = 11                 # budget bar in the HUD row
    g[4, player_col] = 9
    return g


def test_walking_back_to_start_is_not_a_level_reset():
    # Interior returns to the start but the bar keeps shrinking: an ordinary move.
    s = GameSession(_Env([_board(4, 8), _board(3, 7), _board(4, 6)]),
                    GameConfig(game_id="x", hud_border=1))
    assert not s.step(Action("LEFT")).level_reset
    r = s.step(Action("RIGHT"))
    assert r.gameplay_changed and not r.level_reset and r.message == "board changed"


def test_reset_with_bar_refill_is_a_level_reset():
    s = GameSession(_Env([_board(4, 8), _board(3, 7), _board(2, 6), _board(4, 8)]),
                    GameConfig(game_id="x", hud_border=1))
    assert not s.step(Action("LEFT")).level_reset
    assert not s.step(Action("LEFT")).level_reset
    assert s.step(Action("LEFT")).level_reset
