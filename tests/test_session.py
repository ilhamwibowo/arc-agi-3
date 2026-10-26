"""End-to-end on a real public game: Env -> GameSession -> PythonTool, no model."""
from pathlib import Path

from arc3.env import Env, list_public_games
from arc3.sandbox import PythonTool
from arc3.session import GameSession
from arc3.types import Action, GameConfig

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
