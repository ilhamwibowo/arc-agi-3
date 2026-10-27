"""RESET after a real level-up must keep completed levels (ONLY_RESET_LEVELS) and restart the new level."""
from pathlib import Path

from arc3 import percept
from arc3.env import Env, list_public_games
from arc3.session import GameSession
from arc3.types import Action, GameConfig

ENVS = Path(__file__).resolve().parents[1] / "data" / "environments"
# Found by BFS over the engine (seed 0); solves tu93 level 1 in 18 moves.
TU93_L1 = ["RIGHT", "DOWN", "DOWN", "RIGHT", "UP", "RIGHT", "DOWN", "DOWN", "LEFT", "LEFT", "DOWN",
           "RIGHT", "RIGHT", "DOWN", "RIGHT", "UP", "RIGHT", "DOWN"]


def test_reset_after_level_up_keeps_level():
    gid = next(g for g in list_public_games(ENVS) if g.startswith("tu93"))
    s = GameSession(Env.offline(gid, ENVS, seed=0), GameConfig(game_id=gid))
    for a in TU93_L1:
        r = s.step(Action(a))
    assert r.level_completed and s.current.levels_completed == 1
    l2_start = percept.state_hash(s.current.grid)
    assert s.step(Action("UP")).gameplay_changed
    r = s.step(Action("RESET"))
    assert r.executed and s.current.levels_completed == 1
    assert percept.state_hash(s.current.grid) == l2_start
    assert s.step(Action("RESET")).stop_reason == "reset_noop"
    s.close()
