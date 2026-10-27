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


# ------------------------------------------------------------------ 2. wm argument handling

def _hud_record():
    """A 12x12 board with a top-row HUD bar that shrinks every move; the avatar (3) moves right."""
    from arc3.wm import Record
    def board(pos, bar):
        g = np.zeros((12, 12), np.uint8)
        g[0, :bar] = 9
        g[6, pos] = 3
        g[2, 2] = 7            # a decoration the avatar never touches
        return g
    rec = Record(level_start_grid=board(4, 10), level=1,
                 available_actions=["UP", "DOWN", "LEFT", "RIGHT", "MOUSE"])
    rec.transitions = [
        (board(4, 10), {"action": "RIGHT", "row": None, "col": None}, board(5, 9), "ongoing"),
        (board(5, 9), {"action": "LEFT", "row": None, "col": None}, board(4, 8), "ongoing"),
        (board(4, 8), {"action": "UP", "row": None, "col": None}, board(4, 7), "ongoing"),
    ]
    rec.live_grid = board(4, 7)
    return rec, board


def test_untried_actions_ignores_the_hud_bar():
    from arc3.wm import WM
    rec, _ = _hud_record()
    wm = WM(rec, hud_border=1)
    un = wm.untried_actions()
    # The full grid differs every move (bar), but the interior at pos 4 has seen RIGHT and UP.
    assert un["here"] == ["DOWN", "LEFT", "MOUSE"]
    assert un["ever"] == ["DOWN", "MOUSE"]


def test_click_candidates_goal_menu_and_induce_motion_accept_grid_frame_or_nothing():
    from arc3.sandbox import FrameView
    from arc3.wm import WM
    rec, board = _hud_record()
    wm = WM(rec, hud_border=1)
    frame = FrameView(board(4, 7), step=3, level=1)
    by_none = wm.click_candidates()
    assert by_none == wm.click_candidates(board(4, 7)) == wm.click_candidates(frame)
    assert {"action": "MOUSE", "row": 6, "col": 4} in by_none
    assert all(c["row"] >= 1 for c in by_none)       # the HUD bar is not an object
    menu = dict(wm.goal_menu())
    pred = menu["avatar_reaches_color_7"]
    assert pred() is False and pred(board(4, 7)) is False and pred(frame) is False
    assert pred({"grid": board(4, 7)}) is False
    won = board(4, 7); won[2, 2] = 3
    assert pred(won) is True and pred(FrameView(won, 4, 1)) is True
    for arg in (None, board(4, 7), frame, rec):
        m = wm.induce_motion(arg) if arg is not None else wm.induce_motion()
        assert m["actions"]["RIGHT"]["dx"] == 1 and m["avatar_candidates"][0]["color"] == 3
    assert wm.induce_motion(record=rec)["actions"]["LEFT"]["dx"] == -1


def test_prompt_lists_the_free_helpers_with_their_real_signatures():
    p = prompts.build_system_prompt()
    for helper in ("wm.click_candidates()", "wm.goal_menu()", "wm.untried_actions()"):
        assert helper in p
    assert "Grid arguments are optional" in p
