"""Tests for arc3.wm on a synthetic 10x10 maze: avatar (3) moves one cell, walls (5) block,
reaching the goal cell (4) completes the level, stepping on lava (2) is game over."""
import time

import numpy as np
import pytest

from arc3.wm import WM, Record, diff_text, grid_hash

BG, WALL, AVATAR, GOAL, LAVA = 0, 5, 3, 4, 2
DELTA = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}


def start_grid():
    g = np.zeros((10, 10), dtype=np.uint8)
    g[0, :] = g[-1, :] = g[:, 0] = g[:, -1] = WALL
    g[2:7, 4] = WALL          # internal wall
    g[1, 7] = LAVA
    g[8, 8] = GOAL
    g[1, 1] = AVATAR
    return g


def true_step(grid, action, step=1):
    """Ground-truth dynamics, also used as the 'model' (step=2 gives the off-by-one model)."""
    g = grid.copy()
    name = action["action"] if isinstance(action, dict) else action
    if name not in DELTA:
        return g
    (r, c), = np.argwhere(g == AVATAR)
    dr, dc = DELTA[name]
    nr, nc = r + dr * step, c + dc * step
    if not (0 <= nr < 10 and 0 <= nc < 10) or g[nr, nc] == WALL:
        return g
    g[r, c], g[nr, nc] = BG, AVATAR
    return g


class FakeEnv:
    """Env-like stepper that tracks lava/goal positions itself (the avatar overwrites cells)."""

    def __init__(self):
        self.grid = start_grid()

    def step(self, name):
        before = self.grid
        after = true_step(before, {"action": name})
        (r, c), = np.argwhere(after == AVATAR)
        out = "ongoing"
        if (r, c) == (8, 8):
            out = "level_complete"
        elif (r, c) == (1, 7):
            out = "game_over"
        self.grid = after
        return before, {"action": name, "row": None, "col": None}, after, out


def make_record(names):
    env = FakeEnv()
    rec = Record(level_start_grid=env.grid.copy(), level=0,
                 available_actions=["UP", "DOWN", "LEFT", "RIGHT"])
    for n in names:
        rec.transitions.append(env.step(n))
    return rec


# A walk with a wall bump (LEFT at the start is a no-op) and moves in all four directions.
WALK = ["LEFT", "RIGHT", "RIGHT", "DOWN", "DOWN", "UP", "LEFT", "DOWN"]


def model_outcome(state):
    g = state["grid"]
    if not (g == GOAL).any():
        return "level_complete"
    if not (g == LAVA).any():
        return "game_over"
    return "ongoing"


def register(wm, step=1, **kw):
    wm.register(lambda s, a: {**s, "grid": true_step(s["grid"], a, step)}, model_outcome, **kw)


def test_correct_model_passes_all_channels():
    wm = WM(make_record(WALK))
    register(wm)
    res = wm.verify()
    assert res["ok"], res["summary"]
    assert res["dynamics"]["accuracy"] == 1.0
    assert res["dynamics"]["graded"] == len(WALK) - 1   # the wall bump is an ungraded no-op


def test_off_by_one_model_reports_first_divergence():
    wm = WM(make_record(WALK))
    register(wm, step=2)
    res = wm.verify()
    assert not res["ok"]
    fd = res["dynamics"]["first_divergence"]
    assert fd["index"] == 1 and fd["action"] == "RIGHT"
    # expected avatar at (1,2); model put it at (1,3)
    assert fd["diff"] == "3->0 @ r1 c2; 0->3 @ r1 c3"
    assert fd["bbox"] == (1, 2, 1, 3)
    assert res["dynamics"]["accuracy"] < 1.0
    assert "first divergence" in res["summary"]


def test_init_and_outcome_channels():
    wm = WM(make_record(["RIGHT"] * 6))   # walks into lava at (1,7)
    register(wm)
    assert wm.verify()["ok"]
    wm.register(lambda s, a: {**s, "grid": true_step(s["grid"], a)}, lambda s: "ongoing",
                init=lambda g, lv: {"grid": np.zeros_like(g)})
    res = wm.verify()
    assert not res["init"]["ok"]
    assert res["outcome"]["mismatches"][0]["expected"] == "game_over"


def test_diff_text_caps_runs():
    a = np.zeros((20, 20), dtype=np.uint8)
    b = a.copy()
    b[::1, 0] = 7
    d = diff_text(a, b)
    assert d["n_cells"] == 20 and "+8 more runs" in d["diff"]


def test_plan_finds_shortest_path_and_validates():
    rec = make_record(["RIGHT"])
    wm = WM(rec)
    register(wm)
    res = wm.plan()
    assert res["found"], res["reason"]
    # From (1,2) to (8,8): 7 down + 6 right = 13, wall at col 4 rows 2-6 does not lengthen it.
    assert len(res["actions"]) == 13
    v = wm.validate(res["actions"])
    assert v["ok"] and v["final_outcome"] == "level_complete"
    assert wm.expected_hashes(res["actions"]) == res["hashes"]
    # executing the plan in the real env reproduces the expected hashes
    env = FakeEnv()
    env.step("RIGHT")
    observed = [grid_hash(env.step(a["action"])[2]) for a in res["actions"]]
    assert observed == res["hashes"]


def test_plan_prunes_game_over_and_supports_astar_and_goal():
    wm = WM(make_record([]))
    def h(s):
        (r, c), = np.argwhere(s["grid"] == AVATAR)
        return abs(8 - r) + abs(8 - c)
    register(wm, heuristic=h)
    res = wm.plan()
    assert res["found"] and len(res["actions"]) == 14
    res = wm.plan_to(lambda s: s["grid"][1, 6] == AVATAR, mode="bfs")
    assert res["found"] and [a["action"] for a in res["actions"]] == ["RIGHT"] * 5


def test_current_state_carries_hidden_fields():
    wm = WM(make_record(["RIGHT", "DOWN", "LEFT"]))
    wm.register(lambda s, a: {**s, "grid": true_step(s["grid"], a), "n": s["n"] + 1},
                model_outcome, init=lambda g, lv: {"grid": g.copy(), "n": 0})
    st = wm.current_state()
    assert st["n"] == 3 and np.array_equal(st["grid"], wm._rec().current_grid)


def test_induce_motion_recovers_offsets():
    wm = WM(make_record(WALK))
    m = wm.induce_motion()
    acts = m["actions"]
    assert (acts["RIGHT"]["dy"], acts["RIGHT"]["dx"]) == (0, 1)
    assert (acts["DOWN"]["dy"], acts["DOWN"]["dx"]) == (1, 0)
    assert (acts["UP"]["dy"], acts["UP"]["dx"]) == (-1, 0)
    assert (acts["LEFT"]["dy"], acts["LEFT"]["dx"]) == (0, -1)
    assert acts["LEFT"]["blocked_cases"] == [0]
    assert acts["RIGHT"]["moving_colors"] == [AVATAR]
    assert m["avatar_candidates"][0]["color"] == AVATAR


def test_transition_graph_and_untried():
    wm = WM(make_record(WALK))
    g = wm.transition_graph()
    assert g["n_noops"] == 1 and g["noops"][0]["index"] == 0
    assert g["n_deaths"] == 0
    assert any(r["action"] == "UP" for r in g["revisits"])
    un = wm.untried_actions()
    assert un["here"] == ["UP", "DOWN", "LEFT", "RIGHT"]   # (3,2) never visited before
    assert un["ever"] == []
    assert WM(make_record(["RIGHT"] * 6)).transition_graph()["n_deaths"] == 1


def test_goal_menu_predicates_plan():
    wm = WM(make_record(WALK))
    register(wm)
    menu = dict(wm.goal_menu())
    assert f"avatar_reaches_color_{GOAL}" in menu and f"all_color_{LAVA}_removed" in menu
    res = wm.plan(goal=menu[f"avatar_reaches_color_{GOAL}"])
    assert res["found"]
    assert wm.validate(res["actions"])["final_outcome"] == "level_complete"


def test_mouse_click_candidates_default_actions():
    rec = make_record([])
    rec.available_actions = ["MOUSE"]
    wm = WM(rec)
    register(wm)
    acts = wm._actions_for(wm.current_state(), ["MOUSE"])
    assert {"action": "MOUSE", "row": 8, "col": 8} in acts and len(acts) <= 48


def test_timeouts_respected():
    wm = WM(make_record(WALK * 5))
    def slow(s, a):
        time.sleep(0.05)
        return {**s, "grid": true_step(s["grid"], a)}
    wm.register(slow, model_outcome)
    # A cap of 0.2 s plus one in-flight 0.05 s call; 1.5 s leaves room for a loaded machine.
    t = time.monotonic()
    res = wm.verify(timeout=0.2)
    assert res["timeout"] and not res["ok"] and time.monotonic() - t < 1.5
    t = time.monotonic()
    res = wm.plan(timeout=0.2)
    assert not res["found"] and res["reason"].startswith("timeout") and time.monotonic() - t < 1.5


def test_model_errors_are_reported_not_raised():
    wm = WM(make_record(WALK))
    wm.register(lambda s, a: 1 / 0, model_outcome)
    res = wm.verify()
    assert not res["ok"] and "ZeroDivisionError" in res["dynamics"]["first_divergence"]["error"]
    assert not wm.plan()["found"]
    with pytest.raises(RuntimeError):
        WM(make_record([])).verify()


def test_hud_border_ignored_unless_strict():
    rec = make_record(WALK)
    for i, (b, a, after, o) in enumerate(rec.transitions):   # a HUD step counter in cell (0, 0)
        after = after.copy()
        after[0, 0] = i + 1
        rec.transitions[i] = (b, a, after, o)
    wm = WM(rec, hud_border=1)
    register(wm)
    assert wm.verify()["ok"]
    res = wm.verify(strict=True)
    assert not res["ok"] and "@ r0 c0" in res["dynamics"]["first_divergence"]["diff"]
    hashes = wm.expected_hashes(["DOWN"])
    observed = true_step(rec.current_grid, {"action": "DOWN"})
    observed[0, 0] = 99
    assert hashes == [wm.view_hash(observed)]
    assert wm.expected_hashes(["DOWN"], strict=True) != [wm.view_hash(observed, strict=True)]



def test_verify_checks_current_level_unless_all_levels():
    # An earlier level whose recorded dynamics differ (2-cell moves) from the model (1-cell moves).
    old = make_record(WALK)
    old.transitions = [(b, a, true_step(b, a, 2), o) for b, a, _, o in old.transitions]
    rec = make_record(WALK)
    rec.level, rec.previous = 1, [old]
    wm = WM(rec)
    register(wm)
    assert wm.verify()["ok"]                                   # current level only by default
    res = wm.verify(all_levels=True)
    assert not res["ok"] and res["dynamics"]["first_divergence"]["level"] == 0
