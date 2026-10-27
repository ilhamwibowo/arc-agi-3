"""Prompt rendering checks and an executable check of the WORLD_MODEL worked example."""
import numpy as np

from arc3 import prompts
from arc3.wm import WM, Record

FLOOR, WALL, BLOCK, GOAL = 0, 5, 9, 11
STEP = {"UP": (-5, 0), "DOWN": (5, 0), "LEFT": (0, -5), "RIGHT": (0, 5)}


def test_system_prompt_renders_without_format_leftovers():
    p = prompts.build_system_prompt()
    assert "{{" not in p and "}}" not in p and "{legend}" not in p
    assert "wm.register(transition, outcome, init=init)" in p
    assert "wm.check(read, step)" in p and "wm.plan_abstract(read, step, goal)" in p
    assert "`memory` is a dict that persists across calls" in p
    assert "level_reset" in p and "deliberate restart mechanic" in p
    for helper in ("wm.induce_motion()", "wm.untried_actions()", "wm.transition_graph()"):
        assert helper in p
    actions = lambda **kw: prompts.build_system_prompt(**kw).split("## Actions")[1].split("##")[0]
    assert "UNDO" in actions(expose_undo=True) and "UNDO" not in actions(expose_undo=False)
    assert "NEVER burn moves" in actions(expose_reset=True)
    assert "RESET" not in actions(expose_reset=False)


def test_world_model_section_size():
    # Brief: lead with check/plan_abstract and keep the section near 450 tokens (~1600 chars).
    assert len(prompts.WORLD_MODEL) <= 1610
    assert prompts.WORLD_MODEL.index("wm.check") < prompts.WORLD_MODEL.index("wm.register")


# ------------------------------------------------------------------ the worked example

FLOOR, WALL, BLOCK, KEY, DOOR = 0, 5, 9, 11, 14
STEP = {"UP": (-5, 0), "DOWN": (5, 0), "LEFT": (0, -5), "RIGHT": (0, 5)}
KEY_AT, DOOR_AT = (10, 30), (40, 30)


def _static():
    g = np.zeros((64, 64), np.uint8)
    g[:5, :] = g[60:, :] = g[:, :5] = g[:, 60:] = WALL
    g[20:25, 5:40] = WALL               # wall between the key row and the door
    g[DOOR_AT[0]:DOOR_AT[0] + 5, DOOR_AT[1]:DOOR_AT[1] + 5] = DOOR
    return g


def _render(r, c, has_key):
    g = _static()
    if not has_key:
        g[KEY_AT[0]:KEY_AT[0] + 5, KEY_AT[1]:KEY_AT[1] + 5] = KEY
    g[r:r + 5, c:c + 5] = BLOCK
    return g


def _true_step(state, name):
    """Independent ground truth: the 5x5 block moves 5 cells unless a wall is in the way;
    landing on the key takes it."""
    r, c, has_key = state
    dr, dc = STEP[name]
    nr, nc = r + dr, c + dc
    if (_static()[nr:nr + 5, nc:nc + 5] == WALL).any():
        return state
    return nr, nc, has_key or (nr, nc) == KEY_AT


def _record(moves):
    state = (10, 10, False)
    rec = Record(level_start_grid=_render(*state), level=1, available_actions=list(STEP))
    for name in moves:
        nxt = _true_step(state, name)
        rec.transitions.append((_render(*state), {"action": name}, _render(*nxt), "ongoing"))
        state = nxt
    return rec, state


class _Frame:
    def __init__(self, grid):
        self.grid = grid


def test_world_model_example_checks_and_plans():
    moves = ["RIGHT"] * 4 + ["DOWN", "DOWN"]     # takes the key; the second DOWN hits the wall
    rec, state = _record(moves)
    wm = WM(rec, hud_border=4)
    out = []
    ns = {"np": np, "wm": wm, "memory": {}, "current_frame": _Frame(rec.current_grid),
          "print": lambda *a: out.append(" ".join(map(str, a)))}
    exec(prompts.WM_EXAMPLE, ns)
    assert out and out[0].startswith("6/6 predicted"), out
    p = ns["p"]
    assert p["found"], p["reason"]
    for a in p["actions"]:
        state = _true_step(state, a)
    assert state == (*DOOR_AT, True)


def test_check_reports_first_mismatch_and_unseen_actions():
    rec, _ = _record(["RIGHT"] * 4 + ["DOWN", "DOWN"])
    wm = WM(rec, hud_border=4)
    read = lambda g: tuple(int(x) for x in np.argwhere(g == BLOCK).min(axis=0))

    def no_walls(s, a):
        dr, dc = STEP.get(a, (0, 0))
        return (s[0] + dr, s[1] + dc)

    res = wm.check(read, no_walls)
    assert not res["ok"] and res["checked"] == 6 and res["correct"] == 5
    fm = res["first_mismatch"]
    assert fm["index"] == 5 and fm["action"] == "DOWN"
    assert fm["predicted"] == "(20, 30)" and fm["actual"] == "(15, 30)"
    assert res["by_action"] == {"DOWN": "1/2", "RIGHT": "4/4"}
    assert res["unseen_actions"] == ["UP", "LEFT"]
    assert "first mismatch L1 #5 DOWN" in res["summary"]
    # A crashing read is reported, not raised.
    bad = wm.check(lambda g: 1 / 0, no_walls)
    assert not bad["ok"] and "ZeroDivisionError" in bad["first_mismatch"]["error"]


def test_plan_abstract_handles_unhashable_states_and_no_path():
    rec, _ = _record(["RIGHT"])
    wm = WM(rec, hud_border=4)
    read = lambda g: {"pos": [int(x) for x in np.argwhere(g == BLOCK).min(axis=0)]}

    def step(s, a):                     # mutates its argument: plan_abstract must copy first
        dr, dc = STEP.get(a, (0, 0))
        s["pos"] = [s["pos"][0] + dr, s["pos"][1] + dc]
        return s

    p = wm.plan_abstract(read, step, lambda s: s["pos"] == [25, 15])
    assert p["found"] and sorted(p["plan"]) == ["DOWN", "DOWN", "DOWN"]
    none = wm.plan_abstract(read, lambda s, a: s, lambda s: False)
    assert not none["found"] and none["reason"].startswith("exhausted")
