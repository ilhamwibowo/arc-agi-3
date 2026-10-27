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
    assert "`memory` is a dict that persists across calls" in p
    assert "all_levels=True" in p and "level_reset" in p
    for helper in ("wm.induce_motion()", "wm.untried_actions()", "wm.transition_graph()"):
        assert helper in p
    actions = lambda **kw: prompts.build_system_prompt(**kw).split("## Actions")[1].split("##")[0]
    assert "UNDO" in actions(expose_undo=True) and "UNDO" not in actions(expose_undo=False)


def _start():
    g = np.zeros((64, 64), np.uint8)
    g[20:25, 10:30] = WALL          # a wall between the block and the goal
    g[10:15, 10:15] = BLOCK
    g[30:35, 10:15] = GOAL
    return g


def _true_step(g, name):
    """Independent ground truth: the 5x5 block moves 5 cells unless a wall or the HUD edge blocks."""
    g = g.copy()
    (r, c) = np.argwhere(g == BLOCK).min(axis=0)
    dr, dc = STEP[name]
    nr, nc = r + dr, c + dc
    if nr < 4 or nc < 4 or nr + 5 > 60 or nc + 5 > 60 or (g[nr:nr + 5, nc:nc + 5] == WALL).any():
        return g
    g[r:r + 5, c:c + 5] = FLOOR
    g[nr:nr + 5, nc:nc + 5] = BLOCK
    return g


def test_world_model_example_runs_verifies_and_plans():
    rec = Record(level_start_grid=_start(), level=1, available_actions=list(STEP))
    g = rec.level_start_grid
    for name in ["DOWN", "RIGHT", "DOWN", "LEFT"]:   # the second DOWN hits the wall: a no-op
        after = _true_step(g, name)
        rec.transitions.append((g, {"action": name}, after, "ongoing"))
        g = after
    wm = WM(rec, hud_border=4)
    out = []
    ns = {"np": np, "wm": wm, "print": lambda *a: out.append(" ".join(map(str, a)))}
    exec(prompts.WM_EXAMPLE, ns)
    assert out and out[0].startswith("init ok; dynamics 3/3; outcome ok"), out
    plan = wm.plan()
    assert plan["found"], plan["reason"]
    g = rec.current_grid
    for a in plan["actions"]:
        g = _true_step(g, a["action"])
    assert ((g == BLOCK) & (_start() == GOAL)).any()
