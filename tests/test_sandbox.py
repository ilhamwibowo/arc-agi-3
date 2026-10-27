"""Sandbox tests against a scripted fake session (no env, no model)."""
import sys
import types

import numpy as np
import pytest

import arc3
from arc3.types import (COLOR_LETTERS, ActionResult, GameConfig, GameState, Observation,
                         Transition)

try:
    from arc3 import percept  # noqa: F401
except ImportError:  # minimal stand-in until arc3.percept lands
    fake = types.ModuleType("arc3.percept")
    fake.ascii = lambda g: "\n".join("".join(COLOR_LETTERS[v] for v in row) for row in g)
    fake.segment = lambda g: {"nodes": int(len(np.unique(g)))}
    fake.frame_diff = lambda a, b: {"changed_cell_count": int((a != b).sum())}

    def _collapse(before, frames):
        chain = [before]
        for f in frames:
            if not np.array_equal(f, chain[-1]):
                chain.append(f)
        return chain

    def _transient(chain):
        mask = sum((a != b).astype(int) for a, b in zip(chain, chain[1:])) >= 2
        if not mask.any():
            return 0, None
        rr, cc = np.nonzero(mask)
        return int(mask.sum()), [int(rr.min()), int(cc.min()), int(rr.max()), int(cc.max())]

    fake.collapse_frames, fake.transient_cells = _collapse, _transient
    sys.modules["arc3.percept"] = fake
    arc3.percept = fake

from arc3.sandbox import PythonTool, truncate_middle  # noqa: E402

MOVES = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}


class FakeSession:
    """8x8 board, player (3) moves; walls at the edge are no-ops; goal (4) completes the level;
    lava (2) is game over; SPACE flashes a cell (animation with a transient pixel)."""

    def __init__(self, level=1, guards_from_level=1, pos=(4, 4), **cfg):
        self.config = GameConfig(game_id="secret-game", guards_from_level=guards_from_level, **cfg)
        self.level, self.pos = level, pos
        self.history: list[Transition] = []
        self.valid_actions = ["UP", "DOWN", "LEFT", "RIGHT", "SPACE", "MOUSE"]
        self.extra = {}  # (r, c) -> color
        self.previous = None
        self.current = self._obs(GameState.NOT_FINISHED, 0)

    def _grid(self):
        g = np.zeros((8, 8), np.uint8)
        for (r, c), v in self.extra.items():
            g[r, c] = v
        g[self.pos] = 3
        return g

    def _obs(self, state, step, frames=None):
        g = self._grid()
        return Observation(g, frames or [g], state, self.level - 1, 3, list(self.valid_actions),
                           step, game_id="secret-game")

    def step(self, a):
        before = self.current
        frames = None
        if a.name in MOVES:
            dr, dc = MOVES[a.name]
            r, c = self.pos[0] + dr, self.pos[1] + dc
            if 0 <= r < 8 and 0 <= c < 8:
                self.pos = (r, c)
        elif a.name == "MOUSE":
            self.extra[(a.row, a.col)] = 1
        elif a.name == "SPACE":
            flash = before.grid.copy()
            flash[0, 0] = 9
            frames = [flash, before.grid.copy()]
        hit = self.extra.get(self.pos)
        state = GameState.GAME_OVER if hit == 2 else GameState.NOT_FINISHED
        after = self._obs(state, before.step + 1, frames)
        changed = not np.array_equal(before.grid, after.grid)
        res = ActionResult(a, True, changed, changed, hit == 4, hit == 2, False,
                           len(after.frames) > 1, before=before.grid, after=after.grid,
                           frames=after.frames)
        self.history.append(Transition(len(self.history), a, before, after, res))
        self.previous, self.current = before, after
        return res


def test_globals_present_and_no_hidden_state():
    tool = PythonTool(FakeSession())
    r = tool.run("print(sorted(k for k in dir() if not k.startswith('__')))\n"
                 "print(current_frame.shape, current_frame.grid[4,4], previous_frame, valid_actions)\n"
                 "print(current_frame.crop(3,3,6,6).ascii)\n"
                 "result = current_frame.step")
    assert r.error is None, r.error
    for name in ("current_frame", "transitions", "frame_diff", "action", "last_animation_frames"):
        assert name in r.stdout
    assert "(8, 8) 3 None" in r.stdout and "result = 0" in r.stdout
    assert COLOR_LETTERS[0] + COLOR_LETTERS[3] + COLOR_LETTERS[0] in r.stdout
    assert "secret-game" not in r.stdout
    r = tool.run("current_frame.grid[0,0] = 5")
    assert "read-only" in r.error


def test_batch_refreshes_globals_and_frame_diff():
    s = FakeSession(level=1, guards_from_level=5)
    r = PythonTool(s).run(
        "seen = []\n"
        "for a in ['UP', 'UP', {'action': 'MOUSE', 'row': 0, 'col': 0}]:\n"
        "    action([a])\n"
        "    seen.append(int(current_frame.grid[2, 4] == 3))\n"
        "print(seen, len(transitions), last_transition.action, frame_diff())\n"
        "print(last_action_call_result['executed'])")
    assert r.error is None, r.error
    assert "[0, 1, 1] 3 MOUSE(0,0) {'changed_cell_count': 1" in r.stdout
    assert len(r.actions) == 3


def test_noop_stop_and_stale_guard():
    s = FakeSession(pos=(0, 4))
    r = PythonTool(s).run("res = action(['UP', 'DOWN', 'DOWN'])\nprint(res['stop_reason'], "
                          "res['executed'], res['skipped'])\naction(['DOWN'])")
    assert "no_op 1 2" in r.stdout
    assert "StaleStateError" in r.error and len(r.actions) == 1


def test_guards_off_below_level_but_terminal_latch_always_on():
    s = FakeSession(level=1, guards_from_level=2, pos=(0, 4))
    s.extra[(0, 2)] = 2
    r = PythonTool(s).run("action(['UP', 'LEFT'])\nprint(action(['LEFT'])['game_over'])\n"
                          "action(['DOWN'])")
    assert r.stdout.strip() == "True"
    assert "TerminalStateError" in r.error and len(r.actions) == 3


def test_level_complete_stops_batch():
    s = FakeSession()
    s.extra[(4, 5)] = 4
    r = PythonTool(s).run("res = action(['RIGHT', 'RIGHT'])\n"
                          "print(res['stop_reason'], res['level_completed'], res['skipped'])")
    assert "level_completed True 1" in r.stdout


def test_reset_refused_and_action_cap():
    tool = PythonTool(FakeSession(guards_from_level=9, expose_reset=False), max_actions_per_call=3)
    tool.session.valid_actions.append("RESET")
    assert "RESET is not available" in tool.run("action(['RESET'])").error
    r = tool.run("print(action(['UP'] * 5)['stop_reason'])")
    assert "max_actions_per_call" in r.stdout and len(r.actions) == 3


def test_animation_summary():
    tool = PythonTool(FakeSession())
    r = tool.run("print(action(['SPACE'])['animation'], len(last_animation_frames))")
    assert "'transient_pixels': 1" in r.stdout and "[0, 0, 0, 0]" in r.stdout and r.stdout.endswith("3\n")


def test_output_middle_truncation():
    r = PythonTool(FakeSession()).run("print('A' * 10000 + 'B' * 10000)")
    assert r.truncated and r.stdout.startswith("AAA") and r.stdout.rstrip().endswith("BBB")
    assert "chars truncated" in r.stdout and len(r.stdout) < 12_100
    assert truncate_middle("short") == ("short", False)


def test_timeout_discards_and_blocks_actions():
    tool = PythonTool(FakeSession(), timeout=0.3)
    r = tool.run("while True:\n    pass")
    assert "TimeoutError" in r.error
    assert tool.run("print(1)").stdout == "1\n"


def test_imports_and_builtins_blocked():
    tool = PythonTool(FakeSession())
    assert "not allowed" in tool.run("import os").error
    assert "not allowed" in tool.run("import subprocess").error
    assert "not allowed" in tool.run("from os import path").error
    assert "NameError" in tool.run("open('/etc/passwd')").error
    assert tool.run("import collections, math\nprint(math.sqrt(4))").stdout == "2.0\n"
    assert tool.run("import numpy as np\nprint(int(np.sum(current_frame.grid)))").stdout == "3\n"


def test_exception_mid_batch_keeps_actions():
    r = PythonTool(FakeSession(guards_from_level=9)).run(
        "def boom(x):\n    return 1 / x\naction(['UP', 'LEFT'])\nboom(0)")
    assert len(r.actions) == 2
    assert "line 4" in r.error and "line 2, in boom" in r.error and "ZeroDivisionError" in r.error


def test_retained_functions():
    tool = PythonTool(FakeSession())
    r = tool.run(
        "import math\nimport heapq as hq\nK = 3\n"
        "def dist(a, b=(0, 0)):\n    return math.hypot(a[0] - b[0], a[1] - b[1])\n"
        "def player():\n    return tuple(int(v) for v in zip(*(current_frame.grid == 3).nonzero()).__next__())\n"
        "def twice(a):\n    return 2 * dist(a)\n"
        "def uses_k():\n    return K\n"
        "def annotated(x: int):\n    return x\n"
        "def nonliteral(x=len):\n    return x\n"
        "def bumps():\n    global K\n    K += 1\n"
        "def heap(xs):\n    return hq.nsmallest(1, xs)\n"
        "def action(x):\n    return x\n")
    assert r.error is None, r.error
    assert r.retained_functions == ["dist", "heap", "player", "twice"]
    assert set(r.rejected_functions) == {"uses_k", "annotated", "nonliteral", "bumps", "action"}
    assert "K" in r.rejected_functions["uses_k"]
    r = tool.run("print(twice((3, 4)), player(), heap([5, 2]))")
    assert r.stdout == "10.0 (4, 4) [2]\n", r.error
    # redefinition replaces; a dependent of a now-rejected function is dropped too
    r = tool.run("def dist(a, b=K):\n    return 0\n")
    assert "dist" in r.rejected_functions and "twice" in r.rejected_functions
    assert "NameError" in tool.run("twice((1, 1))").error


# ------------------------------------------------------------------ fix-review-1 additions

def test_memory_persists_across_calls_but_variables_do_not():
    tool = PythonTool(FakeSession())
    r = tool.run("memory['plan'] = ['UP', 'LEFT']\nx = 5\n"
                 "def first_step():\n    return memory['plan'][0]")
    assert r.error is None, r.error
    assert r.retained_functions == ["first_step"]          # `memory` counts as provided
    r = tool.run("print(memory, 'x' in dir(), first_step())")
    assert r.error is None, r.error
    assert "{'plan': ['UP', 'LEFT']} False UP" in r.stdout
    tool.run("memory = {}")                                  # rebinding stays local to that call
    assert "plan" in tool.run("print(memory)").stdout


def test_per_action_diff_line_and_tool_text():
    from arc3 import prompts
    s = FakeSession(guards_from_level=5)
    tool = PythonTool(s)
    r = tool.run("res = action(['UP', 'SPACE'])\nprint(res['per_action'][0]['diff'])")
    assert r.error is None, r.error
    assert "moved: 1 (A (4,4)->(3,4))" in r.stdout
    assert r.actions[0].diff.startswith("moved: 1") and len(r.actions[0].diff) <= 200
    assert r.actions[1].diff == "no change"                   # SPACE flash settles back
    text = prompts.tool_result_text(r.stdout, r.error, r.actions)
    assert "[effects]" in text and "  1 UP: moved: 1" in text


def test_tool_text_diff_lines_bounded_for_long_batches():
    from arc3 import prompts
    from arc3.types import Action
    acts = [ActionResult(Action("UP"), True, True, True, False, False, False, False,
                         diff=f"moved: 1 (#{i})") for i in range(10)]
    text = prompts.tool_result_text("", None, acts)
    assert "  1 UP: moved: 1 (#0)" in text and "  10 UP: moved: 1 (#9)" in text
    assert "(#4)" not in text and "6 actions omitted" in text


class ResetSession(FakeSession):
    """Moving onto (2, 4) silently puts the player back at (4, 4) (a life lost, no game over)."""

    def step(self, a):
        res = super().step(a)
        if self.pos == (2, 4):
            res.level_reset = True
            res.message = "LEVEL RESET: ..."
        return res


def test_level_reset_stops_batch_like_game_over():
    tool = PythonTool(ResetSession(guards_from_level=5))
    r = tool.run("res = action(['UP', 'UP', 'LEFT', 'LEFT'])\n"
                 "print(res['stop_reason'], res['executed'], res['level_reset'], "
                 "res['per_action'][-1]['level_reset'])\naction(['LEFT'])")
    assert "level_reset 2 True True" in r.stdout
    assert "TerminalStateError" in r.error and len(r.actions) == 2
    rec = tool.wm._rec()                                       # wm sees the reset as a death
    assert rec.transitions[-1][3] == "game_over"


def test_timeout_actually_stops_a_pure_python_loop():
    import threading
    before = {t.name for t in threading.enumerate()}
    tool = PythonTool(FakeSession(), timeout=0.3)
    r = tool.run("while True:\n    pass")
    assert "TimeoutError" in r.error and "snippet stopped" in r.error
    assert "sandbox" not in {t.name for t in threading.enumerate()} - before


def test_runaway_list_append_is_killed_before_host_memory_runs_out():
    # The run-#4 killer: `while seen[cur]: path.append(...)` with `cur` never advancing.
    tool = PythonTool(FakeSession(), timeout=0.5, mem_growth_limit=64 << 20)
    r = tool.run("path = []\nwhile True:\n    path.append(bytearray(1 << 20))")
    assert ("MemoryError" in r.error or "TimeoutError" in r.error) and "snippet stopped" in r.error
    assert tool.run("print(1)").stdout == "1\n"


def test_snippet_kill_cannot_be_swallowed_by_bare_except():
    tool = PythonTool(FakeSession(), timeout=0.3)
    r = tool.run("while True:\n    try:\n        pass\n    except Exception:\n        pass")
    assert "snippet stopped" in r.error
