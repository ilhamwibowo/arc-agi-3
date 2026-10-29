"""Fixes from docs/trace-review-run3-weak.md: RESET, budget bar, wm.check on a real game, trims,
stop reasons and small sandbox additions. Real games run offline through arc3.env; no model."""
import random
from pathlib import Path

import numpy as np

from arc3 import prompts
from arc3.context import ContextManager
from arc3.env import Env, list_public_games
from arc3.percept import BudgetTracker
from arc3.sandbox import PythonTool
from arc3.session import GameSession
from arc3.types import Action, GameConfig

ENVS = Path(__file__).resolve().parents[1] / "data" / "environments"


def _session(prefix, tmp_path, **cfg):
    gid = next(g for g in list_public_games(ENVS) if g.startswith(prefix))
    env = Env.offline(gid, ENVS, seed=0, recordings_dir=tmp_path / "toolkit")
    return GameSession(env, GameConfig(game_id=gid, **cfg))


def _interior(g):
    return np.asarray(g)[4:-4, 4:-4]


# ------------------------------------------------------------------ 1. RESET

def test_reset_restarts_level_continues_batch_and_marks_wm_record(tmp_path):
    s = _session("ls20", tmp_path)
    assert "RESET" in s.valid_actions
    tool = PythonTool(s)
    start = s.current.grid
    # A RESET before any move would change nothing: refused, and no action is spent.
    r = tool.run("print(action(['RESET'])['stop_reason'])")
    assert r.stdout.strip() == "reset_noop" and s.actions == 0
    # fix-review-5: a model RESET no longer ends the batch; the moves after it run from the start
    # board (run #5 lost 6 calls to action([RESET, *replay]) raising TerminalStateError).
    r = tool.run("res = action(['UP', 'UP', 'RESET', 'DOWN'])\n"
                 "print(res['executed'], res['skipped'], res['stop_reason'])")
    assert r.error is None, r.error
    assert r.stdout.strip() == "4 0 None"
    assert np.array_equal(_interior(r.actions[2].after), _interior(start))
    assert r.actions[2].action.name == "RESET" and r.actions[2].stop_reason is None
    assert r.actions[3].action.name == "DOWN" and r.actions[3].executed
    text = prompts.tool_result_text(r.stdout, r.error, r.actions)
    assert "RESET: level restarted from its start board" in text and "stopped" not in text
    # A RESET right after the DOWN is accepted (one move since the restart); the next one is a
    # no-op and costs nothing.
    r = tool.run("print(action('RESET')['stop_reason'])")
    assert r.stdout.strip() == "None" and s.actions == 5
    r = tool.run("print(action('RESET')['stop_reason'])")
    assert r.stdout.strip() == "reset_noop" and s.actions == 5
    # wm: moves before a RESET stay in this level's record; each RESET is a restart boundary
    # and the planner starts from the post-RESET board.
    rec = tool._wm_record()
    assert len(rec.transitions) == 3 and rec.restarts == [2, 3]
    tool.wm.register(lambda st, a: st, lambda st: "ongoing",
                     init=lambda g, lv: {"grid": g.copy(), "fresh": True})
    cur = tool.wm.current_state()
    assert cur["fresh"] and np.array_equal(cur["grid"], s.current.grid)
    # Moves after the RESET continue the same level record.
    tool.run("action(['DOWN'])")
    rec = tool._wm_record()
    assert len(rec.transitions) == 4 and rec.restarts == [2, 3]


def test_reset_hidden_when_not_exposed(tmp_path):
    s = _session("ls20", tmp_path, expose_reset=False)
    assert "RESET" not in s.valid_actions
    assert "RESET is not available" in PythonTool(s).run("action('RESET')").error


# ------------------------------------------------------------------ 2. budget bar

def test_ls20_budget_readout_counts_down_to_the_reset(tmp_path):
    # ls20's bar empties after 42 moves and the 43rd puts the level back to its start.
    s = _session("ls20", tmp_path)
    left = []
    for i in range(43):
        r = s.step(Action("UP" if i % 2 else "LEFT"))
        left.append(r.budget_left)
    assert left[:2] == [None, None]                      # needs 3 shrinks first
    assert left[2:42] == [42 - n for n in range(3, 43)]  # exact: 1 cell per action
    assert "bottom band" in s.history[10].result.budget_bar
    assert s.history[42].result.level_reset and left[42] is None   # detector restarts
    opener = prompts.build_opener([s.history[10].result], step=11, level=0, win_levels=7,
                                  valid_actions=s.valid_actions)
    assert "Move budget: ~31 actions left on this level (bar on the bottom edge" in opener
    assert "RESET first" in opener
    s.close()


def test_budget_readout_resets_after_reset(tmp_path):
    s = _session("ls20", tmp_path)
    for name in ["UP", "LEFT", "UP", "LEFT"]:
        s.step(Action(name))
    assert s.history[-1].result.budget_left is not None
    assert s.step(Action("RESET")).budget_left is None
    assert [s.step(Action(n)).budget_left for n in ("UP", "LEFT", "UP")][-1] is not None


def test_no_budget_false_positive_without_a_bar(tmp_path):
    for prefix in ("vc33", "sb26"):
        s = _session(prefix, tmp_path)
        rng = random.Random(3)
        for _ in range(20):
            names = [n for n in s.valid_actions if n not in ("RESET", "UNDO")]
            n = rng.choice(names)
            a = Action("MOUSE", rng.randrange(64), rng.randrange(64)) if n == "MOUSE" else Action(n)
            r = s.step(a)
            s.auto_reset_if_needed()
            assert r.budget_left is None, (prefix, r.budget_bar)


def test_budget_tracker_fractional_rate_and_growth():
    def board(bar):
        g = np.zeros((16, 16), np.uint8)
        g[0, :bar] = 9
        return g
    t = BudgetTracker(border=2)
    t.reset(board(12))
    # one cell every 2 actions -> about 2 actions per remaining cell
    out = [t.update(board(b)) for b in (12, 11, 11, 10, 10, 9)]
    assert out[-1] == (18, "top band, color 9, shrinks 1 per ~2.0 actions")
    assert t.update(board(12)) is None      # a bar that grows is not a budget (refill: reset)


# ------------------------------------------------------------------ 3. wm.check on a public game

READ_STEP = """
def read(g):  # top-left of the orange part of the player block
    r, c = np.argwhere(g == 12).min(axis=0)
    return (int(r), int(c))
def step(s, a):  # walls ignored on purpose
    dr, dc = {'UP': (-5, 0), 'DOWN': (5, 0), 'LEFT': (0, -5), 'RIGHT': (0, 5)}.get(a, (0, 0))
    return (s[0] + dr, s[1] + dc)
"""


def test_check_and_plan_abstract_on_ls20(tmp_path):
    s = _session("ls20", tmp_path)
    tool = PythonTool(s)
    r = tool.run(READ_STEP + "action(['UP', 'UP', 'LEFT', 'RIGHT', 'UP'])\n"
                 "c = wm.check(read, step)\nprint(c['summary'])\nmemory['c'] = c")
    assert r.error is None, r.error
    c = tool.memory["c"]
    # LEFT is blocked by a wall at that point; the wall-free step mispredicts exactly that move.
    assert c["checked"] == 5 and c["correct"] == 4 and not c["ok"]
    assert c["first_mismatch"]["action"] == "LEFT" and c["by_action"]["LEFT"] == "0/1"
    assert c["unseen_actions"] == ["DOWN"]
    assert s.actions == 5                         # check itself costs no game action
    # Plan back down two cells over a path already walked, then play it for real.
    r = tool.run("here = read(current_frame.grid)\n"
                 "p = wm.plan_abstract(read, step, lambda st: st == (here[0] + 10, here[1]))\n"
                 "action(p['actions'])\n"
                 "print(p['plan'], read(current_frame.grid) == (here[0] + 10, here[1]))")
    assert r.error is None, r.error
    assert r.stdout.strip() == "['DOWN', 'DOWN'] True"


# ------------------------------------------------------------------ 4. trims

def _block(ctx, n_chars):
    ctx.add_user("u" * 100)
    ctx.add_assistant("", reasoning="r" * n_chars, tool_calls=[{"id": "c", "type": "function",
                      "function": {"name": "python", "arguments": "{}"}}])
    ctx.add_tool("c", "t" * 100)


def test_trim_strips_old_reasoning_before_dropping_turns():
    ctx = ContextManager("sys", keep_reasoning_turns=6)
    ctx.tm.chars_per_token = 1.0
    for _ in range(10):
        _block(ctx, 3000)
    n_before = len(ctx.messages)
    assert ctx.trim(budget_tokens=30_000, drain_tokens=8_000) == 4   # 4 old reasoning blocks
    assert len(ctx.messages) == n_before and not ctx.trimmed_since_last_user
    assert ctx.last_trim == {"stripped": 4, "dropped": 0}
    assistants = [m for m in ctx.messages if m.role == "assistant"]
    assert [bool(m.reasoning) for m in assistants] == [False] * 4 + [True] * 6
    # Still too big: now whole blocks go, and the next user message gets the notice.
    ctx.trim(budget_tokens=10_000, drain_tokens=4_000)
    assert ctx.last_trim["dropped"] > 0 and ctx.trimmed_since_last_user
    ctx.add_user("nudge")                     # only the agent's opener clears the flag
    assert ctx.trimmed_since_last_user


def test_trim_notice_and_memory_summary(tmp_path):
    s = _session("ls20", tmp_path)
    tool = PythonTool(s)
    assert tool.memory_summary() == "(empty)"
    tool.run("memory['rules'] = 'walls black; bar = budget'\nmemory['plan'] = list(range(100))")
    summary = tool.memory_summary(limit=120)
    assert summary.startswith("'rules': 'walls black; bar = budget'") and len(summary) <= 120
    note = prompts.trim_notice(summary, ["read", "step"])
    assert note.startswith("Trim notice: older turns were dropped. Your memory dict still holds: ")
    assert "Retained functions: read, step." in note


# ------------------------------------------------------------------ 5/6. stop reasons, sandbox extras

def test_stop_reason_reaches_actions_line_and_history(tmp_path):
    s = _session("ls20", tmp_path, guards_from_level=1)
    tool = PythonTool(s)
    # After UP, UP the player stands under a wall on its left: LEFT changes nothing inside.
    r = tool.run("print(action(['UP', 'UP', 'LEFT', 'DOWN', 'DOWN'])['stop_reason'])")
    assert r.stdout.strip() == "no_op"
    assert r.actions[-1].stop_reason == "no_op" and r.actions[-1].skipped == 2
    text = prompts.tool_result_text(r.stdout, r.error, r.actions)
    assert "stopped: no_op, 2 requested action(s) not run" in text
    assert "then stopped (no_op)" in text
    opener = prompts.build_opener(r.actions, step=3, level=0, win_levels=7,
                                  valid_actions=s.valid_actions)
    assert "Guard stop (no_op, 2 requested action(s) not run)" in opener
    r = tool.run("print(history[-1].stop_reason, history[-1].step, history[-1].level_reset, "
                 "history[-1]['result']['level_reset'])")
    assert r.stdout.strip() == "no_op 3 False False"


def test_action_cap_stop_reason_set_on_last_result(tmp_path):
    s = _session("ls20", tmp_path)
    tool = PythonTool(s, max_actions_per_call=2)
    r = tool.run("print(action(['UP', 'UP', 'DOWN'])['stop_reason'])")
    assert r.stdout.strip() == "max_actions_per_call"
    assert r.actions[-1].stop_reason == "max_actions_per_call" and r.actions[-1].skipped == 1


def test_preloaded_modules_and_source(tmp_path):
    s = _session("ls20", tmp_path)
    tool = PythonTool(s)
    r = tool.run("def dist(a, b):\n    return math.hypot(a[0] - b[0], a[1] - b[1])\n"
                 "print(dist((0, 0), (3, 4)), collections.Counter('aab')['a'], "
                 "list(itertools.islice(itertools.count(), 3)))")
    assert r.error is None, r.error
    assert r.stdout.strip() == "5.0 2 [0, 1, 2]" and r.retained_functions == ["dist"]
    r = tool.run("print(dist((0, 0), (6, 8)))\nsource('dist')\nsource('nope')")
    assert r.error is None, r.error
    lines = r.stdout.splitlines()
    assert lines[0] == "10.0" and lines[1] == "def dist(a, b):"
    assert "no retained function 'nope'; retained: dist" in r.stdout


def test_level_reset_wording_is_neutral(tmp_path):
    s = _session("ls20", tmp_path)
    for i in range(43):
        s.step(Action("UP" if i % 2 else "LEFT"))
    res = s.history[42].result
    assert res.level_reset and "deliberate restart mechanic" in res.message
    assert "life" not in res.message
    opener = prompts.build_opener([res], step=43, level=0, win_levels=7,
                                  valid_actions=s.valid_actions)
    assert "Causes: a fatal move, an exhausted move budget, or a deliberate restart" in opener


# ------------------------------------------------------------------ pinned notes (solved-level memory)

def test_pinned_rules_survive_and_show_in_opener(tmp_path):
    from arc3.sandbox import PINNED_CHARS
    s = _session("ls20", tmp_path)
    tool = PythonTool(s)
    assert tool.pinned_block() == ""                      # nothing written yet
    tool.run("memory['rules'] = 'arrows move the blue square; green cell ends the level'")
    block = tool.pinned_block()
    assert block.startswith("Rules you have written down (memory['rules']): arrows move")
    tool.pin_level(1)
    assert tool.pinned == ["Level 1 solved with rules: arrows move the blue square; green cell ends the level"]
    # identical current rules are not repeated as a snapshot
    assert tool.pinned_block().count("arrows move") == 1
    tool.run("memory['rules'] = 'level 2: red walls kill'")
    block = tool.pinned_block()
    assert "red walls kill" in block and "Level 1 solved with rules: arrows move" in block
    tool.pin_level(1)                                     # re-pin same level replaces, no duplicate
    assert len(tool.pinned) == 1 and tool.pinned[0].endswith("red walls kill")
    tool.run("memory['rules'] = 'x' * 5000")
    assert len(tool.rules_text()) == 1500 and tool.rules_text().endswith("...")
    assert len(tool.pinned_block()) <= PINNED_CHARS + 100
    opener = prompts.build_opener([], step=3, level=1, win_levels=3, valid_actions=["UP"],
                                  pinned=tool.pinned_block())
    assert "[pinned notes]" in opener and "Level 1 solved" in opener
    assert "[pinned notes]" not in prompts.build_opener([], step=3, level=1, win_levels=3,
                                                        valid_actions=["UP"])
    assert "pinned notes below" in prompts.trim_notice("(empty)")


def test_stall_facts_from_a_real_game(tmp_path):
    """stall_facts never raises and reports clicked cells whose surroundings changed."""
    s = _session("vc33", tmp_path)                    # click game
    tool = PythonTool(s)
    facts = tool.stall_facts()
    assert set(facts) <= {"changed_since_click", "new_vs_previous_level", "goals",
                          "changed_error", "new_error", "goals_error"}
    assert facts.get("new_vs_previous_level") == []    # level 1: no previous level
    valid = s.valid_actions
    if "MOUSE" in valid:
        tool.run("action([('MOUSE', 10, 10)])") if False else None
    # synthetic record check for the two static helpers
    from arc3.wm import Record
    g0 = np.zeros((64, 64), np.int8); g1 = g0.copy(); g1[0:4, 10:21] = 7
    rec = Record(level_start_grid=g1, previous=[Record(level_start_grid=g0)])
    assert PythonTool._new_vs_previous_level(rec) == [(7, 44, 0, 3, 10, 20)]
    before = g1.copy(); after = g1.copy(); after[5:7, 5:7] = 3
    rec.transitions.append((before, {"action": "MOUSE", "row": 5, "col": 9}, after, "ongoing"))
    assert PythonTool._changed_since_click(rec) == [(5, 9)]       # 9x9 window around (5,9) changed
    rec.transitions.append((after, {"action": "MOUSE", "row": 40, "col": 40}, after, "ongoing"))
    assert PythonTool._changed_since_click(rec) == [(5, 9)]       # (40,40) unchanged: not listed
