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


# ------------------------------------------------------------------ 3. opener evidence lines

def _click_record():
    """A click game: two 2x2 buttons (colour 7) and one 1x3 bar (colour 9); clicking a button
    moves it (a toggled copy), so the copy must count as tried."""
    from arc3.wm import Record
    def board(btn_a_col):
        g = np.zeros((12, 12), np.uint8)
        g[2:4, btn_a_col:btn_a_col + 2] = 7
        g[2:4, 8:10] = 7
        g[8, 3:6] = 9
        return g
    rec = Record(level_start_grid=board(2), level=1, available_actions=["MOUSE"])
    rec.transitions = [(board(2), {"action": "MOUSE", "row": 2, "col": 2}, board(5), "ongoing")]
    rec.live_grid = board(5)
    return rec


def test_unclicked_objects_use_colour_and_shape_identity():
    from arc3.wm import WM
    wm = WM(_click_record(), hud_border=0)
    un = wm.unclicked_objects()
    # Both 2x2 buttons share the clicked identity; only the bar is left.
    assert un["total"] == 1 and un["objects"] == [{"color": 9, "size": 3, "row": 8, "col": 3}]
    rec = _click_record()
    rec.transitions = []
    assert WM(rec, hud_border=0).unclicked_objects(max_items=2)["total"] == 3
    assert len(WM(rec, hud_border=0).unclicked_objects(max_items=2)["objects"]) == 2


def test_blocked_moves_report_target_and_entered_colours():
    from arc3.wm import WM
    rec, board = _hud_record()
    g = board(4, 7); g[6, 3] = 5                      # a wall left of the avatar
    rec.transitions.append((g, {"action": "LEFT", "row": None, "col": None}, g, "ongoing"))
    g2 = g.copy(); g2[0, :] = 0; g2[0, :6] = 9
    rec.transitions.append((g, {"action": "LEFT", "row": None, "col": None}, g2, "ongoing"))
    rec.live_grid = g2
    bl = WM(rec, hud_border=1).blocked_moves()
    # The fixture's UP only shrank the bar: refused once, and never seen to move, so no colours.
    assert bl["blocked"] == {"LEFT": {"count": 2, "colors": [5]}, "UP": {"count": 1, "colors": []}}
    assert bl["entered"] == [0]
    assert "LEFT" not in WM(_hud_record()[0], hud_border=1).blocked_moves()["blocked"]
    assert WM(_click_record(), hud_border=0).blocked_moves() == {"blocked": {}, "entered": []}


def test_novelty_counts_states_by_view_hash():
    from arc3.wm import WM
    rec, board = _hud_record()
    g = board(4, 7)
    for _ in range(3):      # a launch button that only shrinks the bar
        g2 = g.copy(); g2[0, :] = 0; g2[0, :max(0, int((g[0] == 9).sum()) - 1)] = 9
        rec.transitions.append((g, {"action": "MOUSE", "row": 6, "col": 4}, g2, "ongoing"))
        g = g2
    rec.live_grid = g
    nv = WM(rec, hud_border=1).novelty(window=20)
    assert nv["actions"] == 6 and nv["window"] == 6
    assert nv["states"] == 2                       # interior: avatar at 4 or 5
    assert nv["new_in_window"] == 1
    assert nv["top_noop"] == ("MOUSE(6,4)", 3)
    assert WM(rec, hud_border=1).novelty(window=2)["new_in_window"] == 0


def test_evidence_lines_format_and_skip_when_empty():
    ev = {"unclicked": {"objects": [{"color": 14, "size": 8, "row": 30, "col": 27},
                                    {"color": 4, "size": 16, "row": 26, "col": 44}], "total": 11},
          "blocked": {"blocked": {"LEFT": {"count": 4, "colors": [4]}, "UP": {"count": 1, "colors": [-1]}},
                      "entered": [9, 2, 8]},
          "novelty": {"actions": 97, "states": 31, "window": 20, "new_in_window": 2,
                      "top_noop": ("MOUSE(58,46)", 15)}}
    lines = prompts.evidence_lines(ev, ["UP", "MOUSE"])
    assert lines == [
        "Objects never clicked on this level: G 8px at (30,27), D 16px at (26,44) (+9 more).",
        "Blocked moves: LEFT refused 4x (target cells colour D), UP refused 1x (target cells "
        "colour edge); successful moves entered B, S, R.",
        "This level: 97 actions, 31 distinct states; last 20 actions reached 2 new states; "
        "MOUSE(58,46) was a no-op 15 times.",
    ]
    assert all(len(line) < 240 for line in lines)
    # No MOUSE: the object line is dropped. Nothing refused: no blocked line. No actions: no novelty.
    assert len(prompts.evidence_lines(ev, ["UP"])) == 2
    empty = {"unclicked": {"objects": [], "total": 0}, "blocked": {"blocked": {}, "entered": [0]},
             "novelty": {"actions": 0, "states": 1, "window": 0, "new_in_window": 0, "top_noop": None}}
    assert prompts.evidence_lines(empty, ["MOUSE"]) == []
    assert prompts.evidence_lines(None, ["MOUSE"]) == []
    opener = prompts.build_opener([], step=5, level=0, win_levels=3, valid_actions=["MOUSE"],
                                  evidence=ev)
    assert opener.index("Objects never clicked") < opener.index("Current state")


def test_sandbox_evidence_and_agent_opener_on_fake_session():
    from agent_fakes import FakeSession as AgentFakeSession, fake_png
    from arc3.agent import Agent, AgentConfig
    from arc3.llm import FakeLLM
    import test_sandbox

    # Sandbox side: an 8x8 click/keyboard fake game, HUD band off so the whole board counts.
    s = test_sandbox.FakeSession(guards_from_level=9, hud_border=0)
    tool = PythonTool(s)
    tool.run("action(['LEFT', 'LEFT', 'LEFT', 'LEFT', 'LEFT', {'action': 'MOUSE', 'row': 1, 'col': 1}])")
    ev = tool.evidence()
    assert ev["novelty"]["actions"] == 6 and ev["novelty"]["top_noop"] == ("LEFT", 1)
    assert ev["blocked"]["blocked"] == {"LEFT": {"count": 1, "colors": [-1]}}
    assert ev["blocked"]["entered"] == [0]
    # The click painted (1,1): it hit background, so the new cell was never itself clicked.
    assert ev["unclicked"]["objects"] == [{"color": 1, "size": 1, "row": 1, "col": 1},
                                          {"color": 3, "size": 1, "row": 4, "col": 0}]
    lines = prompts.evidence_lines(ev, s.valid_actions)
    assert lines[0] == "Objects never clicked on this level: V 1px at (1,1), A 1px at (4,0)."
    assert lines[1] == "Blocked moves: LEFT refused 1x (target cells colour edge); successful moves entered W."
    assert lines[2] == "This level: 6 actions, 6 distinct states; last 6 actions reached 5 new states; LEFT was a no-op 1 time."

    # Agent side: whatever the tool's evidence() returns lands in the opener; a tool without it is fine.
    class EvidenceTool:
        def __init__(self, session):
            from agent_fakes import FakeTool
            self.inner = FakeTool(session)
        def run(self, code):
            return self.inner.run(code)
        def evidence(self):
            return {"novelty": {"actions": 3, "states": 2, "window": 3, "new_in_window": 1, "top_noop": None}}

    llm = FakeLLM([FakeLLM.code("action(['RIGHT'])"), FakeLLM.code("action(['RIGHT'])"),
                   FakeLLM.code("action(['SPACE'])")])
    agent = Agent(llm, AgentConfig(retry_sleep=0), tool_factory=EvidenceTool,
                  render_png=lambda g, sc: fake_png())
    agent.play_game(AgentFakeSession())
    users = [m.content for m in agent.context.messages if m.role == "user"]
    assert any("This level: 3 actions, 2 distinct states; last 3 actions reached 1 new states." in u
               for u in users)


# ------------------------------------------------------------------ 4. cheaper cuts

def test_output_cap_and_short_burst_line():
    from arc3.agent import AgentConfig
    assert AgentConfig().output_tokens == 6000
    p = prompts.build_system_prompt()
    assert "Think in short bursts: when a hypothesis can be checked with code, call the tool" in p
    # Items 2-4 of fix-review-5 together added under 500 chars to the run #5 prompt (8803); the
    # pinned-notes instruction (memory['rules'], 2026-10-29) adds ~330 more. Keep the prompt lean.
    assert len(p) - 8803 < 900


# ------------------------------------------------------------------ 5. helper retention

def test_retained_functions_freeze_snippet_level_data():
    import test_sandbox
    tool = PythonTool(test_sandbox.FakeSession())
    r = tool.run(
        "WALLS = current_frame.grid == 5\n"
        "DELTA = {'UP': (-1, 0), 'DOWN': (1, 0)}\n"
        "ORDER = ('UP', 'DOWN')\n"
        "SEEN = {(1, 2)}\n"
        "def blocked(r, c):\n    return bool(WALLS[r, c])\n"
        "def move(a, d=DELTA):\n    return d[a]\n"
        "def first(o=ORDER):\n    return o[0], (1, 2) in SEEN\n")
    assert r.error is None, r.error
    assert r.retained_functions == ["blocked", "first", "move"]
    assert r.rejected_functions == {}
    b = tool.retained["blocked"]["bound"]
    assert set(b) == {"WALLS"} and not b["WALLS"].flags.writeable
    assert tool.retained["first"]["bound"]["ORDER"] == ("UP", "DOWN")      # tuples stay tuples
    assert tool.retained["first"]["bound"]["SEEN"] == {(1, 2)}
    assert tool.retained["move"]["deps"] == set()
    # Next call: the frozen copies are back, the originals are gone from the snippet's own scope
    # only in the sense that the snippet never set them; the function reads the copy.
    r = tool.run("print(blocked(0, 0), move('UP'), first(), sorted(DELTA))")
    assert r.error is None, r.error
    assert r.stdout == "False (-1, 0) ('UP', True) ['DOWN', 'UP']\n"
    # Rebinding a name the function reads at call time affects that call only (a default such
    # as move's `d=DELTA` was bound when the function was restored and does not change).
    r = tool.run("SEEN = set()\nDELTA = {'UP': 'changed'}\nprint(first(), move('UP'))")
    assert r.stdout == "('UP', False) (-1, 0)\n"
    assert tool.run("print(first())").stdout == "('UP', True)\n"
    # source() names the frozen values.
    out = tool.run("source('blocked')").stdout
    assert out.startswith("# frozen at definition: WALLS\ndef blocked")


def test_retention_refuses_unfreezable_or_oversized_values_with_a_reason():
    import test_sandbox
    tool = PythonTool(test_sandbox.FakeSession())
    r = tool.run(
        "BIG = np.zeros((300, 300), np.uint8)\n"            # 90 KB > 64 KB
        "f = lambda x: x\n"
        "FR = current_frame\n"
        "def big():\n    return BIG.sum()\n"
        "def uses_lambda(x):\n    return f(x)\n"
        "def uses_frame():\n    return FR.step\n"
        "def missing():\n    return NOPE\n")
    assert r.error is None, r.error
    assert r.retained_functions == []
    assert "over 64 KB" in r.rejected_functions["big"] and "memory[" in r.rejected_functions["big"]
    assert "a function, which cannot be frozen" in r.rejected_functions["uses_lambda"]
    assert "a FrameView, which cannot be frozen" in r.rejected_functions["uses_frame"]
    assert "NOPE which is not available later" in r.rejected_functions["missing"]
    assert all("\n" not in why for why in r.rejected_functions.values())
    # A value that fits is frozen even next to a big one in the same snippet.
    r = tool.run("SMALL = [1, 2, 3]\nBIG = np.zeros((300, 300), np.uint8)\n"
                 "def small():\n    return sum(SMALL)\n")
    assert r.retained_functions == ["small"] and tool.run("print(small())").stdout == "6\n"
    assert "frozen copies, up to 64 KB" in prompts.build_system_prompt()


# ------------------------------------------------------------------ 6. slot handover

def test_should_yield_needs_a_waiter_above_the_hysteresis():
    import threading
    from arc3.scheduler import GameProgress, PriorityGate, priority

    gate = PriorityGate(max_active=1)
    holder = GameProgress("holder", started=True, level=1, levels_total=7, actions_on_level=40)
    assert gate.acquire(holder)
    assert not gate.should_yield(holder)                       # nobody waits
    mine = priority(holder)

    def wait(p):
        t = threading.Thread(target=lambda: gate.acquire(p, timeout=5), daemon=True)
        t.start()
        return t

    near = GameProgress("near", started=True, level=1, levels_total=7, actions_on_level=30)
    assert mine < priority(near) < mine * 1.2
    t1 = wait(near)
    assert _wait_for(lambda: "near" in [w for w, _ in gate.snapshot()["waiting"]])
    assert not gate.should_yield(near)                         # not a holder
    assert not gate.should_yield(holder)                       # within hysteresis: keep it
    assert gate.should_yield(holder, hysteresis=1.0)
    far = GameProgress("far", started=True, level=4, levels_total=7)
    assert priority(far) > mine * 1.2
    t2 = wait(far)
    assert _wait_for(lambda: "far" in [w for w, _ in gate.snapshot()["waiting"]])
    assert gate.should_yield(holder)
    gate.release(holder)
    assert _wait_for(lambda: gate.holds(far))
    assert gate.should_yield(far) is False or priority(near) > priority(far) * 1.2
    gate.finish(far); gate.finish(near)
    t1.join(1); t2.join(1)
    # An unstarted waiter always outranks a holder.
    gate = PriorityGate(max_active=1)
    assert gate.acquire(holder)
    t = wait(GameProgress("new"))
    assert _wait_for(lambda: "new" in [w for w, _ in gate.snapshot()["waiting"]])
    assert gate.should_yield(holder)
    gate.release(holder); t.join(1)


def _wait_for(cond, timeout=2.0):
    import time
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.005)
    return False


class _YieldGate:
    def __init__(self):
        self.should, self.acquires, self.releases, self.holding = False, 0, 0, False

    def acquire(self, progress, timeout=None):
        if not self.holding:            # like PriorityGate: re-acquire while holding is a no-op
            self.acquires += 1
        self.holding = True
        return True

    def release(self, progress):
        self.releases += 1
        self.holding = False

    def should_yield(self, progress):
        return self.should


class _Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def test_agent_releases_at_trim_only_when_outranked_and_rotates_after_15_minutes():
    from agent_fakes import FakeSession, FakeTool
    from arc3.agent import Agent, AgentConfig
    from arc3.llm import FakeLLM
    from arc3.scheduler import GameProgress

    gate, clock = _YieldGate(), _Clock()
    llm = FakeLLM([FakeLLM.code("action(['LEFT'])") for _ in range(6)])
    # release_at_trim is off by default since run #10 (a release cost a median 19 min wait)
    agent = Agent(llm, AgentConfig(retry_sleep=0, release_at_trim=True), tool_factory=FakeTool,
                  gate=gate, progress=GameProgress("g"), clock=clock)
    session = FakeSession()
    agent.reset_game(session)
    agent.context.trim = lambda *a, **k: 1          # every call "trims"
    agent.play_turn(session)                        # nobody outranks us: keep the slot
    assert agent.stats.trims == 1 and gate.releases == 0 and gate.acquires == 1
    assert agent._slot_since == 1000.0
    gate.should = True
    agent.play_turn(session)                        # outranked at the trim: release, re-acquire
    assert gate.releases == 1 and gate.acquires == 2
    trims = [e for e in session.events if e.get("kind") == "trim"]
    assert [e["released"] for e in trims] == [False, True]
    # Turn boundary rotation: held for 15 minutes and outranked -> release before the turn.
    agent.context.trim = lambda *a, **k: 0
    clock.t += 100
    agent.play_turn(session)                        # held 100 s: too soon
    assert gate.releases == 1
    clock.t += 900
    gate.should = False
    agent.play_turn(session)                        # long enough, but nobody outranks us
    assert gate.releases == 1
    gate.should = True
    agent.play_turn(session)
    assert gate.releases == 2 and gate.acquires == 3
    rot = [e for e in session.events if e.get("kind") == "slot_rotate"]
    assert len(rot) == 1 and rot[0]["held"] >= 900
    assert agent._slot_since == clock.t              # the hold timer restarted on re-acquire


def test_gate_without_should_yield_keeps_the_old_release_at_trim():
    from agent_fakes import FakeSession, FakeTool
    from arc3.agent import Agent, AgentConfig
    from arc3.llm import FakeLLM
    from arc3.scheduler import GameProgress

    class OldGate:
        def __init__(self):
            self.releases = 0
        def acquire(self, progress, timeout=None):
            return True
        def release(self, progress):
            self.releases += 1

    gate = OldGate()
    agent = Agent(FakeLLM([FakeLLM.code("action(['LEFT'])")]), AgentConfig(retry_sleep=0, release_at_trim=True),
                  tool_factory=FakeTool, gate=gate, progress=GameProgress("g"))
    session = FakeSession()
    agent.reset_game(session)
    agent.context.trim = lambda *a, **k: 1
    agent.play_turn(session)
    assert gate.releases == 1
