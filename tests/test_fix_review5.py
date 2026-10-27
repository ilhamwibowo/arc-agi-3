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
    # Items 2-4 of fix-review-5 together must add under 500 chars to the run #5 prompt (8803).
    assert len(p) - 8803 < 500
