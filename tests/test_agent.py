"""Agent loop, context manager, prompts and LLM client, with FakeLLM and a fake session."""
from __future__ import annotations

import json

import pytest
import requests

from agent_fakes import FakeSession, FakeTool, fake_png
from arc3 import prompts
from arc3.agent import Agent, AgentConfig, AgentFailure
from arc3.context import ContextManager, has_images
from arc3.llm import FakeLLM, LLMClient, LLMError, LLMTimeout, image_part, parse_tool_code
from arc3.types import LLMMessage


def make_agent(script, **cfg):
    llm = FakeLLM(script)
    agent = Agent(llm, AgentConfig(retry_sleep=0, **cfg), tool_factory=FakeTool,
                  render_png=lambda g, s: fake_png(), render_diff_png=lambda a, b, s: fake_png())
    return agent, llm


def user_texts(messages):
    out = []
    for m in messages:
        if m["role"] != "user":
            continue
        c = m["content"]
        out.append(c if isinstance(c, str) else " ".join(p.get("text", "") for p in c))
    return out


def n_images(message):
    c = message["content"]
    return 0 if isinstance(c, str) else sum(p["type"] == "image_url" for p in c)


# ------------------------------------------------------------------ agent loop

def test_two_turn_game_wins():
    agent, llm = make_agent([
        FakeLLM.code("print(int(current_frame.sum()))", reasoning="look first"),
        FakeLLM.code("action(['RIGHT', 'RIGHT'])"),
        FakeLLM.code("action(['SPACE'])"),
    ], send_images=True)  # image plumbing is still exercised with images enabled
    session = FakeSession()
    stats = agent.play_game(session)
    assert stats.end_reason == "win"
    assert stats.turns == 2 and stats.actions == 3 and stats.tool_calls == 3
    assert stats.levels_completed == 1 and stats.state == "WIN"

    first, second = llm.calls[0]["messages"], llm.calls[2]["messages"]
    assert first[0]["role"] == "system" and "python" in first[0]["content"]
    assert n_images(first[1]) == 1                          # current board only
    assert llm.calls[0]["tools"][0]["function"]["name"] == "python"
    # inspection output and reasoning are carried back
    assert any(m["role"] == "tool" and m["content"].strip() == "83" for m in llm.calls[1]["messages"])
    assert llm.calls[1]["messages"][2]["reasoning_content"] == "look first"
    opener2 = second[-1]
    text2 = user_texts([opener2])[0]
    assert "Previous call executed 2 action(s): RIGHT RIGHT" in text2
    assert "gameplay changed" in text2
    assert n_images(opener2) == 2                           # board + diff
    kinds = [e["kind"] for e in session.events]
    assert kinds.count("model_call") == 3 and kinds.count("tool_call") == 3
    assert kinds[-1] == "game_end"


def test_token_yield_then_resume():
    agent, llm = make_agent([
        FakeLLM.code("print(1)", tokens=1500),
        FakeLLM.code("print(2)", tokens=1500),                # 3000 >= 2048 -> yield
        FakeLLM.code("action(['RIGHT'])", tokens=100),
    ])
    session = FakeSession()
    t1 = agent.play_turn(session)
    assert agent.turn_end_reason == "yield" and t1.actions == 0 and t1.generated_tokens == 3000
    t2 = agent.play_turn(session)
    assert agent.turn_end_reason == "action" and t2.actions == 1
    last_user = user_texts(llm.calls[2]["messages"])[-1]
    assert last_user.startswith(prompts.RESUME)
    assert agent.stats.yields == 1


def test_tool_call_cap_and_skipped_calls_answered():
    agent, llm = make_agent([FakeLLM.code("print(1)")] * 3, max_tool_calls_per_turn=2)
    agent.play_turn(FakeSession())
    assert agent.turn_end_reason == "tool_cap" and len(llm.calls) == 2


def test_no_tool_call_nudge():
    agent, llm = make_agent([FakeLLM.text("I will move right."),
                             FakeLLM.code("action(['RIGHT'])")])
    ts = agent.play_turn(FakeSession())
    assert ts.actions == 1 and agent.stats.nudges == 1
    assert user_texts(llm.calls[1]["messages"])[-1] == prompts.NUDGE

    agent, _ = make_agent([FakeLLM.text("hmm"), FakeLLM.text("still thinking")])
    ts = agent.play_turn(FakeSession())
    assert agent.turn_end_reason == "no_tool_call" and ts.actions == 0
    assert agent.resume_next


def test_game_over_opener():
    agent, llm = make_agent([FakeLLM.code("action(['RIGHT', 'DOWN', 'RIGHT'])"),
                             FakeLLM.code("print('diagnose')")], max_turns=2, send_images=True)
    session = FakeSession()
    agent.play_game(session)
    assert session.resets == 1
    opener = llm.calls[1]["messages"][-1]
    text = user_texts([opener])[0]
    for needle in ("GAME OVER", "BAR RULE", "history[-2]", "deterministic",
                   "The fatal action was DOWN", "death frame"):
        assert needle in text
    assert n_images(opener) == 2                            # board + death frame


def test_llm_failures_abandon():
    agent, _ = make_agent([LLMError("down")] * 3, max_llm_failures=3)
    stats = agent.play_game(FakeSession())
    assert stats.end_reason == "llm_failures" and stats.llm_failures == 3


def test_llm_failure_recovers():
    agent, _ = make_agent([LLMError("blip"), FakeLLM.code("action(['RIGHT'])")])
    ts = agent.play_turn(FakeSession())
    assert ts.actions == 1 and agent.consecutive_failures == 0


# ------------------------------------------------------------------ context

def _block(ctx, n_chars):
    ctx.add_user("u" * n_chars)
    ctx.add_assistant("", reasoning="r" * n_chars, tool_calls=[{"id": "c", "type": "function",
                      "function": {"name": "python", "arguments": "{}"}}])
    ctx.add_tool("c", "t" * n_chars)


def test_trim_hysteresis_keeps_system_and_last_block():
    ctx = ContextManager("SYSTEM " * 50)
    ctx.tm.chars_per_token = 1.0
    for _ in range(10):
        _block(ctx, 1000)                                   # ~3000 tokens per block
    total = ctx.total_tokens()
    assert ctx.trim(budget_tokens=total + 1, drain_tokens=10_000) == 0
    assert ctx.trim(budget_tokens=20_000, drain_tokens=10_000) > 0
    dropped = ctx.last_trim["dropped"]       # the return value also counts stripped reasoning
    assert dropped % 3 == 0 and dropped > 0 and ctx.trimmed_since_last_user
    assert ctx.total_tokens() <= 10_000
    assert ctx.messages[0].role == "system" and ctx.messages[0].content.startswith("SYSTEM")
    assert ctx.messages[1].role == "user"
    assert ctx.messages[-1].role == "tool" and len(ctx.messages) >= 4
    # hysteresis: adding a block afterwards does not trigger another trim
    _block(ctx, 1000)
    assert ctx.trim(budget_tokens=20_000, drain_tokens=10_000) == 0


def test_trim_never_drops_last_block():
    ctx = ContextManager("sys")
    ctx.tm.chars_per_token = 1.0
    _block(ctx, 5000)
    assert ctx.trim(budget_tokens=100, drain_tokens=50) == 0
    assert len(ctx.messages) == 4


def test_image_window_and_image_tokens():
    ctx = ContextManager("sys", images_window=2)
    for i in range(4):
        ctx.add_user([{"type": "text", "text": f"turn {i}"}, image_part(fake_png())])
        ctx.add_assistant("ok")
    users = [m for m in ctx.messages if m.role == "user"]
    assert [has_images(m) for m in users] == [False, False, True, True]
    assert "older image removed" in json.dumps(users[0].content)
    assert users[3].tokens > 400 > users[0].tokens          # 640x640 -> 20*20 + overhead


def test_calibration_moves_ratio():
    ctx = ContextManager("x" * 4000)
    before = ctx.ratio
    ctx.calibrate(prompt_tokens=2000)                       # 4000 chars / ~2000 tokens = 2.0
    assert ctx.ratio < before
    for _ in range(30):
        ctx.calibrate(prompt_tokens=2000)
    assert abs(ctx.ratio - 2.0) < 0.1


# ------------------------------------------------------------------ prompts

def test_system_prompt_size_and_sections():
    p = prompts.build_system_prompt()
    # 2800: the WORLD_MODEL rework (worked example, return shapes) added ~400 tokens.
    assert len(p) / 3.5 < 2800
    for needle in ("current_frame", ".grid", ".crop", "last_animation_frames", "frame_diff",
                   "UNDO", "MOUSE", "gameplay_changed", "game_over", "BFS", "free"):
        assert needle in p
    assert "RESET" not in prompts.build_system_prompt(expose_reset=False).split("## Actions")[1].split("##")[0]


# ------------------------------------------------------------------ llm client

class _Resp:
    def __init__(self, status, data):
        self.status_code, self._data, self.text = status, data, json.dumps(data)

    def json(self):
        return self._data


class _HTTP:
    def __init__(self, outcomes):
        self.outcomes, self.bodies = list(outcomes), []

    def post(self, url, json=None, headers=None, timeout=None):
        self.bodies.append(json)
        o = self.outcomes.pop(0)
        if isinstance(o, Exception):
            raise o
        return o


OK = {"choices": [{"message": {"content": "", "reasoning_content": "think",
                               "tool_calls": [{"id": "a1", "type": "function", "function": {
                                   "name": "python", "arguments": "{\"code\": \"print(1)\"}"}}]},
                   "finish_reason": "tool_calls"}],
      "usage": {"prompt_tokens": 50, "completion_tokens": 7,
                "prompt_tokens_details": {"cached_tokens": 40}}}


def test_client_request_and_parse():
    http = _HTTP([requests.ConnectionError("refused"), _Resp(503, {}), _Resp(200, OK)])
    client = LLMClient(base_url="http://x/v1", session=http, backoff=0)
    msgs = [LLMMessage("user", "hi"),
            LLMMessage("assistant", "", reasoning="old thought", tool_calls=[OK["choices"][0]["message"]["tool_calls"][0]]),
            LLMMessage("tool", "1", tool_call_id="a1")]
    r = client.chat(msgs, tools=[{"type": "function"}], top_k=20, seed=3)
    assert r.reasoning == "think" and r.usage == {"prompt_tokens": 50, "completion_tokens": 7, "cached_tokens": 40}
    assert parse_tool_code(r.tool_calls[0]) == ("print(1)", None)
    body = http.bodies[-1]
    assert body["chat_template_kwargs"] == {"enable_thinking": True}
    assert body["top_k"] == 20 and body["seed"] == 3
    assert body["messages"][1]["reasoning_content"] == "old thought"
    assert body["messages"][2]["tool_call_id"] == "a1"


def test_client_timeout_and_errors():
    client = LLMClient(session=_HTTP([requests.Timeout()]), backoff=0)
    with pytest.raises(LLMTimeout):
        client.chat([LLMMessage("user", "hi")])
    client = LLMClient(session=_HTTP([_Resp(400, {"error": "bad"})]), backoff=0)
    with pytest.raises(LLMError):
        client.chat([LLMMessage("user", "hi")])
    client = LLMClient(session=_HTTP([requests.ConnectionError()] * 3), backoff=0, max_retries=2)
    with pytest.raises(LLMError):
        client.chat([LLMMessage("user", "hi")])


def test_parse_tool_code_lenient():
    call = lambda args: {"function": {"name": "python", "arguments": args}}
    assert parse_tool_code(call("print(2)")) == ("print(2)", None)
    assert parse_tool_code(call("{bad json")) [0] is None
    assert parse_tool_code({"function": {"name": "bash", "arguments": "{}"}})[0] is None


# ------------------------------------------------------------------ scheduler integration

class _Gate:
    def __init__(self, grant=True):
        self.grant, self.acquires, self.releases = grant, 0, 0

    def acquire(self, progress, timeout=None):
        self.acquires += 1
        return self.grant

    def release(self, progress):
        self.releases += 1


def test_gate_and_progress():
    from arc3.scheduler import GameProgress
    gate, progress = _Gate(), GameProgress("fake")
    llm = FakeLLM([FakeLLM.code("print(1)", tokens=50), FakeLLM.code("action(['RIGHT','RIGHT'])", tokens=70),
                   FakeLLM.code("action(['SPACE'])", tokens=30)])
    agent = Agent(llm, AgentConfig(retry_sleep=0), tool_factory=FakeTool, gate=gate, progress=progress,
                  render_png=lambda g, s: fake_png(), render_diff_png=lambda a, b, s: fake_png())
    stats = agent.play_game(FakeSession())
    assert stats.end_reason == "win"
    assert gate.acquires == 3 and gate.releases == 1        # released once at game end
    assert progress.level == 1 and progress.levels_total == 1 and progress.started
    assert progress.actions_on_level == 0                   # reset by the level change


def test_gate_timeout_is_deadline():
    agent = Agent(FakeLLM([]), AgentConfig(), tool_factory=FakeTool, gate=_Gate(grant=False),
                  progress=__import__("arc3.scheduler", fromlist=["GameProgress"]).GameProgress("g"),
                  render_png=lambda g, s: fake_png())
    assert agent.play_game(FakeSession()).end_reason == "deadline"


def test_idle_turns_release_slot_then_park():
    # run #6 sb26: 130 consecutive turns of refused RESET no-ops held a decode slot for 2.7 h.
    from arc3.scheduler import GameProgress
    gate, progress, naps = _Gate(), GameProgress("fake"), []
    script = [FakeLLM.code("print('hold')", tokens=20) for _ in range(9)]
    script += [FakeLLM.code("action(['RIGHT', 'RIGHT'])", tokens=30), FakeLLM.code("action(['SPACE'])", tokens=30)]
    llm = FakeLLM(script)
    cfg = AgentConfig(retry_sleep=0, max_tool_calls_per_turn=1, idle_turns_notice=3,
                      idle_turns_sleep=6, idle_sleep_seconds=600)
    agent = Agent(llm, cfg, tool_factory=FakeTool, gate=gate, progress=progress, sleep=naps.append,
                  render_png=lambda g, s: fake_png(), render_diff_png=lambda a, b, s: fake_png())
    stats = agent.play_game(FakeSession())
    assert stats.end_reason == "win"
    assert naps == [600, 600]                  # parked after idle turns 6 and 9 (counter falls back to 3)
    assert gate.releases == 3                  # released only when parked (twice) and at game end
    notices = [m for c in llm.calls for m in user_texts(c["messages"])[-1:] if "Idling earns nothing" in m]
    assert len(notices) >= 3                   # shown from the 3rd idle turn on, and after parking
    assert agent.idle_turns == 0               # the real action cleared the counter


def test_agent_pins_rules_when_a_level_completes():
    """The agent snapshots memory['rules'] the turn after the environment reports a new level,
    and the opener of that turn already carries the pinned block."""
    class PinTool(FakeTool):
        def __init__(self, session):
            super().__init__(session)
            self.pins: list[int] = []

        def pin_level(self, level):
            self.pins.append(level)

        def pinned_block(self):
            return "Level %d solved with rules: SPACE at col 10" % self.pins[-1] if self.pins else ""

    session = FakeSession(win_levels=2)
    llm = FakeLLM([
        FakeLLM.code('action(["RIGHT", "RIGHT", "SPACE"])'),   # completes level 1
        FakeLLM.code('action(["SPACE"])'),                     # still at col 10: completes level 2 -> win
    ])
    agent = Agent(llm, AgentConfig(retry_sleep=0), tool_factory=PinTool,
                  render_png=lambda g, s: fake_png(), render_diff_png=lambda a, b, s: fake_png())
    stats = agent.play_game(session)
    assert stats.levels_completed == 2, stats
    tool = agent.tool
    assert tool.pins == [1], tool.pins          # level 2 ends the game: no further turn to pin in
    # the second turn's opener (after level 1) carried the pinned block
    opener2 = llm.calls[1]["messages"][-1]
    assert opener2["role"] == "user" and "Level 1 solved with rules" in opener2["content"], opener2
