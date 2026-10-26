"""The model-facing loop: one turn = opener, then LLM <-> python tool until an action executes.

A turn also ends when the model has generated `yield_tokens` without acting (the next turn
then starts with the resume prompt instead of a fresh opener), when it hits
`max_tool_calls_per_turn`, or when it twice answers without a tool call. `play_game` repeats
turns until the game is won or a deadline/action/failure cap is reached.

Interfaces assumed from other modules (fakes in tests):
- session: `.current` (Observation), `.valid_actions`, `.config` (GameConfig), `.trace(dict)`,
  `.auto_reset_if_needed()`; optional `.level`.
- tool: `PythonTool(session).run(code) -> ToolResult` (arc3.sandbox).
- percept: `render_png(grid, scale) -> bytes`, `render_diff_png(before, after, scale) -> bytes`.
- scheduler (optional): `PriorityGate.acquire(progress, timeout) -> bool` before each model call
  (re-acquire while holding is a no-op, so the slot is kept across tool calls) and
  `.release(progress)` at a context trim (handover). `GameProgress` counters are kept current.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Optional

from arc3 import prompts
from arc3.context import ContextManager, TokenModel
from arc3.llm import image_part, parse_tool_code, text_part
from arc3.types import ActionResult, GameState, ToolResult, TurnStats

PYTHON_TOOL = {
    "type": "function",
    "function": {
        "name": "python",
        "description": ("Run Python in the game sandbox. Runtime variables (current_frame, "
                        "history, valid_actions, frame_diff, action, ...) are preloaded. "
                        "Print compact results."),
        "parameters": {
            "type": "object",
            "properties": {"code": {"type": "string", "description": "Python source to execute."}},
            "required": ["code"],
        },
    },
}


@dataclass
class AgentConfig:
    context_tokens: int = 116_000       # prompt budget; trimming starts above this
    output_tokens: int = 12_000         # max_tokens per completion
    drain_tokens: int = 58_000          # trim down to context_tokens - drain_tokens
    yield_tokens: int = 2048            # generated tokens per turn before yielding
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: Optional[int] = 20
    seed: Optional[int] = None
    image_scale: int = 10
    images_window: int = 2
    keep_reasoning: bool = True
    max_tool_calls_per_turn: int = 12
    max_llm_failures: int = 10          # consecutive failed model calls before giving up
    retry_sleep: float = 10.0
    llm_timeout: float = 900.0
    max_actions: Optional[int] = None   # overrides session.config.max_actions when set
    max_turns: Optional[int] = None
    tool_timeout: int = 30
    tool_output_tokens: int = 3000
    image_px_per_token: int = 32
    send_images: bool = True


@dataclass
class GameStats:
    turns: int = 0
    model_calls: int = 0
    tool_calls: int = 0
    actions: int = 0
    generated_tokens: int = 0
    prompt_tokens: int = 0
    cached_tokens: int = 0
    llm_failures: int = 0
    yields: int = 0
    nudges: int = 0
    trims: int = 0
    levels_completed: int = 0
    win_levels: int = 0
    state: str = ""
    end_reason: str = ""
    seconds: float = 0.0


class AgentFailure(RuntimeError):
    """Too many consecutive model failures; the game should be abandoned."""


class DeadlineReached(RuntimeError):
    """The game's deadline passed while waiting for a decode slot."""


class Agent:
    def __init__(self, llm, config: Optional[AgentConfig] = None,
                 tool_factory: Optional[Callable[[Any], Any]] = None,
                 render_png: Optional[Callable] = None,
                 render_diff_png: Optional[Callable] = None,
                 sleep: Callable[[float], None] = time.sleep,
                 gate=None, progress=None):
        self.llm = llm
        self.gate = gate
        self.progress = progress
        self.cfg = config or AgentConfig()
        self._tool_factory = tool_factory
        self._render_png = render_png
        self._render_diff_png = render_diff_png
        self._sleep = sleep
        self.session = None
        self.stats = GameStats()

    # ------------------------------------------------------------ game setup

    def reset_game(self, session) -> None:
        cfg = self.cfg
        gc = getattr(session, "config", None)
        system = prompts.build_system_prompt(
            hud_border=getattr(gc, "hud_border", 4),
            expose_undo=getattr(gc, "expose_undo", True),
            expose_reset=getattr(gc, "expose_reset", False),
            tool_timeout=cfg.tool_timeout, output_tokens=cfg.tool_output_tokens)
        self.context = ContextManager(
            system, TokenModel(image_px_per_token=cfg.image_px_per_token),
            images_window=cfg.images_window, keep_reasoning=cfg.keep_reasoning,
            extra_chars=len(json.dumps([PYTHON_TOOL])))
        self.session = session
        self.tool = self._make_tool(session)
        self.stats = GameStats()
        self.last_results: list[ActionResult] = []
        self.retained: list[str] = []
        self.resume_next = False
        self.first_turn = True
        self.consecutive_failures = 0
        self.turn_end_reason = ""

    def _make_tool(self, session):
        if self._tool_factory is not None:
            return self._tool_factory(session)
        from arc3.sandbox import PythonTool
        return PythonTool(session)

    def _trace(self, event: dict) -> None:
        fn = getattr(self.session, "trace", None)
        if callable(fn):
            event.setdefault("t", round(time.time(), 3))
            fn(event)

    # ------------------------------------------------------------ opener

    def _render(self, kind: str, *grids) -> Optional[bytes]:
        try:
            if kind == "diff":
                fn = self._render_diff_png
                if fn is None:
                    from arc3.percept import render_diff_png as fn
            else:
                fn = self._render_png
                if fn is None:
                    from arc3.percept import render_png as fn
            return fn(*grids, self.cfg.image_scale)
        except Exception as e:     # an image is a nice-to-have; never lose a turn over it
            self._trace({"kind": "render_error", "image": kind, "error": repr(e)})
            return None

    def _opener_images(self, session) -> list[tuple[str, bytes]]:
        if not self.cfg.send_images:
            return []
        images = []
        png = self._render("grid", session.current.grid)
        if png:
            images.append(("current board", png))
        executed = [r for r in self.last_results if r.executed and not r.automatic]
        fatal = next((r for r in executed if r.game_over), None)
        if fatal is not None and fatal.after is not None:
            png = self._render("grid", fatal.after)
            if png:
                images.append(("death frame (board right after the fatal action)", png))
        elif (executed and any(r.gameplay_changed for r in executed)
              and not any(r.level_completed for r in executed)
              and executed[0].before is not None and executed[-1].after is not None):
            png = self._render("diff", executed[0].before, executed[-1].after)
            if png:
                images.append(("diff of the previous call (changed cells bright, rest dimmed)", png))
        return images

    def build_opener(self, session) -> Any:
        obs = session.current
        images = self._opener_images(session)
        text = prompts.build_opener(
            self.last_results, step=obs.step, level=obs.levels_completed,
            win_levels=obs.win_levels, valid_actions=list(session.valid_actions),
            first_turn=self.first_turn, images=[c for c, _ in images], retained=self.retained)
        if not images:
            return text
        parts = [text_part(text)]
        for caption, png in images:
            parts.append(text_part(f"Image: {caption}"))
            parts.append(image_part(png))
        return parts

    def _resume_text(self, session) -> str:
        obs = session.current
        return (prompts.RESUME + f"\nCurrent state: step {obs.step}, level {obs.levels_completed + 1}. "
                f"Valid actions: {', '.join(session.valid_actions)}.")

    # ------------------------------------------------------------ model call

    def _call_llm(self, deadline: Optional[float]):
        cfg = self.cfg
        while True:
            timeout = cfg.llm_timeout
            if deadline is not None:
                timeout = max(1.0, min(timeout, deadline - time.time()))
            msgs = list(self.context.messages)
            self._acquire_slot(deadline)
            t0 = time.time()
            try:
                resp = self.llm.chat(msgs, tools=[PYTHON_TOOL], max_tokens=cfg.output_tokens,
                                     temperature=cfg.temperature, top_p=cfg.top_p,
                                     top_k=cfg.top_k, seed=cfg.seed, timeout=timeout)
            except Exception as e:
                self.consecutive_failures += 1
                self.stats.llm_failures += 1
                self._trace({"kind": "model_error", "error": repr(e)[:500],
                             "consecutive": self.consecutive_failures})
                if self.consecutive_failures >= cfg.max_llm_failures:
                    raise AgentFailure(f"{self.consecutive_failures} consecutive model failures") from e
                if deadline is not None and time.time() >= deadline:
                    raise AgentFailure("deadline reached while retrying the model") from e
                self._sleep(cfg.retry_sleep)
                continue
            self.consecutive_failures = 0
            self.context.calibrate(resp.usage.get("prompt_tokens", 0), msgs)
            self._trace({"kind": "model_call", "seconds": round(time.time() - t0, 3),
                         "messages": len(msgs), "estimate": self.context.total_tokens(),
                         "usage": resp.usage, "finish_reason": resp.finish_reason,
                         "tool_calls": len(resp.tool_calls), "content": resp.content[:300],
                         "reasoning_chars": len(resp.reasoning or "")})
            return resp

    def _acquire_slot(self, deadline: Optional[float]) -> None:
        if self.gate is None or self.progress is None:
            return
        timeout = None if deadline is None else max(0.0, deadline - time.time())
        t0 = time.time()
        if not self.gate.acquire(self.progress, timeout=timeout):
            raise DeadlineReached("no decode slot before the deadline")
        waited = time.time() - t0
        if waited > 1.0:
            self._trace({"kind": "slot_wait", "seconds": round(waited, 3)})

    def _release_slot(self) -> None:
        if self.gate is not None and self.progress is not None:
            self.gate.release(self.progress)

    def _update_progress(self, session, ts: TurnStats) -> None:
        p = self.progress
        if p is None:
            return
        p.started = True
        p.actions_on_level += ts.actions
        p.tokens_on_level += ts.generated_tokens
        p.levels_total = session.current.win_levels or p.levels_total
        p.level_completed(session.current.levels_completed)

    # ------------------------------------------------------------ tool call

    def _run_tool(self, call: dict) -> tuple[str, list[ActionResult]]:
        code, err = parse_tool_code(call)
        if code is None:
            self._trace({"kind": "tool_call", "error": err, "code": None})
            return f"[error] {err}", []
        t0 = time.time()
        try:
            res: ToolResult = self.tool.run(code)
        except Exception as e:     # the sandbox should not raise; if it does, tell the model
            self._trace({"kind": "tool_call", "code": code, "error": repr(e)})
            return f"[error] tool failed: {e!r}", []
        if res.retained_functions:
            self.retained = list(res.retained_functions)
        self._trace({"kind": "tool_call", "code": code, "seconds": round(time.time() - t0, 3),
                     "stdout": res.stdout[:2000], "error": res.error, "truncated": res.truncated,
                     "actions": [r.action.display() for r in res.actions if r.executed],
                     "stops": [r.stop_reason for r in res.actions if r.stop_reason]})
        text = prompts.tool_result_text(res.stdout, res.error, res.actions, res.truncated,
                                        res.retained_functions, res.rejected_functions)
        return text, list(res.actions)

    # ------------------------------------------------------------ turn

    def play_turn(self, session, deadline: Optional[float] = None) -> TurnStats:
        if self.session is not session:
            self.reset_game(session)
        reset = getattr(session, "auto_reset_if_needed", None)
        if callable(reset):
            reset()
        cfg, ts, t0 = self.cfg, TurnStats(), time.time()
        if self.resume_next:
            self.context.add_user(self._resume_text(session))
        else:
            self.context.add_user(self.build_opener(session))
        self.first_turn = False

        results: list[ActionResult] = []
        acted, nudged, reason = False, False, ""
        while not reason:
            if self.context.trim(cfg.context_tokens, cfg.drain_tokens):
                # the prefix cache is gone anyway: a good moment to hand the slot over
                self._release_slot()
                self.stats.trims += 1
                self._trace({"kind": "trim", "estimate": self.context.total_tokens(),
                             "messages": len(self.context.messages)})
            resp = self._call_llm(deadline)
            self.stats.model_calls += 1
            ts.generated_tokens += resp.usage.get("completion_tokens", 0)
            ts.prompt_tokens += resp.usage.get("prompt_tokens", 0)
            ts.cached_tokens += resp.usage.get("cached_tokens", 0)
            self.context.add_assistant(resp.content, resp.reasoning, resp.tool_calls)

            if not resp.tool_calls:
                if nudged:
                    reason = "no_tool_call"
                    break
                nudged = True
                self.stats.nudges += 1
                self.context.add_user(prompts.NUDGE)
                continue

            for call in resp.tool_calls:
                # every tool_call id needs an answer, even the ones we skip
                if acted:
                    text = "[not run] an action already executed earlier in this turn."
                elif ts.tool_calls >= cfg.max_tool_calls_per_turn:
                    text = "[not run] tool-call limit for this turn reached."
                else:
                    ts.tool_calls += 1
                    text, actions = self._run_tool(call)
                    results.extend(actions)
                    if any(r.executed for r in actions):
                        acted = True
                self.context.add_tool(call["id"], text)

            if acted:
                reason = "action"
            elif ts.generated_tokens >= cfg.yield_tokens:
                reason = "yield"
                self.stats.yields += 1
            elif ts.tool_calls >= cfg.max_tool_calls_per_turn:
                reason = "tool_cap"

        executed = [r for r in results if r.executed]
        ts.actions = len(executed)
        if acted:
            self.last_results = results
        self.resume_next = not acted
        self.turn_end_reason = reason
        ts.seconds = time.time() - t0
        self._accumulate(ts)
        self._update_progress(session, ts)
        self._trace({"kind": "turn", "reason": reason, **asdict(ts)})
        return ts

    def _accumulate(self, ts: TurnStats) -> None:
        s = self.stats
        s.turns += 1
        s.tool_calls += ts.tool_calls
        s.actions += ts.actions
        s.generated_tokens += ts.generated_tokens
        s.prompt_tokens += ts.prompt_tokens
        s.cached_tokens += ts.cached_tokens

    # ------------------------------------------------------------ game

    def _stop_reason(self, session, deadline: Optional[float]) -> str:
        obs = session.current
        if obs.state == GameState.WIN or any(r.run_complete for r in self.last_results):
            return "win"
        if deadline is not None and time.time() >= deadline:
            return "deadline"
        cap = self.cfg.max_actions or getattr(getattr(session, "config", None), "max_actions", None)
        if cap is not None and self.stats.actions >= cap:
            return "action_cap"
        if self.cfg.max_turns is not None and self.stats.turns >= self.cfg.max_turns:
            return "turn_cap"
        return ""

    def play_game(self, session, deadline: Optional[float] = None) -> GameStats:
        t0 = time.time()
        self.reset_game(session)
        reason = ""
        wait_ready = getattr(self.llm, "wait_ready", None)
        if callable(wait_ready):
            remaining = 1800.0 if deadline is None else max(1.0, deadline - time.time())
            if not wait_ready(timeout=remaining):
                reason = "server_unavailable"
        if self.progress is not None:
            self.progress.levels_total = session.current.win_levels
            self.progress.level_completed(session.current.levels_completed)
        while not reason:
            session.auto_reset_if_needed()
            reason = self._stop_reason(session, deadline)
            if reason:
                break
            try:
                self.play_turn(session, deadline)
            except AgentFailure as e:
                reason = "llm_failures"
                self._trace({"kind": "abandon", "error": str(e)})
                break
            except DeadlineReached:
                reason = "deadline"
                break
        self._release_slot()
        s = self.stats
        s.end_reason = reason
        s.levels_completed = session.current.levels_completed
        s.win_levels = session.current.win_levels
        s.state = getattr(session.current.state, "value", str(session.current.state))
        s.seconds = time.time() - t0
        self._trace({"kind": "game_end", **asdict(s)})
        return s
