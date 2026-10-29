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
  (re-acquire while holding is a no-op, so the slot is kept across tool calls),
  `.should_yield(progress) -> bool` and `.release(progress)`. The slot is released at a context
  trim only when `should_yield` says a waiting game outranks us, and at a turn boundary once we
  have held it for `slot_rotate_seconds` and `should_yield` is true (the next model call
  re-acquires). A gate without `should_yield` is released at every trim. `GameProgress` counters
  are kept current.
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
    context_tokens: int = 116_000       # prompt budget; trimming starts above this (80k in runs #8-#9 doubled trims and cost ~15 levels)
    output_tokens: int = 6_000          # max_tokens per completion; a cut at 8k cost 150-285 s in run #5
    drain_tokens: int = 58_000          # trim down to context_tokens - drain_tokens
    yield_tokens: int = 2048            # generated tokens per turn before yielding
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: Optional[int] = 20
    seed: Optional[int] = None
    image_scale: int = 10
    images_window: int = 1   # >1 breaks the server prefix cache on every new image (run #1)
    keep_reasoning: bool = True
    max_tool_calls_per_turn: int = 12
    idle_turns_notice: int = 3    # consecutive turns with no executed action → opener notice
    idle_turns_sleep: int = 8     # ... → release the slot and park for idle_sleep_seconds (run #6 sb26: 130 idle turns)
    idle_sleep_seconds: float = 600.0
    max_llm_failures: int = 100_000     # consecutive failed model calls before giving up. In the
                                        # competition a server relaunch can take 60 min; a game that
                                        # gives up scores nothing, so keep retrying until the deadline
                                        # (the runner's abort ends a run whose server never returns).
    retry_sleep: float = 10.0
    release_at_trim: bool = False       # run #10: a trim release cost a median 19 min wait; the
                                        # 15-min rotation already redistributes slots
    output_tokens_after_cut: int = 2500 # a reply cut at the cap without a tool call is re-asked
                                        # with this smaller cap (run #10: 227 cuts = 18% of time)
    stall_notice_every: int = 6         # turns between stall notices while a level is stalled
    llm_timeout: float = 900.0
    max_actions: Optional[int] = None   # overrides session.config.max_actions when set
    max_turns: Optional[int] = None
    tool_timeout: int = 30
    tool_output_tokens: int = 3000
    image_px_per_token: int = 32
    send_images: bool = False  # run #4 A/B: text-only calls cache 96 % vs 43 % with images, 22 % faster
    slot_rotate_seconds: float = 900.0  # hand the decode slot over after this long, if outranked


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


def _prepend_text(prefix: str, opener):
    if isinstance(opener, str):
        return prefix + "\n" + opener
    return [text_part(prefix)] + list(opener)


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
                 gate=None, progress=None, clock: Callable[[], float] = time.time):
        self.llm = llm
        self.gate = gate
        self.progress = progress
        self._clock = clock
        self._slot_since: Optional[float] = None   # when we last took the decode slot
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
            # list UNDO only when the game actually offers it (run #1: the model tried an invalid UNDO)
            expose_undo=getattr(gc, "expose_undo", True)
            and "UNDO" in getattr(session.current, "available_actions", ["UNDO"]),
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
        self.idle_turns = 0           # consecutive turns that executed no action
        self._pinned_levels = 0       # levels whose rules have been pinned (see _pin_if_level_completed)
        self._last_stall_notice = -10**9
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

    def _evidence(self) -> Optional[dict]:
        """Per-level facts from the sandbox (unclicked objects, blocked moves, novelty)."""
        fn = getattr(self.tool, "evidence", None)
        if not callable(fn):
            return None
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 - evidence is a nice-to-have; never lose a turn
            self._trace({"kind": "evidence_error", "error": repr(e)[:300]})
            return None

    def build_opener(self, session) -> Any:
        obs = session.current
        images = self._opener_images(session)
        text = prompts.build_opener(
            self.last_results, step=obs.step, level=obs.levels_completed,
            win_levels=obs.win_levels, valid_actions=list(session.valid_actions),
            first_turn=self.first_turn, images=[c for c, _ in images], retained=self.retained,
            evidence=self._evidence(), pinned=self._pinned_block())
        if not images:
            return text
        parts = [text_part(text)]
        for caption, png in images:
            parts.append(text_part(f"Image: {caption}"))
            parts.append(image_part(png))
        return parts

    # ---- stall signals (run #10: 35 of 65 lost levels were rules never found, and the model kept
    # looping after writing "unsolvable"; the scheduler demotes such levels and the opener asks for
    # new hypotheses instead of repeats) -------------------------------------------------------

    def _update_stall(self) -> float:
        """Set progress.stall from three signals and return it: (1) novelty: >= 40 actions on the
        level and under 3 new states in the last 20; (2) actions on this level over 3x the median of
        this game's completed levels; (3) the model wrote 'impossible'/'unsolvable'/... into memory."""
        p = self.progress
        signals = 0
        ev = self._evidence() or {}
        nv = ev.get("novelty") or {}
        if nv.get("actions", 0) >= 40 and nv.get("window", 0) >= 20 and nv.get("new_in_window", 99) < 3:
            signals += 1
        if p is not None and p.level_actions:
            done = sorted(p.level_actions)
            median = done[len(done) // 2]
            if median > 0 and p.actions_on_level > 3 * median:
                signals += 1
        words = ("impossible", "unsolvable", "unwinnable", "exhausted", "give up", "no way to")
        text = ""
        try:
            fn = getattr(self.tool, "rules_text", None)
            text = (fn() if callable(fn) else "") + " " + self._memory_summary()
        except Exception:  # noqa: BLE001
            pass
        if any(w in text.lower() for w in words):
            signals += 1
        stall = 0.0 if signals == 0 else (0.5 if signals == 1 else 1.0)
        if p is not None:
            p.stall = stall
        return stall

    def _stall_facts(self) -> dict:
        fn = getattr(self.tool, "stall_facts", None)
        try:
            return fn() if callable(fn) else {}
        except Exception as e:  # noqa: BLE001
            self._trace({"kind": "evidence_error", "error": repr(e)[:300]})
            return {}

    def _pinned_block(self) -> str:
        fn = getattr(self.tool, "pinned_block", None)
        try:
            return fn() if callable(fn) else ""
        except Exception:  # noqa: BLE001 - a broken repr must not cost a turn
            return ""

    def _pin_if_level_completed(self, session) -> None:
        """Snapshot memory['rules'] the moment the environment reports a new completed level."""
        done = session.current.levels_completed
        if done > self._pinned_levels:
            fn = getattr(self.tool, "pin_level", None)
            if callable(fn):
                try:
                    fn(done)
                except Exception as e:  # noqa: BLE001 - a broken repr must not cost a turn
                    self._trace({"kind": "pin_error", "error": repr(e)[:300]})
            self._pinned_levels = done

    def _memory_summary(self) -> str:
        fn = getattr(self.tool, "memory_summary", None)
        return fn() if callable(fn) else ""

    def _resume_text(self, session) -> str:
        obs = session.current
        return (prompts.RESUME + f"\nCurrent state: step {obs.step}, level {obs.levels_completed + 1}. "
                f"Valid actions: {', '.join(session.valid_actions)}.")

    # ------------------------------------------------------------ model call

    def _call_llm(self, deadline: Optional[float], max_tokens: Optional[int] = None):
        cfg = self.cfg
        max_tokens = max_tokens or cfg.output_tokens
        while True:
            timeout = cfg.llm_timeout
            if deadline is not None:
                timeout = max(1.0, min(timeout, deadline - time.time()))
            msgs = list(self.context.messages)
            self._acquire_slot(deadline)
            t0 = time.time()
            try:
                resp = self.llm.chat(msgs, tools=[PYTHON_TOOL], max_tokens=max_tokens,
                                     temperature=cfg.temperature, top_p=cfg.top_p,
                                     top_k=cfg.top_k, seed=cfg.seed, timeout=timeout)
            except Exception as e:
                if deadline is not None and time.time() >= deadline - 1.0:
                    # the request was cut by our own deadline: not a model failure
                    raise DeadlineReached("deadline reached during the model call") from e
                self.consecutive_failures += 1
                self.stats.llm_failures += 1
                self._trace({"kind": "model_error", "error": repr(e)[:500],
                             "consecutive": self.consecutive_failures})
                if self.consecutive_failures >= cfg.max_llm_failures:
                    raise AgentFailure(f"{self.consecutive_failures} consecutive model failures") from e
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
        if self._slot_since is None:
            self._slot_since = self._clock()

    def _park(self, deadline: Optional[float]) -> None:
        """The model keeps ending turns without acting: give the slot to other games for a while.
        The idle counter drops back to the notice threshold so the notice keeps showing."""
        self._release_slot()
        wait = self.cfg.idle_sleep_seconds
        if deadline is not None:
            wait = min(wait, max(0.0, deadline - time.time()))
        self._trace({"kind": "idle_park", "idle_turns": self.idle_turns, "seconds": round(wait, 1)})
        if wait > 0:
            self._sleep(wait)
        self.idle_turns = self.cfg.idle_turns_notice

    def _release_slot(self) -> None:
        if self.gate is not None and self.progress is not None:
            self.gate.release(self.progress)
        self._slot_since = None

    def _should_yield(self) -> bool:
        """Does a waiting game outrank us? Gates without should_yield always yield (old rule)."""
        if self.gate is None or self.progress is None:
            return False
        fn = getattr(self.gate, "should_yield", None)
        return True if fn is None else bool(fn(self.progress))

    def _rotate_slot(self) -> None:
        """Turn boundary: hand the slot over once we have held it long enough and are outranked,
        so the priority formula rotates slots instead of waiting for a trim."""
        if self._slot_since is None:
            return
        held = self._clock() - self._slot_since
        if held >= self.cfg.slot_rotate_seconds and self._should_yield():
            self._release_slot()
            self._trace({"kind": "slot_rotate", "held": round(held, 1)})

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
        self._trace({"kind": "tool_result", "text": text[:3000]})   # what the model actually saw
        return text, list(res.actions)

    # ------------------------------------------------------------ turn

    def play_turn(self, session, deadline: Optional[float] = None) -> TurnStats:
        if self.session is not session:
            self.reset_game(session)
        reset = getattr(session, "auto_reset_if_needed", None)
        if callable(reset):
            reset()
        self._rotate_slot()
        self._pin_if_level_completed(session)
        cfg, ts, t0 = self.cfg, TurnStats(), time.time()
        stalled = self._update_stall()
        opener = self._resume_text(session) if self.resume_next else self.build_opener(session)
        if stalled and self.stats.turns - self._last_stall_notice >= cfg.stall_notice_every:
            opener = _prepend_text(prompts.stall_notice(self._stall_facts()), opener)
            self._last_stall_notice = self.stats.turns
            self._trace({"kind": "stall_notice", "turn": self.stats.turns, "stall": stalled})
        if getattr(self.context, "trimmed_since_last_user", False):
            # older turns were dropped: remind the model what still persists (memory, functions)
            notice = prompts.trim_notice(self._memory_summary(), self.retained)
            opener = _prepend_text(notice, opener)
            self.context.trimmed_since_last_user = False
        if self.idle_turns >= cfg.idle_turns_notice:
            opener = _prepend_text(prompts.idle_notice(self.idle_turns), opener)
        self.context.add_user(opener)
        self.first_turn = False

        results: list[ActionResult] = []
        acted, nudged, reason = False, False, ""
        next_max_tokens: Optional[int] = None
        while not reason:
            if self.context.trim(cfg.context_tokens, cfg.drain_tokens):
                # The prefix cache is gone anyway, so hand the slot over if someone outranks us.
                # Not unconditionally: run #5 games waited 6,677-8,952 s to get it back.
                released = cfg.release_at_trim and self._should_yield()
                if released:
                    self._release_slot()
                self.stats.trims += 1
                self._trace({"kind": "trim", "estimate": self.context.total_tokens(),
                             "messages": len(self.context.messages), "released": released})
            resp = self._call_llm(deadline, max_tokens=next_max_tokens)
            next_max_tokens = None
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
                cut = resp.finish_reason == "length"
                if cut and cfg.output_tokens_after_cut:
                    next_max_tokens = cfg.output_tokens_after_cut
                self.context.add_user(getattr(prompts, "NUDGE_CUT", prompts.NUDGE) if cut else prompts.NUDGE)
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
        self.idle_turns = 0 if ts.actions else self.idle_turns + 1
        # Idle turns only get the notice; the slot is released by _park at idle_turns_sleep.
        # Run #9 released at the notice threshold: 169 releases, each re-waiting ~8 min for a slot
        # and dropping the warm prefix, on turns that were mostly legitimate analysis.
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
            if self.idle_turns >= self.cfg.idle_turns_sleep:
                self._park(deadline)
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
