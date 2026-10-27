"""The single `python` tool the model drives the game with.

`PythonTool(session).run(code)` executes a model snippet with game globals (frame views, history,
`frame_diff`, `action`) and returns a ToolResult. Actions go through `session.step`, which owns the
environment; this module adds the per-snippet guards (batch no-op stop, stale-state block, terminal
latch), output capping and retained helper functions.

Design notes:
- In-process, one worker thread per call with a 30 s wall clock. A runaway snippet cannot be killed
  from Python; on timeout we cancel cooperatively (the next `action()` raises), wait for any
  in-flight env step to finish, discard the snippet's globals and move on. A pure CPU loop keeps
  burning a core until it ends. This is a guardrail against model mistakes, not a security boundary.
- Unlike Duck, `.grid` exposes the raw numpy board (read-only). Code over exact cell values is more
  reliable for a 27B model than parsing ascii, and nothing hidden is in it.
- Model-visible output contains no game id, wall times or hidden engine state.
"""
from __future__ import annotations

import ast
import builtins
import random
import symtable
import threading
import time
import traceback
from typing import Any, Optional

import numpy as np

from arc3.types import Action, ActionResult, ToolResult
from arc3.wm import WM, Record

# numpy>=2 prints scalars as np.int64(3); the model should see plain 3 in tool output.
np.set_printoptions(legacy="1.25")

ALLOWED_MODULES = frozenset({
    "bisect", "collections", "copy", "difflib", "fractions", "functools", "heapq", "itertools",
    "json", "math", "operator", "random", "re", "statistics", "string",
    "numpy",  # search/scoring code over .grid
})
BLOCKED_BUILTINS = frozenset({
    "open", "exec", "eval", "compile", "input", "breakpoint", "globals", "locals", "vars", "exit",
    "quit", "help", "memoryview", "__build_class__", "__loader__", "__spec__", "copyright",
    "credits", "license",
})
SAFE_BUILTIN_NAMES = frozenset(k for k in vars(builtins) if k not in BLOCKED_BUILTINS)
PROVIDED = ("current_frame", "previous_frame", "history", "transitions", "last_transition",
            "valid_actions", "last_action_call_result", "last_animation_frames", "frame_diff",
            "action", "result", "wm", "np", "memory")


class AttrDict(dict):
    """Result dict that also answers attribute access (`res.executed` or `res['executed']`)."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None
TIMEOUT_S = 30.0
MAX_OUTPUT_CHARS = 12_000          # ~3000 tokens
MAX_ACTIONS_PER_CALL = 64
DIFF_CHARS = 200                   # cap for the one-line per-action object diff
DIFF_ITEMS = 3                     # objects detailed per diff category
SNIPPET = "<snippet>"


class StaleStateError(RuntimeError):
    """action() called again after an ineffective action in the same snippet."""


class TerminalStateError(RuntimeError):
    """action() called after a death or level-up in the same snippet."""


class ActionLimitError(RuntimeError):
    """The snippet exhausted its per-call action budget."""


def _percept():
    from arc3 import percept  # imported lazily so the sandbox loads before percept exists
    return percept


class FrameView:
    """Read-only view of one board as the model sees it."""

    def __init__(self, grid: np.ndarray, step: int, level: int):
        self.grid = grid.view()
        self.grid.flags.writeable = False
        self.step, self.level, self.shape = step, level, tuple(grid.shape)
        self._ascii: Optional[str] = None
        self._seg: Any = None

    @classmethod
    def of(cls, obs) -> Optional["FrameView"]:
        return None if obs is None else cls(obs.grid, obs.step, obs.levels_completed + 1)

    @property
    def ascii(self) -> str:
        if self._ascii is None:
            self._ascii = _percept().ascii(self.grid)
        return self._ascii

    @property
    def segmentation(self):
        if self._seg is None:
            self._seg = _percept().segment(self.grid)
        return self._seg

    def crop(self, r0: int, c0: int, r1: int, c1: int) -> "FrameView":
        return FrameView(self.grid[r0:r1, c0:c1], self.step, self.level)

    def __repr__(self) -> str:
        return f"<Frame step={self.step} level={self.level} shape={self.shape}>"


class TransitionView:
    def __init__(self, t):
        self.index, self.action = t.index, t.action.display()
        self.before, self.after = FrameView.of(t.before), FrameView.of(t.after)
        # Aliases the prompt mentions / models reach for.
        self.before_frame, self.after_frame, self.frame = self.before, self.after, self.after
        self.level = self.before.level
        r = t.result
        self.board_changed, self.gameplay_changed = r.board_changed, r.gameplay_changed
        self.level_completed, self.game_over = r.level_completed, r.game_over
        self.message = r.message
        self.result = AttrDict(action=self.action, executed=r.executed,
                               board_changed=r.board_changed, gameplay_changed=r.gameplay_changed,
                               level_completed=r.level_completed, game_over=r.game_over,
                               run_complete=r.run_complete, animated=r.animated,
                               message=r.message)

    def __getitem__(self, key):
        """Dict-style access (`history[-1]['action']`) for models that treat entries as dicts."""
        if isinstance(key, str) and not key.startswith("_") and hasattr(self, key):
            return getattr(self, key)
        raise KeyError(key)

    def keys(self):
        return ["index", "action", "level", "frame", "before_frame", "after_frame", "result",
                "gameplay_changed", "level_completed", "game_over", "message"]

    def __repr__(self) -> str:
        return f"<Transition #{self.index} {self.action} changed={self.gameplay_changed}>"


def truncate_middle(text: str, limit: int = MAX_OUTPUT_CHARS) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    head = limit * 2 // 3
    tail = limit - head
    cut = len(text) - head - tail
    return f"{text[:head]}\n... [{cut} chars truncated] ...\n{text[-tail:]}", True


def parse_action(spec) -> Action:
    if isinstance(spec, Action):
        return spec
    if isinstance(spec, str):
        name = spec.strip().upper()
        if name.startswith("MOUSE(") and name.endswith(")"):
            r, c = name[6:-1].split(",")
            return Action("MOUSE", int(r), int(c))
        if name == "MOUSE":
            raise ValueError("MOUSE needs coordinates: {'action':'MOUSE','row':r,'col':c}")
        return Action(name)
    if isinstance(spec, dict):
        name = str(spec.get("action", "")).upper()
        if name == "MOUSE":
            return Action("MOUSE", int(spec["row"]), int(spec["col"]))
        return Action(name)
    raise ValueError(f"cannot parse action {spec!r}")


def _cell(bbox) -> str:
    return f"({bbox[0]},{bbox[1]})"


def diff_line(before: Optional[np.ndarray], after: Optional[np.ndarray]) -> str:
    """Compact one-line object diff of one action, e.g. 'moved: 1 (K (10,4)->(10,6)); ...'.

    Every game in run 1 hand-wrote a "what moved" tracker; this gives it for free. It uses the
    heuristic percept.frame_diff on the settled boards, so matches are best-effort.
    """
    if before is None or after is None:
        return ""
    if np.array_equal(before, after):
        return "no change"
    d = _percept().frame_diff(before, after)
    fmt = {
        "moved": lambda m: f"{m['color']} {_cell(m['from'])}->{_cell(m['to'])}",
        "rotated": lambda m: f"{m['color']} {m['degrees']}deg at {_cell(m['from'])}",
        "appeared": lambda m: f"{m['color']} {m['pixels']}px at {_cell(m['bbox'])}",
        "disappeared": lambda m: f"{m['color']} {m['pixels']}px at {_cell(m['bbox'])}",
        "changed_color": lambda m: f"{m['from']}->{m['to']} at {_cell(m['bbox'])}",
        "resized": lambda m: f"{m['color']} {m['pixels'][0]}->{m['pixels'][1]}px at "
                             f"{_cell(m['to'])}",
    }
    parts = []
    for key, f in fmt.items():
        items = d.get(key) or []
        if not items and key in ("rotated", "resized"):
            continue   # rarer categories only when present, to keep the line short
        shown = ", ".join(f(m) for m in items[:DIFF_ITEMS])
        more = f", +{len(items) - DIFF_ITEMS}" if len(items) > DIFF_ITEMS else ""
        parts.append(f"{key}: {len(items)}" + (f" ({shown}{more})" if items else ""))
    text = "; ".join(parts) + f"; cells: {d['changed_cell_count']}"
    return text if len(text) <= DIFF_CHARS else text[:DIFF_CHARS - 3] + "..."


def _animation_summary(before: Optional[np.ndarray], frames: list[np.ndarray]):
    """(summary dict or None, collapsed frame chain) for one action's animation."""
    if before is None or len(frames) < 2:
        return None, []
    p = _percept()
    chain = p.collapse_frames(before, frames)
    if len(chain) <= 2:
        return None, chain
    count, bbox = p.transient_cells(chain)
    return {"frames": len(chain), "transient_pixels": count, "transient_bbox": bbox}, chain


class _Run:
    """Per-snippet mutable state: guards, executed actions, output."""

    def __init__(self, max_actions: int):
        self.results: list[ActionResult] = []
        self.out: list[str] = []
        self.ineffective = False
        self.terminal = False
        self.cancelled = False
        self.budget = max_actions
        self.lock = threading.Lock()


class PythonTool:
    def __init__(self, session, timeout: float = TIMEOUT_S,
                 max_actions_per_call: int = MAX_ACTIONS_PER_CALL):
        self.session = session
        self.timeout = timeout
        self.max_actions = max_actions_per_call
        self.retained: dict[str, dict] = {}   # name -> {"source": str, "imports": [str]}
        self.last_action_call_result: Optional[dict] = None
        self.last_animation_frames: list = []
        # Plain dict that survives across calls (variables do not); the model stores plans here.
        self.memory: dict = {}
        self._views: dict[int, TransitionView] = {}
        self._calls = 0
        self.wm = WM(record=self._wm_record, hud_border=session.config.hud_border)

    # ---- globals -------------------------------------------------------------------------------

    def _builtins(self, run: _Run) -> dict:
        safe = {k: getattr(builtins, k) for k in SAFE_BUILTIN_NAMES}

        def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
            if level != 0 or name.split(".")[0] not in ALLOWED_MODULES:
                raise ImportError(f"import of {name!r} is not allowed; allowed: "
                                  + ", ".join(sorted(ALLOWED_MODULES)))
            return __import__(name, globals, locals, fromlist, level)

        def sandbox_print(*args, sep=" ", end="\n", file=None, flush=False):
            run.out.append(sep.join(str(a) for a in args) + end)

        safe["__import__"] = guarded_import
        safe["print"] = sandbox_print
        return safe

    def _refresh(self, ns: dict) -> None:
        s = self.session
        for t in s.history:
            if t.index not in self._views:
                self._views[t.index] = TransitionView(t)
        views = [self._views[t.index] for t in s.history]
        ns["current_frame"] = FrameView.of(s.current)
        ns["previous_frame"] = FrameView.of(s.previous)
        ns["transitions"] = views
        ns["history"] = views   # same objects: attribute and dict-style access both work
        ns["last_transition"] = views[-1] if views else None
        ns["valid_actions"] = list(s.valid_actions)
        ns["last_action_call_result"] = self.last_action_call_result
        ns["last_animation_frames"] = self.last_animation_frames

    def _namespace(self, run: _Run) -> dict:
        ns: dict[str, Any] = {"__builtins__": self._builtins(run), "__name__": "__sandbox__"}

        def frame_diff(before=None, after=None):
            before = ns["previous_frame"] if before is None else before
            after = ns["current_frame"] if after is None else after
            if before is None or after is None:
                raise ValueError("frame_diff needs two frames (no previous frame yet)")
            grid = lambda f: f.grid if isinstance(f, FrameView) else np.asarray(f)
            return _percept().frame_diff(grid(before), grid(after))

        ns["frame_diff"] = frame_diff
        ns["np"] = np   # models assume numpy is already imported
        ns["action"] = lambda actions: self._action(run, ns, actions)
        # One WM per game (keeps the registered model across calls), rebound to this snippet.
        self.wm.executor = ns["action"]
        self.wm._cancel = lambda: run.cancelled
        ns["wm"] = self.wm
        ns["result"] = None
        ns["memory"] = self.memory
        self._refresh(ns)
        return ns

    def _wm_record(self) -> Record:
        """Settled transitions per level from the session history (automatic RESETs dropped)."""
        s = self.session
        by_level: dict[int, Record] = {}
        for t in s.history:
            if t.action.name == "RESET" or not t.result.executed:
                continue
            lv = t.before.levels_completed + 1
            rec = by_level.get(lv)
            if rec is None:
                rec = by_level[lv] = Record(level_start_grid=t.before.grid, level=lv)
            r = t.result
            # A silent level reset is a death in all but name; recording it as game_over lets
            # verify skip grading its dynamics and the planner prune it.
            outcome = ("game_over" if r.game_over or r.level_reset else
                       "level_complete" if r.level_completed or r.run_complete else "ongoing")
            rec.transitions.append((t.before.grid, {"action": t.action.name, "row": t.action.row,
                                                    "col": t.action.col}, t.after.grid, outcome))
        cur = s.level
        current = by_level.pop(cur, None) or Record(level_start_grid=s.current.grid, level=cur)
        current.available_actions = list(s.valid_actions)
        current.live_grid = s.current.grid   # differs from the last `after` after an auto RESET
        current.previous = [by_level[k] for k in sorted(by_level) if k < cur]
        return current

    # ---- action() ------------------------------------------------------------------------------

    def _action(self, run: _Run, ns: dict, actions) -> dict:
        if run.cancelled:
            raise TimeoutError("snippet timed out; no further actions")
        if run.terminal:
            raise TerminalStateError("the level ended, was reset, or the game is over in this "
                                     "snippet; end the snippet and look at the new state")
        if isinstance(actions, (str, dict, Action)):
            actions = [actions]
        cfg = self.session.config
        guards = self.session.level >= cfg.guards_from_level
        if run.ineffective and guards:
            raise StaleStateError("the previous action changed nothing; end the snippet and "
                                  "re-examine the board before acting again")
        if run.budget <= 0:
            raise ActionLimitError(f"at most {self.max_actions} actions per python call")
        parsed = [parse_action(a) for a in actions]
        valid = set(self.session.valid_actions)
        for a in parsed:
            if a.name == "RESET" and not cfg.expose_reset:
                raise ValueError("RESET is not available")
            if a.name not in valid:
                raise ValueError(f"{a.display()} is not a valid action now; valid: {sorted(valid)}")

        summary = AttrDict({"executed": 0, "skipped": len(parsed), "stop_reason": None,
                            "level_completed": False, "game_over": False, "run_complete": False,
                            "level_reset": False, "animation": None, "per_action": [],
                            "frame": ns["current_frame"]})
        for a in parsed:
            if run.budget <= 0:
                summary["stop_reason"] = "max_actions_per_call"
                break
            with run.lock:
                if run.cancelled:
                    summary["stop_reason"] = "timeout"
                    break
                res = self.session.step(a)
                run.results.append(res)  # under the lock so a timed-out call still reports it
            run.budget -= 1
            summary["skipped"] -= 1
            if res.executed:
                try:
                    res.diff = diff_line(res.before, res.after)
                except Exception as e:  # a percept bug must not break acting
                    res.diff = f"diff unavailable ({type(e).__name__})"
            summary["per_action"].append({
                "action": a.display(), "executed": res.executed,
                "gameplay_changed": res.gameplay_changed, "board_changed": res.board_changed,
                "level_completed": res.level_completed, "game_over": res.game_over,
                "level_reset": res.level_reset, "stop_reason": res.stop_reason,
                "message": res.message, "diff": res.diff})
            if not res.executed:
                summary["stop_reason"] = res.stop_reason or "refused"
                break
            summary["executed"] += 1
            for key in ("level_completed", "game_over", "run_complete", "level_reset"):
                summary[key] = summary[key] or getattr(res, key)
            self._refresh(ns)
            anim, chain = _animation_summary(res.before, res.frames)
            if anim is not None:  # keep only the latest animated action, as frame views
                summary["animation"] = anim
                cur = ns["current_frame"]
                self.last_animation_frames = [FrameView(g, cur.step, cur.level) for g in chain]
            if res.level_completed or res.game_over or res.run_complete or res.level_reset:
                run.terminal = True
                summary["stop_reason"] = "game_over" if res.game_over else (
                    "level_completed" if res.level_completed else
                    "level_reset" if res.level_reset else "run_complete")
                break
            if not res.gameplay_changed:
                run.ineffective = True
                if guards:
                    summary["stop_reason"] = "no_op"
                    break
        summary["frame"] = ns["current_frame"]   # board after the batch settled
        self.last_action_call_result = summary
        ns["last_action_call_result"] = summary
        ns["last_animation_frames"] = self.last_animation_frames
        return summary

    # ---- run() ---------------------------------------------------------------------------------

    def run(self, code: str) -> ToolResult:
        t0 = time.monotonic()
        self._calls += 1
        run = _Run(self.max_actions)
        ns = self._namespace(run)
        restore_errors = self._restore(ns)
        random.seed(self.session.config.seed * 100_003 + self._calls)
        error: Optional[str] = None
        try:
            compiled = compile(code, SNIPPET, "exec")
        except SyntaxError as e:
            compiled, error = None, f"SyntaxError: {e.msg} (line {e.lineno})"

        box: dict[str, Optional[str]] = {"error": None}
        if compiled is not None:
            def target():
                try:
                    exec(compiled, ns)
                except BaseException as e:  # SystemExit/KeyboardInterrupt too: never kill host
                    box["error"] = _compact_traceback(e, code)

            worker = threading.Thread(target=target, daemon=True, name="sandbox")
            worker.start()
            worker.join(self.timeout)
            if worker.is_alive():
                with run.lock:  # wait for an in-flight env step, then block further steps
                    run.cancelled = True
                error = f"TimeoutError: python call exceeded {self.timeout:g} s; globals discarded"
            else:
                error = box["error"]
                if ns.get("result") is not None:
                    run.out.append(f"result = {ns['result']!r}\n")

        retained, rejected = ([], {})
        if compiled is not None and not run.cancelled:
            try:
                retained, rejected = self._collect(code, ns)
            except Exception as e:  # never lose the call's actions over a retention bug
                rejected = {"*": f"function retention failed: {type(e).__name__}: {e}"}
        rejected.update(restore_errors)
        stdout, truncated = truncate_middle("".join(run.out))
        return ToolResult(stdout=stdout, error=error, actions=list(run.results),
                          truncated=truncated, retained_functions=retained,
                          rejected_functions=rejected, seconds=time.monotonic() - t0)

    # ---- retained functions --------------------------------------------------------------------

    def _restore(self, ns: dict) -> dict[str, str]:
        errors = {}
        for name, entry in list(self.retained.items()):
            try:
                exec(compile("\n".join(entry["imports"] + [entry["source"]]), "<retained>", "exec"),
                     ns)
            except Exception as e:
                del self.retained[name]
                errors[name] = f"dropped: failed to restore ({type(e).__name__}: {e})"
        return errors

    def _collect(self, code: str, ns: dict) -> tuple[list[str], dict[str, str]]:
        tree = ast.parse(code)
        imports: dict[str, str] = {}   # bound name -> import line
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                mod = node.module if isinstance(node, ast.ImportFrom) else None
                for alias in node.names:
                    if getattr(node, "level", 0) or alias.name == "*":
                        continue
                    if (mod or alias.name).split(".")[0] not in ALLOWED_MODULES:
                        continue
                    as_ = f" as {alias.asname}" if alias.asname else ""
                    line = (f"from {mod} import {alias.name}{as_}" if mod
                            else f"import {alias.name}{as_}")
                    imports[(alias.asname or alias.name).split(".")[0]] = line
        rejected: dict[str, str] = {}
        fresh: dict[str, dict] = {}
        for node in tree.body:
            if not isinstance(node, ast.FunctionDef):
                continue
            name = node.name
            self.retained.pop(name, None)  # redefinition replaces, even if the new one is rejected
            reason = _shape_problem(node, ns)
            if reason:
                rejected[name] = reason
                continue
            src = ast.get_source_segment(code, node)
            deps = _global_names(src, name)
            fresh[name] = {"source": src, "deps": deps - set(imports),
                           "imports": sorted({imports[d] for d in deps if d in imports})}
        known = SAFE_BUILTIN_NAMES | set(PROVIDED)
        candidates = {**self.retained, **fresh}
        changed = True
        while changed:  # drop functions with unresolved names until a fixed point
            changed = False
            for name, entry in list(candidates.items()):
                missing = [d for d in entry["deps"] if d not in known and d not in candidates]
                if missing:
                    rejected[name] = (f"uses {', '.join(sorted(missing))} which is not available "
                                      "later; pass it as an argument or import it at top level")
                    del candidates[name]
                    changed = True
        self.retained = candidates
        return sorted(n for n in candidates if n in fresh), rejected


def _shape_problem(node: ast.FunctionDef, ns: dict) -> Optional[str]:
    if node.name in PROVIDED:
        return f"name {node.name} conflicts with a provided global"
    if not callable(ns.get(node.name)):
        return "definition was not executed"
    if node.decorator_list:
        return "decorators are not retained; use a plain top-level def"
    if getattr(node, "type_params", None):
        return "generic type parameters are not retained"
    a = node.args
    if node.returns or any(x.annotation for x in a.posonlyargs + a.args + a.kwonlyargs
                           + [v for v in (a.vararg, a.kwarg) if v]):
        return "annotations are not retained; remove them"
    for d in a.defaults + [d for d in a.kw_defaults if d is not None]:
        try:
            ast.literal_eval(d)
        except ValueError:
            return "defaults must be literals"
    if any(isinstance(n, (ast.Global, ast.Nonlocal)) for n in ast.walk(node)):
        return "functions that write globals are not retained; return values instead"
    return None


def _global_names(src: str, name: str) -> set[str]:
    """Names a function (and its nested scopes) read from module scope."""
    top = symtable.symtable(src, SNIPPET, "exec")
    out: set[str] = set()
    stack = [c for c in top.get_children() if c.get_name() == name]
    while stack:
        t = stack.pop()
        out.update(s.get_name() for s in t.get_symbols() if s.is_global() and s.is_referenced())
        stack.extend(t.get_children())
    out.discard(name)
    return out


def _compact_traceback(exc: BaseException, code: str) -> str:
    lines = code.splitlines()
    frames = [f for f in traceback.extract_tb(exc.__traceback__) if f.filename == SNIPPET]
    parts = []
    for f in frames[-3:]:
        text = lines[f.lineno - 1].strip() if f.lineno and f.lineno <= len(lines) else ""
        parts.append(f"  line {f.lineno}, in {f.name}: {text}")
    head = "Traceback (snippet lines only):\n" if parts else ""
    return head + "\n".join(parts) + ("\n" if parts else "") + f"{type(exc).__name__}: {exc}"
