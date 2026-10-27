"""Executable world-model tools exposed to the model as `wm` in the sandbox.

The model registers a simulator (`init`, `transition`, `outcome`, optional `actions`, `heuristic`,
`planner_key`) over a state dict that holds at least `'grid'`. The harness supplies the settled
transitions of the current level as a `Record`. On top of that this module offers:

- `verify()`  three channels (init render, dynamics, outcome) with a compact first-divergence diff,
- `plan()`    BFS / A* over the model from the replayed current state,
- `check()` / `plan_abstract()`  the light path: the model's own read(grid)->state and
  step(state, action)->state, checked against history and searched, no grid rendering,
- `validate()` / `expected_hashes()` for hash-anchored plan execution by the harness,
- mechanical helpers that need no model: `induce_motion`, `transition_graph`, `untried_actions`,
  `goal_menu`.

Pure: numpy + stdlib only, no env or sandbox imports. The sandbox runs snippets in a worker thread,
so time caps are cooperative (checked between model calls); one call that never returns cannot be
interrupted here, the sandbox's own wall clock covers that case.
"""
from __future__ import annotations

import copy
import hashlib
import heapq
import time
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

OUTCOMES = ("ongoing", "level_complete", "game_over")
DIRECTIONS = ("UP", "DOWN", "LEFT", "RIGHT")
VERIFY_SECONDS = 10.0
PLAN_SECONDS = 20.0
MAX_DIFF_RUNS = 12
MAX_CLICKS = 48


@dataclass
class Record:
    """Settled transitions of one level: (before, action, after, outcome) tuples."""
    level_start_grid: np.ndarray
    transitions: list = field(default_factory=list)
    level: int = 0
    available_actions: list = field(default_factory=list)
    previous: list = field(default_factory=list)   # Records of earlier levels, oldest first
    live_grid: Optional[np.ndarray] = None         # observed board now, if not the last `after`
    # Transition indices preceded by a RESET (len(transitions) = a RESET after the last one). The
    # model's state restarts from init(before) there, whatever the boards look like.
    restarts: list = field(default_factory=list)

    @property
    def current_grid(self) -> np.ndarray:
        if self.live_grid is not None:
            return self.live_grid
        return self.transitions[-1][2] if self.transitions else self.level_start_grid


# ---------------------------------------------------------------- small helpers (no percept dep)

class _Timeout(Exception):
    pass


def grid_hash(grid) -> str:
    """Stable 16-hex hash of a grid (shape + uint8 cells). The harness compares observed grids
    against `expected_hashes()` with this same function."""
    g = np.ascontiguousarray(np.asarray(grid), dtype=np.uint8)
    h = hashlib.blake2b(digest_size=8)
    h.update(str(g.shape).encode())
    h.update(g.tobytes())
    return h.hexdigest()


def norm_action(a) -> dict:
    """Accept 'UP', {'action': 'UP'}, ('MOUSE', r, c) or an Action-like object; return a dict."""
    if isinstance(a, str):
        return {"action": a.upper(), "row": None, "col": None}
    if isinstance(a, dict):
        return {"action": str(a.get("action", a.get("name"))).upper(),
                "row": a.get("row"), "col": a.get("col")}
    if isinstance(a, (tuple, list)):
        return {"action": str(a[0]).upper(), "row": a[1] if len(a) > 1 else None,
                "col": a[2] if len(a) > 2 else None}
    return {"action": str(a.name).upper(), "row": getattr(a, "row", None),
            "col": getattr(a, "col", None)}


def action_str(a) -> str:
    a = norm_action(a)
    return f"MOUSE({a['row']},{a['col']})" if a["action"] == "MOUSE" else a["action"]


def components(grid, colors=None) -> list[dict]:
    """4-connected same-colour components: {color, cells (frozenset of (r,c)), bbox, size}."""
    g = np.asarray(grid)
    seen = np.zeros(g.shape, dtype=bool)
    out = []
    rows, cols = g.shape
    for r0 in range(rows):
        for c0 in range(cols):
            if seen[r0, c0] or (colors is not None and int(g[r0, c0]) not in colors):
                continue
            color = g[r0, c0]
            stack, cells = [(r0, c0)], []
            seen[r0, c0] = True
            while stack:
                r, c = stack.pop()
                cells.append((r, c))
                for nr, nc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                    if 0 <= nr < rows and 0 <= nc < cols and not seen[nr, nc] and g[nr, nc] == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            rs = [p[0] for p in cells]
            cs = [p[1] for p in cells]
            out.append({"color": int(color), "cells": frozenset(cells), "size": len(cells),
                        "bbox": (min(rs), min(cs), max(rs), max(cs))})
    return out


def background(grid) -> int:
    vals, counts = np.unique(np.asarray(grid), return_counts=True)
    return int(vals[np.argmax(counts)])


def diff_text(expected, predicted, cap: int = MAX_DIFF_RUNS) -> dict:
    """Run-length diff 'exp->pred @ rR cA-B; ...' plus bbox (r0,c0,r1,c1) and mismatched cell count."""
    e, p = np.asarray(expected), np.asarray(predicted)
    if e.shape != p.shape:
        return {"diff": f"shape {e.shape} expected, {p.shape} predicted", "bbox": None,
                "n_cells": -1}
    rr, cc = np.nonzero(e != p)
    if len(rr) == 0:
        return {"diff": "", "bbox": None, "n_cells": 0}
    runs = []
    for r in range(int(rr.min()), int(rr.max()) + 1):
        c = 0
        while c < e.shape[1]:
            if e[r, c] == p[r, c]:
                c += 1
                continue
            key, start = (int(e[r, c]), int(p[r, c])), c
            while c + 1 < e.shape[1] and (int(e[r, c + 1]), int(p[r, c + 1])) == key:
                c += 1
            span = f"c{start}" if start == c else f"c{start}-{c}"
            runs.append(f"{key[0]}->{key[1]} @ r{r} {span}")
            c += 1
    text = "; ".join(runs[:cap])
    if len(runs) > cap:
        text += f"; ... +{len(runs) - cap} more runs"
    return {"diff": text, "bbox": (int(rr.min()), int(cc.min()), int(rr.max()), int(cc.max())),
            "n_cells": int(len(rr))}


def _clone(state: dict) -> dict:
    out = {}
    for k, v in state.items():
        if isinstance(v, np.ndarray):
            out[k] = v.copy()
        elif isinstance(v, (int, float, str, bool, bytes, type(None), frozenset)):
            out[k] = v
        else:
            out[k] = copy.deepcopy(v)
    return out


def _key_part(v):
    if isinstance(v, np.ndarray):
        return (v.shape, v.tobytes())
    if isinstance(v, dict):
        return tuple(sorted((str(k), _key_part(x)) for k, x in v.items()))
    if isinstance(v, (list, tuple)):
        return tuple(_key_part(x) for x in v)
    if isinstance(v, set):
        return tuple(sorted(repr(x) for x in v))
    return repr(v)


def default_key(state: dict):
    g = np.asarray(state["grid"])
    hidden = tuple(sorted((k, _key_part(v)) for k, v in state.items() if k != "grid"))
    return (g.shape, g.tobytes(), hidden)


def _err(e: Exception) -> str:
    return f"{type(e).__name__}: {e}"[:300]


# ---------------------------------------------------------------- the model-facing namespace

class WM:
    """World-model tools. The harness constructs one per game and keeps `record` current
    (a Record or a zero-arg callable returning one)."""

    def __init__(self, record=None, hud_border: Optional[int] = None,
                 cancel: Optional[Callable[[], bool]] = None):
        self._record_src = record
        self.hud_border = hud_border
        self._cancel = cancel or (lambda: False)
        self.model: Optional[dict] = None
        # Set by the sandbox per snippet: runs a list of actions for real, returns the call summary.
        self.executor: Optional[Callable[[list], dict]] = None

    # ---- harness side
    def set_record(self, record) -> None:
        self._record_src = record

    def _rec(self, record=None) -> Record:
        r = record if record is not None else self._record_src
        r = r() if callable(r) else r
        if r is None:
            raise ValueError("no record available yet")
        return r

    # ---- model side
    def register(self, transition, outcome, init=None, planner_key=None, actions=None,
                 heuristic=None) -> str:
        """Register your simulator. State is a dict with 'grid' (numpy int array) plus any hidden
        fields. transition(state, action_dict)->new_state; outcome(state)->'ongoing'|
        'level_complete'|'game_over'; init(grid0, level)->state (default {'grid': grid0.copy()});
        actions(state)->list of action dicts for the planner; heuristic(state)->float (enables A*);
        planner_key(state)->hashable. Action dicts look like {'action':'UP','row':None,'col':None}
        - compare a['action'] == 'UP', never a == 'UP'. Call wm.verify() next."""
        if not callable(transition) or not callable(outcome):
            raise TypeError("transition and outcome must be functions")
        self.model = {"transition": transition, "outcome": outcome,
                      "init": init or (lambda g, level: {"grid": np.array(g, copy=True)}),
                      "planner_key": planner_key or default_key, "actions": actions,
                      "heuristic": heuristic}
        return "registered; now call wm.verify()"

    def _need_model(self):
        if self.model is None:
            raise RuntimeError("no model registered: call wm.register(transition, outcome, ...)")

    def _check(self, deadline: float):
        if time.monotonic() > deadline or self._cancel():
            raise _Timeout()

    def _init(self, grid, level) -> dict:
        s = self.model["init"](np.array(grid, copy=True), level)
        if not isinstance(s, dict) or "grid" not in s:
            raise TypeError("init must return a dict containing 'grid'")
        s["grid"] = np.asarray(s["grid"])
        return s

    def _step(self, state: dict, action: dict) -> dict:
        s = self.model["transition"](_clone(state), dict(action))
        if not isinstance(s, dict) or "grid" not in s:
            raise TypeError("transition must return a dict containing 'grid'")
        s["grid"] = np.asarray(s["grid"])
        return s

    def _outcome(self, state: dict) -> str:
        o = self.model["outcome"](state)
        if o not in OUTCOMES:
            raise ValueError(f"outcome returned {o!r}; expected one of {OUTCOMES}")
        return o

    def border(self, grid) -> int:
        """Width of the HUD band ignored by non-strict comparisons."""
        if self.hud_border is not None:
            return self.hud_border
        return 4 if min(np.asarray(grid).shape) >= 32 else 0

    def view(self, grid, strict: bool = False) -> np.ndarray:
        """The grid as compared: HUD border cells blanked to 255 unless strict."""
        g = np.array(grid, dtype=np.int16, copy=True)
        b = 0 if strict else self.border(g)
        if b and g.ndim == 2:
            g[:b, :] = g[-b:, :] = g[:, :b] = g[:, -b:] = 255
        return g

    def view_hash(self, grid, strict: bool = False) -> str:
        """Hash used by expected_hashes/execute: grid_hash of view(grid, strict)."""
        return grid_hash(self.view(grid, strict))

    def verify(self, record=None, all_levels: bool = False, timeout: float = VERIFY_SECONDS,
               strict: bool = False) -> dict:
        """Check the registered model against recorded settled transitions of the current level
        (plus earlier levels if all_levels=True; off by default so a new mechanic does not force
        re-modelling every solved level). The outer HUD border (4 cells) is ignored unless
        strict=True. Channels: init (init(start)['grid'] == observed start),
        dynamics (replay; a step is graded if the observed grid changed or the model predicted a
        change), outcome (model outcome == recorded outcome at every step). Returns
        {ok, summary, init, dynamics:{accuracy, graded, correct, first_divergence}, outcome}.
        first_divergence.diff reads 'expected->predicted @ row cols'."""
        self._need_model()
        rec = self._rec(record)
        levels = (list(rec.previous) if all_levels else []) + [rec]
        t0 = time.monotonic()
        deadline = t0 + timeout
        res = {"ok": False, "timeout": False,
               "init": {"ok": True, "failures": []},
               "dynamics": {"accuracy": 1.0, "graded": 0, "correct": 0, "first_divergence": None},
               "outcome": {"ok": True, "mismatches": []}}
        try:
            for lv in levels:
                self._verify_level(lv, res, deadline, strict)
        except _Timeout:
            res["timeout"] = True
        d = res["dynamics"]
        d["accuracy"] = round(d["correct"] / d["graded"], 4) if d["graded"] else 1.0
        res["outcome"]["ok"] = not res["outcome"]["mismatches"]
        res["ok"] = (not res["timeout"] and res["init"]["ok"] and res["outcome"]["ok"]
                     and d["correct"] == d["graded"])
        res["seconds"] = round(time.monotonic() - t0, 3)
        res["summary"] = self._summary(res)
        return res

    def _verify_level(self, lv: Record, res: dict, deadline: float, strict: bool):
        v = lambda g: self.view(g, strict)
        init_ch, dyn, out_ch = res["init"], res["dynamics"], res["outcome"]
        try:
            state = self._init(lv.level_start_grid, lv.level)
            dt = diff_text(v(lv.level_start_grid), v(state["grid"]))
            if dt["n_cells"]:
                init_ch["ok"] = False
                init_ch["failures"].append({"level": lv.level, **dt})
        except Exception as e:  # noqa: BLE001 - model code may raise anything
            init_ch["ok"] = False
            init_ch["failures"].append({"level": lv.level, "error": _err(e)})
            return
        prev_after = lv.level_start_grid
        restarts = set(lv.restarts)
        for i, (before, action, after, rec_out) in enumerate(lv.transitions):
            self._check(deadline)
            action = norm_action(action)
            before, after = np.asarray(before), np.asarray(after)
            if i in restarts or before.shape != prev_after.shape or \
                    not np.array_equal(before, prev_after):
                # Discontinuity (reset after game over, unmodelled jump): restart from what we saw.
                try:
                    state = self._init(before, lv.level)
                except Exception:  # noqa: BLE001
                    state = {"grid": before.copy()}
            state["grid"] = before.copy()  # teacher-force the grid, keep hidden fields
            where = {"level": lv.level, "index": i, "action": action_str(action)}
            try:
                pred = self._step(state, action)
            except Exception as e:  # noqa: BLE001
                dyn["graded"] += 1
                if dyn["first_divergence"] is None:
                    dyn["first_divergence"] = {**where, "error": _err(e)}
                out_ch["mismatches"].append({**where, "expected": rec_out, "predicted": "error"})
                state, prev_after = {"grid": after.copy()}, after
                continue
            if rec_out == "ongoing":
                vb, va, vp = v(before), v(after), v(pred["grid"])
                observed_change = not np.array_equal(vb, va)
                predicted_change = vp.shape != vb.shape or not np.array_equal(vp, vb)
                if observed_change or predicted_change:
                    dyn["graded"] += 1
                    dt = diff_text(va, vp)
                    if dt["n_cells"] == 0:
                        dyn["correct"] += 1
                    elif dyn["first_divergence"] is None:
                        dyn["first_divergence"] = {**where, **dt}
            try:
                p_out = self._outcome(pred)
            except Exception as e:  # noqa: BLE001
                p_out = "error: " + _err(e)
            if p_out != rec_out and len(out_ch["mismatches"]) < 10:
                out_ch["mismatches"].append({**where, "expected": rec_out, "predicted": p_out})
            state, prev_after = pred, after

    @staticmethod
    def _summary(res: dict) -> str:
        if res["timeout"]:
            return "timeout: model too slow or record too long; partial results"
        d = res["dynamics"]
        parts = [f"init {'ok' if res['init']['ok'] else 'FAIL'}",
                 f"dynamics {d['correct']}/{d['graded']}",
                 f"outcome {'ok' if res['outcome']['ok'] else 'FAIL'}"]
        fd = d["first_divergence"]
        if fd:
            detail = fd.get("error") or fd.get("diff")
            parts.append(f"first divergence L{fd['level']} #{fd['index']} {fd['action']}: {detail}")
        return "; ".join(parts)

    def current_state(self, record=None, _deadline: Optional[float] = None) -> dict:
        """Model state for the live grid: replays this level's real history through your model so
        hidden fields are carried; the grid is always the observed one. Falls back to
        init(current grid) if your model raises."""
        self._need_model()
        rec = self._rec(record)
        cur = np.asarray(rec.current_grid)
        try:
            state = self._init(rec.level_start_grid, rec.level)
            prev_after = np.asarray(rec.level_start_grid)
            restarts = set(rec.restarts)
            # Only the segment after the last RESET shapes the current state.
            first = max([i for i in restarts if i < len(rec.transitions)], default=0)
            for i, (before, action, after, _) in enumerate(rec.transitions):
                if i < first:
                    continue
                if _deadline is not None:
                    self._check(_deadline)
                before = np.asarray(before)
                if i in restarts or before.shape != prev_after.shape or \
                        not np.array_equal(before, prev_after):
                    state = self._init(before, rec.level)
                state["grid"] = before.copy()
                state = self._step(state, norm_action(action))
                prev_after = np.asarray(after)
            if len(rec.transitions) in restarts or cur.shape != prev_after.shape or \
                    not np.array_equal(cur, prev_after):
                state = self._init(cur, rec.level)   # board jumped (e.g. reset after game over)
            state["grid"] = cur.copy()
            return state
        except _Timeout:
            raise
        except Exception:  # noqa: BLE001
            try:
                return self._init(cur, rec.level)
            except Exception:  # noqa: BLE001
                return {"grid": cur.copy()}

    def _available(self, rec: Record) -> list[str]:
        names = [str(a).upper() for a in rec.available_actions]
        if not names:
            seen = {norm_action(t[1])["action"] for lv in rec.previous + [rec] for t in lv.transitions}
            names = sorted(seen) or list(DIRECTIONS)
        return names

    def _grid_arg(self, x=None) -> np.ndarray:
        """Resolve the model's grid argument: None (the current board), a FrameView-like object
        with `.grid`, a model state dict with 'grid', a Record, or an array."""
        if x is None:
            return np.asarray(self._rec().current_grid)
        if isinstance(x, Record):
            return np.asarray(x.current_grid)
        if isinstance(x, dict) and "grid" in x:
            return np.asarray(x["grid"])
        if hasattr(x, "grid") and not isinstance(x, np.ndarray):
            return np.asarray(x.grid)
        return np.asarray(x)

    def click_candidates(self, grid=None, max_clicks: int = MAX_CLICKS) -> list[dict]:
        """One MOUSE action per segmented component (background and HUD border band excluded),
        clicking the component cell nearest its centroid; biggest components first. `grid` may
        be an array, a frame (`current_frame`) or omitted (the current board)."""
        g = self._grid_arg(grid)
        bg = background(g)
        hud = self.border(g)
        rows, cols = g.shape
        cands = []
        for comp in components(g):
            if comp["color"] == bg:
                continue
            r0, c0, r1, c1 = comp["bbox"]
            if hud and (r1 < hud or c1 < hud or r0 >= rows - hud or c0 >= cols - hud):
                continue
            cr = sum(p[0] for p in comp["cells"]) / comp["size"]
            cc = sum(p[1] for p in comp["cells"]) / comp["size"]
            r, c = min(comp["cells"], key=lambda p: (p[0] - cr) ** 2 + (p[1] - cc) ** 2)
            cands.append((comp["size"], {"action": "MOUSE", "row": int(r), "col": int(c)}))
        cands.sort(key=lambda x: -x[0])
        return [a for _, a in cands[:max_clicks]]

    def _actions_for(self, state: dict, names: list[str]) -> list[dict]:
        if self.model["actions"] is not None:
            return [norm_action(a) for a in self.model["actions"](state)]
        acts = [norm_action(n) for n in names if n not in ("MOUSE", "UNDO", "RESET")]
        if "MOUSE" in names:
            acts += self.click_candidates(state["grid"])
        return acts

    # ---- light path: the model's own abstraction, no grid rendering

    @property
    def level_start(self) -> np.ndarray:
        """Copy of this level's first board. Build static maps (walls, targets, pickups) from it:
        the current board may already lack an item that the recorded moves picked up."""
        return np.array(self._rec().level_start_grid, copy=True)

    def check(self, read, step, all_levels: bool = False, max_mismatches: int = 3,
              timeout: float = VERIFY_SECONDS) -> dict:
        """Grade your abstraction against this level's recorded moves (all_levels=True adds earlier
        levels). For every recorded (before, action, after): predicted = step(read(before), action),
        actual = read(after). Moves that ended the level, the game or a life are skipped (their
        `after` is not a normal board). `action` is 'UP'..'SPACE' or ('MOUSE', row, col). Costs no
        game actions. Returns {ok, accuracy, checked, correct, skipped, by_action {name: 'c/n'},
        first_mismatch {level, index, action, before, predicted, actual} or None,
        mismatches (up to max_mismatches), unseen_actions, summary}."""
        rec = self._rec()
        levels = (list(rec.previous) if all_levels else []) + [rec]
        deadline = time.monotonic() + timeout
        out = {"ok": False, "accuracy": 1.0, "checked": 0, "correct": 0, "skipped": 0,
               "by_action": {}, "first_mismatch": None, "mismatches": [], "timeout": False}
        tally: dict = defaultdict(lambda: [0, 0])
        try:
            for lv in levels:
                for i, (before, action, after, outcome) in enumerate(lv.transitions):
                    self._check(deadline)
                    if outcome != "ongoing":
                        out["skipped"] += 1
                        continue
                    a = abstract_action(action)
                    name = a if isinstance(a, str) else a[0]
                    out["checked"] += 1
                    tally[name][1] += 1
                    where = {"level": lv.level, "index": i, "action": action_str(action)}
                    try:
                        predicted = step(read(np.array(before, copy=True)), a)
                        actual = read(np.array(after, copy=True))
                        same = same_state(predicted, actual)
                    except Exception as e:  # noqa: BLE001 - model code may raise anything
                        bad = {**where, "error": _err(e)}
                    else:
                        if same:
                            out["correct"] += 1
                            tally[name][0] += 1
                            continue
                        try:
                            start = _short(read(np.array(before, copy=True)))
                        except Exception as e:  # noqa: BLE001
                            start = _err(e)
                        bad = {**where, "before": start, "predicted": _short(predicted),
                               "actual": _short(actual)}
                    if len(out["mismatches"]) < max_mismatches:
                        out["mismatches"].append(bad)
        except _Timeout:
            out["timeout"] = True
        out["first_mismatch"] = out["mismatches"][0] if out["mismatches"] else None
        out["by_action"] = {k: f"{c}/{n}" for k, (c, n) in sorted(tally.items())}
        if out["checked"]:
            out["accuracy"] = round(out["correct"] / out["checked"], 4)
        out["ok"] = not out["timeout"] and out["correct"] == out["checked"]
        seen = {norm_action(t[1])["action"] for lv in levels for t in lv.transitions}
        out["unseen_actions"] = [n for n in self._available(rec)
                                 if n not in seen and n not in ("RESET", "UNDO")]
        out["summary"] = _check_summary(out)
        return out

    def plan_abstract(self, read, step, goal, actions=None, max_nodes: int = 20000,
                      timeout: float = PLAN_SECONDS) -> dict:
        """BFS over your abstraction from read(current board) to a state with goal(state) True.
        actions: the moves to try (default: the valid keyboard actions; with MOUSE available, also
        one click per object on the current board as ('MOUSE', row, col)). step must not modify
        the state it is given in place (return a new one). States must be hashable or plain data
        (tuples, dicts, lists, sets, numpy arrays). Costs no game actions. Returns {found, actions
        (pass to action(...)), plan (strings), nodes, reason}. Run check(read, step) first."""
        rec = self._rec()
        t0 = time.monotonic()
        deadline = t0 + timeout
        out = {"found": False, "actions": [], "plan": [], "nodes": 0, "reason": "",
               "seconds": 0.0}

        def done(reason, path=None):
            out["reason"] = reason
            out["seconds"] = round(time.monotonic() - t0, 3)
            if path is not None:
                out.update(found=True, actions=path,
                           plan=[a if isinstance(a, str) else action_str(a) for a in path])
            return out

        if actions is None:
            names = self._available(rec)
            actions = [n for n in names if n not in ("MOUSE", "UNDO", "RESET")]
            if "MOUSE" in names:
                actions += [("MOUSE", c["row"], c["col"])
                            for c in self.click_candidates(rec.current_grid)]
        actions = [abstract_action(a) for a in actions]
        try:
            start = read(np.array(rec.current_grid, copy=True))
            if goal(start):
                return done("already at goal", [])
        except Exception as e:  # noqa: BLE001
            return done("read/goal failed on the current board: " + _err(e))
        parents = {state_key(start): None}
        frontier = deque([start])
        n_errors = 0
        try:
            while frontier:
                state = frontier.popleft()
                skey = state_key(state)
                for a in actions:
                    self._check(deadline)
                    try:
                        nxt = step(copy.deepcopy(state), a)
                        key = state_key(nxt)
                        if key in parents:
                            continue
                        parents[key] = (skey, a)
                        out["nodes"] += 1
                        if goal(nxt):
                            return done("found", self._path(parents, key))
                    except Exception:  # noqa: BLE001
                        n_errors += 1
                        continue
                    if out["nodes"] >= max_nodes:
                        return done(f"node cap {max_nodes} reached")
                    frontier.append(nxt)
        except _Timeout:
            return done(f"timeout after {timeout}s")
        reason = "exhausted: goal unreachable in your step()"
        if n_errors:
            reason += f" ({n_errors} step/goal errors)"
        return done(reason)

    def plan(self, max_nodes: int = 20000, max_depth: int = 200, goal=None, mode: str = "auto",
             timeout: float = PLAN_SECONDS) -> dict:
        """Search your model from wm.current_state() to outcome 'level_complete' (or goal(state)
        -> True). Game-over states are pruned. mode: 'auto' (A* if you registered a heuristic,
        else BFS), 'bfs', 'astar'. Returns {found, actions, plan (strings), nodes, seconds, reason,
        hashes (expected grid hash after each step)}."""
        self._need_model()
        rec = self._rec()
        names = self._available(rec)
        t0 = time.monotonic()
        deadline = t0 + timeout
        m = self.model
        is_goal = goal or (lambda s: self._outcome(s) == "level_complete")
        use_astar = m["heuristic"] is not None and mode in ("auto", "astar")
        h = m["heuristic"] or (lambda s: 0.0)
        out = {"found": False, "actions": [], "plan": [], "nodes": 0, "seconds": 0.0,
               "reason": "", "hashes": []}

        def done(reason, path=None):
            out["reason"] = reason
            out["seconds"] = round(time.monotonic() - t0, 3)
            if path is not None:
                out.update(found=True, actions=path, plan=[action_str(a) for a in path])
                out["hashes"] = [s["hash"] for s in self.validate(path)["steps"]]
            return out

        try:
            start = self.current_state(_deadline=deadline)
            if is_goal(start):
                return done("already at goal", [])
        except _Timeout:
            return done(f"timeout after {timeout}s")
        except Exception as e:  # noqa: BLE001
            return done("start state failed: " + _err(e))
        parents = {m["planner_key"](start): None}
        counter = 0
        frontier = [(h(start), 0, start, 0)] if use_astar else deque([(start, 0)])
        n_errors = 0
        try:
            while frontier:
                self._check(deadline)
                if use_astar:
                    _, _, state, depth = heapq.heappop(frontier)
                else:
                    state, depth = frontier.popleft()
                if depth >= max_depth:
                    continue
                skey = m["planner_key"](state)
                for a in self._actions_for(state, names):
                    self._check(deadline)
                    try:
                        nxt = self._step(state, a)
                        key = m["planner_key"](nxt)
                        if key in parents:
                            continue
                        parents[key] = (skey, a)
                        out["nodes"] += 1
                        if self._outcome(nxt) == "game_over":
                            continue
                        if is_goal(nxt):
                            return done("found", self._path(parents, key))
                    except Exception:  # noqa: BLE001
                        n_errors += 1
                        continue
                    if out["nodes"] >= max_nodes:
                        return done(f"node cap {max_nodes} reached")
                    counter += 1
                    if use_astar:
                        heapq.heappush(frontier, (depth + 1 + h(nxt), counter, nxt, depth + 1))
                    else:
                        frontier.append((nxt, depth + 1))
        except _Timeout:
            return done(f"timeout after {timeout}s")
        reason = "exhausted: goal unreachable in model"
        if n_errors:
            reason += f" ({n_errors} transition errors)"
        return done(reason)

    def plan_to(self, predicate, **kw) -> dict:
        """plan(goal=predicate): search to any state where predicate(state) is True."""
        return self.plan(goal=predicate, **kw)

    @staticmethod
    def _path(parents: dict, key) -> list[dict]:
        path = []
        while parents[key] is not None:
            key, a = parents[key]
            path.append(a)
        return path[::-1]

    def validate(self, actions, strict: bool = False) -> dict:
        """Replay actions through your model from wm.current_state(). Returns {ok (ends in
        level_complete), final_outcome, steps:[{action, hash, outcome}], error}; hash is
        wm.view_hash (HUD border ignored unless strict). Stops at the first terminal outcome or
        model error."""
        return self._rollout(actions, keep_grids=False, strict=strict)

    def _rollout(self, actions, keep_grids: bool, strict: bool = False) -> dict:
        self._need_model()
        state = self.current_state()
        steps, final, error = [], "ongoing", None
        for a in actions:
            a = norm_action(a)
            try:
                state = self._step(state, a)
                final = self._outcome(state)
            except Exception as e:  # noqa: BLE001
                error = f"step {len(steps)} {action_str(a)}: {_err(e)}"
                break
            steps.append({"action": action_str(a), "hash": self.view_hash(state["grid"], strict),
                          "outcome": final})
            if keep_grids:
                steps[-1]["grid"] = state["grid"]
            if final != "ongoing":
                break
        return {"ok": error is None and final == "level_complete", "final_outcome": final,
                "steps": steps, "error": error}

    def expected_hashes(self, actions, strict: bool = False) -> list[str]:
        """Expected board hash after each action of a plan; compare with wm.view_hash(observed)."""
        return [s["hash"] for s in self.validate(actions, strict)["steps"]]

    def execute(self, actions, stop_on_mismatch: bool = True, strict: bool = False) -> dict:
        """Play a plan in the REAL game one action at a time (this costs actions), checking each
        observed board against your model's prediction. Stops at the first mismatch (when
        stop_on_mismatch), a game over, a level completion or a refused action. Returns {executed,
        mismatch_index (None if all matched), expected_diff ('expected->observed @ ...'),
        observed_outcome, stop_reason}."""
        if self.executor is None:
            raise RuntimeError("wm.execute is only available inside the python tool")
        actions = [norm_action(a) for a in actions]
        expected = self._rollout(actions, keep_grids=True, strict=strict)["steps"]
        out = {"executed": 0, "mismatch_index": None, "expected_diff": None,
               "observed_outcome": "ongoing", "stop_reason": None}
        for i, a in enumerate(actions):
            try:
                res = self.executor([a])
            except Exception as e:  # noqa: BLE001 - guards raise to stop the batch
                out["stop_reason"] = _err(e)
                break
            if not res.get("executed"):
                out["stop_reason"] = res.get("stop_reason") or "refused"
                break
            out["executed"] += 1
            if res.get("level_completed") or res.get("run_complete"):
                out["observed_outcome"] = "level_complete"
                out["stop_reason"] = "level_completed"
                break
            if res.get("game_over") or res.get("level_reset"):
                # A silent level reset (life lost) is a death; the sandbox reports it separately.
                out["observed_outcome"] = "game_over"
            observed = np.asarray(self._rec().current_grid)
            exp = expected[i] if i < len(expected) else None
            if exp is None or exp["outcome"] != out["observed_outcome"] or \
                    self.view_hash(observed, strict) != exp["hash"]:
                if out["mismatch_index"] is None:
                    out["mismatch_index"] = i
                    out["expected_diff"] = ("model gave no prediction" if exp is None else
                                            diff_text(self.view(exp["grid"], strict),
                                                      self.view(observed, strict))["diff"]
                                            or f"outcome {exp['outcome']} predicted")
                if stop_on_mismatch:
                    out["stop_reason"] = "mismatch"
                    break
            if out["observed_outcome"] == "game_over":
                out["stop_reason"] = "game_over"
                break
            if res.get("stop_reason"):
                out["stop_reason"] = res["stop_reason"]
                break
        return out

    grid_hash = staticmethod(grid_hash)
    diff = staticmethod(diff_text)

    # ---- mechanical analysis (no model needed)

    def induce_motion(self, grid=None, record=None) -> dict:
        """Per directional action, find same-colour components that translated by a consistent
        (dy, dx) between settled before/after grids. Returns {'actions': {name: {dx, dy,
        moving_colors, n_support, n_contradictions, blocked_cases}}, 'avatar_candidates': [...]}.
        blocked_cases = record indices where the action moved nothing. `grid` (array, frame or
        omitted = current board) is where avatar candidates are located; a Record is accepted
        there too."""
        if isinstance(grid, Record) and record is None:
            grid, record = None, grid
        rec = self._rec(record)
        per = defaultdict(lambda: {"offsets": Counter(), "colors": defaultdict(Counter),
                                   "blocked": [], "by_case": []})
        for i, (before, action, after, out) in enumerate(rec.transitions):
            name = norm_action(action)["action"]
            if name not in DIRECTIONS or out != "ongoing":
                continue
            moves = _component_moves(np.asarray(before), np.asarray(after))
            p = per[name]
            if not moves:
                p["blocked"].append(i)
                continue
            offs = Counter(off for _, off in moves)
            off = offs.most_common(1)[0][0]
            p["offsets"][off] += 1
            p["by_case"].append(off)
            for color, o in moves:
                p["colors"][o][color] += 1
        result, color_actions = {}, defaultdict(set)
        for name, p in per.items():
            if not p["offsets"]:
                result[name] = {"dx": 0, "dy": 0, "moving_colors": [], "n_support": 0,
                                "n_contradictions": 0, "blocked_cases": p["blocked"]}
                continue
            (dy, dx), support = p["offsets"].most_common(1)[0]
            colors = sorted(p["colors"][(dy, dx)])
            for c in colors:
                color_actions[c].add(name)
            result[name] = {"dx": int(dx), "dy": int(dy), "moving_colors": colors,
                            "n_support": support, "n_contradictions": len(p["by_case"]) - support,
                            "blocked_cases": p["blocked"]}
        cur = self._grid_arg(grid) if grid is not None else np.asarray(rec.current_grid)
        avatars = []
        for c, acts in sorted(color_actions.items(), key=lambda x: -len(x[1])):
            comps = [k for k in components(cur, {c})]
            if comps:
                k = max(comps, key=lambda k: k["size"])
                avatars.append({"color": c, "actions": sorted(acts), "bbox": k["bbox"],
                                "size": k["size"], "n_components": len(comps)})
        return {"actions": result, "avatar_candidates": avatars}

    def transition_graph(self, record=None) -> dict:
        """Observed graph state_hash -> {action: next_hash} for this level, with no-ops (grid
        unchanged), deaths (game_over) and revisits (actions that led to an already-seen state)."""
        rec = self._rec(record)
        graph = defaultdict(dict)
        seen = {grid_hash(rec.level_start_grid)}
        noops, deaths, revisits = [], [], []
        for i, (before, action, after, out) in enumerate(rec.transitions):
            hb, ha, a = grid_hash(before), grid_hash(after), action_str(action)
            graph[hb][a] = ha
            if out == "game_over":
                deaths.append({"index": i, "state": hb, "action": a})
            elif hb == ha:
                noops.append({"index": i, "state": hb, "action": a})
            elif ha in seen:
                revisits.append({"index": i, "state": ha, "action": a})
            seen.add(ha)
        return {"graph": dict(graph), "n_states": len(seen), "noops": noops, "n_noops": len(noops),
                "deaths": deaths, "n_deaths": len(deaths), "revisits": revisits,
                "current": grid_hash(rec.current_grid)}

    def untried_actions(self, record=None, available=None) -> dict:
        """Available actions never tried from the current grid ('here') and never tried at all in
        this level ('ever'). MOUSE counts as one action name here."""
        rec = self._rec(record)
        names = [str(a).upper() for a in (available or self._available(rec))]
        # view_hash, not grid_hash: with a per-action HUD bar no two full grids ever match, so
        # 'here' would always list every action (run #5, ka59 and dc22).
        cur = self.view_hash(rec.current_grid)
        here, ever = set(), set()
        for before, action, _, _ in rec.transitions:
            n = norm_action(action)["action"]
            ever.add(n)
            if self.view_hash(before) == cur:
                here.add(n)
        return {"here": [n for n in names if n not in here],
                "ever": [n for n in names if n not in ever]}

    def goal_menu(self, record=None, max_items: int = 20) -> list:
        """Candidate goal predicates (name, fn) built from the record, for wm.plan(goal=fn):
        avatar_reaches_color_c (avatar colour now covers a cell that was colour c),
        all_color_c_removed, grid_matches_region (object region B equals object region A's
        current content). Each fn accepts a model state dict, a grid, a frame or no argument
        (the current board) and returns bool."""
        rec = self._rec(record)
        cur = np.asarray(rec.current_grid).copy()
        bg = background(cur)
        colors = [int(c) for c in np.unique(cur) if int(c) != bg]
        avatar_colors = [a["color"] for a in self.induce_motion(rec)["avatar_candidates"]]
        menu = []
        for av in avatar_colors[:2]:
            for c in colors:
                if c == av:
                    continue
                mask = cur == c
                menu.append((f"avatar_reaches_color_{c}" + (f"(avatar={av})" if len(avatar_colors) > 1 else ""),
                             _reach_pred(mask, av)))
        for c in colors:
            if c not in avatar_colors:
                menu.append((f"all_color_{c}_removed", _removed_pred(c)))
        menu += _region_preds(cur, bg)
        return [(name, self._flex_pred(fn)) for name, fn in menu[:max_items]]

    def _flex_pred(self, fn):
        """Wrap a grid predicate so the model can call it on a state, a grid, a frame or nothing
        (run #5: five calls were lost to predicates that only took state['grid'])."""
        def pred(state=None):
            return fn({"grid": self._grid_arg(state)})
        pred.__doc__ = "pred(state_or_grid_or_frame=None) -> bool; no argument = current board"
        return pred


# ---------------------------------------------------------------- light-path helpers

def abstract_action(a):
    """The action form check/plan_abstract hand to step(): 'UP' or ('MOUSE', row, col)."""
    n = norm_action(a)
    return ("MOUSE", n["row"], n["col"]) if n["action"] == "MOUSE" else n["action"]


def same_state(a, b) -> bool:
    """Equality for model states that may hold numpy arrays (where == is elementwise)."""
    try:
        eq = a == b
        if isinstance(eq, bool):
            return eq
        if isinstance(eq, np.ndarray):
            return np.shape(a) == np.shape(b) and bool(eq.all())
        return bool(eq)
    except Exception:  # noqa: BLE001 - ambiguous truth value of containers holding arrays
        return state_key(a) == state_key(b)


def state_key(state):
    """Hashable key for a model state: the state itself when hashable, else a canonical form."""
    try:
        hash(state)
        if not isinstance(state, np.ndarray):
            return state
    except TypeError:
        pass
    return _key_part(state)


def _short(x, limit: int = 160) -> str:
    try:
        r = repr(x)
    except Exception as e:  # noqa: BLE001
        r = f"<unprintable {type(x).__name__}: {e}>"
    return r if len(r) <= limit else r[:limit - 3] + "..."


def _check_summary(out: dict) -> str:
    if out["timeout"]:
        head = "timeout; partial: "
    else:
        head = ""
    text = (f"{head}{out['correct']}/{out['checked']} predicted"
            + (f" ({out['skipped']} terminal moves skipped)" if out["skipped"] else "")
            + (f"; by action {out['by_action']}" if out["by_action"] else ""))
    fm = out["first_mismatch"]
    if fm:
        detail = (f"error {fm['error']}" if "error" in fm else
                  f"from {fm['before']} predicted {fm['predicted']}, actual {fm['actual']}")
        text += f"; first mismatch L{fm['level']} #{fm['index']} {fm['action']}: {detail}"
    if out.get("unseen_actions"):
        text += f"; never tried here: {', '.join(out['unseen_actions'])}"
    return text


# ---------------------------------------------------------------- mechanical helpers

def _component_moves(before: np.ndarray, after: np.ndarray) -> list[tuple[int, tuple]]:
    """(color, (dy, dx)) for components of `before` that vanished and reappear translated with the
    same shape in `after`."""
    if before.shape != after.shape or np.array_equal(before, after):
        return []
    changed = before != after
    bcomps = [k for k in components(before) if any(changed[p] for p in k["cells"])]
    acomps = [k for k in components(after) if any(changed[p] for p in k["cells"])]
    bg = background(before)
    moves = []
    for b in bcomps:
        if b["color"] == bg:
            continue
        bshape = _shape(b)
        for a in acomps:
            if a["color"] == b["color"] and a["size"] == b["size"] and _shape(a) == bshape:
                off = (a["bbox"][0] - b["bbox"][0], a["bbox"][1] - b["bbox"][1])
                if off != (0, 0):
                    moves.append((b["color"], off))
                    break
    return moves


def _shape(comp: dict) -> frozenset:
    r0, c0 = comp["bbox"][0], comp["bbox"][1]
    return frozenset((r - r0, c - c0) for r, c in comp["cells"])


def _reach_pred(mask: np.ndarray, avatar: int):
    def pred(state):
        g = np.asarray(state["grid"])
        return g.shape == mask.shape and bool(((g == avatar) & mask).any())
    return pred


def _removed_pred(color: int):
    return lambda state: not bool((np.asarray(state["grid"]) == color).any())


def _region_preds(grid: np.ndarray, bg: int, limit: int = 6) -> list:
    """Pairs of non-background objects with equal bbox size but different content."""
    objs = components((grid != bg).astype(np.uint8), {1})
    boxes = []
    for o in objs:
        r0, c0, r1, c1 = o["bbox"]
        if o["size"] >= 4:
            boxes.append((r0, c0, r1, c1))
    out = []
    for i, a in enumerate(boxes):
        for b in boxes[i + 1:]:
            if (a[2] - a[0], a[3] - a[1]) != (b[2] - b[0], b[3] - b[1]):
                continue
            ra = grid[a[0]:a[2] + 1, a[1]:a[3] + 1].copy()
            rb = grid[b[0]:b[2] + 1, b[1]:b[3] + 1].copy()
            if np.array_equal(ra, rb):
                continue
            for src, dst, target in ((a, b, ra), (b, a, rb)):
                out.append((f"grid_matches_region({dst} := {src})", _match_pred(dst, target)))
            if len(out) >= limit:
                return out[:limit]
    return out


def _match_pred(dst: tuple, target: np.ndarray):
    def pred(state):
        g = np.asarray(state["grid"])
        return np.array_equal(g[dst[0]:dst[2] + 1, dst[1]:dst[3] + 1], target)
    return pred
