"""Prompt text: the system prompt, the per-turn opener, the resume prompt and the nudge.

The system prompt is static per game (so the server can cache it); everything that changes
each turn goes into the opener. Wording is ours; the content follows what the public
solutions found useful (see reference/duck-architecture-brief.md, dfranzen-writeup-notes.md).
"""
from __future__ import annotations

from typing import Optional, Sequence

from arc3.types import COLOR_LEGEND, COLOR_LETTERS, ActionResult

DEFAULT_MODULES = ("bisect", "collections", "copy", "fractions", "functools", "heapq",
                   "itertools", "json", "math", "operator", "random", "re", "statistics",
                   "string", "numpy")


# ------------------------------------------------------------------ system prompt

ROLE = "You are a coding agent playing an unknown turn-based grid game through a Python tool."

OVERVIEW = """\
## Game
- The game has several levels; your job is to clear all of them. Nobody tells you the rules or the goal: discover them from how the board reacts to your actions.
- Boards are 64x64 grids of 16 colors. Values 0-15; ascii uses one letter per value: {legend}.
- You are called once per turn. Each turn: look at what changed, update your understanding, then act.
- Score depends on how FEW actions you take per level (roughly quadratic in efficiency). Thinking and Python are free; only `action(...)` costs."""

RUNTIME = """\
## Variables inside every `python` call
- `current_frame`: the latest board. `.grid` (numpy uint8 array, rows x cols), `.ascii` (letter rows), `.crop(r0, c0, r1, c1)` (a frame of the sub-rectangle, end-exclusive; use `.crop(...).ascii` to read a region), `.segmentation` (below), `.step`, `.level`.
- `current_frame.segmentation`: 4-connected same-color objects, ids in reading order (top row first, then left to right). It acts as a list: `len(seg)`, `for node in seg`, `seg[i]`, `seg.nodes`.
  - Each node supports `node.color` and `node['color']`, `.get`, `'color' in node`, `dict(node)`. Keys: `id`, `color` (int 0-15), `letter`, `pixels` (count), `bbox` [r0, c0, r1, c1] inclusive, `cells` [(row, col), ...], `center` (row, col), `hash` (same color+shape in any 90-degree rotation), `shape_hash` (shape only), `pose_hash` (exact orientation), `rotation` (degrees from canonical; None if 4-fold symmetric), `rotational_symmetry` (1/2/4), `children` (ids of objects directly enclosed), `parent`.
  - `seg.adjacency`: list of (id_a, id_b) for objects sharing an edge; `seg.neighbors(id)`, `seg.node_at(row, col)`, `seg.by_color(c)`, `seg.labels` (object id per cell).
  - print(current_frame.segmentation) gives a compact object table
- `previous_frame`: the board before the most recent action (or None).
- `history`: list of entries with `.action`, `.frame` (board AFTER that action) and `.result`. `history[-1].frame` is `current_frame`.
- `transitions` / `last_transition`: per-action records with `.action`, `.before_frame`, `.after_frame`, `.result`, `.step`, `.level_reset`, `.stop_reason`.
- `valid_actions`: action names allowed right now.
- `last_action_call_result`: the result of your latest `action(...)` call (executed actions, flags, stop reasons). It survives later inspection calls.
- `last_animation_frames`: every intermediate frame of the last action that animated (index 0 first, last = settled board); empty if none. Reading them costs nothing.
- `frame_diff(before=None, after=None)`: object-level diff (moved, rotated, appeared, disappeared, changed_color, resized, changed_cell_count); defaults to previous_frame vs current_frame. Object matching is heuristic, not identity.
- `action(actions)`: executes real game actions, e.g. `action(['LEFT', 'LEFT'])` or `action([{'action': 'MOUSE', 'row': 12, 'col': 30}])`. It returns the call result, and all variables above are refreshed before your next statement."""

ACTIONS = """\
## Actions
- `UP`, `DOWN`, `LEFT`, `RIGHT`: directional inputs; what they move (if anything) is game-specific.
- `SPACE`: a game-specific trigger (select, rotate, interact, confirm...). Test it, do not assume.
- `MOUSE`: click a cell; `row` (0-63, downward) and `col` (0-63, rightward) are required.{undo}{reset}
- Only actions in `valid_actions` are accepted."""

UNDO = "\n- `UNDO`: reverts the previous action when available. Check what it restored. It cannot undo a game over."
RESET = ("\n- `RESET`: restarts the current level from its start board and keeps completed levels. "
         "It costs one action. Use it when the position is lost or the move budget left cannot "
         "cover your plan. NEVER burn moves to exhaust a budget or to die on purpose: that costs "
         "dozens of actions; RESET costs one. The start board is deterministic, so actions may "
         "follow RESET in the same `action([...])` call.")

SIGNALS = """\
## Reading results
- `board_changed`: any cell changed. `gameplay_changed`: a cell changed more than {border} cells away from the edge. Timers and step bars live in that edge band, so an action that only shrinks a bar is `board_changed` but not `gameplay_changed` and did nothing useful.
- A bar along the edge that shrinks every action is usually a move budget. Do not mistake its segments for clickable objects.
- One action may play a short animation. `transient_pixels` > 0 means some cells changed and changed back mid-animation; that evidence exists only in `last_animation_frames`, so inspect them then.
- `game_over`: this attempt FAILED (hazard, or budget ran out). The harness resets the level automatically; completed levels are kept. It never means you won.
- `level_reset`: the board returned to the level's start without a game over: a fatal move, an exhausted move budget, or a deliberate restart mechanic. Check `history[-1]`.
- `level_completed`: you advanced a level; the next board is already in `current_frame`. `run_complete`: the whole game is won.
- After any of these flags, stop acting in the current call and re-examine the board next turn. Guards may also stop a batch early (for example after an action that changed nothing in the interior); the stop reason is reported."""

TOOL = """\
## The python tool
- One tool, `python(code)`. Each call has a {timeout}s limit and its printed output is cut to about {out_tokens} tokens, so print compact summaries: object lists, counts, coordinates, small crops. Never print a whole board.
- Variables reset between calls, but top-level functions you define are kept and can be called later (no decorators/annotations; top-level values they read are frozen copies, up to 64 KB); `source('name')` prints one. `np`, `math`, `collections`, `itertools` are preloaded. Importable modules: {modules}.
- `memory` is a dict that persists across calls; variables do not. Store plans, maps, rule notes there (`memory['plan'] = ...`). Mutate it; do not reassign it.
- Keep `memory['rules']` as a short plain-text note of what you know for sure: what each action does, the goal, what kills you, how levels differ. Update it when you learn or disprove something. It is shown back to you every turn and survives when older turns are dropped, so write it for a reader who has seen nothing else.
- Call the tool as often as you like. Think in short bursts: when a hypothesis can be checked with code, call the tool instead of reasoning further."""

STRATEGY = """\
## Strategy
- On a new game, summarise the board structurally (objects, colors, likely player/cursor, HUD), then probe each available action once, cheaply, and measure its effect with `frame_diff`.
- Once you understand the mechanics and the goal, stop probing: write an explicit search (BFS/A*/simulation over the objects you extracted) and execute the found plan in one batched `action([...])` call.
- Prefer the shortest reliable sequence. Inside a batch, check in code that each step did what you predicted and stop when it did not.
- Do not frame goals as absolute coordinates; reason about objects and relations (reach, align, match, fill, sort, mirror, count).
- Levels usually extend earlier mechanics: carry what you learned forward as a hypothesis, look closely at new kinds of objects (they usually matter), and re-check whether the goal changed.
- The game is deterministic: a sequence that failed once will fail again.
- Before 20 actions of trial and error on a level, click each unclicked object once. If the last 20 actions reached fewer than 5 new states, stop probing variants and test something not yet examined."""


# Worked example shown in WORLD_MODEL. Kept separate so a test can execute it (tests/test_prompts.py).
WM_EXAMPLE = """\
g = wm.level_start  # static maps from the first board
memory.update(wall=g == 5, key=g == 11, door=g == 14)
def read(g):  # block top-left, key taken?
    r, c = np.argwhere(g == 9).min(axis=0)
    return (int(r), int(c), not (g == 11).any())
def step(s, a):  # move 5 unless a wall blocks; landing on the key takes it
    dr, dc = {'UP': (-5, 0), 'DOWN': (5, 0), 'LEFT': (0, -5), 'RIGHT': (0, 5)}.get(a, (0, 0))
    r, c = s[0] + dr, s[1] + dc
    if memory['wall'][r:r + 5, c:c + 5].any(): return s
    return (r, c, s[2] or bool(memory['key'][r:r + 5, c:c + 5].any()))
print(wm.check(read, step)['summary'])
p = wm.plan_abstract(read, step, lambda s: s[2] and memory['door'][s[0]:s[0] + 5, s[1]:s[1] + 5].any())"""

WORLD_MODEL = """\
## World model: check your rules, then search
Write your rules as two functions; the global `wm` checks and searches them free.
- `read(grid)` -> the facts that matter (positions, keys held) as a tuple/dict; `step(state, action)` -> next state; action is 'UP'... or ('MOUSE', row, col).
- `wm.check(read, step)` replays this level's moves: `ok`, `summary`, `first_mismatch`, `unseen_actions`. Reach `ok` before executing over ~8 actions from them; after a death, add its cause to `step`.
- `wm.plan_abstract(read, step, goal)`: BFS to goal(state); `found`, `actions`, `reason`.
Example:
""" + WM_EXAMPLE + """
- Full simulator option: `wm.register(transition, outcome, init=init)` (transition over `state['grid']`; outcome 'ongoing'/'level_complete'/'game_over'), then `wm.verify()`, `p = wm.plan()`, `wm.execute(p['actions'])`.
- Free (no model): `wm.induce_motion()`, `wm.untried_actions()`, `wm.transition_graph()`, `wm.click_candidates()` (one MOUSE per object), `wm.goal_menu()` -> [(name, pred)]. Grid arguments are optional (default current board; a frame works)."""


def build_system_prompt(hud_border: int = 4, expose_undo: bool = True, expose_reset: bool = True,
                        tool_timeout: int = 30, output_tokens: int = 3000,
                        modules: Sequence[str] = DEFAULT_MODULES) -> str:
    parts = [
        ROLE,
        OVERVIEW.format(legend=COLOR_LEGEND),
        RUNTIME,
        ACTIONS.format(undo=UNDO if expose_undo else "", reset=RESET if expose_reset else ""),
        SIGNALS.format(border=hud_border),
        TOOL.format(timeout=tool_timeout, out_tokens=output_tokens, modules=", ".join(modules)),
        STRATEGY,
        WORLD_MODEL,
    ]
    return "\n\n".join(parts)


# ------------------------------------------------------------------ per-turn opener

LEVEL_START = (
    "LEVEL COMPLETE. `current_frame` is now the first board of the next level. Do not continue "
    "the old plan. Keep the mechanics you established, look for unfamiliar objects or "
    "arrangements and test those with small probes, and ask whether the goal is the same or "
    "has changed before you search for a plan.")

GAME_OVER = (
    "GAME OVER: the previous attempt FAILED. The level was reset automatically to its start; "
    "completed levels are kept. Diagnose before acting: `history[-1]` is the automatic RESET, "
    "`history[-2].action` is the fatal action, `history[-2].frame` is the death frame and "
    "`history[-3].frame` is the board just before it. Check what the fatal action touched.\n"
    "BAR RULE: an edge bar that shrinks every action is normal; a slightly shorter bar proves "
    "nothing. Only a bar that is empty (or nearly) in the death frame means the move budget ran "
    "out - then you need a shorter route or something that refills the bar. Otherwise the fatal "
    "action itself killed you (hazard or forbidden move).\n"
    "The game is deterministic: do NOT replay the sequence that just failed. Change the plan "
    "before the fatal point.")

LEVEL_RESET = (
    "LEVEL RESET: the board returned to this level's start position. Causes: a fatal move, an "
    "exhausted move budget, or a deliberate restart mechanic. Check `history[-1]`: its "
    "`.action` is the move after which it happened and `.before_frame` the board just before. "
    "If that move was meant to restart (a game mechanic), carry on; otherwise do not replay the "
    "same sequence.")

TRIM_NOTICE = ("Trim notice: older turns were dropped. Your memory dict still holds: {memory}. "
               "Retained functions: {functions}. Your pinned notes below are complete; do not "
               "re-probe mechanics they already settle.")

TAIL_FIRST = ("Start with a compact structural summary of `current_frame` from Python (not a "
              "printout of the board), then probe. When ready, call `action(...)` from inside "
              "`python`.")
TAIL = ("Inspect what changed (gameplay vs HUD) from Python, update your plan, then call "
        "`action(...)` from inside `python` with the best action or a batched sequence your code "
        "selected.")

RESUME = ("You paused on the turn budget. Your earlier tool results above are still valid; do "
          "not restart the analysis. Continue from them and call `action(...)` with your best "
          "next action or short batch.")

STALL_NOTICE = ("STALL NOTICE: this level is not progressing (no new board states, far more actions "
                "than your solved levels, or you wrote it off). A note saying a level is impossible "
                "is a hypothesis, not a fact: every level is solvable. Do not repeat sequences that "
                "already failed and do not burn the budget with 'hold' or no-op moves. Before acting, "
                "pick ONE untested hypothesis from this list and test it with real actions:")


def stall_notice(facts: Optional[dict]) -> str:
    """Opener prefix when a level stalls (agent.py decides when). `facts` is
    PythonTool.stall_facts(): objects clicked only in an older board state, elements new since the
    previous level, candidate goals."""
    f = facts or {}
    lines = [STALL_NOTICE]
    changed = f.get("changed_since_click") or []
    if changed:
        items = ", ".join(f"({r},{c})" for r, c in changed[:8])
        lines.append(f"- Objects you clicked only while the board around them looked different "
                     f"from now (re-try them in the current state): {items}.")
    new = f.get("new_vs_previous_level") or []
    if new:
        items = "; ".join(f"{_colour(c)} {n} cells at rows {r0}-{r1}, cols {c0}-{c1}"
                          for c, n, r0, r1, c0, c1 in new[:6])
        lines.append(f"- Elements this level has that the previous level's start did not "
                     f"(likely the new mechanic): {items}.")
    goals = f.get("goals") or []
    if goals:
        lines.append("- Alternative goals you have not tested: " + ", ".join(goals[:5]) + ".")
    if len(lines) == 1:
        lines.append("- Re-list every object and every action's effect from scratch in code; look "
                     "for what you have assumed but never measured.")
    return "\n".join(lines) + "\n\n"


def idle_notice(idle_turns: int) -> str:
    return (f"NOTICE: your last {idle_turns} turns executed no action. Idling earns nothing: completed "
            "levels are never lost whatever happens next, a no-op RESET is refused, and no move is "
            "'parked' or saved by waiting. The only way to score more is to attempt this level now: "
            "take your best hypothesis, even an uncertain one, and test it with real actions.\n\n")


NUDGE_CUT = ("Your reply hit the length limit before any `python` tool call, so nothing happened. "
             "Think briefly, then call the tool: run code to test a hypothesis instead of reasoning "
             "it through. Keep this reply short.")
NUDGE = ("You have not acted yet: no `python` tool call was made. Use the `python` tool to "
         "inspect the board if needed, and call `action(...)` inside it. Do not write actions "
         "as plain text.")


# Batch stops that only restate the action's own effect; describe_result does not repeat them.
EFFECT_STOPS = ("game_over", "level_completed", "level_reset", "run_complete", "reset")


def trim_notice(memory_summary: str, retained: Sequence[str] = ()) -> str:
    """Prefix for the first user message after a trim dropped turns (agent.py adds it)."""
    return TRIM_NOTICE.format(memory=memory_summary or "(empty)",
                              functions=", ".join(retained) or "none") + "\n"


def describe_result(r: ActionResult) -> str:
    """One compact line per action for the opener."""
    name = r.action.display()
    if not r.executed:
        return f"{name}: refused ({r.stop_reason or 'guard'})"
    if r.game_over:
        effect = "GAME OVER"
    elif r.level_reset:
        effect = "LEVEL RESET (back to the level start)"
    elif r.action.name == "RESET" and not r.automatic:
        effect = "level restarted from its start board"
    elif r.level_completed:
        effect = "level completed"
    elif r.gameplay_changed:
        effect = "gameplay changed"
    elif r.board_changed:
        effect = "only HUD/edge cells changed"
    else:
        effect = "no change"
    if r.transient_pixels:
        effect += f", animated with {r.transient_pixels} transient cells"
    if r.stop_reason and r.stop_reason not in EFFECT_STOPS:
        effect += f", then stopped ({r.stop_reason})"
    return f"{name}: {effect}"


def budget_line(results: Sequence[ActionResult], can_reset: bool = True) -> Optional[str]:
    """Opener line with the move-budget estimate after the last executed action, if any."""
    last = next((r for r in reversed(results) if r.executed and not r.automatic), None)
    if last is None or last.budget_left is None:
        return None
    band = (last.budget_bar or "edge").split(" band")[0]
    text = (f"Move budget: ~{last.budget_left} actions left on this level (bar on the {band} "
            f"edge: {last.budget_bar}). Plan length must fit")
    return text + ("; if it cannot, RESET first." if can_reset else ".")


def _colour(c: int) -> str:
    return "edge" if c < 0 else (COLOR_LETTERS[c] if 0 <= c < len(COLOR_LETTERS) else str(c))


def evidence_lines(evidence: Optional[dict], valid_actions: Sequence[str] = ()) -> list[str]:
    """Opener lines from PythonTool.evidence(): unclicked objects (click games), blocked moves
    (when at least one move was refused) and the novelty counter. Each part is skipped when it
    has nothing to say."""
    if not evidence:
        return []
    lines = []
    un = evidence.get("unclicked") or {}
    if "MOUSE" in valid_actions and un.get("objects"):
        items = ", ".join(f"{_colour(o['color'])} {o['size']}px at ({o['row']},{o['col']})"
                          for o in un["objects"])
        more = un["total"] - len(un["objects"])
        lines.append("Objects never clicked on this level: " + items
                     + (f" (+{more} more)" if more > 0 else "") + ".")
    bl = evidence.get("blocked") or {}
    if bl.get("blocked"):
        parts = []
        for name, e in bl["blocked"].items():
            cols = ", ".join(_colour(c) for c in e["colors"])
            parts.append(f"{name} refused {e['count']}x"
                         + (f" (target cells colour {cols})" if cols else ""))
        text = "Blocked moves: " + ", ".join(parts)
        if bl.get("entered"):
            text += "; successful moves entered " + ", ".join(_colour(c) for c in bl["entered"])
        lines.append(text + ".")
    nv = evidence.get("novelty") or {}
    if nv.get("actions"):
        text = (f"This level: {nv['actions']} actions, {nv['states']} distinct states; last "
                f"{nv['window']} actions reached {nv['new_in_window']} new states")
        if nv.get("top_noop"):
            name, times = nv["top_noop"]
            text += f"; {name} was a no-op {times} time{'s' if times != 1 else ''}"
        lines.append(text + ".")
    return lines


def summarize_sequence(results: Sequence[ActionResult], max_lines: int = 12) -> list[str]:
    real = [r for r in results if not r.automatic]
    if not real:
        return []
    executed = [r for r in real if r.executed]
    lines = [f"Previous call executed {len(executed)} action(s): "
             + " ".join(r.action.display() for r in executed) + "."]
    gameplay = sum(r.gameplay_changed for r in executed)
    hud_only = sum(r.board_changed and not r.gameplay_changed for r in executed)
    nothing = sum(not r.board_changed for r in executed)
    lines.append(f"Effects: {gameplay} changed gameplay, {hud_only} changed only the HUD edge, "
                 f"{nothing} changed nothing.")
    shown = real if len(real) <= max_lines else real[:max_lines // 2] + real[-max_lines // 2:]
    lines.extend("- " + describe_result(r) for r in shown)
    if len(real) > len(shown):
        lines.insert(2 + max_lines // 2, f"- ... {len(real) - len(shown)} more ...")
    if any(r.transient_pixels for r in executed):
        lines.append("Some cells changed and changed back during an animation; that is visible "
                     "only in `last_animation_frames` - inspect it before concluding nothing "
                     "happened.")
    for r in real:
        if r.stop_reason and r.stop_reason not in EFFECT_STOPS and r.message:
            skipped = f", {r.skipped} requested action(s) not run" if r.skipped else ""
            lines.append(f"Guard stop ({r.stop_reason}{skipped}): {r.message}")
    return lines


def build_opener(results: Sequence[ActionResult], step: int, level: int, win_levels: int,
                 valid_actions: Sequence[str], first_turn: bool = False,
                 images: Sequence[str] = (), retained: Sequence[str] = (),
                 evidence: Optional[dict] = None, pinned: str = "") -> str:
    """Text of the user message that starts a turn. `images` are captions of attached images;
    `evidence` is PythonTool.evidence() (see evidence_lines); `pinned` is
    PythonTool.pinned_block(): rules notes that survive trims and level changes."""
    lines: list[str] = []
    lines.extend(summarize_sequence(results))
    fatal = next((r for r in results if r.game_over), None)
    if fatal is not None:
        lines.append(GAME_OVER)
        lines.append(f"The fatal action was {fatal.action.display()}.")
    elif any(r.level_reset for r in results):
        lines.append(LEVEL_RESET)
    elif any(r.level_completed for r in results):
        lines.append(LEVEL_START)
    budget = budget_line(results, can_reset="RESET" in valid_actions)
    if budget:
        lines.append(budget)
    lines.extend(evidence_lines(evidence, valid_actions))
    if pinned:
        lines.append("[pinned notes]\n" + pinned)
    total = f" of {win_levels}" if win_levels else ""
    lines.append(f"Current state: step {step}, level {level + 1}{total} "
                 f"({level} completed). Valid actions: {', '.join(valid_actions)}.")
    if retained:
        lines.append("Retained functions: " + ", ".join(retained) + ".")
    if images:
        lines.append("Attached images: " + "; ".join(images) + ".")
    lines.append(TAIL_FIRST if first_turn else TAIL)
    return "\n".join(lines)


def diff_lines(actions: Sequence[ActionResult], max_full: int = 8, edge: int = 2) -> list[str]:
    """One object-diff line per executed action; long batches show only both ends."""
    numbered = [(i + 1, r) for i, r in enumerate(actions) if r.executed and r.diff]
    if not numbered:
        return []
    shown, omitted = numbered, 0
    if len(numbered) > max_full:
        shown = numbered[:edge] + numbered[-edge:]
        omitted = len(numbered) - 2 * edge
    lines = ["[effects]"]
    for k, (i, r) in enumerate(shown):
        if omitted and k == edge:
            lines.append(f"  ... {omitted} actions omitted ...")
        lines.append(f"  {i} {r.action.display()}: {r.diff}")
    return lines


def tool_result_text(stdout: str, error: Optional[str], actions: Sequence[ActionResult],
                     truncated: bool = False, retained: Sequence[str] = (),
                     rejected: Optional[dict] = None) -> str:
    """Content of the tool message returned to the model."""
    parts = []
    if stdout:
        parts.append(stdout.rstrip())
    if truncated:
        parts.append("[output truncated]")
    if error:
        parts.append("[error] " + error.rstrip())
    if actions:
        line = "[actions] " + "; ".join(describe_result(r) for r in actions)
        stop = next((r for r in reversed(actions) if r.stop_reason), None)
        if stop is not None:
            line += f" | stopped: {stop.stop_reason}"
            if stop.skipped:
                line += f", {stop.skipped} requested action(s) not run"
        last = next((r for r in reversed(actions) if r.executed), None)
        if last is not None and last.budget_left is not None:
            line += f" | move budget ~{last.budget_left} left"
        parts.append(line)
        parts.extend(diff_lines(actions))
    if retained:
        parts.append("[retained functions] " + ", ".join(retained))
    for name, why in (rejected or {}).items():
        parts.append(f"[not retained] {name}: {why}")
    return "\n".join(parts) if parts else "(no output)"
