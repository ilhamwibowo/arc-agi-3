"""Prompt text: the system prompt, the per-turn opener, the resume prompt and the nudge.

The system prompt is static per game (so the server can cache it); everything that changes
each turn goes into the opener. Wording is ours; the content follows what the public
solutions found useful (see reference/duck-architecture-brief.md, dfranzen-writeup-notes.md).
"""
from __future__ import annotations

from typing import Optional, Sequence

from arc3.types import COLOR_LEGEND, ActionResult

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
- `current_frame`: the latest board. `.grid` (numpy uint8 array, rows x cols), `.ascii` (letter rows), `.crop(r0, c0, r1, c1)` (a frame of the sub-rectangle, end-exclusive; use `.crop(...).ascii` to read a region), `.segmentation` (4-connected same-color objects: id, color, bbox, pixels, rotation-invariant hash, shape hash, enclosed children, plus adjacency pairs), `.step`, `.level`.
- `previous_frame`: the board before the most recent action (or None).
- `history`: list of entries with `.action`, `.frame` (board AFTER that action) and `.result`. `history[-1].frame` is `current_frame`.
- `transitions` / `last_transition`: per-action records with `.action`, `.before_frame`, `.after_frame`, `.result`.
- `valid_actions`: action names allowed right now.
- `last_action_call_result`: the result of your latest `action(...)` call (executed actions, flags, stop reasons). It survives later inspection calls.
- `last_animation_frames`: every intermediate frame of the last action that animated (index 0 first, last = settled board); empty if none. Reading them costs nothing.
- `frame_diff(before=None, after=None)`: object-level diff (moved, rotated, appeared, disappeared, changed_color, resized, changed_cell_count); defaults to previous_frame vs current_frame. Object matching is heuristic, not identity.
- `action(actions)`: executes real game actions, e.g. `action(['LEFT', 'LEFT'])` or `action([{{'action': 'MOUSE', 'row': 12, 'col': 30}}])`. It returns the call result, and all variables above are refreshed before your next statement."""

ACTIONS = """\
## Actions
- `UP`, `DOWN`, `LEFT`, `RIGHT`: directional inputs; what they move (if anything) is game-specific.
- `SPACE`: a game-specific trigger (select, rotate, interact, confirm...). Test it, do not assume.
- `MOUSE`: click a cell; `row` (0-63, downward) and `col` (0-63, rightward) are required.{undo}{reset}
- Only actions in `valid_actions` are accepted."""

UNDO = "\n- `UNDO`: reverts the previous action when available. Check what it restored. It cannot undo a game over."
RESET = ("\n- `RESET`: restarts the current level (completed levels are kept). It costs one action and the "
         "actions already spent still count. Use it only to escape a hopeless position.")

SIGNALS = """\
## Reading results
- `board_changed`: any cell changed. `gameplay_changed`: a cell changed more than {border} cells away from the edge. Timers and step bars live in that edge band, so an action that only shrinks a bar is `board_changed` but not `gameplay_changed` and did nothing useful.
- A bar along the edge that shrinks every action is usually a move budget. Do not mistake its segments for clickable objects.
- One action may play a short animation. `transient_pixels` > 0 means some cells changed and changed back mid-animation; that evidence exists only in `last_animation_frames`, so inspect them then.
- `game_over`: this attempt FAILED (hazard, or budget ran out). The harness resets the level automatically; completed levels are kept. It never means you won.
- `level_completed`: you advanced a level; the next board is already in `current_frame`. `run_complete`: the whole game is won.
- After any of these flags, stop acting in the current call and re-examine the board next turn. Guards may also stop a batch early (for example after an action that changed nothing in the interior); the stop reason is reported."""

TOOL = """\
## The python tool
- One tool, `python(code)`. Each call has a {timeout}s limit and its printed output is cut to about {out_tokens} tokens, so print compact summaries: object lists, counts, coordinates, small crops. Never print a whole board.
- Variables reset between calls, but top-level functions you define are kept and can be called later (no decorators/annotations; pass data as arguments). Importable modules: {modules}.
- Call the tool as often as you like. When unsure, run code rather than reasoning longer: code answers in a second what thinking only guesses."""

STRATEGY = """\
## Strategy
- On a new game, summarise the board structurally (objects, colors, likely player/cursor, HUD), then probe each available action once, cheaply, and measure its effect with `frame_diff`.
- Once you understand the mechanics and the goal, stop probing: write an explicit search (BFS/A*/simulation over the objects you extracted) and execute the found plan in one batched `action([...])` call.
- Prefer the shortest reliable sequence. Inside a batch, check in code that each step did what you predicted and stop when it did not.
- Do not frame goals as absolute coordinates; reason about objects and relations (reach, align, match, fill, sort, mirror, count).
- Levels usually extend earlier mechanics: carry what you learned forward as a hypothesis, look closely at new kinds of objects (they usually matter), and re-check whether the goal changed.
- The game is deterministic: a sequence that failed once will fail again."""


def build_system_prompt(hud_border: int = 4, expose_undo: bool = True, expose_reset: bool = False,
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

TAIL_FIRST = ("Start with a compact structural summary of `current_frame` from Python (not a "
              "printout of the board), then probe. When ready, call `action(...)` from inside "
              "`python`.")
TAIL = ("Inspect what changed (gameplay vs HUD) from Python, update your plan, then call "
        "`action(...)` from inside `python` with the best action or a batched sequence your code "
        "selected.")

RESUME = ("You paused on the turn budget. Your earlier tool results above are still valid; do "
          "not restart the analysis. Continue from them and call `action(...)` with your best "
          "next action or short batch.")

NUDGE = ("You have not acted yet: no `python` tool call was made. Use the `python` tool to "
         "inspect the board if needed, and call `action(...)` inside it. Do not write actions "
         "as plain text.")


def describe_result(r: ActionResult) -> str:
    """One compact line per action for the opener."""
    name = r.action.display()
    if not r.executed:
        return f"{name}: refused ({r.stop_reason or 'guard'})"
    if r.game_over:
        effect = "GAME OVER"
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
    if r.stop_reason:
        effect += f", then stopped ({r.stop_reason})"
    return f"{name}: {effect}"


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
        if r.stop_reason and r.message:
            lines.append(f"Guard stop ({r.stop_reason}): {r.message}")
    return lines


def build_opener(results: Sequence[ActionResult], step: int, level: int, win_levels: int,
                 valid_actions: Sequence[str], first_turn: bool = False,
                 images: Sequence[str] = (), retained: Sequence[str] = ()) -> str:
    """Text of the user message that starts a turn. `images` are captions of attached images."""
    lines: list[str] = []
    lines.extend(summarize_sequence(results))
    fatal = next((r for r in results if r.game_over), None)
    if fatal is not None:
        lines.append(GAME_OVER)
        lines.append(f"The fatal action was {fatal.action.display()}.")
    elif any(r.level_completed for r in results):
        lines.append(LEVEL_START)
    total = f" of {win_levels}" if win_levels else ""
    lines.append(f"Current state: step {step}, level {level + 1}{total} "
                 f"({level} completed). Valid actions: {', '.join(valid_actions)}.")
    if retained:
        lines.append("Retained functions: " + ", ".join(retained) + ".")
    if images:
        lines.append("Attached images: " + "; ".join(images) + ".")
    lines.append(TAIL_FIRST if first_turn else TAIL)
    return "\n".join(lines)


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
        parts.append("[actions] " + "; ".join(describe_result(r) for r in actions))
    if retained:
        parts.append("[retained functions] " + ", ".join(retained))
    for name, why in (rejected or {}).items():
        parts.append(f"[not retained] {name}: {why}")
    return "\n".join(parts) if parts else "(no output)"
