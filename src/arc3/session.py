"""GameSession: the glue between Env, the sandbox and the agent loop.

Owns the environment handle and the public interaction history for one game. The sandbox calls
`step(action)`; the agent calls `auto_reset_if_needed()` after a death and reads `current`.
Everything written to the trace is public observation data only (frame hashes, actions, flags).
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from arc3 import percept
from arc3.types import Action, ActionResult, GameConfig, GameState, Observation, Transition


class GameSession:
    def __init__(self, env, config: GameConfig, trace_path: Optional[str | Path] = None,
                 on_action: Optional[Callable[[ActionResult], None]] = None):
        self.env = env
        self.config = config
        self.history: list[Transition] = []
        self.previous: Optional[Observation] = None
        self.current: Observation = env.observation or env.reset()
        self.actions = 0            # scored actions issued by the model (excludes initial reset)
        self.resets = 0             # automatic resets after GAME_OVER
        self.started = time.time()
        self.on_action = on_action
        self._trace_file = open(trace_path, "a") if trace_path else None
        # First board of the current level; used to spot silent resets (see _is_level_reset).
        self.level_start: np.ndarray = self.current.grid
        # Non-RESET actions since the level started or was last RESET; a RESET with none is a no-op
        # at best and, at a fresh level, a full game reset in some engine builds.
        self.since_restart = 0
        self.budget = percept.BudgetTracker(border=config.hud_border)
        self.budget.reset(self.current.grid)
        self.trace({"event": "start", "step": self.current.step,
                    "levels": self.current.levels_completed, "win_levels": self.current.win_levels,
                    "grid": percept.state_hash(self.current.grid)})

    # ---- views used by sandbox / agent ---------------------------------------------------------

    @property
    def level(self) -> int:
        """1-based current level (matches FrameView.level)."""
        return self.current.levels_completed + 1

    @property
    def valid_actions(self) -> list[str]:
        names = list(self.current.available_actions)
        if not self.config.expose_reset:
            names = [n for n in names if n != "RESET"]
        elif "RESET" not in names and self.current.state != GameState.WIN:
            names.append("RESET")   # the engine never lists RESET; it is always accepted
        if not self.config.expose_undo:
            names = [n for n in names if n != "UNDO"]
        return names

    @property
    def finished(self) -> bool:
        return self.current.state == GameState.WIN

    # ---- acting --------------------------------------------------------------------------------

    def step(self, action: Action, automatic: bool = False) -> ActionResult:
        before = self.current
        if before.state == GameState.WIN:
            return self._refused(action, "run_complete", "the game is already won")
        if before.state == GameState.GAME_OVER and action.name != "RESET":
            return self._refused(action, "game_over", "attempt is over; waiting for automatic reset")
        if self.config.max_actions is not None and self.actions >= self.config.max_actions:
            return self._refused(action, "action_budget", "action budget exhausted")
        if (action.name == "RESET" and not automatic and self.since_restart == 0
                and before.state != GameState.GAME_OVER):
            return self._refused(action, "reset_noop", "the level is already at its start board; "
                                 "RESET would change nothing")

        after = self.env.step(action)
        board_changed, gameplay_changed = percept.changed_masks(
            before.grid, after.grid, border=self.config.hud_border)
        chain = percept.collapse_frames(before.grid, after.frames)
        transient, _bbox = percept.transient_cells(chain) if len(chain) > 2 else (0, None)
        level_completed = after.levels_completed > before.levels_completed
        alive = after.state not in (GameState.GAME_OVER, GameState.WIN)
        level_reset = (alive and not level_completed
                       and action.name not in ("RESET", "UNDO") and board_changed
                       and self._is_level_reset(before.grid, after.grid, int(transient)))
        res = ActionResult(
            action=action, executed=True, board_changed=board_changed,
            gameplay_changed=gameplay_changed, level_completed=level_completed,
            game_over=after.state == GameState.GAME_OVER, run_complete=after.state == GameState.WIN,
            animated=len(chain) > 2, transient_pixels=int(transient), automatic=automatic,
            before=before.grid, after=after.grid, frames=list(after.frames),
            level_reset=level_reset)
        self._track_budget(res, after)
        res.message = self._describe(res)
        self.history.append(Transition(len(self.history), action, before, after, res))
        self.previous, self.current = before, after
        if level_completed:
            self.level_start = after.grid
        if level_completed or action.name == "RESET":
            self.since_restart = 0
        else:
            self.since_restart += 1
        if automatic:
            self.resets += 1
        else:
            self.actions += 1
        self.trace({"event": "action", "action": action.display(), "automatic": automatic,
                    "step": after.step, "levels": after.levels_completed, "state": after.state.value,
                    "board_changed": board_changed, "gameplay_changed": gameplay_changed,
                    "level_reset": level_reset, "frames": len(chain) - 1,
                    "transient": int(transient),
                    "grid": percept.state_hash(after.grid), "t": round(time.time() - self.started, 2)})
        if self.on_action:
            self.on_action(res)
        return res

    def auto_reset_if_needed(self) -> Optional[ActionResult]:
        if self.current.state != GameState.GAME_OVER:
            return None
        return self.step(Action("RESET"), automatic=True)

    # ---- helpers -------------------------------------------------------------------------------

    def _track_budget(self, res: ActionResult, after: Observation) -> None:
        """Feed the budget-bar detector; it restarts wherever the bar may legitimately refill."""
        if (res.level_completed or res.level_reset or res.game_over
                or res.action.name in ("RESET", "UNDO")):
            self.budget.reset(after.grid)   # UNDO may give the spent cells back
            return
        est = self.budget.update(after.grid)
        if est is not None:
            res.budget_left, res.budget_bar = est

    def _is_level_reset(self, before: np.ndarray, after: np.ndarray, transient: int) -> bool:
        """True when `after` is the level's start board again although `before` was not.

        Some games (ls20) answer a fatal move or an exhausted move budget by putting the level
        back to its start and taking a life, without a GAME_OVER state. Only the interior is
        compared: the HUD band (lives, budget bar) differs from the start after a life is lost.
        Walking back to the start square also restores the interior, so we additionally require
        a reset-like signature: the HUD band moved back toward its start state (the budget bar
        refilled) or a large flash animation played.
        """
        b = self.config.hud_border
        start = self.level_start
        if before.shape != start.shape or after.shape != start.shape:
            return False
        inner = (slice(b, start.shape[0] - b), slice(b, start.shape[1] - b)) if b else \
            (slice(None), slice(None))
        if not np.array_equal(after[inner], start[inner]) or \
                np.array_equal(before[inner], start[inner]):
            return False
        if not b:
            return True   # no HUD band configured: nothing else to check
        hud_off = lambda g: int((g != start).sum() - (g[inner] != start[inner]).sum())
        flash = transient >= (start.shape[0] - 2 * b) * (start.shape[1] - 2 * b) // 4
        return hud_off(after) < hud_off(before) or flash

    def _refused(self, action: Action, reason: str, message: str) -> ActionResult:
        return ActionResult(action=action, executed=False, board_changed=False,
                            gameplay_changed=False, level_completed=False, game_over=False,
                            run_complete=False, animated=False, stop_reason=reason, message=message,
                            before=self.current.grid, after=self.current.grid)

    @staticmethod
    def _describe(r: ActionResult) -> str:
        if r.run_complete:
            return "run complete: the whole game is won"
        if r.game_over:
            return "GAME OVER: this attempt failed"
        if r.level_completed:
            return "level completed"
        if r.level_reset:
            return ("LEVEL RESET: the board returned to this level's start position (a fatal move, "
                    "an exhausted move budget, or a deliberate restart mechanic)")
        if r.action.name == "RESET":
            return "RESET: the level restarted from its start board; completed levels are kept"
        if not r.board_changed:
            return "nothing changed"
        if not r.gameplay_changed:
            return "only border/HUD cells changed"
        return "board changed" + (f", animated ({r.transient_pixels} transient cells)"
                                  if r.animated else "")

    def trace(self, event: dict) -> None:
        if self._trace_file:
            self._trace_file.write(json.dumps(event, default=str) + "\n")
            self._trace_file.flush()

    def close(self) -> None:
        if self._trace_file:
            self._trace_file.close()
            self._trace_file = None
