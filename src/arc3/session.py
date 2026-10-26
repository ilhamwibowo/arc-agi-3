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

        after = self.env.step(action)
        board_changed, gameplay_changed = percept.changed_masks(
            before.grid, after.grid, border=self.config.hud_border)
        chain = percept.collapse_frames(before.grid, after.frames)
        transient, _bbox = percept.transient_cells(chain) if len(chain) > 2 else (0, None)
        level_completed = after.levels_completed > before.levels_completed
        res = ActionResult(
            action=action, executed=True, board_changed=board_changed,
            gameplay_changed=gameplay_changed, level_completed=level_completed,
            game_over=after.state == GameState.GAME_OVER, run_complete=after.state == GameState.WIN,
            animated=len(chain) > 2, transient_pixels=int(transient), automatic=automatic,
            before=before.grid, after=after.grid, frames=list(after.frames))
        res.message = self._describe(res)
        self.history.append(Transition(len(self.history), action, before, after, res))
        self.previous, self.current = before, after
        if automatic:
            self.resets += 1
        else:
            self.actions += 1
        self.trace({"event": "action", "action": action.display(), "automatic": automatic,
                    "step": after.step, "levels": after.levels_completed, "state": after.state.value,
                    "board_changed": board_changed, "gameplay_changed": gameplay_changed,
                    "frames": len(chain) - 1, "transient": int(transient),
                    "grid": percept.state_hash(after.grid), "t": round(time.time() - self.started, 2)})
        if self.on_action:
            self.on_action(res)
        return res

    def auto_reset_if_needed(self) -> Optional[ActionResult]:
        if self.current.state != GameState.GAME_OVER:
            return None
        return self.step(Action("RESET"), automatic=True)

    # ---- helpers -------------------------------------------------------------------------------

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
