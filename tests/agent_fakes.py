"""Fakes for agent-loop tests: a tiny deterministic game session and an exec-based tool."""
from __future__ import annotations

import contextlib
import io
import struct
import zlib

import numpy as np

from arc3.types import Action, ActionResult, GameConfig, GameState, Observation, ToolResult


def fake_png(w: int = 640, h: int = 640) -> bytes:
    """Header-valid PNG (only IHDR is read by the context manager)."""
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    chunk = struct.pack(">I", 13) + b"IHDR" + ihdr + struct.pack(">I", zlib.crc32(b"IHDR" + ihdr))
    return b"\x89PNG\r\n\x1a\n" + chunk


class FakeSession:
    """Player cell moves with arrows; SPACE completes the level when the player is at col 10;
    DOWN is fatal. One level, so completing it wins."""

    def __init__(self, win_levels: int = 1):
        self.config = GameConfig(game_id="fake")
        self.events: list[dict] = []
        self.history: list[ActionResult] = []
        self.pos = [8, 8]
        self.levels = 0
        self.win_levels = win_levels
        self.step_count = 0
        self.state = GameState.NOT_FINISHED
        self.resets = 0
        self.current = self._obs()
        self.previous = None

    def trace(self, event: dict) -> None:
        self.events.append(event)

    @property
    def valid_actions(self):
        return ["UP", "DOWN", "LEFT", "RIGHT", "SPACE"]

    @property
    def level(self):
        return self.levels

    def _grid(self):
        g = np.zeros((64, 64), dtype=np.uint8)
        g[self.pos[0], self.pos[1]] = 3
        g[0, : max(0, 20 - self.step_count)] = 4      # HUD bar
        return g

    def _obs(self):
        g = self._grid()
        return Observation(grid=g, frames=[g], state=self.state, levels_completed=self.levels,
                           win_levels=self.win_levels, available_actions=self.valid_actions,
                           step=self.step_count)

    def step(self, action: Action, automatic: bool = False) -> ActionResult:
        before = self.current.grid
        self.step_count += 1
        level_done = game_over = False
        if action.name == "RESET":
            self.pos, self.state = [8, 8], GameState.NOT_FINISHED
        elif action.name == "DOWN":
            game_over = True
            self.state = GameState.GAME_OVER
        elif action.name == "RIGHT":
            self.pos[1] += 1
        elif action.name == "LEFT":
            self.pos[1] -= 1
        elif action.name == "SPACE" and self.pos[1] == 10:
            level_done = True
            self.levels += 1
            if self.levels >= self.win_levels:
                self.state = GameState.WIN
        self.previous, self.current = self.current, self._obs()
        after = self.current.grid
        changed = before != after
        r = ActionResult(action=action, executed=True, board_changed=bool(changed.any()),
                         gameplay_changed=bool(changed[4:-4, 4:-4].any()),
                         level_completed=level_done, game_over=game_over,
                         run_complete=self.state == GameState.WIN, animated=False,
                         automatic=automatic, before=before, after=after, frames=[after])
        self.history.append(r)
        return r

    def auto_reset_if_needed(self):
        if self.state == GameState.GAME_OVER:
            self.resets += 1
            self.step(Action("RESET"), automatic=True)


class FakeTool:
    """Executes snippets with `exec`; exposes `current_frame` (grid), `action(list)`."""

    def __init__(self, session: FakeSession):
        self.session = session
        self.codes: list[str] = []

    def run(self, code: str) -> ToolResult:
        self.codes.append(code)
        executed: list[ActionResult] = []

        def action(actions):
            for a in actions:
                if self.session.state != GameState.NOT_FINISHED:
                    break
                r = self.session.step(Action(a))
                executed.append(r)
                if r.game_over or r.level_completed:
                    break
            return executed

        buf, err = io.StringIO(), None
        ns = {"action": action, "current_frame": self.session.current.grid}
        with contextlib.redirect_stdout(buf):
            try:
                exec(code, ns)
            except Exception as e:
                err = repr(e)
        return ToolResult(stdout=buf.getvalue(), error=err, actions=executed)
