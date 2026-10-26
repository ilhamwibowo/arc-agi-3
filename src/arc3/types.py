"""Shared contracts for arc3. Every module codes against these; keep them small and stable.

Grid convention: numpy uint8 array shape (rows, cols), values 0-15, (0,0) top-left.
Action names seen by the model: UP DOWN LEFT RIGHT SPACE MOUSE(row,col) UNDO. RESET is harness-only
(auto after GAME_OVER) unless a config exposes it. Engine mapping: ACTION1..4 = UP/DOWN/LEFT/RIGHT,
ACTION5 = SPACE, ACTION6 = MOUSE, ACTION7 = UNDO (verify against arc_agi GameAction at runtime).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np

MODEL_ACTIONS = ("UP", "DOWN", "LEFT", "RIGHT", "SPACE", "MOUSE", "UNDO", "RESET")
ENGINE_TO_MODEL = {
    "ACTION1": "UP", "ACTION2": "DOWN", "ACTION3": "LEFT", "ACTION4": "RIGHT",
    "ACTION5": "SPACE", "ACTION6": "MOUSE", "ACTION7": "UNDO", "RESET": "RESET",
}
MODEL_TO_ENGINE = {v: k for k, v in ENGINE_TO_MODEL.items()}

# ARC palette letters for ascii rendering (index = color value 0..15)
COLOR_LETTERS = "KBRGYAMOCWPLEDFH"  # 0 black(K) 1 blue 2 red 3 green 4 yellow 5 gray(A) 6 magenta
# 7 orange 8 cyan 9 white 10 pink 11 lime 12 maroon(E) 13 dark(D) 14 forest(F) 15 hot(H)


class GameState(str, Enum):
    NOT_FINISHED = "NOT_FINISHED"
    NOT_PLAYED = "NOT_PLAYED"
    WIN = "WIN"
    GAME_OVER = "GAME_OVER"


@dataclass(frozen=True)
class Action:
    name: str                      # model name: UP/DOWN/LEFT/RIGHT/SPACE/MOUSE/UNDO/RESET
    row: Optional[int] = None      # MOUSE only
    col: Optional[int] = None

    def engine_name(self) -> str:
        return MODEL_TO_ENGINE[self.name]

    def display(self) -> str:
        return f"MOUSE({self.row},{self.col})" if self.name == "MOUSE" else self.name


@dataclass
class Observation:
    """One environment response. `frames` holds every animation frame (last == grid)."""
    grid: np.ndarray                      # final frame
    frames: list[np.ndarray]              # all frames returned for this step (len>=1)
    state: GameState
    levels_completed: int
    win_levels: int                       # total levels in the game (0 if unknown)
    available_actions: list[str]          # model action names available now
    step: int                             # engine step counter after this observation
    game_id: str = ""
    raw: Any = None                       # toolkit object, never shown to the model


@dataclass
class ActionResult:
    """Result of executing ONE action in the environment, as reported to the model."""
    action: Action
    executed: bool
    board_changed: bool                   # any cell differs from before
    gameplay_changed: bool                # any cell differs outside the HUD border
    level_completed: bool
    game_over: bool
    run_complete: bool
    animated: bool                        # more than one distinct frame
    transient_pixels: int = 0             # cells that changed and changed back within animation
    automatic: bool = False               # harness-issued (e.g. auto RESET)
    stop_reason: Optional[str] = None     # guard name when refused/stopped
    message: str = ""                     # human-readable explanation for the model
    before: Optional[np.ndarray] = None
    after: Optional[np.ndarray] = None
    frames: list[np.ndarray] = field(default_factory=list)


@dataclass
class Transition:
    index: int
    action: Action
    before: Observation
    after: Observation
    result: ActionResult


@dataclass
class GameConfig:
    game_id: str
    max_actions: Optional[int] = None
    max_seconds: float = 30 * 60
    hud_border: int = 4                   # cells; changes only within this border are HUD-only
    guards_from_level: int = 2
    expose_reset: bool = False
    expose_undo: bool = True
    image_scale: int = 10
    seed: int = 0


@dataclass
class ToolResult:
    """Return of one `python` tool call."""
    stdout: str                           # possibly middle-truncated
    error: Optional[str]                  # exception text or None
    actions: list[ActionResult]           # actions executed during the call, in order
    truncated: bool = False
    retained_functions: list[str] = field(default_factory=list)
    rejected_functions: dict[str, str] = field(default_factory=dict)
    seconds: float = 0.0


@dataclass
class LLMMessage:
    role: str                             # system/user/assistant/tool
    content: Any                          # str or list of parts (text/image_url)
    reasoning: Optional[str] = None       # assistant thinking, preserved in history
    tool_calls: Optional[list[dict]] = None
    tool_call_id: Optional[str] = None
    tokens: int = 0                       # calibrated estimate incl. images


@dataclass
class TurnStats:
    generated_tokens: int = 0
    prompt_tokens: int = 0
    cached_tokens: int = 0
    tool_calls: int = 0
    actions: int = 0
    seconds: float = 0.0
