"""Environment adapter around the arc_agi toolkit (0.9.8) and arcengine (0.9.3).

`Arena` owns one toolkit `Arcade` (offline or competition) and the scorecard; `Env` wraps one game
and turns toolkit FrameDataRaw responses into `arc3.types.Observation`.

Verified toolkit facts (see docs/env-percept.md and tests/test_env.py):
- One step returns a list of frames (`FrameDataRaw.frame`), one per internal engine tick; the last
  one is the settled board. All public games render 64x64 int8 grids with values 0..15.
- MOUSE is engine ACTION6 with data={'x': col, 'y': row}: x is the column, y the row of the 64x64
  display grid (verified by clicking cells on ft09/tn36 and checking where the board changes).
- `win_levels` and `levels_completed` come with every response (`_win_score`/`_score` in the engine).
- Any non-RESET action after GAME_OVER or WIN returns an EMPTY frame list with levels_completed=0
  (an engine quirk). We keep the previous grid and level count in that case.
- RESET restarts the current level, keeping completed levels, EXCEPT when no action has been taken
  since the level started (fresh game, right after a level-up, or right after another RESET): then
  it is a FULL reset and levels_completed drops to 0. Env refuses such a RESET by default
  (`safe_reset`). With env var ONLY_RESET_LEVELS=true the engine always does a level reset (except
  after WIN); Arena always sets it before creating the Arcade. In competition mode the engine runs
  behind the gateway, so the local env var may have no effect there; the guard covers that.

Thread safety: Arcade.make holds the toolkit's lock, Arena.open_scorecard holds ours, and each Env
owns its own game (local) or HTTP session (remote), so different games can step concurrently.
One Env must not be stepped from two threads at once.

The model must never see game ids, hidden state or scorecards: `Observation.raw` holds the toolkit
object for the harness only, and `log_dict()` reports only what the model may see plus the game id.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Optional

import numpy as np
from arc_agi import Arcade, OperationMode
from arcengine import GameAction

from .types import ENGINE_TO_MODEL, Action, GameState, Observation

log = logging.getLogger("arc3.env")


def _toolkit_logger() -> logging.Logger:
    # The toolkit logs every scorecard/recording event at INFO; keep only warnings.
    lg = logging.getLogger("arc3.toolkit")
    lg.setLevel(logging.WARNING)
    return lg


def list_public_games(environments_dir: str | Path) -> list[str]:
    """Return sorted full game ids ('ft09-0d8bbf25') found under environments_dir."""
    games = []
    for meta in Path(environments_dir).rglob("metadata.json"):
        try:
            games.append(json.loads(meta.read_text(encoding="utf-8"))["game_id"])
        except (OSError, ValueError, KeyError):
            log.warning("skipping unreadable %s", meta)
    return sorted(set(games))


def engine_action_names(ids: list[int]) -> list[str]:
    """Map toolkit action ids (0..7) to model action names, dropping unknown ids."""
    out = []
    for i in ids:
        key = "RESET" if i == 0 else f"ACTION{i}"
        if key in ENGINE_TO_MODEL:
            out.append(ENGINE_TO_MODEL[key])
    return out


def _dedupe_adjacent(frames: list[np.ndarray]) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for f in frames:
        if not out or not np.array_equal(out[-1], f):
            out.append(f)
    return out


class Arena:
    """Holds one Arcade and its scorecard; creates Envs. Competition mode shares one scorecard."""

    def __init__(self, mode: str = "offline", environments_dir: str | Path | None = None,
                 base_url: Optional[str] = None, api_key: str = "",
                 recordings_dir: str | Path = "runs/toolkit"):
        """mode='offline' needs environments_dir; mode='competition' needs base_url (+ api_key)."""
        # Read by arcengine at every RESET (process-wide): RESET never wipes completed levels.
        os.environ["ONLY_RESET_LEVELS"] = "true"
        self.competition = mode == "competition"
        self.scorecard_id: Optional[str] = None
        self._lock = threading.Lock()
        recordings = str(Path(recordings_dir).resolve())
        if self.competition:
            if not base_url:
                raise ValueError("competition mode needs base_url")
            self.arcade = Arcade(arc_api_key=api_key, arc_base_url=base_url.rstrip("/"),
                                 operation_mode=OperationMode.COMPETITION,
                                 recordings_dir=recordings, logger=_toolkit_logger())
        elif mode == "offline":
            if environments_dir is None:
                raise ValueError("offline mode needs environments_dir")
            self.arcade = Arcade(operation_mode=OperationMode.OFFLINE,
                                 environments_dir=str(Path(environments_dir).resolve()),
                                 recordings_dir=recordings, logger=_toolkit_logger())
        else:
            raise ValueError(f"unknown mode {mode!r}")

    @classmethod
    def offline(cls, environments_dir: str | Path,
                recordings_dir: str | Path = "runs/toolkit") -> "Arena":
        return cls("offline", environments_dir=environments_dir, recordings_dir=recordings_dir)

    @classmethod
    def competition_mode(cls, arc_base_url: str, api_key: str,
                         recordings_dir: str | Path = "runs/toolkit") -> "Arena":
        return cls("competition", base_url=arc_base_url, api_key=api_key,
                   recordings_dir=recordings_dir)

    def game_ids(self) -> list[str]:
        return sorted({e.game_id for e in self.arcade.available_environments})

    def open_scorecard(self, tags: Optional[list[str]] = None) -> str:
        """Open the (single) scorecard if none is open; return its id."""
        with self._lock:  # many game threads call make() at once; open exactly one card
            if self.scorecard_id is None:
                self.scorecard_id = self.arcade.open_scorecard(tags=tags or ["arc3"])
            return self.scorecard_id

    def close_scorecard(self) -> Optional[dict]:
        """Close the scorecard; returns its JSON dump without the api key (harness-only data)."""
        with self._lock:
            card_id, self.scorecard_id = self.scorecard_id, None
        if card_id is None:
            return None
        card = self.arcade.close_scorecard(card_id)
        if card is None:
            return None
        data = card.model_dump(mode="json")
        data.pop("api_key", None)
        return data

    def make(self, game_id: str, seed: int = 0, safe_reset: bool = True) -> "Env":
        card = self.open_scorecard()
        wrapper = self.arcade.make(game_id, seed=seed, scorecard_id=card)
        if wrapper is None:
            raise RuntimeError(f"toolkit could not create game {game_id}")
        return Env(wrapper, game_id, safe_reset=safe_reset)


class Env:
    """One game. `reset()` / `step(action)` return Observations with every animation frame."""

    def __init__(self, wrapper, game_id: str, safe_reset: bool = True):
        self._wrapper = wrapper
        self.game_id = game_id
        self.safe_reset = safe_reset
        self.steps = 0                      # non-RESET actions we sent
        self._actions_since_level_start = 0
        self._last: Optional[Observation] = None
        self.arena: Optional[Arena] = None  # set by the offline() convenience constructor
        first = wrapper.observation_space   # toolkit already reset the game in make()
        if first is not None and len(first.frame) > 0:
            self._last = self._convert(first, previous=None)

    # -- constructors -------------------------------------------------------------------------
    @classmethod
    def offline(cls, game_id: str, environments_dir: str | Path, seed: int = 0,
                recordings_dir: str | Path = "runs/toolkit", safe_reset: bool = True) -> "Env":
        arena = Arena.offline(environments_dir, recordings_dir)
        env = arena.make(game_id, seed=seed, safe_reset=safe_reset)
        env.arena = arena
        return env

    @classmethod
    def competition(cls, arc_base_url: str, api_key: str) -> Arena:
        """Competition play is many games on one scorecard, so this returns the Arena."""
        return Arena.competition_mode(arc_base_url, api_key)

    # -- properties ---------------------------------------------------------------------------
    @property
    def observation(self) -> Optional[Observation]:
        return self._last

    @property
    def state(self) -> GameState:
        return self._last.state if self._last else GameState.NOT_PLAYED

    @property
    def levels_completed(self) -> int:
        return self._last.levels_completed if self._last else 0

    @property
    def win_levels(self) -> int:
        return self._last.win_levels if self._last else 0

    @property
    def available_actions(self) -> list[str]:
        return list(self._last.available_actions) if self._last else []

    # -- actions ------------------------------------------------------------------------------
    def reset(self) -> Observation:
        """Send RESET. Restarts the current level (see module docstring for the full-reset trap)."""
        return self.step(Action("RESET"))

    def would_full_reset(self) -> bool:
        """True when the engine's default rule would make a RESET now wipe completed levels.

        Applied even with ONLY_RESET_LEVELS=true: the gateway's engine setting is unknown, and a
        RESET with no action since the level start would only return the same board anyway.
        """
        return self._actions_since_level_start == 0 and self.levels_completed > 0

    def close(self) -> None:
        """Nothing to release per game; the Arena owns the scorecard."""

    def step(self, action: Action) -> Observation:
        if action.name == "RESET":
            if self.safe_reset and self._last is not None and self.would_full_reset():
                log.warning("%s: refused RESET that would wipe %d completed levels",
                            self.game_id, self.levels_completed)
                return self._last
            raw = self._wrapper.step(GameAction.RESET)
        else:
            raw = self._wrapper.step(GameAction[action.engine_name()], data=self._data(action))
            self.steps += 1
        if raw is None:
            raise RuntimeError(f"{self.game_id}: toolkit step failed for {action.display()}")
        obs = self._convert(raw, previous=self._last)
        self._track_level(action, obs)
        self._last = obs
        log.debug("%s", json.dumps(self.log_dict(action, obs)))
        return obs

    @staticmethod
    def _data(action: Action) -> dict:
        if action.name != "MOUSE":
            return {}
        if action.row is None or action.col is None:
            raise ValueError("MOUSE needs row and col")
        if not (0 <= action.row <= 63 and 0 <= action.col <= 63):
            raise ValueError(f"MOUSE out of range: {action.display()}")
        return {"x": int(action.col), "y": int(action.row)}

    def _track_level(self, action: Action, obs: Observation) -> None:
        prev_levels = self._last.levels_completed if self._last else 0
        if action.name == "RESET" or obs.levels_completed != prev_levels:
            self._actions_since_level_start = 0
        else:
            self._actions_since_level_start += 1

    def _convert(self, raw, previous: Optional[Observation]) -> Observation:
        frames = [np.asarray(f).astype(np.uint8) for f in (raw.frame or [])]
        state = GameState(raw.state.value if hasattr(raw.state, "value") else str(raw.state))
        if not frames:
            # Engine returns no frames (and a bogus level count) for actions on a finished game.
            if previous is None:
                raise RuntimeError(f"{self.game_id}: empty first observation")
            frames = [previous.grid]
            levels, win_levels = previous.levels_completed, previous.win_levels
        else:
            levels, win_levels = int(raw.levels_completed), int(raw.win_levels)
            if previous is not None and not win_levels:
                win_levels = previous.win_levels
        frames = _dedupe_adjacent(frames)
        return Observation(
            grid=frames[-1], frames=frames, state=state, levels_completed=levels,
            win_levels=win_levels, available_actions=engine_action_names(list(raw.available_actions)),
            step=self.steps, game_id=self.game_id, raw=raw,
        )

    def log_dict(self, action: Action, obs: Observation) -> dict:
        """Trace record: no raw toolkit object, no hidden state, grids as hashes."""
        from .percept import state_hash
        return {
            "game": self.game_id, "action": action.display(), "step": obs.step,
            "state": obs.state.value, "levels": obs.levels_completed, "win_levels": obs.win_levels,
            "frames": len(obs.frames), "grid": state_hash(obs.grid),
            "available": obs.available_actions,
        }
