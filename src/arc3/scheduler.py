"""GPU slot scheduling across games: a priority gate over the server's decode slots.

All games run as threads, but only `max_active` of them may hold a decode slot (one in-flight model
request stream each). A game calls `gate.acquire(progress)` before a burst of model requests and keeps
the slot across tool calls; it calls `gate.release(progress)` at handover points (after a context
trim, when the prefix cache is cold anyway) and `gate.finish(progress)` when the game ends.

When a slot frees up it goes to the waiting game with the highest priority

    P = (A + B) * C
    A = l * (25 / (25 + a))^2 * 55 / (N (N + 1) / 2)       value of finishing the current level
    B = {>=3: 8, 2: 7, 1: 5, 0: 0}[levels after this one]   value of the levels still ahead,
                                                            faded linearly to 0 over the last 20%
    C = 0.25 * 2^-(a/115)^2 + 0.75 * 2^-(t/62000)^2       chance of still solving this level

with l = current level number (1-based), a = actions on this level, t = generated tokens on this
level, N = total levels clipped to 6..10 (10 when unknown). This is the published Milestone-2
formula; the 55/(N(N+1)/2) factor normalises a level's score weight to a 10-level game.

Games that have not had a turn yet outrank everything, so every game gets a first turn (in arrival
order) before the formula starts choosing.
"""
from __future__ import annotations

import asyncio
import itertools
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

TAIL_BY_REMAINING = {0: 0.0, 1: 5.0, 2: 7.0}
TAIL_MAX = 8.0                    # three or more levels after the current one
TAIL_FADE_FRACTION = 0.2
DEFAULT_LEVELS = 10
UNSTARTED_PRIORITY = float("inf")


@dataclass
class GameProgress:
    """Per-game counters the agent keeps current; the gate reads them when ranking waiters."""
    game_id: str
    level: int = 0                # levels completed so far (current level number is level + 1)
    actions_on_level: int = 0
    tokens_on_level: int = 0      # generated (completion) tokens on the current level
    levels_total: int = 0         # 0 when unknown
    started: bool = False         # had at least one model turn
    finished: bool = False

    def level_completed(self, levels_completed: int) -> None:
        """Call when the environment reports a new level count; resets the per-level counters."""
        if levels_completed != self.level:
            self.level = levels_completed
            self.actions_on_level = 0
            self.tokens_on_level = 0


def _clip_levels(levels_total: int) -> int:
    if levels_total <= 0:
        return DEFAULT_LEVELS
    return max(6, min(10, levels_total))


def tail_value(levels_after_current: int, run_fraction: float) -> float:
    """B term: lookup by levels remaining after the current one, faded over the run's tail."""
    base = TAIL_BY_REMAINING.get(levels_after_current, TAIL_MAX) if levels_after_current >= 0 else 0.0
    fade_start = 1.0 - TAIL_FADE_FRACTION
    if run_fraction <= fade_start:
        return base
    return base * max(0.0, (1.0 - run_fraction) / TAIL_FADE_FRACTION)


def priority(progress: GameProgress, run_fraction: float = 0.0) -> float:
    """P = (A + B) * C for one game; `run_fraction` is elapsed / scheduled runtime in [0, 1]."""
    if not progress.started:
        return UNSTARTED_PRIORITY
    n = _clip_levels(progress.levels_total)
    l = min(progress.level + 1, n)
    a = max(0, progress.actions_on_level)
    t = max(0, progress.tokens_on_level)
    a_term = l * (25.0 / (25.0 + a)) ** 2 * 55.0 / (n * (n + 1) / 2.0)
    b_term = tail_value(n - l, run_fraction)
    c_term = 0.25 * 2.0 ** (-((a / 115.0) ** 2)) + 0.75 * 2.0 ** (-((t / 62000.0) ** 2))
    return (a_term + b_term) * c_term


class PriorityGate:
    """Admits at most `max_active` games to the GPU at once, highest priority first.

    Thread-safe: games are threads that block in `acquire`. Ties (including all unstarted games)
    are broken by arrival order of the acquire call, so a game that releases and immediately
    re-acquires queues behind equally ranked waiters.
    """

    def __init__(self, max_active: int = 10, clock: Callable[[], float] = time.time,
                 poll_seconds: float = 5.0):
        if max_active < 1:
            raise ValueError("max_active must be >= 1")
        self.max_active = max_active
        self._clock = clock
        self._poll = poll_seconds         # re-rank periodically: the tail fade moves with time
        self._cond = threading.Condition()
        self._active: dict[str, GameProgress] = {}
        self._waiting: dict[str, tuple[int, GameProgress]] = {}
        self._seq = itertools.count()
        self._run_start: Optional[float] = None
        self._run_end: Optional[float] = None
        self.grants: dict[str, int] = {}
        self.wait_seconds: dict[str, float] = {}

    # ---- run window (drives the tail fade) ----
    def set_run_window(self, start: float, end: float) -> None:
        with self._cond:
            self._run_start, self._run_end = start, end
            self._cond.notify_all()

    def run_fraction(self) -> float:
        if self._run_start is None or self._run_end is None or self._run_end <= self._run_start:
            return 0.0
        frac = (self._clock() - self._run_start) / (self._run_end - self._run_start)
        return min(1.0, max(0.0, frac))

    # ---- slot handling ----
    def acquire(self, game: GameProgress, timeout: Optional[float] = None) -> bool:
        """Block until `game` holds a slot. Returns False on timeout or if the game is finished.
        Re-acquiring while already holding the slot returns immediately."""
        deadline = None if timeout is None else self._clock() + timeout
        t0 = self._clock()
        with self._cond:
            if game.game_id in self._active:
                return True
            if game.finished:
                return False
            self._waiting[game.game_id] = (next(self._seq), game)
            try:
                while not self._can_grant(game.game_id):
                    wait = self._poll
                    if deadline is not None:
                        remaining = deadline - self._clock()
                        if remaining <= 0:
                            return False
                        wait = min(wait, remaining)
                    self._cond.wait(wait)
                    if game.finished:
                        return False
                self._active[game.game_id] = game
                game.started = True
                self.grants[game.game_id] = self.grants.get(game.game_id, 0) + 1
                self.wait_seconds[game.game_id] = (self.wait_seconds.get(game.game_id, 0.0)
                                                   + self._clock() - t0)
                return True
            finally:
                self._waiting.pop(game.game_id, None)
                # Another waiter may now be in the top-k (we left the queue either way).
                self._cond.notify_all()

    async def acquire_async(self, game: GameProgress, timeout: Optional[float] = None) -> bool:
        """`await`-able form of acquire for asyncio callers; blocks a worker thread, not the loop."""
        return await asyncio.to_thread(self.acquire, game, timeout)

    def release(self, game: GameProgress) -> None:
        with self._cond:
            if self._active.pop(game.game_id, None) is not None:
                self._cond.notify_all()

    def finish(self, game: GameProgress) -> None:
        """Mark the game finished and free its slot; a pending acquire for it returns False."""
        with self._cond:
            game.finished = True
            self._active.pop(game.game_id, None)
            self._cond.notify_all()

    def holds(self, game: GameProgress) -> bool:
        with self._cond:
            return game.game_id in self._active

    def _can_grant(self, game_id: str) -> bool:
        free = self.max_active - len(self._active)
        if free <= 0:
            return False
        return game_id in {gid for gid, _ in self._ranked_waiters()[:free]}

    def _ranked_waiters(self) -> list[tuple[str, float]]:
        frac = self.run_fraction()
        scored = [(-priority(p, frac), seq, gid) for gid, (seq, p) in self._waiting.items()]
        scored.sort()
        return [(gid, -neg) for neg, _, gid in scored]

    def snapshot(self) -> dict:
        """JSON-friendly state for logs."""
        with self._cond:
            return {
                "run_fraction": round(self.run_fraction(), 4),
                "active": sorted(self._active),
                "waiting": [(gid, None if p == UNSTARTED_PRIORITY else round(p, 4))
                            for gid, p in self._ranked_waiters()],
            }
