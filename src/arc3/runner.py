"""Run many games concurrently: one thread per game, GPU slots arbitrated by a PriorityGate.

Contract with the other modules (duck-typed so tests can use fakes):
- `make_env(game_id) -> env`: an environment session; must have `reset()` and `step(action)`.
  In competition mode pass `arena=` instead and the runner uses `arena.make` and opens/closes the
  single competition scorecard around the whole run.
- `make_session(game_id, env) -> session` (optional; default: the env itself) builds what the agent
  plays on, e.g. arc3.session.GameSession.
- `agent_factory(game_id, progress) -> agent` where `agent.play_game(session, deadline)` plays until
  done or until `deadline` (epoch seconds, time.time()) and may return a summary object.
  The agent owns `progress` updates and gate acquire/release.

Keep-alive: the competition gateway ends the run after 15 minutes without an action, while the model
server needs ~10+ minutes to load. The first game is the "keeper": it RESETs immediately and every
`keepalive_seconds` until any game executes a non-RESET action or `model_ready()` reports true, then
plays normally. Actions are observed through a thin env proxy that watches `step` calls.
"""
from __future__ import annotations

import json
import logging
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Optional

from arc3.scheduler import GameProgress, PriorityGate

log = logging.getLogger("arc3.runner")

RUN_MARGIN_SECONDS = 15 * 60
KEEPALIVE_SECONDS = 300
JOIN_GRACE_SECONDS = 120


class _ActionWatch:
    """Env proxy: flags the shared event on the first non-RESET step, delegates everything else."""

    def __init__(self, env: Any, first_action: threading.Event):
        self._env = env
        self._first_action = first_action

    def step(self, action, *args, **kwargs):
        name = getattr(action, "name", action)
        result = self._env.step(action, *args, **kwargs)
        if str(name).upper() != "RESET":
            self._first_action.set()
        return result

    def __getattr__(self, item):
        return getattr(self._env, item)


def _json_default(obj):
    if hasattr(obj, "__dict__"):
        return {k: v for k, v in vars(obj).items() if not k.startswith("_")}
    return str(obj)


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=_json_default))
    tmp.replace(path)


class _Run:
    """Shared state of one run_games call."""

    def __init__(self, gate, log_dir, run_start, run_end, per_game_cap, model_ready,
                 keepalive_seconds, clock, make_session=None):
        self.gate = gate
        self.make_session = make_session
        self.log_dir = log_dir
        self.run_start = run_start
        self.run_end = run_end                  # already includes the safety margin
        self.per_game_cap = per_game_cap
        self.model_ready = model_ready
        self.keepalive_seconds = keepalive_seconds
        self.clock = clock
        self.first_action = threading.Event()
        self.stop = threading.Event()           # set when the run is over; ends the keep-alive loop
        self.keepalive_resets = 0
        self.summaries: dict[str, dict] = {}
        self.lock = threading.Lock()

    def deadline_for(self, game_start: float) -> float:
        return min(self.run_end, game_start + self.per_game_cap)

    def keep_alive(self, env, game_id: str) -> None:
        """Keeper loop: RESET now and every keepalive_seconds until the model takes over."""
        while True:
            if self.first_action.is_set() or self.stop.is_set() or self.clock() >= self.run_end:
                return
            try:
                env.reset()
                self.keepalive_resets += 1
                log.info("keep-alive RESET #%d on %s", self.keepalive_resets, game_id)
            except Exception:
                log.exception("keep-alive RESET failed on %s", game_id)
            next_reset = min(self.clock() + self.keepalive_seconds, self.run_end)
            while (left := next_reset - self.clock()) > 0:
                if self.stop.is_set() or self._model_up():
                    return
                if self.first_action.wait(min(5.0, left)):
                    return

    def _model_up(self) -> bool:
        if self.model_ready is None:
            return False
        try:
            return bool(self.model_ready())
        except Exception:
            return False

    def play_one(self, game_id, make_env, agent_factory, keeper: bool) -> None:
        progress = GameProgress(game_id=game_id)
        start = self.clock()
        deadline = self.deadline_for(start)
        summary: dict[str, Any] = {"game_id": game_id, "keeper": keeper, "start": start,
                                   "deadline": deadline, "status": "running"}
        env = session = None
        try:
            env = _ActionWatch(make_env(game_id), self.first_action)
            if keeper:
                self.keep_alive(env, game_id)
            session = self.make_session(game_id, env) if self.make_session else env
            agent = agent_factory(game_id, progress)
            result = agent.play_game(session, deadline)
            summary["status"] = "ok"
            summary["result"] = result
        except BaseException as exc:            # one game must never take the run down
            summary["status"] = "error"
            summary["error"] = f"{type(exc).__name__}: {exc}"
            summary["traceback"] = traceback.format_exc()
            log.error("game %s failed: %s", game_id, summary["error"])
        finally:
            self.gate.finish(progress)
            for obj in {id(o): o for o in (session, env) if o is not None}.values():
                close = getattr(obj, "close", None)
                if callable(close):
                    try:
                        close()
                    except Exception:
                        log.exception("close failed for %s", game_id)
            end = self.clock()
            summary.update({
                "end": end, "seconds": round(end - start, 1),
                "levels_completed": progress.level, "levels_total": progress.levels_total,
                "actions_on_last_level": progress.actions_on_level,
                "gate_grants": self.gate.grants.get(game_id, 0),
                "gate_wait_seconds": round(self.gate.wait_seconds.get(game_id, 0.0), 1),
            })
            with self.lock:
                self.summaries[game_id] = summary
            try:
                _write_json(self.log_dir / game_id / "summary.json", summary)
            except Exception:
                log.exception("could not write summary for %s", game_id)


def run_games(game_ids: list[str], make_env: Optional[Callable[[str], Any]],
              agent_factory: Callable[[str, GameProgress], Any], gate: PriorityGate,
              total_seconds: float, per_game_cap_seconds: float, log_dir,
              *, arena: Any = None, margin_seconds: float = RUN_MARGIN_SECONDS,
              keepalive_seconds: float = KEEPALIVE_SECONDS,
              model_ready: Optional[Callable[[], bool]] = None,
              abort: Optional[threading.Event] = None,
              make_session: Optional[Callable[[str, Any], Any]] = None,
              clock: Callable[[], float] = time.time) -> dict:
    """Play all games concurrently and return the run summary (also written to run_summary.json).

    `total_seconds` is the hard wall from now; games must end `margin_seconds` before it.
    Setting `abort` (e.g. the model server died) stops waiting for games: the scorecard is closed and
    the summary written; still-running game threads are daemons and are reported as hung.
    """
    if not game_ids:
        raise ValueError("run_games: empty game list")
    if make_env is None:
        if arena is None:
            raise ValueError("run_games: need make_env or arena")
        make_env = arena.make
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    run_start = clock()
    run_end = run_start + max(0.0, total_seconds - margin_seconds)
    run = _Run(gate, log_dir, run_start, run_end, per_game_cap_seconds, model_ready,
               keepalive_seconds, clock, make_session)
    gate.set_run_window(run_start, run_end)

    scorecard = None
    if arena is not None:
        scorecard = arena.open_scorecard()       # one scorecard for the whole competition run
    threads = []
    try:
        for i, game_id in enumerate(game_ids):
            t = threading.Thread(target=run.play_one, name=f"game-{game_id}", daemon=True,
                                 args=(game_id, make_env, agent_factory, i == 0))
            t.start()
            threads.append(t)
        # Join with a hard bound so a hung agent cannot keep the scorecard open past the wall.
        hard_stop = run_end + min(JOIN_GRACE_SECONDS, margin_seconds)
        for t in threads:
            while t.is_alive() and clock() < hard_stop and not (abort and abort.is_set()):
                t.join(min(5.0, max(0.0, hard_stop - clock())))
        if abort is not None and abort.is_set():
            log.error("run aborted")
    finally:
        run.stop.set()
        if arena is not None:
            try:
                arena.close_scorecard()
            except Exception:
                log.exception("closing the competition scorecard failed")

    hung = [t.name.removeprefix("game-") for t in threads if t.is_alive()]
    for game_id in hung:
        run.summaries.setdefault(game_id, {"game_id": game_id, "status": "timeout"})
    statuses = [s.get("status") for s in run.summaries.values()]
    summary = {
        "games": len(game_ids),
        "ok": statuses.count("ok"),
        "errors": statuses.count("error"),
        "hung": hung,
        "aborted": bool(abort is not None and abort.is_set()),
        "levels_completed": sum(s.get("levels_completed", 0) for s in run.summaries.values()),
        "run_start": run_start, "run_end": run_end, "finished": clock(),
        "keepalive_resets": run.keepalive_resets,
        "first_action_seen": run.first_action.is_set(),
        "scorecard": str(scorecard) if scorecard is not None else None,
        "per_game": {gid: {k: run.summaries.get(gid, {}).get(k)
                           for k in ("status", "levels_completed", "seconds", "error")}
                     for gid in game_ids},
    }
    _write_json(log_dir / "run_summary.json", summary)
    return summary
