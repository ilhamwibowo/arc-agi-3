"""run_games with fake envs/agents (no model), plus one pass over real offline games."""
import json
import threading
import time
from pathlib import Path

import pytest

from arc3.runner import run_games
from arc3.scheduler import PriorityGate
from arc3.types import Action

ENV_DIR = Path(__file__).resolve().parents[1] / "data" / "environments"


class FakeEnv:
    def __init__(self, game_id, log):
        self.game_id = game_id
        self.log = log
        self.closed = False

    def reset(self):
        self.log.append((self.game_id, "RESET", time.time()))

    def step(self, action):
        self.log.append((self.game_id, action.name, time.time()))

    def close(self):
        self.closed = True


class FakeAgent:
    def __init__(self, game_id, progress, gate, actions=3, fail=False, wait_for=None):
        self.game_id, self.progress, self.gate = game_id, progress, gate
        self.actions, self.fail, self.wait_for = actions, fail, wait_for

    def play_game(self, env, deadline):
        if self.wait_for is not None:
            self.wait_for.wait(5)
        if self.fail:
            raise ValueError("boom")
        self.gate.acquire(self.progress)
        for _ in range(self.actions):
            env.step(Action("UP"))
            self.progress.actions_on_level += 1
        self.progress.level_completed(1)
        self.gate.release(self.progress)
        return {"deadline": deadline}


class FakeArena:
    def __init__(self, log):
        self.log = log
        self.calls = []

    def open_scorecard(self):
        self.calls.append("open")
        return "card-1"

    def make(self, game_id):
        self.calls.append(f"make:{game_id}")
        return FakeEnv(game_id, self.log)

    def close_scorecard(self):
        self.calls.append("close")


def test_all_games_run_and_one_failure_is_contained(tmp_path):
    log = []
    envs = {}

    def make_env(gid):
        envs[gid] = FakeEnv(gid, log)
        return envs[gid]
    gate = PriorityGate(max_active=2)

    def factory(gid, progress):
        return FakeAgent(gid, progress, gate, fail=(gid == "g2"))
    ids = [f"g{i}" for i in range(6)]
    summary = run_games(ids, make_env, factory, gate, total_seconds=3600, per_game_cap_seconds=600,
                        log_dir=tmp_path, model_ready=lambda: True)
    assert summary["ok"] == 5 and summary["errors"] == 1 and summary["hung"] == []
    assert summary["per_game"]["g2"]["status"] == "error" and "boom" in summary["per_game"]["g2"]["error"]
    assert all(e.closed for e in envs.values())
    per_game = json.loads((tmp_path / "g0" / "summary.json").read_text())
    assert per_game["status"] == "ok" and per_game["levels_completed"] == 1
    assert json.loads((tmp_path / "run_summary.json").read_text())["games"] == 6
    assert summary["first_action_seen"]


def test_deadline_is_min_of_run_end_minus_margin_and_cap(tmp_path):
    now = time.time()
    gate = PriorityGate()
    results = {}

    class Recorder:
        def __init__(self, gid):
            self.gid = gid

        def play_game(self, env, deadline):
            results[self.gid] = deadline
            env.step(Action("UP"))
    run_games(["a"], lambda g: FakeEnv(g, []), lambda g, p: Recorder(g), gate,
              total_seconds=3600, per_game_cap_seconds=60, log_dir=tmp_path, model_ready=lambda: True)
    assert results["a"] == pytest.approx(now + 60, abs=5)
    run_games(["b"], lambda g: FakeEnv(g, []), lambda g, p: Recorder(g), gate,
              total_seconds=1000, per_game_cap_seconds=10_000, log_dir=tmp_path, margin_seconds=900,
              model_ready=lambda: True)
    assert results["b"] == pytest.approx(now + 100, abs=5)


def test_competition_uses_one_scorecard_around_all_games(tmp_path):
    log = []
    arena = FakeArena(log)
    gate = PriorityGate()
    summary = run_games(["x", "y", "z"], None, lambda g, p: FakeAgent(g, p, gate), gate,
                        total_seconds=3600, per_game_cap_seconds=600, log_dir=tmp_path, arena=arena,
                        model_ready=lambda: True)
    assert arena.calls[0] == "open" and arena.calls[-1] == "close"
    assert arena.calls.count("open") == 1 and arena.calls.count("close") == 1
    assert summary["scorecard"] == "card-1"


def test_scorecard_closed_even_if_every_game_fails(tmp_path):
    arena = FakeArena([])
    gate = PriorityGate()
    summary = run_games(["x", "y"], None, lambda g, p: FakeAgent(g, p, gate, fail=True), gate,
                        total_seconds=3600, per_game_cap_seconds=600, log_dir=tmp_path, arena=arena,
                        model_ready=lambda: True)
    assert summary["errors"] == 2 and arena.calls[-1] == "close"


def test_keeper_resets_until_first_model_action(tmp_path):
    log = []
    gate = PriorityGate()
    go = threading.Event()

    def factory(gid, progress):
        return FakeAgent(gid, progress, gate, wait_for=go)
    timer = threading.Timer(0.35, go.set)
    timer.start()
    summary = run_games(["keeper", "other"], lambda g: FakeEnv(g, log), factory, gate,
                        total_seconds=3600, per_game_cap_seconds=600, log_dir=tmp_path,
                        keepalive_seconds=0.1)
    resets = [e for e in log if e[1] == "RESET"]
    assert all(e[0] == "keeper" for e in resets)
    assert summary["keepalive_resets"] >= 2
    first_up = min(e[2] for e in log if e[1] == "UP")
    assert max(e[2] for e in resets) <= first_up + 0.01   # no keep-alive after the model acted


def test_keeper_stops_when_model_ready(tmp_path):
    log = []
    gate = PriorityGate()
    summary = run_games(["solo"], lambda g: FakeEnv(g, log), lambda g, p: FakeAgent(g, p, gate), gate,
                        total_seconds=3600, per_game_cap_seconds=600, log_dir=tmp_path,
                        keepalive_seconds=60, model_ready=lambda: True)
    assert summary["keepalive_resets"] == 1 and summary["ok"] == 1


def test_abort_returns_without_waiting_for_hung_games(tmp_path):
    gate = PriorityGate()
    abort = threading.Event()
    never = threading.Event()

    class Hung:
        def play_game(self, env, deadline):
            never.wait(30)
    threading.Timer(0.2, abort.set).start()
    t0 = time.time()
    arena = FakeArena([])
    summary = run_games(["h"], None, lambda g, p: Hung(), gate, total_seconds=3600,
                        per_game_cap_seconds=600, log_dir=tmp_path, arena=arena, abort=abort,
                        model_ready=lambda: True)
    never.set()
    assert time.time() - t0 < 10
    assert summary["aborted"] and summary["hung"] == ["h"] and arena.calls[-1] == "close"


@pytest.mark.skipif(not ENV_DIR.is_dir(), reason="public games not present")
def test_real_offline_games_with_session(tmp_path):
    from arc3.env import Arena
    from arc3.session import GameSession
    from arc3.types import GameConfig
    arena = Arena("offline", environments_dir=ENV_DIR, recordings_dir=tmp_path / "rec")
    ids = arena.game_ids()[:3]
    gate = PriorityGate(max_active=2)

    class Mover:
        def __init__(self, progress):
            self.progress = progress

        def play_game(self, session, deadline):
            gate.acquire(self.progress)
            for name in session.valid_actions[:2]:
                if name != "MOUSE":
                    session.step(Action(name))
            gate.release(self.progress)
            return {"levels": session.current.levels_completed}

    summary = run_games(ids, None, lambda g, p: Mover(p), gate, total_seconds=600,
                        per_game_cap_seconds=300, log_dir=tmp_path, arena=arena,
                        make_session=lambda g, env: GameSession(env, GameConfig(game_id=g)),
                        model_ready=lambda: True)
    assert summary["ok"] == 3, summary
