"""Env adapter tests against the public games (offline) plus a fake wrapper for engine quirks."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from arc3.env import Env, list_public_games
from arc3.types import MODEL_ACTIONS, Action, GameState

ROOT = Path(__file__).resolve().parents[1]
ENVS = ROOT / "data" / "environments"
GAMES = {"ft09": "ft09-0d8bbf25", "ls20": "ls20-9607627b", "tu93": "tu93-0768757b"}


@pytest.fixture
def make_env(tmp_path):
    def _make(key):
        return Env.offline(GAMES[key], ENVS, seed=0, recordings_dir=tmp_path / "rec")
    return _make


def check_obs(obs, prev_step=None):
    assert obs.frames and obs.grid is obs.frames[-1]
    for f in obs.frames:
        assert f.shape == (64, 64) and f.dtype == np.uint8 and f.max() <= 15
    for a, b in zip(obs.frames, obs.frames[1:]):
        assert not np.array_equal(a, b), "adjacent duplicate frames must be collapsed"
    assert set(obs.available_actions) <= set(MODEL_ACTIONS)
    assert obs.win_levels > 0
    assert 0 <= obs.levels_completed <= obs.win_levels


def test_list_public_games():
    games = list_public_games(ENVS)
    assert len(games) == 25 and set(GAMES.values()) <= set(games)


@pytest.mark.parametrize("key", list(GAMES))
def test_reset_and_steps(make_env, key):
    env = make_env(key)
    obs = env.reset()
    check_obs(obs)
    assert obs.state == GameState.NOT_FINISHED and obs.levels_completed == 0 and obs.step == 0
    initial = obs.grid.copy()
    acts = [a for a in obs.available_actions if a not in ("RESET", "UNDO")]
    for i in range(4):
        name = acts[i % len(acts)]
        a = Action(name, row=32, col=32) if name == "MOUSE" else Action(name)
        obs = env.step(a)
        check_obs(obs)
        assert obs.step == i + 1
    # RESET mid-level restores the level start without touching levels_completed.
    obs = env.reset()
    assert np.array_equal(obs.grid, initial) and obs.levels_completed == 0
    rec = env.log_dict(Action("UP"), obs)
    assert "raw" not in rec and isinstance(rec["grid"], str)


def test_available_actions_names(make_env):
    assert make_env("ft09").available_actions == ["MOUSE"]
    assert make_env("ls20").available_actions == ["UP", "DOWN", "LEFT", "RIGHT"]


def test_mouse_x_is_col_y_is_row(make_env):
    """Clicks change cells around (row, col), never around the transposed point."""
    env = make_env("ft09")
    hits_rc = hits_cr = 0
    for r in range(2, 62, 10):
        for c in range(3, 62, 9):
            before = env.reset().grid.copy()
            after = env.step(Action("MOUSE", row=r, col=c)).grid
            d = np.argwhere(before != after)
            if not len(d):
                continue
            near_rc = np.any((abs(d[:, 0] - r) <= 1) & (abs(d[:, 1] - c) <= 1))
            near_cr = np.any((abs(d[:, 0] - c) <= 1) & (abs(d[:, 1] - r) <= 1))
            hits_rc += bool(near_rc and not near_cr)
            hits_cr += bool(near_cr and not near_rc)
    assert hits_rc >= 2 and hits_cr == 0


def test_animation_frames_kept(make_env):
    """tu93 moves animate over several engine frames; all of them reach the Observation."""
    env = make_env("tu93")
    counts = [len(env.step(Action(a)).frames) for a in ["UP", "DOWN", "LEFT", "RIGHT"] * 3]
    assert max(counts) > 1


def test_game_over_then_action_keeps_board(make_env):
    import random
    env = make_env("ls20")
    rnd = random.Random(1)
    obs = env.observation
    for _ in range(600):
        obs = env.step(Action(rnd.choice(["UP", "DOWN", "LEFT", "RIGHT"])))
        if obs.state != GameState.NOT_FINISHED:
            break
    assert obs.state == GameState.GAME_OVER
    after = env.step(Action("UP"))
    assert np.array_equal(after.grid, obs.grid) and after.win_levels == obs.win_levels
    assert env.reset().state == GameState.NOT_FINISHED


# ---- fake wrapper: engine quirks that are slow to reach with real games

class FakeWrapper:
    def __init__(self):
        self.calls = []
        self.levels = 1
        self.observation_space = self._resp([np.zeros((64, 64), np.int8)])

    def _resp(self, frames, state="NOT_FINISHED", levels=None):
        return SimpleNamespace(frame=frames, state=GameState(state), win_levels=3,
                               levels_completed=self.levels if levels is None else levels,
                               available_actions=[1, 2, 6, 7])

    def step(self, action, data=None):
        self.calls.append((action.name, data))
        g = np.zeros((64, 64), np.int8)
        g[0, len(self.calls)] = 1
        if action.name == "ACTION2":
            return self._resp([], state="GAME_OVER", levels=0)
        return self._resp([g, g.copy(), g.copy()])


def test_fake_quirks():
    w = FakeWrapper()
    env = Env(w, "fake-0")
    assert env.available_actions == ["UP", "DOWN", "MOUSE", "UNDO"]
    # Fresh game with completed levels and no action yet: RESET would be a full reset.
    obs = env.reset()
    assert w.calls == [] and obs.levels_completed == 1
    obs = env.step(Action("MOUSE", row=5, col=9))
    assert w.calls[-1] == ("ACTION6", {"x": 9, "y": 5})
    assert len(obs.frames) == 1  # three identical frames collapse to one
    prev = obs.grid
    obs = env.step(Action("DOWN"))  # empty frame list => previous board, previous levels kept
    assert np.array_equal(obs.grid, prev) and obs.levels_completed == 1
    env.reset()
    assert w.calls[-1][0] == "RESET"
    env.reset()  # second RESET in a row is refused
    assert [c[0] for c in w.calls].count("RESET") == 1


def test_arena_threads(tmp_path):
    import os
    from concurrent.futures import ThreadPoolExecutor
    from arc3.env import Arena
    arena = Arena(mode="offline", environments_dir=ENVS, recordings_dir=tmp_path / "rec")
    assert os.environ["ONLY_RESET_LEVELS"] == "true" and len(arena.game_ids()) == 25

    def play(game):
        env = arena.make(game)
        acts = env.available_actions
        for _ in range(5):
            name = acts[0]
            env.step(Action(name, row=10, col=10) if name == "MOUSE" else Action(name))
        return env.observation.step

    with ThreadPoolExecutor(8) as pool:
        steps = list(pool.map(play, arena.game_ids()[:8]))
    assert steps == [5] * 8 and arena.scorecard_id is not None
    assert arena.close_scorecard() is not None and arena.scorecard_id is None


# ---- gateway hiccups: the toolkit returns None on any HTTP failure; a 9 h run must ride them out

class FlakyWrapper(FakeWrapper):
    def __init__(self, fail_first):
        super().__init__()
        self.fail_first = fail_first

    def step(self, action, data=None):
        if self.fail_first > 0:
            self.fail_first -= 1
            self.calls.append((action.name, "FAILED"))
            return None
        return super().step(action, data)


def test_step_retries_toolkit_failures(monkeypatch):
    env = Env(FlakyWrapper(fail_first=2), "fake-0")
    env.step_retry_sleep = 0.0
    obs = env.step(Action("UP"))
    assert obs.state == GameState.NOT_FINISHED
    assert [c[1] for c in env._wrapper.calls] == ["FAILED", "FAILED", {}]
    assert env.steps == 1                       # counted once, not per attempt


def test_step_gives_up_after_attempts():
    env = Env(FlakyWrapper(fail_first=99), "fake-0")
    env.step_retry_sleep = 0.0
    env.step_attempts = 3
    with pytest.raises(RuntimeError, match="after 3 attempts"):
        env.step(Action("UP"))


def test_arena_make_retries_none(tmp_path, monkeypatch):
    from arc3.env import Arena
    arena = Arena.offline(Path("data/environments"), recordings_dir=tmp_path)
    real = arena.arcade.make
    calls = {"n": 0}

    def flaky(*a, **kw):
        calls["n"] += 1
        return None if calls["n"] < 3 else real(*a, **kw)
    monkeypatch.setattr(arena.arcade, "make", flaky)
    env = arena.make(arena.game_ids()[0], retry_sleep=0.0)
    assert calls["n"] == 3 and env.reset().state == GameState.NOT_FINISHED
    monkeypatch.setattr(arena.arcade, "make", lambda *a, **kw: None)
    with pytest.raises(RuntimeError, match="after 2 attempts"):
        arena.make(arena.game_ids()[0], attempts=2, retry_sleep=0.0)
