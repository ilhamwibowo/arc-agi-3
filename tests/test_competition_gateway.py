"""Competition-mode path against a real gateway: the toolkit's own HTTP server (competition_mode=True)
serving the public games, our Arena in competition mode as the client, the runner's keeper and
scorecard handling, and kaggle_main.wait_for_gateway. sub-001 (2026-10-28) scored 0.00 after an
early exit and the rerun gives no logs; this is the only way to exercise that path locally."""
import logging
import threading
import time
from pathlib import Path

import pytest

from arc3.runner import run_games
from arc3.scheduler import PriorityGate
from arc3.types import Action

ENV_DIR = Path(__file__).resolve().parents[1] / "data" / "environments"
pytestmark = pytest.mark.skipif(not ENV_DIR.is_dir(), reason="public games not present")


@pytest.fixture
def gateway(tmp_path):
    from arc_agi import Arcade, OperationMode
    from arc_agi.server import create_app
    from werkzeug.serving import make_server
    quiet = logging.getLogger("arc3-test-gateway")
    quiet.setLevel(logging.ERROR)
    logging.getLogger("werkzeug").setLevel(logging.ERROR)
    arcade = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=str(ENV_DIR),
                    recordings_dir=str(tmp_path / "server-rec"), logger=quiet)
    app, _ = create_app(arcade, competition_mode=True)
    seen = []

    @app.before_request
    def _record():
        from flask import request
        seen.append((request.method, request.path))

    server = make_server("127.0.0.1", 0, app)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/", seen
    finally:
        server.shutdown()
        thread.join(timeout=10)
        server.server_close()


def test_wait_for_gateway_returns_once_up(gateway):
    from arc3.kaggle_main import wait_for_gateway
    base, _ = gateway
    wait_for_gateway(base, timeout=30)


def test_wait_for_gateway_fails_loudly_when_down():
    from arc3.kaggle_main import wait_for_gateway
    with pytest.raises(RuntimeError):
        wait_for_gateway("http://127.0.0.1:9/", timeout=6)


def test_competition_arena_lists_games_and_plays_on_one_scorecard(gateway, tmp_path):
    from arc3.env import Arena
    from arc3.session import GameSession
    from arc3.types import GameConfig
    base, seen = gateway
    arena = Arena(mode="competition", base_url=base, api_key="test-key-123",
                  recordings_dir=tmp_path / "client-rec")
    ids = arena.game_ids()
    assert len(ids) == 25 and all("-" in g for g in ids), ids
    ids = ids[:3]
    gate = PriorityGate(max_active=2)
    model_up = threading.Event()

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

    # model "boots" 2 s in: the keeper must RESET on the open scorecard meanwhile
    threading.Timer(2.0, model_up.set).start()
    summary = run_games(ids, None, lambda g, p: Mover(p), gate, total_seconds=600,
                        per_game_cap_seconds=300, log_dir=tmp_path, arena=arena,
                        make_session=lambda g, env: GameSession(env, GameConfig(game_id=g)),
                        model_ready=model_up.is_set, keepalive_seconds=1)
    assert summary["ok"] == 3 and summary["errors"] == 0, summary
    paths = [p for _, p in seen]
    assert paths.count("/api/scorecard/open") == 1, paths
    assert paths.count("/api/scorecard/close") == 1, paths
    assert any(p.startswith("/api/cmd/RESET") or p.endswith("/RESET") or "reset" in p.lower() for p in paths), paths
    assert arena.scorecard_id is None            # closed by the runner


def test_main_competition_ignores_offline_smoke_cap(gateway, tmp_path, monkeypatch):
    """sub-001/sub-002 (2026-10-28/29) ended after ~33 min: the notebook's offline smoke cap
    (180 s per game) was passed through to competition mode, 110 games / 10 slots * 180 s.
    main() must replace any short cap with the full run when competition=True."""
    from arc3 import kaggle_main, runner
    base, _ = gateway
    monkeypatch.setenv("ARC_BASE_URL", base)
    monkeypatch.setenv("ARC_API_KEY", "test-key-123")
    captured = {}

    def fake_run_games(game_ids, make_env, agent_factory, gate, **kw):
        captured.update(kw)
        captured["n_games"] = len(game_ids)
        return {"ok": 0, "errors": 0, "hung": [], "levels_completed": 0}

    monkeypatch.setattr(kaggle_main, "run_games", fake_run_games)
    monkeypatch.setattr(kaggle_main, "cap_address_space", lambda: None)
    kaggle_main.main(competition=True, run_end_epoch=time.time() + 9 * 3600, log_dir=str(tmp_path),
                     llm_url="http://127.0.0.1:1/v1", model_name="x", offline_subset=1,
                     per_game_cap_seconds=180.0)
    assert captured["n_games"] == 25                      # offline_subset ignored
    assert captured["per_game_cap_seconds"] >= 8 * 3600   # smoke cap ignored


def test_main_retries_when_arcade_lists_zero_games(tmp_path, monkeypatch):
    """The toolkit swallows a failed /api/games and exposes zero games; main() must retry."""
    from arc3 import kaggle_main
    calls = {"n": 0}

    class FakeArena:
        def game_ids(self):
            calls["n"] += 1
            return [] if calls["n"] < 3 else ["aa00-1"]

    monkeypatch.setattr(kaggle_main, "build_components",
                        lambda *a, **kw: (FakeArena(), None, None))
    monkeypatch.setattr(kaggle_main, "cap_address_space", lambda: None)
    monkeypatch.setattr(kaggle_main.time, "sleep", lambda s: None)
    monkeypatch.setattr(kaggle_main, "run_games",
                        lambda ids, *a, **kw: {"ok": 0, "errors": 0, "hung": [], "levels_completed": 0,
                                               "ids": ids})
    monkeypatch.setattr(kaggle_main, "select_subset", lambda ids, s: ids)
    s = kaggle_main.main(competition=False, run_end_epoch=time.time() + 9 * 3600,
                         log_dir=str(tmp_path), llm_url="x", model_name="x", env_dir=str(ENV_DIR))
    assert calls["n"] == 3 and s["ids"] == ["aa00-1"]
