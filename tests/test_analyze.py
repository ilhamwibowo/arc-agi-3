"""Tests for scripts/analyze_run.py on the smoke run and a synthetic trace."""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("analyze_run", ROOT / "scripts" / "analyze_run.py")
ar = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ar)


def write(path, events):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(e) for e in events) + "\n{truncated")


def test_synthetic(tmp_path):
    act = lambda step, lv, st="NOT_FINISHED", auto=False: {"event": "action", "action": "UP", "automatic": auto,
                                                           "step": step, "levels": lv, "state": st}
    write(tmp_path / "g1" / "trace.jsonl", [
        {"event": "start", "step": 0, "levels": 0, "win_levels": 3},
        {"kind": "model_call", "seconds": 2.0, "usage": {"prompt_tokens": 100, "completion_tokens": 10,
                                                        "cached_tokens": 50}},
        {"kind": "tool_call", "code": "x", "error": "boom", "actions": [], "stops": []},
        act(1, 0), act(2, 1), act(3, 1, "GAME_OVER"), act(4, 1, auto=True), act(5, 2)])
    (tmp_path / "empty").mkdir()
    (tmp_path / "empty" / "summary.json").write_text(json.dumps({"result": {"levels_completed": 1}}))
    rep = ar.analyze_run(tmp_path)
    g = next(x for x in rep["games"] if x["game"] == "g1")
    assert (g["levels_completed"], g["win_levels"], g["scored_actions"], g["resets"]) == (2, 3, 4, 1)
    assert g["level_actions"] == [2, 4] and g["actions_per_level"] == [2, 2]
    assert g["tool_errors"] == 1 and g["cached_fraction"] == 0.5
    assert "no game_end" in g["end_reason"]
    assert rep["total"]["levels_completed"] == 3
    assert "TOTAL" in ar.text_table(rep)


def test_smoke_run(tmp_path):
    run = ROOT / "runs/local/smoke-ollama-1"
    if not run.exists():
        return
    out_h, out_j = tmp_path / "o.html", tmp_path / "o.json"
    assert ar.main([str(run), "--html", str(out_h), "--json", str(out_j)]) == 0
    g = json.loads(out_j.read_text())["games"][0]
    assert g["scored_actions"] == 40 and g["model_calls"] == 19 and g["win_levels"] == 7
    assert g["end_reason"] == "action_cap"
    assert "<details" in out_h.read_text()
