"""wm inside the live loop: real Env (ls20) -> GameSession -> PythonTool, no model."""
from pathlib import Path

from arc3.env import Env, list_public_games
from arc3.sandbox import PythonTool
from arc3.session import GameSession
from arc3.types import Action, GameConfig

ENVS = Path(__file__).resolve().parents[1] / "data" / "environments"

SNIPPET = """
res = action(['UP', 'LEFT', 'DOWN', 'RIGHT'])
m = wm.induce_motion()
u = wm.untried_actions()
g = wm.transition_graph()
print('motion', sorted(m), sorted(m['actions']))
print('untried', sorted(u))
print('graph', g['n_states'] >= 1, 'n_noops' in g)
def transition(state, a):
    return dict(state)
def outcome(state):
    return 'ongoing'
wm.register(transition, outcome)
v = wm.verify()
print('verify', sorted(v), v['init']['ok'], v['dynamics']['graded'] >= 0)
print('summary', v['summary'])
"""


def _session(tmp_path):
    gid = next(g for g in list_public_games(ENVS) if g.startswith("ls20"))
    env = Env.offline(gid, ENVS, seed=0, recordings_dir=tmp_path / "toolkit")
    return GameSession(env, GameConfig(game_id=gid))


def test_wm_tools_in_sandbox(tmp_path):
    s = _session(tmp_path)
    tool = PythonTool(s)
    out = tool.run(SNIPPET)
    assert out.error is None, out.error
    assert "motion ['actions', 'avatar_candidates']" in out.stdout
    assert "untried ['ever', 'here']" in out.stdout
    assert "graph True True" in out.stdout
    assert "verify ['dynamics', 'init', 'ok', 'outcome', 'seconds', 'summary', 'timeout'] True True" \
        in out.stdout
    rec = tool.wm._rec()
    assert rec.level == 1 and len(rec.transitions) == len(out.actions) >= 1
    assert rec.available_actions == s.valid_actions

    # The registered model survives into the next call; execute stops where identity is wrong.
    out = tool.run("r = wm.execute(['UP', 'UP'])\nprint(sorted(r), r['executed'], r['mismatch_index'])\n"
                   "print(r['expected_diff'][:80] if r['expected_diff'] else None)")
    assert out.error is None, out.error
    assert "['executed', 'expected_diff', 'mismatch_index', 'observed_outcome', 'stop_reason']" \
        in out.stdout
    if any(r.board_changed for r in out.actions):
        assert " 1 0" in out.stdout and len(out.actions) == 1
    s.close()


def test_record_after_auto_reset_uses_live_grid(tmp_path):
    s = _session(tmp_path)
    s.step(Action("UP"))
    tool = PythonTool(s)
    rec = tool.wm._rec()
    assert (rec.current_grid == s.current.grid).all()
    assert rec.previous == [] and rec.transitions[0][1]["action"] == "UP"
    s.close()
