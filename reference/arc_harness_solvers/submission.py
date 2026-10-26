"""Sequential public-observation dispatch through the official ARC toolkit.

The gateway owns competition scoring and submission generation. This module writes
diagnostics only; it never creates or edits a competition submission file.
"""
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
import traceback

from arc_agi import Arcade, OperationMode
from arcengine import GameAction

from .agents import AGENTS
from .core import Observation, validate_action


@dataclass(frozen=True)
class Budget:
    max_actions: int = 2048
    per_game_seconds: float = 300
    total_seconds: float = 8 * 3600
    cleanup_seconds: float = 120

    def __post_init__(self):
        if (self.max_actions < 1 or self.per_game_seconds <= 0 or
                self.cleanup_seconds < 0 or self.total_seconds <= self.cleanup_seconds):
            raise ValueError("Budgets must leave positive execution time")


def _save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def dispatch(arc, output, *, policy="balanced", seed=0, budget=Budget(),
             clock=time.monotonic, policy_factory=None, started=None):
    """Use one card and at most one make per game; never expose arc to a policy.

    Deadlines are checked between toolkit operations and before executing actions.
    They cannot interrupt a blocked toolkit call. Caller must reserve cleanup time.
    Game IDs label artifacts and toolkit calls only, never policy construction/input.
    """
    start = clock() if started is None else started
    deadline = start + budget.total_seconds - budget.cleanup_seconds
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    factory = policy_factory or AGENTS[policy]
    games = sorted({item.game_id for item in arc.get_environments()})
    result = {"policy": policy, "seed": seed, "budget": asdict(budget),
              "games": [], "status": "running", "submission_generated_by": "gateway",
              "action_accounting": "explicit step calls; toolkit initialization is separate"}
    card = None
    try:
        if not games:
            raise RuntimeError("Toolkit returned no games")
        card = arc.open_scorecard(tags=["arc-harness", policy])
        if not card:
            raise RuntimeError("Toolkit did not open a scorecard")
        for index, game in enumerate(games):
            row = {"game": game, "actions": 0, "levels": 0, "resets": 0,
                   "status": "running", "level_completions": []}
            result["games"].append(row)
            if clock() >= deadline:
                row["status"] = "global_time_budget"
                continue
            game_start = clock()
            # Share remaining time across unstarted games, retaining cleanup reserve.
            game_deadline = min(deadline, game_start + budget.per_game_seconds,
                                game_start + (deadline-game_start)/(len(games)-index))
            game_out = out / f"game-{index:03d}"
            game_out.mkdir()
            try:
                env = arc.make(game, seed=seed, scorecard_id=card)
                if env is None:
                    raise RuntimeError("Toolkit failed to initialize environment")
                obs = Observation.from_raw(env.observation_space)
                agent = factory(seed)
                if hasattr(agent, "deadline"):
                    agent.deadline = game_deadline
                row.update(levels=obs.levels, total_levels=obs.total_levels, final_state=obs.state)
                with (game_out / "trace.jsonl").open("w", buffering=1) as trace:
                    def event(value):
                        trace.write(json.dumps(value, separators=(",", ":")) + "\n")
                    event({"type": "initial", "observation": obs.record()})
                    for step in range(1, budget.max_actions + 1):
                        if obs.state == "WIN":
                            row["status"] = "win"
                            break
                        if clock() >= game_deadline:
                            row["status"] = "time_budget"
                            break
                        action, decision = agent.choose(obs)
                        if clock() >= game_deadline:
                            row["status"] = "time_budget"
                            break
                        validate_action(action, obs)
                        event({"type": "action", "step": step, "action": action.record(),
                               "decision": decision})
                        name = "RESET" if action.id == 0 else f"ACTION{action.id}"
                        before = obs
                        obs = Observation.from_raw(env.step(GameAction[name], data=action.data()))
                        event({"type": "transition", "step": step, "observation": obs.record()})
                        row["actions"] += 1
                        row["resets"] += int(action.id == 0)
                        row.update(levels=obs.levels, final_state=obs.state)
                        if obs.levels < before.levels:
                            raise RuntimeError("Completed level progress regressed")
                        if obs.levels > before.levels:
                            row["level_completions"].append({"levels": obs.levels, "action": step})
                        agent.observe(before, action, obs)
                        if getattr(agent, "latest_feedback", None) is not None:
                            event({"type": "feedback", "step": step,
                                   "measurements": agent.latest_feedback})
                    else:
                        row["status"] = "win" if obs.state == "WIN" else "action_budget"
                if hasattr(agent, "diagnostics"):
                    row["agent_diagnostics"] = agent.diagnostics()
            except Exception:
                row.update(status="error", error=traceback.format_exc())
            finally:
                row["elapsed_seconds"] = clock() - game_start
                _save(game_out / "summary.json", row)
                _save(out / "summary.json", result)
        result["status"] = ("error" if any(r["status"] == "error" for r in result["games"])
                            else "global_time_budget" if any(r["status"] == "global_time_budget"
                                                             for r in result["games"])
                            else "completed")
    except Exception:
        result.update(status="error", error=traceback.format_exc())
    finally:
        if card is not None:
            try:
                closed = arc.close_scorecard(card)
                if closed is None:
                    raise RuntimeError("Toolkit did not close scorecard")
                value = closed.model_dump(mode="json")
                value.pop("api_key", None)
                _save(out / "scorecard.json", value)
            except Exception:
                result.update(status="error", scorecard_error=traceback.format_exc())
        result["elapsed_seconds"] = clock() - start
        _save(out / "summary.json", result)
    return result


def run_gateway(output, *, policy="balanced", seed=0, budget=Budget()):
    """Kaggle entry point. Gateway service credentials follow the official starter."""
    started = time.monotonic()
    arc = Arcade(arc_api_key="test-key-123", arc_base_url="http://gateway:8001",
                 operation_mode=OperationMode.COMPETITION,
                 environments_dir="/tmp/arc-harness-no-environments",
                 recordings_dir=str(Path(output).parent / "toolkit-recordings"))
    return dispatch(arc, output, policy=policy, seed=seed, budget=budget, started=started)
