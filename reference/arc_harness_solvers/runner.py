import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time
import traceback

from .adapter import LocalAdapter
from .agents import AGENTS
from .core import validate_action
from .compact_llm import CompactAgent, CompactClient
from .transition_llm import TransitionAgent, TransitionClient
from .llm import LLMAgent, LocalClient, ModelBudgetExhausted
from .advisor import AdvisedController, AdvisorClient
from .program_agent import ProgramAgent, ProgramClient


def dump(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def provenance():
    root = Path(__file__).resolve().parents[2]
    files = sorted((root / "src").rglob("*.py")) + [root / "pyproject.toml", root / "uv.lock"]
    return {"python": platform.python_version(),
            "packages": {p: importlib.metadata.version(p) for p in ("arc-agi", "arcengine")},
            "source_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in files if p.exists()}}


def run(config, adapter_factory=LocalAdapter):
    out = Path(config["output"])
    out.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    summary = {**config, "status": "running", "actions": 0, "resets": 0,
               "levels": 0, "unchanged_frames": 0, "repeated_observations": 0,
               "policy_seconds": 0.0, "environment_seconds": 0.0,
               "level_completions": [], "provenance": provenance()}
    dump(out / "config.json", config)
    adapter = agent = None
    try:
        adapter = adapter_factory(config["game"], config["environment_seed"],
                                  config["environments"], out)
        if config["agent"] == "program_llm":
            agent = ProgramAgent(config["agent_seed"], ProgramClient(config["llm"], out, config["agent_seed"]), config["max_actions"])
            agent.deadline = start + config["max_seconds"]
        elif config["agent"] == "llm":
            agent_class, client_class = (CompactAgent, CompactClient) if config["llm"].get("interface") == "compact" else (LLMAgent, LocalClient)
            if config["llm"].get("interface") == "transitions":
                agent_class, client_class = TransitionAgent, TransitionClient
            if config["llm"].get("interface") == "advisor":
                agent_class, client_class = AdvisedController, AdvisorClient
            agent = agent_class(config["agent_seed"], client_class(config["llm"], out, config["agent_seed"]), config["max_actions"])
            agent.deadline = start + config["max_seconds"]
            if isinstance(agent, AdvisedController):
                agent.advice_time_limit = min(90, config["max_seconds"] / 2)
        else:
            agent = AGENTS[config["agent"]](config["agent_seed"])
        obs = adapter.initial
        seen = {obs.key()}
        summary["levels"] = obs.levels
        summary["total_levels"] = obs.total_levels
        with (out / "trace.jsonl").open("w", buffering=1) as trace:
            def event(data):
                trace.write(json.dumps(data, separators=(",", ":")) + "\n")
            event({"type": "initial", "observation": obs.record()})
            for step in range(config["max_actions"]):
                if obs.state == "WIN":
                    summary["status"] = "win"
                    break
                if time.monotonic() - start >= config["max_seconds"]:
                    summary["status"] = "time_budget"
                    break
                before = obs
                t = time.monotonic()
                try:
                    action, decision = agent.choose(obs)
                except ModelBudgetExhausted as exc:
                    summary["status"] = "model_budget"
                    summary["budget_reason"] = str(exc)
                    break
                except TimeoutError:
                    if time.monotonic() - start < config["max_seconds"]:
                        raise
                    summary["status"] = "time_budget"
                    summary["budget_reason"] = "Pending model call exhausted remaining wall time"
                    break
                finally:
                    summary["policy_seconds"] += time.monotonic() - t
                if time.monotonic() - start >= config["max_seconds"]:
                    summary["status"] = "time_budget"
                    break
                policy_time = time.monotonic() - t
                validate_action(action, obs)
                event({"type": "action", "step": step + 1, "action": action.record(),
                       "decision": decision, "policy_seconds": policy_time})
                t = time.monotonic()
                obs = adapter.step(action)
                env_time = time.monotonic() - t
                if obs.levels < before.levels:
                    raise RuntimeError("Reset lost completed levels")
                unchanged = obs.grid == before.grid
                repeated = obs.key() in seen
                event({"type": "transition", "step": step + 1,
                       "observation": obs.record(), "environment_seconds": env_time,
                       "unchanged_frame": unchanged, "repeated_observation": repeated})
                t = time.monotonic()
                agent.observe(before, action, obs)
                if getattr(agent, "latest_feedback", None) is not None:
                    event({"type": "feedback", "step": step + 1, "measurements": agent.latest_feedback})
                summary["policy_seconds"] += time.monotonic() - t
                seen.add(obs.key())
                summary["actions"] += 1
                summary["resets"] += int(action.id == 0)
                summary["unchanged_frames"] += int(unchanged)
                summary["repeated_observations"] += int(repeated)
                summary["environment_seconds"] += env_time
                if obs.levels > before.levels:
                    summary["level_completions"].append({"levels": obs.levels, "action": step + 1})
                summary["levels"] = obs.levels
                summary["final_state"] = obs.state
                if step % 25 == 0:
                    dump(out / "summary.json", summary)
            else:
                summary["status"] = "win" if obs.state == "WIN" else "action_budget"
        summary["unique_observations"] = len(seen)
        if hasattr(agent, "diagnostics"):
            summary["agent_diagnostics"] = agent.diagnostics()
    except Exception:
        summary["status"] = "error"
        summary["error"] = traceback.format_exc()
    finally:
        if agent is not None and hasattr(agent, "close"):
            agent.close()
        if agent is not None and hasattr(agent, "diagnostics"):
            summary["agent_diagnostics"] = agent.diagnostics()
        if adapter is not None:
            try:
                card = adapter.close()
                dump(out / "scorecard.json", card)
                summary["toolkit_score"] = card.get("score")
            except Exception:
                summary["scorecard_error"] = traceback.format_exc()
                summary["status"] = "error"
        summary["elapsed_seconds"] = time.monotonic() - start
        dump(out / "summary.json", summary)
    return summary
