"""Official-style per-game scoring (RHAE) from per-level agent action counts.

Follows docs.arcprize.org/methodology as the lead specified: level_score = min((human/agent)^2, 1.15);
game score = sum(level_score_i * i) / sum(i over ALL levels), i 1-indexed, uncompleted levels = 0;
run score = mean over games. Assumption: automatic RESETs after game over count as actions (Tycho paper).
Baselines are read from data/environments/<game>/<hash>/metadata.json for offline analysis only;
they must never be shown to the model.
"""
import json
from pathlib import Path
from typing import Optional, Sequence

CAP = 1.15
ENV_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "environments"


def load_baselines(game: str, env_dir: Path = ENV_DIR) -> Optional[list[int]]:
    """Human baseline actions per level for `game` ("ls20" or "ls20-9607627b"), or None if unknown."""
    base = game.split("-")[0]
    for meta in sorted((Path(env_dir) / base).glob("*/metadata.json")):
        try:
            b = json.loads(meta.read_text()).get("baseline_actions")
        except (OSError, ValueError):
            continue
        if b:
            return [int(x) for x in b]
    return None


def level_score(human: float, agent: float) -> float:
    return min((human / agent) ** 2, CAP) if agent > 0 else 0.0


def game_score(baselines: Sequence[int], agent_actions: Sequence[int]) -> float:
    """agent_actions[i] = actions spent on level i+1 (only completed levels are listed)."""
    total = sum(range(1, len(baselines) + 1))
    got = sum(level_score(h, a) * (i + 1) for i, (h, a) in enumerate(zip(baselines, agent_actions)))
    return got / total if total else 0.0
