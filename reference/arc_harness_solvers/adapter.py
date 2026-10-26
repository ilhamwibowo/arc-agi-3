import logging
import os
from pathlib import Path
from arc_agi import Arcade, OperationMode
from arcengine import GameAction
from .core import Observation


class LocalAdapter:
    """Only this adapter holds the official environment; agents see observations."""
    def __init__(self, game, seed, environments, output):
        if os.getenv("OPERATION_MODE", "").lower() == "competition":
            raise RuntimeError("Local evaluation cannot run in competition mode")
        self.arc = Arcade(operation_mode=OperationMode.OFFLINE,
                          environments_dir=str(Path(environments).resolve()),
                          recordings_dir=str(Path(output).resolve() / "toolkit"),
                          logger=logging.getLogger("arc_harness.toolkit"))
        self.env = self.arc.make(game, seed=seed)
        if self.env is None:
            raise RuntimeError(f"Cannot load {game}; fetch public environments first")
        self.initial = Observation.from_raw(self.env.observation_space)

    def step(self, action):
        name = "RESET" if action.id == 0 else f"ACTION{action.id}"
        obs = Observation.from_raw(self.env.step(GameAction[name], data=action.data()))
        if obs.levels < self.initial.levels:
            raise RuntimeError("Level progress regressed")
        return obs

    def close(self):
        card = self.arc.close_scorecard()
        if card is None:
            raise RuntimeError("Toolkit did not produce a scorecard")
        result = card.model_dump(mode="json")
        result.pop("api_key", None)
        return result


def fetch_public(directory):
    arc = Arcade(environments_dir=str(Path(directory).resolve()),
                 recordings_dir=str(Path(directory).resolve().parent / "fetch-scorecards"),
                 logger=logging.getLogger("arc_harness.toolkit"))
    if arc.operation_mode != OperationMode.NORMAL:
        raise RuntimeError("Fetch requires normal toolkit mode")
    games = sorted(e.game_id for e in arc.get_environments())
    for game in games:
        if arc.make(game) is None:
            raise RuntimeError(f"Failed to download {game}")
        print(f"Fetched {game}", flush=True)
    arc.close_scorecard()
    return games
