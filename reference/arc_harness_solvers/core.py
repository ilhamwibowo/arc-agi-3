from dataclasses import dataclass
import hashlib
import json


@dataclass(frozen=True)
class Action:
    id: int
    x: int | None = None
    y: int | None = None

    def data(self):
        return {"x": self.x, "y": self.y} if self.id == 6 else {}

    def record(self):
        return {"id": self.id, "data": self.data()}


@dataclass(frozen=True)
class Observation:
    frames: tuple
    state: str
    levels: int
    total_levels: int
    actions: tuple[int, ...]

    @classmethod
    def from_raw(cls, raw):
        if raw is None:
            raise RuntimeError("Environment returned no observation")
        frames = tuple(tuple(tuple(int(v) for v in row) for row in frame)
                       for frame in raw.frame)
        if not frames or any(not f or not f[0] for f in frames):
            raise ValueError("Empty frame")
        for frame in frames:
            w = len(frame[0])
            if len(frame) > 64 or w > 64 or any(len(row) != w for row in frame):
                raise ValueError("Invalid frame dimensions")
            if any(v < 0 or v > 15 for row in frame for v in row):
                raise ValueError("Invalid palette value")
        return cls(frames, raw.state.value, raw.levels_completed,
                   raw.win_levels, tuple(sorted(set(raw.available_actions))))

    @property
    def grid(self):
        return self.frames[-1]

    def key(self):
        # Observation identity is not hidden-state identity.
        return hashlib.sha256(json.dumps(
            [self.grid, self.state, self.levels, self.actions],
            separators=(",", ":")).encode()).hexdigest()

    def record(self):
        return {"frames": self.frames, "state": self.state,
                "levels": self.levels, "total_levels": self.total_levels,
                "actions": self.actions}

    @classmethod
    def from_record(cls, d):
        return cls(tuple(tuple(tuple(row) for row in f) for f in d["frames"]),
                   d["state"], d["levels"], d["total_levels"], tuple(d["actions"]))


def validate_action(action, obs):
    if action.id == 0:
        if obs.state not in ("GAME_OVER", "NOT_PLAYED"):
            raise ValueError("Automatic reset is only permitted after game over")
    elif obs.state != "NOT_FINISHED" or action.id not in obs.actions:
        raise ValueError("Unavailable action")
    if action.id == 6 and not (isinstance(action.x, int) and isinstance(action.y, int)
                              and 0 <= action.x < 64 and 0 <= action.y < 64):
        raise ValueError("Invalid coordinates")
