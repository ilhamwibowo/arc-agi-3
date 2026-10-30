"""Discrete-time simulation of the PriorityGate + turn-boundary rotation under the competition's
shape (110 games, 10 slots, ~8.3 h) vs the offline shape (25 games, 10 slots, 3 h), to size the
"unstarted-game sweep" the run #10 trace review flagged. No GPU, no model: each game is a list of
per-level slot-second costs calibrated from run #10 (runs/kaggle/arc3-full25f/1).

  .venv/bin/python scripts/sim_scheduler.py            # prints levels per rule per shape

Rules compared (all use scheduler.priority for ranking; `unstarted` = how unstarted waiters rank):
  inf      : current code (UNSTARTED_PRIORITY = inf, holder yields at 900 s to any unstarted waiter)
  fresh    : unstarted waiters rank like a fresh started game (finite), so holders mid-level with a
             higher priority keep the slot past 900 s
  fresh+cap: as `fresh`, plus a holder's first slice on a game is capped at 300 s while unstarted
             waiters exist (quick probe, then the formula decides)
"""
from __future__ import annotations

import glob
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arc3 import scheduler as sch  # noqa: E402
from arc3.scheduler import GameProgress, priority  # noqa: E402

TICK = 30.0                      # seconds of slot time per tick for a holder
ACTIONS_PER_MIN = 4.0            # run #10: ~2.7 calls/min, ~1.5 actions/call
TOKENS_PER_MIN = 35 * 60.0       # ~35 tok/s per stream
ROTATE = 900.0
HYST = 1.2


def load_profiles(run_dir: str) -> list[list[float]]:
    """Per-game list of per-level slot-second costs. Completed levels share the measured slot time
    equally; the level that was not completed is given 2x the game's mean level cost plus what was
    already sunk into it (an estimate), later levels the mean cost."""
    profiles = []
    for f in sorted(glob.glob(os.path.join(run_dir, "*/summary.json"))):
        s = json.load(open(f))
        levels, total = int(s.get("levels_completed") or 0), int(s.get("levels_total") or 0)
        slot = float(s.get("seconds") or 0) - float(s.get("gate_wait_seconds") or 0)
        if total <= 0:
            continue
        if levels >= total:
            per = slot / levels
            profiles.append([per] * total)
            continue
        if levels > 0:
            per = slot * 0.75 / levels           # assume ~25% of slot time sank into the unfinished level
            sunk = slot * 0.25
        else:
            per, sunk = slot, slot
        costs = [per] * levels + [sunk + 2 * per] + [per] * (total - levels - 1)
        profiles.append(costs)
    return profiles


def simulate(profiles: list[list[float]], n_games: int, slots: int, hours: float, rule: str,
             seed: int = 0, v5: bool = False) -> int:
    rng = random.Random(seed)
    games = [rng.choice(profiles) for _ in range(n_games)]
    prog = [GameProgress(game_id=str(i), levels_total=len(g)) for i, g in enumerate(games)]
    spent = [0.0] * n_games                  # slot seconds on the current level
    held_since = {}                          # gid -> time slot acquired
    first_slice = set()                      # gids in their first slice
    done = [False] * n_games
    order = list(range(n_games))             # arrival order for unstarted ties
    t, end = 0.0, hours * 3600
    active: list[int] = []

    def prio(i, frac):
        p = prog[i]
        if not p.started:
            if rule == "inf":
                return float("inf")
            fresh = GameProgress(game_id="fresh", started=True, levels_total=p.levels_total)
            return priority(fresh, frac, now=t)
        if not v5:
            p.stall = 0.0
        return priority(p, frac, now=t)

    while t < end:
        frac = t / end
        # holders accumulate slot time
        for i in list(active):
            spent[i] += TICK
            p = prog[i]
            p.actions_on_level += ACTIONS_PER_MIN * TICK / 60
            p.tokens_on_level += TOKENS_PER_MIN * TICK / 60
            if spent[i] >= games[i][p.level]:
                p.level_completed(p.level + 1, now=t)
                spent[i] = 0.0
                if p.level >= p.levels_total:
                    done[i] = True
                    active.remove(i)
                    held_since.pop(i, None)
            elif v5 and p.level_actions:
                med = sorted(p.level_actions)[len(p.level_actions) // 2]
                p.stall = 1.0 if (med > 0 and p.actions_on_level > 3 * med) else 0.0
        # turn-boundary rotation (every tick is a turn boundary here)
        waiting = [i for i in range(n_games) if not done[i] and i not in active]
        for i in list(active):
            held = t - held_since[i]
            cap = ROTATE
            if rule == "fresh+cap" and i in first_slice and any(not prog[j].started for j in waiting):
                cap = 300.0
            if held < cap:
                continue
            mine = prio(i, frac)
            if any(prio(j, frac) > mine * HYST for j in waiting):
                active.remove(i)
                held_since.pop(i, None)
                first_slice.discard(i)
        # fill free slots
        waiting = [i for i in range(n_games) if not done[i] and i not in active]
        waiting.sort(key=lambda j: (-prio(j, frac), order.index(j)))
        for j in waiting[: max(0, slots - len(active))]:
            active.append(j)
            held_since[j] = t
            if not prog[j].started:
                prog[j].started = True
                first_slice.add(j)
        t += TICK
    return sum(p.level for p in prog)


def main():
    profiles = load_profiles(str(ROOT / "runs/kaggle/arc3-full25f/1/output/runs/kaggle"))
    print(f"{len(profiles)} game profiles; levels available offline: {sum(len(p) for p in profiles)}")
    shapes = [("offline 25 games / 10 slots / 3.0 h", 25, 10, 3.0),
              ("competition 110 games / 10 slots / 8.3 h", 110, 10, 8.3),
              ("competition 110 games / 10 slots / 7.6 h (40 min lost)", 110, 10, 7.6)]
    for name, n, slots, hours in shapes:
        print(f"\n== {name}")
        for rule in ("inf", "fresh", "fresh+cap"):
            for v5 in (False, True):
                vals = [simulate(profiles, n, slots, hours, rule, seed=s, v5=v5) for s in range(5)]
                print(f"  {rule:9} {'v5' if v5 else 'v4'}: levels mean {sum(vals)/len(vals):6.1f}  "
                      f"(seeds: {vals})")


if __name__ == "__main__":
    main()
