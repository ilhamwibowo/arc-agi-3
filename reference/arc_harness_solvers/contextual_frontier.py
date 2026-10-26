"""Neighborhood-conditioned terminal evidence for tentative observed-move navigation."""
from collections import OrderedDict, deque
import hashlib

from .core import Action
from .frontier import TerminalAwareFrontierAgent


def neighborhood(grid, region):
    """Hash public pixels around a tracked region, including out-of-frame padding."""
    x0, y0, x1, y1 = region['bounds']
    margin = 2 * max(x1-x0+1, y1-y0+1)
    height, width = len(grid), len(grid[0])
    pixels = bytes(grid[y][x] if 0 <= y < height and 0 <= x < width else 255
                   for y in range(y0-margin, y1+margin+1)
                   for x in range(x0-margin, x1+margin+1))
    return hashlib.sha256(pixels).hexdigest()[:16]


class ContextRisk:
    """Repeated terminal outcomes are scoped to an observed neighborhood, not a rule."""
    def __init__(self):
        self.successes = 0
        self.first_episode = None
        self.multiple_episodes = False
        self.min_age = self.max_age = None
        self.failures = 0
        self.steps = deque(maxlen=4)

    def note(self, terminal, episode, age, step):
        if not terminal:
            self.successes += 1
            return
        self.failures += 1
        self.steps.append(step)
        if self.first_episode is None:
            self.first_episode = episode
        self.multiple_episodes |= episode != self.first_episode
        self.min_age = age if self.min_age is None else min(self.min_age, age)
        self.max_age = age if self.max_age is None else max(self.max_age, age)

    def avoid(self):
        return (not self.successes and self.multiple_episodes
                and self.min_age != self.max_age)

    def record(self):
        return {'nonterminal_outcomes': self.successes, 'terminal_outcomes': self.failures,
                'multiple_attempts': self.multiple_episodes, 'age_range': [self.min_age, self.max_age],
                'recent_terminal_steps': list(self.steps), 'avoid': self.avoid()}


class ContextualFrontierAgent(TerminalAwareFrontierAgent):
    """Filter repeated local-context failures while preserving the baseline navigator."""
    def __init__(self, seed):
        super().__init__(seed)
        self.context_risks = OrderedDict()

    def _new_context(self, obs):
        super()._new_context(obs)
        self.context_risks.clear()

    def risk_key(self, key, region, grid, action):
        return (key, (*region['position'], region['shape']), neighborhood(grid, region), action)

    def choose(self, obs):
        action, decision = super().choose(obs)
        if obs.state != 'NOT_FINISHED' or action.id in (0, 6):
            return action, decision
        visible = self.regions(obs)
        tracked = {key: region for key, region in visible.items() if self.graphs[key].moves >= 2}
        blocked = {}
        for kind in obs.actions:
            if kind in (0, 6):
                continue
            evidence = []
            for key, region in tracked.items():
                risk_key = self.risk_key(key, region, obs.grid, kind)
                risk = self.context_risks.get(risk_key)
                if risk and risk.avoid():
                    evidence.append({'track': key, 'pose': risk_key[1], 'neighborhood': risk_key[2],
                                     **risk.record()})
            if evidence:
                blocked[kind] = evidence
        if action.id not in blocked:
            return action, decision
        alternatives = [kind for kind in obs.actions if kind not in (0, 6) and kind not in blocked]
        if not alternatives:
            self.counts['context_avoidance_no_alternative'] += 1
            return action, decision
        replacement = Action(self.rng.choice(alternatives))
        self.pending = None
        self.counts['context_terminal_overrides'] += 1
        return replacement, {'reason': 'avoid_repeated_terminal_neighborhood',
                             'avoided_action': action.id, 'evidence': blocked[action.id],
                             'alternatives': alternatives,
                             'warning': 'Observed local context is not hidden-state identity; this is a reversible exploration heuristic.'}

    def observe(self, before, action, after):
        evidence = []
        compatible = (before.levels == after.levels and before.actions == after.actions and
                      len(before.grid) == len(after.grid) and len(before.grid[0]) == len(after.grid[0]))
        if (compatible and before.state == 'NOT_FINISHED' and action.id not in (0, 6)
                and after.state in ('NOT_FINISHED', 'GAME_OVER')):
            for key, region in self.regions(before).items():
                if self.graphs[key].moves < 2:
                    continue
                risk_key = self.risk_key(key, region, before.grid, action.id)
                risk = self.context_risks.setdefault(risk_key, ContextRisk())
                self.context_risks.move_to_end(risk_key)
                risk.note(after.state == 'GAME_OVER', self.episode, self.attempt_age+1, self.steps+1)
                if after.state == 'GAME_OVER':
                    evidence.append({'track': key, 'pose': risk_key[1], 'neighborhood': risk_key[2],
                                     **risk.record()})
                while len(self.context_risks) > 8192:
                    self.context_risks.popitem(last=False)
        super().observe(before, action, after)
        if evidence:
            self.latest_feedback['context_terminal_evidence'] = evidence

    def diagnostics(self):
        result = super().diagnostics()
        result['context_risks'] = {'entries': len(self.context_risks),
                                  'avoided': sum(r.avoid() for r in self.context_risks.values())}
        return result
