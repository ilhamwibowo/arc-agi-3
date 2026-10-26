"""Choose one specialist from the first playable public image and retain ownership."""
from shape_assembly import AdaptiveShapeAgent
from arc_harness.retry_frontier import RetryingFrontierAgent
from arc_harness.core import Action
from arc_harness.visual_alignment import candidates


class SpecialistSelectorAgent:
    """Initial marker relationships select assembly; other scenes select navigation.

    This is a visual heuristic, not a claim that a game belongs to one fixed family.
    Only the selected policy receives decisions and outcomes, including across resets.
    """
    def __init__(self, seed):
        self.seed = seed
        self.specialist = None
        self.policy = None
        self.selection = None
        self.steps = 0
        self.latest_feedback = None

    def choose(self, obs):
        if self.specialist is None:
            if obs.state in ('NOT_PLAYED', 'GAME_OVER'):
                return Action(0), {'reason': 'selector_waiting_for_playable_observation'}
            if obs.state != 'NOT_FINISHED':
                raise ValueError('No playable observation for specialist selection')
            _, relations = candidates(obs.grid)
            self.policy = 'adaptive_shape' if relations else 'frontier_retry'
            factory = AdaptiveShapeAgent if relations else RetryingFrontierAgent
            self.specialist = factory(self.seed)
            self.selection = {'policy': self.policy, 'step': self.steps,
                              'initial_marker_relations': relations,
                              'scope': 'One initial visual choice retained across the run; no game identifier.'}
        action, decision = self.specialist.choose(obs)
        return action, {**decision, 'selector': {'policy': self.policy,
                       'selection_step': self.selection['step'],
                       **({'evidence': self.selection} if self.steps == self.selection['step'] else {})}}

    def observe(self, before, action, after):
        self.steps += 1
        if self.specialist is None:
            self.latest_feedback = {'selector': {'policy': None, 'waiting': True}}
            return
        self.specialist.observe(before, action, after)
        self.latest_feedback = {'selector': {'policy': self.policy},
                                'specialist': self.specialist.latest_feedback}

    def diagnostics(self):
        return {'steps': self.steps, 'selection': self.selection,
                'specialist': self.specialist.diagnostics() if self.specialist else None}
