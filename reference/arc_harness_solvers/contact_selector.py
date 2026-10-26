"""Select measured contact planning from visible framed targets and movable stencils."""
from cross_targets import CrossTargetAgent, read_scene
from relational_selector import RelationalSelectorAgent


def contact_evidence(obs):
    if obs.state != 'NOT_FINISHED' or 5 not in obs.actions or 6 in obs.actions:
        return None
    shapes, markers = read_scene(obs.grid)
    supported = [color for color in shapes if len(markers.get(color, [])) >= 2]
    if len(supported) < 2 or sum(s['selected'] for s in shapes.values()) != 1:
        return None
    return {'shape_colors': sorted(supported),
            'framed_markers': sum(len(markers[c]) for c in supported)}


class ContactSelectorAgent(RelationalSelectorAgent):
    def choose(self, obs):
        if self.specialist is None:
            evidence = contact_evidence(obs)
            if evidence:
                self.policy = 'contact'
                self.specialist = CrossTargetAgent(self.seed)
                self.selection = {'policy': self.policy, 'step': self.steps,
                                  **evidence,
                                  'scope': 'Visible framed targets and selected stencils; no game identifier.'}
        return super().choose(obs)
