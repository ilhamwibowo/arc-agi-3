"""Select a demonstrated pattern relation before delegating to existing specialists."""
from arc_harness.core import Action
from pattern_copy import PatternCopyAgent, proposals
from specialist_selector import SpecialistSelectorAgent


class PatternSelectorAgent(SpecialistSelectorAgent):
    def choose(self,obs):
        if self.specialist is None and obs.state=='NOT_FINISHED':
            evidence=list(proposals(obs.grid)) if 6 in obs.actions else []
            if evidence:
                self.policy='pattern_copy'
                self.specialist=PatternCopyAgent(self.seed)
                best=max(evidence,key=lambda p:p['priority'])
                self.selection={'policy':self.policy,'step':self.steps,
                    'demonstration_origins':best['support'],'mapping':best['mapping'],
                    'scope':'Repeated visible miniature/board mapping; no game identifier.'}
        return super().choose(obs)
