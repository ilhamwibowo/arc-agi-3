"""Route demonstrated visual relations to bounded solvers; preserve fixed fallback."""
from pattern_selector import PatternSelectorAgent
from pattern_copy import proposals
from symbol_mapping import mapping_problem
from inverse_symbol import GrammarAgent


class RelationalSelectorAgent(PatternSelectorAgent):
    def choose(self,obs):
        if self.specialist is None and obs.state=='NOT_FINISHED':
            has_pattern=6 in obs.actions and bool(list(proposals(obs.grid)))
            problem=None if has_pattern else mapping_problem(obs.grid)
            if problem:
                self.policy='grammar';self.specialist=GrammarAgent(self.seed)
                self.selection={'policy':self.policy,'step':self.steps,'examples':problem['example_count'],
                                'rule_path':problem['rule_path'],'scope':'Observed paired glyph substitutions; no game identifier.'}
        return super().choose(obs)
