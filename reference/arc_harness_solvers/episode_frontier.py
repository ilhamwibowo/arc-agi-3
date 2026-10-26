"""Diversify repeated failed public histories without inferring a causal hazard."""
from collections import Counter
from dataclasses import dataclass, field
import hashlib
import json

from .core import Action
from .contextual_frontier import ContextualFrontierAgent


@dataclass
class HistoryEdge:
    terminal_count: int = 0
    children: set = field(default_factory=set)


@dataclass
class HistoryNode:
    actions: tuple = ()
    edges: dict = field(default_factory=dict)
    exhausted: bool = False


class EpisodeFrontierAgent(ContextualFrontierAgent):
    """Prefer a different button after repeated failure of an exact public prefix.

    Nodes include the starting observation and every executed action/observation
    since reset. Two terminal repeats mark an edge unproductive only within that
    prefix. Observed nonterminal successors remain alternatives until every button
    from each successor is likewise unproductive. This backs failure evidence up
    toward an earlier decision; it is not proof that any state or action is unsafe.
    Click branches prevent exhaustion. If all branches are exhausted, retry the
    parent choice and preserve counterevidence. Memory clears on context changes
    or at its capacity, making avoidance a reversible exploration heuristic.
    """
    history_limit = 8192

    def __init__(self, seed):
        super().__init__(seed)
        self.history = {}
        self.prefix = None
        self.history_path = []
        self.history_counts = Counter()

    def public_key(self, obs):
        return hashlib.sha256(json.dumps(obs.record(),sort_keys=True,separators=(',',':')).encode()).hexdigest()

    def _root(self, obs):
        return hashlib.sha256(('root:'+self.public_key(obs)).encode()).hexdigest()

    def _new_context(self, obs):
        super()._new_context(obs)
        self.history.clear()
        self.history_path.clear()
        self.prefix = self._root(obs)
        self.history[self.prefix] = HistoryNode(obs.actions)

    def edge_blocked(self, edge):
        if edge is None:
            return False
        if edge.terminal_count and edge.terminal_count < 2:
            return False
        if edge.children:
            return all(child in self.history and self.history[child].exhausted for child in edge.children)
        return edge.terminal_count >= 2

    def refresh(self, prefix):
        node = self.history[prefix]
        buttons = [a for a in node.actions if a not in (0,6)]
        previous = node.exhausted
        node.exhausted = (bool(buttons) and 6 not in node.actions and
                          all(self.edge_blocked(node.edges.get(a)) for a in buttons))
        return node.exhausted != previous

    def choose(self, obs):
        action, decision = super().choose(obs)
        if obs.state != 'NOT_FINISHED' or action.id in (0,6):
            return action, decision
        node = self.history[self.prefix]
        blocked = [kind for kind in obs.actions if kind not in (0,6) and self.edge_blocked(node.edges.get(kind))]
        if action.id not in blocked:
            return action, decision
        alternatives = [kind for kind in obs.actions if kind not in (0,6) and kind not in blocked]
        if not alternatives:
            self.history_counts['all_button_branches_exhausted'] += 1
            return action, decision
        # Prefer alternatives compatible with the existing spatial-risk filter.
        visible = self.regions(obs)
        safe = []
        for kind in alternatives:
            flagged = False
            for key, region in visible.items():
                graph = self.graphs[key]
                if graph.moves < 2:
                    continue
                risk = self.context_risks.get(self.risk_key(key, region, obs.grid, kind))
                pose = (*region['position'], region['shape'])
                flagged |= graph.retired(pose, kind) or bool(risk and risk.avoid())
            if not flagged:
                safe.append(kind)
        if not safe:
            self.history_counts['spatial_risk_conflicts'] += 1
            return action, decision
        replacement = Action(self.rng.choice(safe))
        self.pending = None
        self.history_counts['failed_history_overrides'] += 1
        edge = node.edges[action.id]
        return replacement, {'reason':'diversify_repeated_failed_history',
            'history_prefix':self.prefix,'history_depth':len(self.history_path),
            'avoided_action':action.id,'terminal_repeats':edge.terminal_count,
            'exhausted_successors':len(edge.children),'alternatives':safe,
            'warning':'Repeated public history is not hidden-state identity; this tests another branch rather than declaring a causal hazard.'}

    def observe(self, before, action, after):
        boundary = (before.levels != after.levels or
                    (action.id != 0 and before.state == after.state == 'NOT_FINISHED' and
                     (before.actions != after.actions or
                      (len(before.grid),len(before.grid[0])) != (len(after.grid),len(after.grid[0])))))
        feedback = None
        if (not boundary and action.id != 0 and before.state == 'NOT_FINISHED'
                and self.prefix in self.history):
            node = self.history[self.prefix]
            edge_key = action.id if action.id != 6 else (6,action.x,action.y)
            edge = node.edges.setdefault(edge_key, HistoryEdge())
            if after.state == 'GAME_OVER':
                edge.terminal_count += 1
                if self.refresh(self.prefix):
                    for parent in reversed(self.history_path):
                        if not self.refresh(parent):
                            break
                feedback = {'prefix':self.prefix,'terminal_repeats':edge.terminal_count,
                            'edge_unproductive':self.edge_blocked(edge),'depth':len(self.history_path)}
                self.history_counts['terminal_history_observations'] += 1
            elif after.state == 'NOT_FINISHED':
                child = hashlib.sha256((self.prefix+json.dumps(action.record(),sort_keys=True)+self.public_key(after)).encode()).hexdigest()
                edge.children.add(child)
                self.history.setdefault(child,HistoryNode(after.actions))
                self.history_path.append(self.prefix)
                self.prefix = child
                # A new nonterminal observation can reopen a previously exhausted branch.
                for parent in reversed(self.history_path):
                    if not self.refresh(parent):
                        break
        super().observe(before,action,after)
        if action.id == 0:
            self.prefix = self._root(after)
            self.history.setdefault(self.prefix,HistoryNode(after.actions))
            self.history_path.clear()
        if len(self.history) >= self.history_limit:
            self.history.clear();self.history_path.clear()
            self.prefix = self._root(after)
            self.history[self.prefix] = HistoryNode(after.actions)
            self.history_counts['capacity_resets'] += 1
        if feedback:
            self.latest_feedback['failed_public_history'] = feedback

    def diagnostics(self):
        result = super().diagnostics()
        result['episode_history'] = {'nodes':len(self.history),'depth':len(self.history_path),
            'exhausted_nodes':sum(n.exhausted for n in self.history.values()),
            'counts':dict(self.history_counts)}
        return result
