"""Tentative navigation over observed compact-region positions, without a simulator."""
from collections import Counter, defaultdict, deque
import random

from .core import Action
from .object_memory import components


def compact_color_regions(grid):
    """Describe colors with one compact connected region; ambiguous colors stay out.

    A key describes color, area and bounding-box size, not semantic object identity.
    Shape/orientation can change. Every position correspondence remains a hypothesis.
    """
    groups = defaultdict(list)
    for component in components(grid):
        groups[component['color']].append(component)
    result = {}
    for color, parts in groups.items():
        if len(parts) != 1:
            continue
        c = parts[0]
        x0, y0, x1, y1 = c['bounds']
        w, h = x1-x0+1, y1-y0+1
        if not (2 <= c['pixels'] <= 256 and max(w, h) <= 16 and max(w, h) <= 4*min(w, h)):
            continue
        key = (color, c['pixels'], *sorted((w, h)))
        result[key] = {'color': color, 'pixels': c['pixels'], 'bounds': c['bounds'],
                       'position': (x0+x1, y0+y1), 'shape': c['shape']}
    return result


class MoveGraph:
    """Observed edges are tentative and must be checked at each actual traversal."""
    def __init__(self):
        self.samples = defaultdict(Counter)
        self.steps = defaultdict(lambda: deque(maxlen=4))
        self.moves = 0
        self.contradictions = 0
        self.unresolved_edges = set()
        self.terminal_examples = defaultdict(lambda: deque(maxlen=8))

    def note_terminal(self, pose, action, episode, age, step):
        self.terminal_examples[pose, action].append({'episode': episode, 'age': age, 'step': step})

    def retired(self, pose, action):
        key = (pose, action)
        examples = self.terminal_examples.get(key, ())
        # Repeated failures at one attempt age can be a common resource limit.
        # Any observed nonterminal transition blocks this broad avoidance rule.
        return (not self.samples.get(key) and len({e['episode'] for e in examples}) >= 2
                and len({e['age'] for e in examples}) >= 2)

    def update(self, position, action, successor, step, *, moved=None):
        key = (position, action)
        previous = self.prediction(position, action)
        if previous is not None and previous != successor:
            self.contradictions += 1
        self.samples[key][successor] += 1
        self.steps[key].append(step)
        self.moves += int(successor != position if moved is None else moved)

    def prediction(self, position, action):
        values = self.samples.get((position, action))
        # Conflicting observations are not resolved by majority vote into certainty.
        return (next(iter(values)) if values and len(values) == 1
                and (position, action) not in self.unresolved_edges else None)

    def frontier_action(self, start, actions, rng):
        """Shortest observed route to a node with an untested available control."""
        queue = deque([(start, None, 0)])
        visited = {start}
        while queue and len(visited) <= 4096:
            node, first, distance = queue.popleft()
            untested = [a for a in actions if (node, a) not in self.samples and not self.retired(node, a)]
            if untested:
                return (first if first is not None else rng.choice(untested)), distance
            order = list(actions)
            rng.shuffle(order)
            for action in order:
                nxt = self.prediction(node, action)
                if nxt is None or nxt == node or nxt in visited:
                    continue
                visited.add(nxt)
                queue.append((nxt, action if first is None else first, distance+1))
        return None


class FrontierAgent:
    learn_terminal = False

    def __init__(self, seed):
        # Lazy import avoids a registry dependency at module initialization.
        from .agents import BalancedDirectedAgent
        self.fallback = BalancedDirectedAgent(seed)
        self.rng = random.Random(seed)
        self.graphs = defaultdict(MoveGraph)
        self.steps = 0
        self.anchor = None
        self.anchor_context = None
        self.pending = None
        self.focus = None
        self.counts = Counter()
        self.latest_feedback = None
        self.cached_grid = None
        self.cached_regions = None
        self.episode = self.attempt_age = 0

    def regions(self, obs):
        if obs.grid != self.cached_grid:
            self.cached_grid = obs.grid
            self.cached_regions = compact_color_regions(obs.grid)
        return self.cached_regions

    def _new_context(self, obs):
        self.graphs.clear()
        self.focus = self.pending = None
        self.anchor = obs.grid
        self.anchor_context = (obs.levels, obs.actions)

    def choose(self, obs):
        if self.anchor is None:
            self._new_context(obs)
        self.pending = None
        if obs.state in ('GAME_OVER', 'NOT_PLAYED'):
            return self.fallback.choose(obs)
        controls = [a for a in obs.actions if a not in (0, 6)]
        visible = self.regions(obs)
        candidates = [key for key, c in visible.items() if self.graphs[key].moves >= 2]
        candidates.sort(key=lambda key: (-self.graphs[key].moves, -key[1], key))
        if self.focus in candidates:
            candidates.remove(self.focus)
            candidates.insert(0, self.focus)
        for key in candidates:
            graph = self.graphs[key]
            position = (*visible[key]['position'], visible[key]['shape'])
            route = graph.frontier_action(position, controls, self.rng)
            if route is None:
                continue
            action, distance = route
            expected = graph.prediction(position, action)
            self.focus = key
            self.pending = {'track': key, 'from_pose': position,
                            'expected_pose': expected, 'action': action,
                            'frontier_distance': distance,
                            'evidence_steps': list(graph.steps.get((position, action), []))}
            self.counts['navigation_actions'] += 1
            return Action(action), {'reason': 'observed_route_to_untested_control',
                                    'navigation': self.pending,
                                    'warning': 'Position and visible shape are an abstraction, not hidden-state identity; recheck next frame.'}
        self.counts['balanced_fallback_actions'] += 1
        action, decision = self.fallback.choose(obs)
        if self.learn_terminal and action.id != 6:
            def retired(kind):
                return any(self.graphs[key].retired((*c['position'], c['shape']), kind)
                           for key, c in visible.items())
            if retired(action.id):
                alternatives = [a for a in controls if not retired(a)]
                if alternatives:
                    action = Action(self.rng.choice(alternatives))
                    decision = {'reason': 'avoid_repeated_terminal_frontier',
                                'scope': 'Repeated attempt ages differ; no observed successful transition.'}
                    self.counts['terminal_fallback_overrides'] += 1
        return action, decision

    def observe(self, before, action, after):
        self.steps += 1
        self.attempt_age += 1
        self.fallback.observe(before, action, after)
        pending = self.pending
        feedback = {'step': self.steps, 'levels_delta': after.levels-before.levels,
                    'navigation': pending, 'position_check': 'not_requested'}
        if after.levels != before.levels:
            self.episode += 1
            self.attempt_age = 0
            self._new_context(after)
            feedback['boundary'] = 'level_change'
        elif action.id == 0:
            self.episode += 1
            self.attempt_age = 0
            reusable = after.grid == self.anchor and (after.levels, after.actions) == self.anchor_context
            if not reusable:
                self._new_context(after)
            self.focus = self.pending = None
            self.counts['reset_graph_retained' if reusable else 'reset_graph_cleared'] += 1
            feedback.update(boundary='reset', tentative_graph_retained=reusable)
        elif before.state != 'NOT_FINISHED' or after.state != 'NOT_FINISHED':
            failures = []
            if (self.learn_terminal and before.state == 'NOT_FINISHED' and
                    after.state == 'GAME_OVER' and action.id not in (0, 6)):
                for key, c in self.regions(before).items():
                    graph = self.graphs[key]
                    if graph.moves < 2:
                        continue
                    pose = (*c['position'], c['shape'])
                    graph.note_terminal(pose, action.id, self.episode, self.attempt_age, self.steps)
                    failures.append({'track': key, 'pose': pose, 'action': action.id,
                                     'attempt_age': self.attempt_age,
                                     'retired_frontier': graph.retired(pose, action.id),
                                     'evidence': list(graph.terminal_examples[pose, action.id])})
                self.counts['terminal_outcomes_recorded'] += int(bool(failures))
            if failures:
                feedback['terminal_evidence'] = failures
            self.pending = None
            feedback['boundary'] = 'terminal_state'
        elif (before.actions != after.actions or
              (len(before.grid), len(before.grid[0])) != (len(after.grid), len(after.grid[0]))):
            self._new_context(after)
            feedback['boundary'] = 'controls_or_dimensions'
        else:
            old = self.regions(before)
            new = self.regions(after)
            changes = []
            if action.id != 6:
                for key in old.keys() & new.keys():
                    a, b = old[key], new[key]
                    graph = self.graphs[key]
                    graph.update((*a['position'], a['shape']), action.id,
                                 (*b['position'], b['shape']), self.steps,
                                 moved=a['position'] != b['position'])
                    if a['position'] != b['position']:
                        changes.append({'track': key, 'from_position': a['position'],
                                        'to_position': b['position'],
                                        'shape_changed': a['shape'] != b['shape']})
            if pending and pending['expected_pose'] is not None:
                actual = new.get(pending['track'])
                check = ('unknown' if actual is None else 'matched'
                         if (*actual['position'], actual['shape']) == pending['expected_pose'] else 'contradicted')
                feedback['position_check'] = check
                self.counts['position_' + check] += 1
                if check != 'matched':
                    self.focus = None
                if check == 'unknown':
                    self.graphs[pending['track']].unresolved_edges.add((pending['from_pose'], pending['action']))
            feedback['measured_compact_region_moves'] = changes
            # Full frames remain in the trace; the residual count does not discard pixels.
            feedback['changed_pixels'] = sum(a != b for ra, rb in zip(before.grid, after.grid)
                                             for a, b in zip(ra, rb))
        self.latest_feedback = feedback

    def diagnostics(self):
        return {'steps': self.steps, 'counts': dict(self.counts),
                'tracks': [{'key': key, 'moves': graph.moves, 'position_action_pairs': len(graph.samples),
                            'contradictory_pairs': sum(len(v) > 1 for v in graph.samples.values()),
                            'counterexamples': graph.contradictions,
                            'unresolved_edges': len(graph.unresolved_edges),
                            'terminal_pairs': len(graph.terminal_examples),
                            'retired_frontiers': sum(graph.retired(pose, action) for pose, action in graph.terminal_examples)}
                           for key, graph in sorted(self.graphs.items())]}


class TerminalAwareFrontierAgent(FrontierAgent):
    """Avoid repeatedly terminal unknown controls; keep nonterminal counterevidence."""
    learn_terminal = True
