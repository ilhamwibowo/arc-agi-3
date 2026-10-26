"""Test visible sequence-copy hypotheses and retain observed input mappings."""
from collections import Counter, defaultdict

from arc_harness.agents import BalancedDirectedAgent
from arc_harness.core import Action
from arc_harness.scene import regions
from contact_selector import ContactSelectorAgent


def sequence_problem(grid):
    """Pair an outlined symbol sequence with a filled, same-palette control row.

    Row placement is a visual prior, not a rule of the environment. A candidate
    must be confirmed by interactions; no color is assigned a fixed meaning.
    """
    rows = defaultdict(list)
    for color, x, y, x1, y1, area in regions(grid)[0]:
        size = x1 - x + 1
        if size != y1-y+1 or not 2 <= size <= 12:
            continue
        if area == size*size:
            kind = 'filled'
        elif size >= 3 and area == 4*size-4:
            kind = 'outline'
        else:
            continue
        rows[kind, y, size].append((x, color))
    candidates = []
    for (kind, y, size), row in rows.items():
        if kind != 'outline' or not 2 <= len(row) <= 12:
            continue
        target = tuple(c for x, c in sorted(row))
        palette = set(target)
        if len(palette) < 2:
            continue
        for (other_kind, cy, cs), control in rows.items():
            if other_kind != 'filled' or cy <= y+size:
                continue
            if set(c for x, c in control) != palette or len(control) != len(palette):
                continue
            buttons = {c: (x+cs//2, cy+cs//2) for x, c in control}
            slot_groups = defaultdict(list)
            for (sk, sy, ss), cells in rows.items():
                if sk == 'filled' and y+size < sy < cy:
                    for x, color in cells:
                        if color not in palette:
                            slot_groups[color, ss].append((x+ss//2, sy+ss//2))
            slots = [tuple(sorted(points, key=lambda p: (p[1], p[0])))
                     for points in slot_groups.values() if 2 <= len(points) <= len(target)]
            if slots:
                slots = [points for points in slots if len(points) == max(map(len, slots))]
            if len(slots) != 1:
                continue
            candidates.append({'target': target, 'buttons': buttons, 'slots': slots[0],
                               'blank': tuple(grid[sy][sx] for sx,sy in slots[0]),
                               'target_points': tuple((x+size//2, y+size//2) for x,c in sorted(row)),
                               'target_row': (y, size), 'control_row': (cy, cs)})
    return candidates[0] if len(candidates) == 1 else None


def highlighted(grid, reference, point):
    """Detect a changed solid square border around an observed component center."""
    x,y = point
    for radius in range(2, 6):
        x0,y0,x1,y1 = x-radius,y-radius,x+radius-1,y+radius-1
        if x0<0 or y0<0 or x1>=len(grid[0]) or y1>=len(grid):
            continue
        border = [(xx,yy) for yy in range(y0,y1+1) for xx in range(x0,x1+1)
                  if xx in (x0,x1) or yy in (y0,y1)]
        values = {grid[yy][xx] for xx,yy in border}
        if len(values)==1 and sum(grid[yy][xx]!=reference[yy][xx] for xx,yy in border)>len(border)//2:
            return True
    return False


def animation_constraints(problem, before, frames):
    """Associate uniquely co-highlighted input and target symbols across frames.

    This is a testable execution-order hypothesis from public animation, not an
    interpreter of environment code or an assumption of row-major traversal.
    """
    pairs = Counter()
    for frame in frames:
        targets = [i for i,p in enumerate(problem['target_points']) if highlighted(frame,before,p)]
        slots = [i for i,p in enumerate(problem['slots']) if highlighted(frame,before,p)]
        if len(targets)==len(slots)==1:
            pairs[slots[0], problem['target'][targets[0]]] += 1
    by_slot = defaultdict(set)
    for (slot,token), count in pairs.items():
        if count>=3:
            by_slot[slot].add(token)
    return {slot:next(iter(tokens)) for slot,tokens in by_slot.items() if len(tokens)==1}


class SequenceRuleAgent:
    """Test select-and-place controls against a visible sequence goal.

    Placement predictions are checked at visible slot centers. Unexpected effects
    reject this hypothesis rather than counting the intended action as successful.
    """
    def __init__(self, seed):
        self.fallback = BalancedDirectedAgent(seed)
        self.level = None
        self.problem = None
        self.submit = None
        self.probed_buttons = set()
        self.counts = Counter()
        self.pending = None
        self.latest_feedback = None
        self.exhausted = False
        self.selected = None
        self.constraints = {}
        self.undo = None
        self.rebuilding = False
        self.undo_trials = set()

    def choose(self, obs):
        if self.level != obs.levels:
            self.level = obs.levels
            self.problem = sequence_problem(obs.grid) if 6 in obs.actions else None
            self.probed_buttons = set()
            self.constraints = {}
            self.rebuilding = False
            self.undo_trials = set()
            self.exhausted = False
            self.selected = None
        self.pending = None
        if obs.state in ('NOT_PLAYED', 'GAME_OVER'):
            self.selected = None
            return Action(0), {'reason': 'terminal_reset'}
        if self.problem and not self.exhausted:
            target = self.problem['target']
            if self.rebuilding:
                occupied = sum(obs.grid[y][x] != blank for (x,y),blank in
                               zip(self.problem['slots'],self.problem['blank']))
                if occupied:
                    choices = [a for a in obs.actions if a != 6 and a != self.submit
                               and (a not in self.undo_trials or a == self.undo)]
                    if choices:
                        button = self.undo if self.undo in choices else choices[0]
                        self.undo_trials.add(button)
                        self.pending = {'kind':'undo', 'occupied':occupied}
                        return Action(button), {'reason':'test_reversible_edit', 'occupied':occupied}
                    self.exhausted = True
                    return self.fallback.choose(obs)
                self.rebuilding = False
            remaining = Counter(target)
            for token in self.constraints.values():
                remaining[token] -= 1
            if any(n<0 for n in remaining.values()):
                self.exhausted = True
                return self.fallback.choose(obs)
            available = [token for token in target if remaining[token]>0]
            desired = dict(self.constraints)
            for i in range(len(self.problem['slots'])):
                if i not in desired:
                    token = next((c for c in available if remaining[c]>0), None)
                    if token is None:
                        self.exhausted = True
                        return self.fallback.choose(obs)
                    desired[i] = token
                    remaining[token] -= 1
            for i, (token, point) in enumerate(zip(target, self.problem['slots'])):
                token = desired[i]
                if obs.grid[point[1]][point[0]] == token:
                    continue
                if self.selected != token:
                    self.pending = {'kind': 'select', 'token': token}
                    return Action(6, *self.problem['buttons'][token]), {
                        'reason': 'test_palette_selection', 'token': token, 'slot': i}
                self.pending = {'kind': 'place', 'token': token, 'point': point}
                return Action(6, *point), {'reason': 'test_select_then_place',
                    'goal_confirmations': self.counts['completed_levels'],
                    'target': target, 'slot': i, 'predicted_color': token}
            buttons = [a for a in obs.actions if a != 6 and a not in self.probed_buttons]
            if self.submit in buttons:
                buttons.remove(self.submit)
                buttons.insert(0, self.submit)
            if buttons:
                action = Action(buttons[0])
                self.pending = {'kind': 'submit', 'id': action.id}
                self.probed_buttons.add(action.id)
                return action, {'reason': 'test_sequence_commit', 'target': target,
                                'confirmed_commit': self.submit, 'constraints': self.constraints}
            self.exhausted = True
            self.counts['unconfirmed_goal'] += 1
        return self.fallback.choose(obs)

    def observe(self, before, action, after):
        self.fallback.observe(before, action, after)
        progress = after.levels > before.levels
        self.latest_feedback = {'hypothesis': self.pending, 'level_progress': progress,
                                'frame_changed': before.grid != after.grid}
        if not self.pending:
            return
        if progress:
            self.counts['completed_levels'] += after.levels-before.levels
            if self.pending['kind'] == 'submit':
                self.submit = action.id
        elif self.pending['kind'] == 'undo':
            occupied = sum(after.grid[y][x] != blank for (x,y),blank in
                           zip(self.problem['slots'],self.problem['blank']))
            if occupied < self.pending['occupied']:
                self.undo = action.id
                self.counts['confirmed_undo'] += 1
            else:
                self.undo = None
                self.counts['undo_mismatch'] += 1
        elif self.pending['kind'] == 'select':
            self.selected = self.pending['token']
        elif self.pending['kind'] == 'place':
            x,y = self.pending['point']
            actual = after.grid[y][x]
            expected = self.pending['token']
            self.counts['confirmed_placements' if actual == expected else 'placement_mismatch'] += 1
            self.latest_feedback['predicted_color'] = expected
            self.latest_feedback['actual_color'] = actual
            self.selected = None
            if actual != expected:
                self.exhausted = True
        else:
            self.counts['unconfirmed_commits'] += 1
            self.selected = None
            inferred = animation_constraints(self.problem, before.grid, after.frames)
            changed = {i:c for i,c in inferred.items()
                       if self.constraints.get(i,self.problem['target'][i]) != c}
            self.constraints.update(inferred)
            self.latest_feedback['animation_constraints'] = inferred
            if changed and self.counts['unconfirmed_commits'] <= 24:
                self.probed_buttons.discard(action.id)
                self.counts['order_revisions'] += 1
                self.rebuilding = True

    def diagnostics(self):
        return {'counts': dict(self.counts), 'confirmed_commit': self.submit, 'constraints': self.constraints}


class SequenceSelectorAgent(ContactSelectorAgent):
    def choose(self, obs):
        if self.specialist is None and obs.state == 'NOT_FINISHED' and 6 in obs.actions:
            problem = sequence_problem(obs.grid)
            if problem:
                self.policy = 'sequence_rule'
                self.specialist = SequenceRuleAgent(self.seed)
                self.selection = {'policy': self.policy, 'step': self.steps,
                                  'evidence': problem,
                                  'scope': 'Outlined sequence and same-palette filled row; no game ID.'}
        return super().choose(obs)
