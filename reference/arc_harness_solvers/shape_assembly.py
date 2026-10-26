"""Observed shape construction followed by per-piece rotation checks and assembly."""
from collections import Counter
from copy import deepcopy

from shape_variants import ShapeGrowthProbeAttempt, ShapeProbeAgent, footprint
from reveal_memory import RevealMemory, interior
from assembly_reveal import RevealAttempt
from arc_harness.assembly_control import AssemblyAttempt, CompositeEffects, body_color, effect
from arc_harness.anchor_agent import AnchorSelectiveAgent
from arc_harness.assembly_geometry import solve, validate
from arc_harness.core import Action


class ShapeAssemblyAttempt(ShapeGrowthProbeAttempt):
    """A single bounded experiment; construction, calibration and execution share192 actions."""
    def __init__(self, obs, marker, effects, step=0):
        super().__init__(obs, marker, effects, step)
        self.rotation_checks = []
        self.calibration_models = {}
        self.calibration_control = None
        self.execution = None
        self.required_calibration = []
        self.proposed_geometry = None

    def choose(self, obs):
        if self.status != 'active':
            return None
        if obs.state != 'NOT_FINISHED' or obs.levels != self.level:
            self.stop('context_changed'); return None
        if self.actions >= self.action_limit:
            self.stop('action_budget'); return None
        if self.execution is not None:
            result = self.execution.choose(obs)
            if result is None:
                self.stop('execution_'+self.execution.status)
            else:
                self.pending = {'phase': 'execution'}
            return result
        if (self.phase == 'variant_probe' and self.variants
                and all(body_color(p, self.marker) is not None for p in self.memory.pieces)):
            changed = Counter(s['action'] for s in self.samples if s['outcome'] == 'nonrigid_observation')
            if not changed:
                self.stop('missing_shape_control'); return None
            self.calibration_control = min(changed, key=lambda k:(-changed[k], k))
            # Rotated placements are proposals only. Each nonidentity operation
            # requires its own piece's successful live calibration before execution.
            self.proposed_geometry = solve(self.memory.pieces, self.marker, len(obs.grid[0]), len(obs.grid),
                                           (0, 1, 2, 3), max_nodes=20000, max_seconds=1)
            if self.proposed_geometry['status'] != 'candidate':
                self.stop('proposal_'+self.proposed_geometry['status']); return None
            check = validate(self.memory.pieces, self.marker, len(obs.grid[0]), len(obs.grid),
                             self.proposed_geometry['solution'], (0, 1, 2, 3))
            if not check['valid']:
                self.stop('invalid_proposal'); return None
            self.required_calibration = [p['piece'] for p in self.proposed_geometry['solution']['placements'] if p['op']]
            self.phase = 'rotation_calibration'
        if self.phase != 'rotation_calibration':
            return super().choose(obs)
        if not self.memory.matches(obs.grid)[0]:
            self.stop('remembered_scene_mismatch'); return None
        remaining = [i for i in self.required_calibration
                     if self.calibration_models.get(i, CompositeEffects()).samples[self.calibration_control] < 3]
        if not remaining:
            kinds = [m.model(self.calibration_control) for m in self.calibration_models.values()]
            if kinds and (len(set(kinds)) != 1 or not kinds[0] or kinds[0][0] != 'turn'):
                self.stop('inconsistent_piece_controls'); return None
            controls = deepcopy(self.effects)
            k = self.calibration_control
            controls.samples.pop(k, None); controls.modes.pop(k, None); controls.steps.pop(k, None)
            for sample in self.rotation_checks:
                controls.add(k, sample['before_pixels'], sample['after_pixels'], sample['step'])
            self.execution = RevealAttempt(obs, self.marker, controls)
            memory = RevealMemory(obs, self.marker)
            memory.__dict__.update(deepcopy(self.memory.__dict__))
            self.execution.memory = memory
            self.execution.status = 'active'
            self.execution.action_limit = self.action_limit-self.actions
            self.execution.plan(obs)
            if self.execution.status != 'active':
                self.stop('execution_'+self.execution.status); return None
            unsupported = [p['piece'] for p in self.execution.geometry['solution']['placements']
                           if p['op'] and p['piece'] not in self.calibration_models]
            if unsupported:
                self.execution = None
                self.required_calibration.extend(i for i in unsupported if i not in self.required_calibration)
                return self.choose(obs)
            return self.choose(obs)
        i = remaining[0]
        if self.memory.selected != i:
            if 6 not in obs.actions:
                self.stop('selection_unavailable'); return None
            coords = footprint(self.memory.pieces[i])
            self.pending = {'phase': 'select', 'coords': coords, 'piece': i}
            return Action(6, *interior(coords)), {'reason': 'shape_calibration_select', 'piece': i}
        k = self.calibration_control
        if k not in obs.actions:
            self.stop('control_unavailable'); return None
        self.pending = {'phase': 'calibrate_rotation', 'piece': i}
        return Action(k), {'reason': 'shape_per_piece_control_test', 'piece': i,
                          'samples': self.calibration_models.get(i, CompositeEffects()).samples[k],
                          'warning': 'Rotation is a hypothesis for this piece; observe each result before planning.'}

    def observe(self, before, action, after, step):
        phase = self.pending['phase'] if self.pending else None
        if phase == 'execution':
            self.actions += 1; self.counts['execution_actions'] += 1
            self.execution.observe(before, action, after, step)
            self.feedback = self.execution.feedback
            self.pending = None
            if self.execution.status != 'active':
                self.stop(self.execution.status)
            return
        if phase != 'calibrate_rotation':
            return super().observe(before, action, after, step)
        i = self.pending['piece']; self.pending = None
        self.actions += 1; self.counts['rotation_calibration_actions'] += 1
        sample = {'step': step, 'piece': i, 'action': action.id}
        if after.levels > before.levels or after.state == 'WIN':
            self.stop('progress'); sample['outcome'] = 'progress'
        elif (after.state != 'NOT_FINISHED' or after.actions != before.actions
              or len(after.grid) != len(before.grid) or len(after.grid[0]) != len(before.grid[0])):
            self.stop('context_changed'); sample['outcome'] = self.status
        else:
            old = self.memory.pieces[i]
            found = self.memory.changed_piece(before, after)
            measured = effect(old, found) if found else None
            if measured is None or measured[0] != 'turn':
                self.stop('rotation_hypothesis_rejected'); sample['outcome'] = self.status
                sample['measured_effect'] = measured
            else:
                self.calibration_models.setdefault(i, CompositeEffects()).add(action.id, old, found, step)
                self.memory.pieces[i] = found
                self.memory.steps[i] = (self.memory.steps[i]+[step])[-8:]
                sample.update({'outcome': 'piece_rotation_observed', 'measured_effect': measured,
                               'before_pixels': sorted(old), 'after_pixels': sorted(found)})
        self.rotation_checks.append(sample)
        self.feedback = sample

    def record(self):
        return {**super().record(), 'rotation_checks': self.rotation_checks,
                'per_piece_controls': {i:m.record() for i,m in self.calibration_models.items()},
                'proposed_geometry': self.proposed_geometry, 'required_calibration': self.required_calibration,
                'execution': self.execution.record() if self.execution else None}


class ShapeAssemblyAgent(ShapeProbeAgent):
    attempt_class = ShapeAssemblyAttempt


class AdaptiveShapeAttempt(ShapeAssemblyAttempt):
    """Discover every palette before choosing rigid assembly or bounded construction."""
    variant_limit = 48
    probe_limit = 64

    def plan(self, obs):
        if len(self.memory.pieces) < 2 or self.memory.mask_color is None:
            self.stop('insufficient_revealed_geometry'); return
        if all(body_color(p, self.marker) is not None for p in self.memory.pieces):
            self.execution = RevealAttempt(obs, self.marker, self.effects)
            memory = RevealMemory(obs, self.marker)
            memory.__dict__.update(deepcopy(self.memory.__dict__))
            self.execution.memory = memory
            self.execution.status = 'active'
            self.action_limit = 160
            self.execution.action_limit = self.action_limit-self.actions
            self.execution.plan(obs)
            self.phase = 'execute'
            if self.execution.status != 'active':
                self.stop('execution_'+self.execution.status)
        else:
            super().plan(obs)

    def choose(self, obs):
        if self.status != 'active':
            return None
        if obs.state != 'NOT_FINISHED' or obs.levels != self.level:
            self.stop('context_changed'); return None
        if self.actions >= self.action_limit:
            self.stop('action_budget'); return None
        if self.phase == 'discover' and not self.memory.candidates(obs.grid):
            self.plan(obs)
        if self.phase == 'variant_probe':
            incomplete = [i for i,p in enumerate(self.memory.pieces) if body_color(p,self.marker) is None]
            if incomplete:
                if self.probe_piece not in incomplete:
                    self.probe_piece = incomplete[0]
                if sum(v['piece'] == self.probe_piece for v in self.variants) >= 6:
                    self.stop('piece_variant_budget'); return None
        return super().choose(obs)


class AdaptiveShapeAgent(ShapeProbeAgent):
    """Multicolor discovery also applies when the initially selected piece is two-color."""
    def choose(self, obs):
        self.assembly_pending = False
        if self.role and obs.state == 'NOT_FINISHED' and obs.levels not in self.attempted_levels:
            if len(self.attempted_levels) < 64:
                self.attempted_levels.add(obs.levels)
                self.assembly = AssemblyAttempt(obs, *self.role, self.composite_effects)
                if self.assembly.status == 'ambiguous_scene':
                    self.assembly = AdaptiveShapeAttempt(obs, self.role[0], self.composite_effects, self.steps)
                self.assembly_counts['attempts'] += 1
        if self.assembly and self.assembly.status == 'active':
            proposed = self.assembly.choose(obs)
            if proposed:
                self.assembly_pending = True
                self.pending = {'phase': 'assembly'}
                return proposed
        return AnchorSelectiveAgent.choose(self, obs)
