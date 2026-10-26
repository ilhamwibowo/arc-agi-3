"""Conservative component correspondence and measured transitions from public pixels."""
from collections import Counter, defaultdict, deque
import hashlib

import numpy as np

from .scene import regions


def components(grid):
    descriptors, labels = regions(grid)
    result = []
    for i, (color, x0, y0, x1, y1, area) in enumerate(descriptors):
        ys, xs = np.where(labels == i)
        shape = tuple(zip((xs-x0).tolist(), (ys-y0).tolist()))
        signature = (color, shape)
        nearest = int(((2*xs-x0-x1)**2 + (2*ys-y0-y1)**2).argmin())
        result.append({'color': color, 'bounds': [x0,y0,x1,y1], 'pixels': area,
                       'point': [int(xs[nearest]),int(ys[nearest])], 'signature': signature,
                       'shape': hashlib.sha256(repr(signature).encode()).hexdigest()[:12],
                       'coords': set(zip(xs.tolist(), ys.tolist()))})
    return result


def public_component(c):
    return {k: c[k] for k in ('id', 'color', 'bounds', 'pixels', 'point', 'shape')}


class ObjectMemory:
    """IDs express pixel correspondences, not physical or semantic object identity."""
    def __init__(self):
        self.serial = self.epoch = 0
        self.current = None
        self.tracks = []
        self.events = deque(maxlen=8)
        self.layouts = deque(maxlen=32)
        self.actions = deque(maxlen=24)
        self.movers = set()
        self.effects = defaultdict(Counter)
        self.effect_steps = defaultdict(lambda: deque(maxlen=3))
        self.counts = Counter()

    def assign(self, c):
        self.serial += 1
        c['id'] = f'o{self.serial}'

    def initialize(self, obs):
        self.current = obs
        self.tracks = components(obs.grid)
        for c in self.tracks:
            self.assign(c)

    def view(self, limit=24):
        ranked = sorted(self.tracks, key=lambda c: (c['id'] not in self.movers, c['pixels'], c['id']))
        effects = []
        for key, counts in sorted(self.effects.items(), key=lambda pair: -sum(pair[1].values()))[:12]:
            action, shape = key
            effects.append({'action': action, 'shape': shape,
                            'observed_displacements': [{'dx': d[0], 'dy': d[1], 'count': n}
                                                       for d,n in counts.most_common(4)],
                            'evidence_steps': list(self.effect_steps[key])})
        recent = []
        for event in list(self.events)[-4:]:
            summary = {k: v for k,v in event.items() if k not in
                       ('unmatched_after', 'unmatched_before', 'repeated_control_cycle')}
            if 'translations' in summary:
                summary['omitted_translations'] = max(0,len(summary['translations'])-12)
                summary['translations'] = summary['translations'][:12]
            if event.get('repeated_control_cycle'):
                summary['repeated_control_period'] = event['repeated_control_cycle']['period']
            recent.append(summary)
        return {'epoch': self.epoch, 'components': [public_component(c) for c in ranked[:limit]],
                'omitted_components': max(0,len(ranked)-limit),
                'measured_transitions': recent, 'sampled_effects': effects}

    def update(self, before, action, after, step, prediction):
        if self.current is None:
            self.initialize(before)
        boundary = (action.id == 0 or before.levels != after.levels or before.actions != after.actions
                    or before.state != after.state or
                    (len(before.grid),len(before.grid[0])) != (len(after.grid),len(after.grid[0])))
        if boundary:
            self.epoch += 1
            self.events.clear(); self.layouts.clear(); self.actions.clear(); self.movers.clear()
            self.effects.clear(); self.effect_steps.clear()
            self.initialize(after)
            requested = prediction and prediction['kind'] != 'unknown'
            if requested: self.counts['prediction_unknown'] += 1
            event = {'step': step, 'prediction': prediction, 'boundary': True, 'levels_delta': after.levels-before.levels,
                     'prediction_check': 'unknown' if requested else 'not_requested', 'reason': 'context boundary; correspondence reset'}
            self.events.append(event)
            return event
        old, new = self.tracks, components(after.grid)
        groups_old, groups_new = defaultdict(list), defaultdict(list)
        for i,c in enumerate(old): groups_old[c['signature']].append(i)
        for i,c in enumerate(new): groups_new[c['signature']].append(i)
        matches = {}
        # Exact stationary shapes first. A unique remaining pair can translate.
        # Multiple residual candidates stay unmatched rather than nearest-neighbor guesses.
        for signature, old_indices in groups_old.items():
            remaining_old = set(old_indices)
            remaining_new = set(groups_new[signature])
            positions = {tuple(new[j]['bounds']):j for j in remaining_new}
            for i in old_indices:
                j = positions.get(tuple(old[i]['bounds']))
                if j is not None:
                    matches[i]=j; remaining_old.remove(i); remaining_new.remove(j)
            if len(remaining_old)==len(remaining_new)==1:
                matches[remaining_old.pop()]=remaining_new.pop()
        movement = []; matched_ids = {}; covered = set()
        for i,j in matches.items():
            a,b=old[i],new[j]; b['id']=a['id']
            dx,dy=b['bounds'][0]-a['bounds'][0],b['bounds'][1]-a['bounds'][1]
            matched_ids[a['id']] = (dx,dy)
            if dx or dy:
                self.movers.add(a['id']); covered.update(a['coords'] | b['coords'])
                movement.append({'id':a['id'],'color':a['color'],'pixels':a['pixels'],
                                 'from_bounds':a['bounds'],'to_bounds':b['bounds'],'dx':dx,'dy':dy})
            # Record stationary samples only for components with observed motion.
            if a['id'] in self.movers:
                key=(str(action.record()),a['shape'])
                self.effects[key][dx,dy]+=1; self.effect_steps[key].append(step)
        for c in new:
            if 'id' not in c: self.assign(c)
        changed={(x,y) for y,(ra,rb) in enumerate(zip(before.grid,after.grid))
                 for x,(a,b) in enumerate(zip(ra,rb)) if a!=b}
        check='not_requested'
        if prediction and prediction['kind']!='unknown':
            actual=matched_ids.get(prediction['target'])
            if actual is None: check='unknown'
            else:
                expected=(prediction['dx'],prediction['dy']) if prediction['kind']=='translation' else (0,0)
                check='matched' if actual==expected else 'contradicted'
            self.counts[f'prediction_{check}']+=1
        self.actions.append(action.record())
        cycle=None
        actions=list(self.actions)
        for period in range(1,min(8,len(actions)//3)+1):
            if actions[-period:]==actions[-2*period:-period]==actions[-3*period:-2*period]:
                cycle={'period':period,'last_three_repeats':actions[-3*period:]}; break
        layout=tuple(sorted((c['id'],tuple(c['bounds'])) for c in new if c['id'] in self.movers))
        returned=None
        if layout:
            for k,(old_step,prior_layout,grid) in enumerate(self.layouts):
                if prior_layout==layout and any(p!=layout for _,p,_ in list(self.layouts)[k+1:]):
                    residual=sum(a!=b for ra,rb in zip(grid,after.grid) for a,b in zip(ra,rb))
                    returned={'prior_step':old_step,'tracked_ids':[x[0] for x in layout],
                              'full_frame_pixel_difference':residual}; break
        self.layouts.append((step,layout,after.grid))
        event={'step':step,'action':action.record(),'levels_delta':after.levels-before.levels,
               'translations':movement,'stationary_correspondences':len(matches)-len(movement),
               'unmatched_before':[c['id'] for i,c in enumerate(old) if i not in matches][:24],
               'unmatched_after':[public_component(c) for j,c in enumerate(new) if j not in matches.values()][:12],
               'unmatched_before_count':len(old)-len(matches),'unmatched_after_count':len(new)-len(matches),
               'changed_pixels':len(changed),'changed_pixels_outside_translated_components':len(changed-covered),
               'prediction':prediction,'prediction_check':check,
               'repeated_control_cycle':cycle,'tracked_layout_return':returned}
        self.counts['translations']+=len(movement)
        self.counts['control_cycle_warnings']+=int(cycle is not None)
        self.counts['layout_returns']+=int(returned is not None)
        self.events.append(event); self.current=after; self.tracks=new
        return event
