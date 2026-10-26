"""Measured rigid-region effects with explicit correspondence ambiguity."""
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from functools import lru_cache

from arc_harness.object_memory import components

OPS = tuple(range(8))  # Reflect x for 4..7, then apply quarter turns in screen coordinates.
MAX_PIXELS = 1024
MAX_COMPONENTS = 256


def normalize(coords):
    x0=min(x for x,y in coords);y0=min(y for x,y in coords)
    return tuple(sorted((x-x0,y-y0) for x,y in coords))


@lru_cache(maxsize=2048)
def transform(shape, op):
    coords=[(-x if op>=4 else x,y) for x,y in shape]
    for _ in range(op%4):
        coords=[(-y,x) for x,y in coords]
    return normalize(coords)


@lru_cache(maxsize=2048)
def canonical(shape, mode):
    return min(transform(shape,op) for op in (OPS if mode=='d4' else (0,)))


@lru_cache(maxsize=8)
def _described(grid):
    parts=components(grid)
    eligible=[c for c in parts if c['pixels']<=MAX_PIXELS]
    eligible.sort(key=lambda c:(c['pixels'],c['color'],c['bounds']))
    kept=eligible[:MAX_COMPONENTS]
    for c in kept:
        c['normalized']=normalize(c['coords'])
    return tuple(kept),len(parts)-len(kept)


def describe(grid):
    parts,omitted=_described(grid)
    return [dict(c) for c in parts],omitted


def correspondence(old,new,mode):
    """Exact stationary components, then unique residual pairs; never nearest guesses."""
    groups_old=defaultdict(list);groups_new=defaultdict(list)
    for i,c in enumerate(old):groups_old[c['color'],canonical(c['normalized'],mode)].append(i)
    for j,c in enumerate(new):groups_new[c['color'],canonical(c['normalized'],mode)].append(j)
    matches={};ambiguous=0
    for group,indices in groups_old.items():
        left=set(indices);right=set(groups_new[group])
        positions={frozenset(new[j]['coords']):j for j in right}
        for i in indices:
            j=positions.get(frozenset(old[i]['coords']))
            if j is not None:
                matches[i]=j;left.remove(i);right.remove(j)
        if len(left)==len(right)==1:
            matches[left.pop()]=right.pop()
        elif left and right:
            ambiguous+=len(left)
    return matches,ambiguous


def project(c,op,dx2,dy2):
    shape=transform(c['normalized'],op)
    width=max(x for x,y in shape)+1;height=max(y for x,y in shape)+1
    x0,y0,x1,y1=c['bounds']
    tx2=x0+x1+dx2-width+1;ty2=y0+y1+dy2-height+1
    if tx2%2 or ty2%2:return None
    return frozenset((x+tx2//2,y+ty2//2) for x,y in shape)


def explanations(a,b,mode):
    ax0,ay0,ax1,ay1=a['bounds'];bx0,by0,bx1,by1=b['bounds']
    dx2=bx0+bx1-ax0-ax1;dy2=by0+by1-ay0-ay1
    return tuple((op,dx2,dy2) for op in (OPS if mode=='d4' else (0,))
                 if project(a,op,dx2,dy2)==b['coords'])


def action_key(action):
    return action.id,action.x,action.y


@dataclass
class Evidence:
    samples:int=0
    counts:Counter=field(default_factory=Counter)
    steps:dict=field(default_factory=lambda:defaultdict(lambda:deque(maxlen=8)))

    def add(self,modes,step):
        self.samples+=1
        for mode in modes:
            if mode not in self.counts and len(self.counts)>=128:
                weakest=min(self.counts,key=lambda m:(self.counts[m],m))
                del self.counts[weakest]
                self.steps.pop(weakest,None)
            self.counts[mode]+=1
            self.steps[mode].append(step)

    def predict(self,c):
        if self.samples<3:return None,'insufficient_samples'
        modes=[mode for mode,count in self.counts.items() if count/self.samples>=.8]
        if not modes:return None,'contradictory_calibration'
        variants={project(c,*mode) for mode in modes}
        if None in variants or len(variants)!=1:return None,'ambiguous_prediction'
        coords=variants.pop()
        return {'coords':coords,'modes':sorted(modes),'samples':self.samples,
                'support':min(self.counts[m] for m in modes),
                'evidence_steps':sorted(set(s for m in modes for s in self.steps[m])),
                'nonidentity':coords!=c['coords']},None


class RigidEffectMemory:
    """Frozen effect parameters with causal, observation-only identity bookkeeping.

    Evaluation can update correspondence from an already returned observation for
    the next decision, but cannot add effect samples. Lost identities discard their
    effect evidence. Context boundaries discard all calibration. Pixel agreement
    checks a visual prediction, not physical identity, controllability or a goal.
    At most8192 active effect keys retain128 operation modes each and eight recent
    support steps per mode. Pruning never reduces the total sample denominator.
    """
    effect_limit=8192

    def __init__(self,mode):
        assert mode in ('exact','d4')
        self.mode=mode;self.serial=0;self.current=None;self.tracks=[];self.effects={}
        self.counts=Counter()

    def initialize(self,obs):
        self.current=obs;self.effects.clear()
        self.tracks,omitted=describe(obs.grid)
        self.counts['omitted_components']+=omitted
        for c in self.tracks:
            self.serial+=1;c['id']=self.serial

    def predict(self,before,action):
        if self.current is None:self.initialize(before)
        assert self.current==before,'Identity bookkeeping must precede the next prediction.'
        predictions=[];abstentions=Counter()
        if action.id==0 or before.state!='NOT_FINISHED':return predictions,abstentions
        for c in self.tracks:
            evidence=self.effects.get((c['id'],action_key(action)))
            if evidence is None:
                abstentions['no_calibration']+=1;continue
            prediction,reason=evidence.predict(c)
            if prediction:
                predictions.append({'id':c['id'],'color':c['color'],
                                    'before_coords':frozenset(c['coords']),**prediction})
            else:abstentions[reason]+=1
        return predictions,abstentions

    def observe(self,before,action,after,step,learn):
        if self.current is None:self.initialize(before)
        assert self.current==before
        boundary=(action.id==0 or before.levels!=after.levels or before.state!=after.state or
                  before.actions!=after.actions or len(before.grid)!=len(after.grid) or
                  len(before.grid[0])!=len(after.grid[0]))
        if boundary:
            self.initialize(after);self.counts['boundaries']+=1
            return {'boundary':True,'matches':0,'unmatched':0,'ambiguous':0}
        new,omitted=describe(after.grid);self.counts['omitted_components']+=omitted
        matches,ambiguous=correspondence(self.tracks,new,self.mode)
        for i,c in enumerate(self.tracks):
            j=matches.get(i)
            if learn:
                key=(c['id'],action_key(action))
                if key not in self.effects and len(self.effects)>=self.effect_limit:
                    self.effects.pop(next(iter(self.effects)))
                    self.counts['effect_key_evictions']+=1
                evidence=self.effects.setdefault(key,Evidence())
                modes=explanations(c,new[j],self.mode) if j is not None else ()
                evidence.add(modes,step)
                self.counts['calibration_samples']+=1
                self.counts['unexplained_calibration']+=not bool(modes)
            if j is not None:new[j]['id']=c['id']
        for c in new:
            if 'id' not in c:self.serial+=1;c['id']=self.serial
        live={c['id'] for c in new}
        self.effects={key:value for key,value in self.effects.items() if key[0] in live}
        self.counts['ambiguous_correspondences']+=ambiguous
        event={'boundary':False,'matches':len(matches),'unmatched':len(self.tracks)-len(matches),'ambiguous':ambiguous}
        self.tracks=new;self.current=after
        return event
