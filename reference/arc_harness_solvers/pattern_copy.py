"""Infer miniature-to-board palette mappings from repeated public demonstrations."""
from collections import Counter, defaultdict
import numpy as np

from arc_harness.agents import BalancedDirectedAgent
from arc_harness.core import Action
from arc_harness.scene import regions


def transforms(grid):
    a=np.asarray(grid)
    return [tuple(tuple(int(v) for v in row) for row in np.rot90(a if flip==0 else np.fliplr(a),turn))
            for flip in range(2) for turn in range(4)]


def panels(grid):
    """Find spaced majority-color square lattices with a block-uniform miniature."""
    rectangles=defaultdict(dict)
    for color,x,y,x1,y1,area in regions(grid)[0]:
        w,h=x1-x+1,y1-y+1
        if w==h and 3<=w<=15 and area>w*h/2:
            rectangles[w][x,y]=color
    found=[]
    for size,tiles in rectangles.items():
        for n in (3,5):
            if size%n:continue
            pitches={xx-x for x,y in tiles for xx,yy in tiles if yy==y and size<xx-x<=3*size}
            for pitch in sorted(pitches):
                origins={(x-j*pitch,y-i*pitch) for x,y in tiles for i in range(n) for j in range(n)}
                for x,y in sorted(origins,key=lambda p:(p[1],p[0])):
                    coords=[(x+j*pitch,y+i*pitch) for i in range(n) for j in range(n)]
                    center=n*n//2
                    if sum(p in tiles for k,p in enumerate(coords) if k!=center)<max(2,n*n//4):continue
                    cx,cy=coords[center]
                    if cy<0 or cx<0 or cy+size>len(grid) or cx+size>len(grid[0]):continue
                    unit=size//n; mini=[]; valid=True
                    for i in range(n):
                        row=[]
                        for j in range(n):
                            colors={grid[yy][xx] for yy in range(cy+i*unit,cy+(i+1)*unit)
                                    for xx in range(cx+j*unit,cx+(j+1)*unit)}
                            if len(colors)!=1:valid=False;break
                            row.append(next(iter(colors)))
                        if not valid:break
                        mini.append(tuple(row))
                    if not valid or len({v for row in mini for v in row})<2:continue
                    found.append({'origin':(x,y),'size':size,'pitch':pitch,'n':n,'coords':coords,
                                  'mini':tuple(mini),'colors':tuple(tiles.get(p) for p in coords)})
    # A lattice also admits every second/third slot. Retain its smallest observed pitch.
    unique={}
    for board in found:
        key=(board['coords'][board['n']**2//2],board['size'],board['n'])
        if key not in unique or board['pitch']<unique[key]['pitch']:unique[key]=board
    minima={(b['size'],b['n']):min(p['pitch'] for p in unique.values() if (p['size'],p['n'])==(b['size'],b['n'])) for b in unique.values()}
    return [b for b in unique.values() if b['pitch']==minima[b['size'],b['n']]]


def proposals(grid, learned=()):
    boards=panels(grid); hypotheses={}
    tile_palette={c for b in boards for c in b['colors'] if c is not None}
    tile_palette.update(b['mini'][b['n']//2][b['n']//2] for b in boards)
    for rule in learned:
        for b in boards:
            if b['n']!=rule['n']:continue
            n=b['n'];center=n*n//2
            mini=transforms(b['mini'])[rule['transform']]
            paint=b['mini'][n//2][n//2]
            others=tile_palette-{paint}
            if len(others)!=1:continue
            other=next(iter(others))
            mapping={token:paint if role=='center' else other
                     for token,role in rule['roles'].items()}
            small=[v for row in mini for v in row]
            if any(v not in mapping for k,v in enumerate(small) if k!=center):continue
            desired=tuple(None if k==center else mapping[v] for k,v in enumerate(small))
            mismatch=[k for k in range(n*n) if k!=center and b['colors'][k]!=desired[k]]
            if not mismatch:continue
            yield {'key':(b['origin'],b['size'],b['pitch'],desired),'board':b,'desired':desired,
                   'mapping':mapping,'transform':rule['transform'],'support':[],
                   'transfer_evidence':rule['evidence'],'priority':(1,0,-len(mismatch))}
    for source in boards:
        n=source['n']; center=n*n//2
        if any(c is None for k,c in enumerate(source['colors']) if k!=center):continue
        for t,mini in enumerate(transforms(source['mini'])):
            mapping={};valid=True
            for k,value in enumerate(v for row in mini for v in row):
                if k==center:continue
                color=source['colors'][k]
                if value in mapping and mapping[value]!=color:valid=False;break
                mapping[value]=color
            if not valid or len(set(mapping.values()))<2:continue
            key=(n,t,tuple(sorted(mapping.items())))
            if key in hypotheses:continue
            support=[];targets=[]
            for b in boards:
                if b['n']!=n or any(c is None for k,c in enumerate(b['colors']) if k!=center):continue
                small=[v for row in transforms(b['mini'])[t] for v in row]
                if any(v not in mapping for k,v in enumerate(small) if k!=center):continue
                desired=tuple(None if k==center else mapping[v] for k,v in enumerate(small))
                mismatch=[k for k in range(n*n) if k!=center and b['colors'][k]!=desired[k]]
                if not mismatch:support.append(b['origin'])
                else:targets.append((b,desired,mismatch))
            if len(support)<2:continue
            hypotheses[key]=True
            for b,desired,mismatch in targets:
                # A blank-looking board is an exploration prior, not proof of editability.
                uniform=max(Counter(c for c in b['colors'] if c is not None).values())
                targets_key=(b['origin'],b['size'],b['pitch'],desired)
                yield {'key':targets_key,'board':b,'desired':desired,'mapping':mapping,'transform':t,
                       'support':support,'priority':(len(support),uniform,-len(mismatch))}


def binary_plan(requirements, values, effects):
    """Solve observed binary action effects against equal/not-equal goal predicates."""
    palette=set(values.values())
    if len(palette)!=2:return None
    buttons=sorted(effects);rows=[]
    for point,allowed in sorted(requirements.items()):
        targets=palette&allowed
        if not targets:return None
        if len(targets)==2:continue
        mask=sum(1<<i for i,button in enumerate(buttons) if point in effects[button])
        rows.append([mask,int(values[point] not in targets)])
    pivots=[];rank=0
    for col in range(len(buttons)):
        index=next((i for i in range(rank,len(rows)) if rows[i][0]>>col&1),None)
        if index is None:continue
        rows[rank],rows[index]=rows[index],rows[rank]
        for i in range(len(rows)):
            if i!=rank and rows[i][0]>>col&1:
                rows[i][0]^=rows[rank][0];rows[i][1]^=rows[rank][1]
        pivots.append(col);rank+=1
    if any(mask==0 and rhs for mask,rhs in rows):return None
    return [buttons[col] for i,col in enumerate(pivots) if rows[i][1]]


class PatternCopyAgent:
    """Bounded pattern hypothesis tests; every click is checked on the next frame."""
    def __init__(self,seed):
        self.fallback=BalancedDirectedAgent(seed);self.level=None;self.current=None;self.tried=set()
        self.attempts=Counter();self.steps=0;self.pending=None;self.latest_feedback=None
        self.counts=Counter();self.successful=[];self.learned=[];self.cycles={};self.constraint_failed=False
        self.variables=();self.action_effects={};self.coupled=False

    def constraint_action(self,obs):
        """Intersect learned equal/not-equal predicates on overlapping board cells."""
        rule=self.learned[-1];requirements={};boards=panels(obs.grid)
        selected=[]
        for b in boards:
            if b['n']!=rule['n']:continue
            miniature=[v for row in transforms(b['mini'])[rule['transform']] for v in row]
            if all(miniature[k] in rule['roles'] for k,c in enumerate(b['colors']) if k!=b['n']**2//2 and c is not None):selected.append(b)
        clue_origins={b['coords'][b['n']**2//2] for b in selected}
        for b in selected:
            colors=list(b['colors'])
            for k,(x,y) in enumerate(b['coords']):
                if colors[k] is not None or (x,y) in clue_origins:continue
                size=b['size']
                if x<0 or y<0 or x+size>len(obs.grid[0]) or y+size>len(obs.grid):continue
                counts=Counter(v for row in obs.grid[y:y+size] for v in row[x:x+size])
                value,count=counts.most_common(1)[0]
                if 1<len(counts)<=2 and count>size*size/2:colors[k]=value
            b['colors']=tuple(colors)
        if not selected:return None
        for b in selected:
            n=b['n'];paint=b['mini'][n//2][n//2]
            miniature=[v for row in transforms(b['mini'])[rule['transform']] for v in row]
            if any(miniature[k] not in rule['roles'] for k,c in enumerate(b['colors']) if k!=n*n//2 and c is not None):continue
            for k,(x,y) in enumerate(b['coords']):
                if k==n*n//2 or b['colors'][k] is None:continue
                token=miniature[k]
                if token not in rule['roles']:return None
                allowed={paint} if rule['roles'][token]=='center' else set(range(16))-{paint}
                point=(x+b['size']//2,y+b['size']//2)
                requirements[point]=requirements.get(point,set(range(16)))&allowed
                if not requirements[point]:return None
        self.variables=tuple(sorted(requirements))
        values={p:obs.grid[p[1]][p[0]] for p in self.variables}
        signature=tuple((p,tuple(sorted(v))) for p,v in sorted(requirements.items()))
        if self.coupled:
            unknown=[p for p in self.variables if p not in self.action_effects]
            if unknown:
                point=unknown[0];reason='learn_coupled_tile_effect'
            else:
                plan=binary_plan(requirements,values,self.action_effects)
                if not plan:
                    self.constraint_failed=True;self.counts['coupled_model_unsolved']+=int(plan is None);return None
                point=plan[0];reason='solve_observed_binary_effects'
            self.pending={'point':point,'before':values[point],'predicted':None,
                          'allowed':sorted(requirements[point]),'hypothesis':signature}
            self.counts[reason]+=1
            return Action(6,*point),{'reason':reason,'known_operators':len(self.action_effects),
                'goal_constraints':len(requirements),'expected_changed_points':self.action_effects.get(point)}
        for point,allowed in sorted(requirements.items(),key=lambda item:(len(item[1]),item[0][1],item[0][0])):
            x,y=point;color=obs.grid[y][x]
            if color in allowed:continue
            key=(signature,point,color)
            if self.attempts[key]:
                self.constraint_failed=True;self.counts['constraint_cycle_rejections']+=1;return None
            self.attempts[key]+=1
            self.pending={'point':point,'before':color,'predicted':self.cycles.get((obs.levels,point,color)),
                          'allowed':sorted(allowed),'hypothesis':signature}
            self.counts['constraint_clicks']+=1
            return Action(6,*point),{'reason':'satisfy_shared_pattern_constraints',
                 'point':point,'allowed_colors':sorted(allowed),'rule':rule,
                 'contributing_boards':[b['origin'] for b in selected],
                 'predicted_click_color':self.pending['predicted']}
        self.constraint_failed=True
        self.counts['all_constraints_satisfied_without_progress']+=1
        return None

    def choose(self,obs):
        if self.level!=obs.levels:
            self.level=obs.levels;self.current=None;self.tried.clear();self.attempts.clear();self.constraint_failed=False;self.variables=();self.action_effects={};self.coupled=False
        self.pending=None
        if obs.state!='NOT_FINISHED' or 6 not in obs.actions:
            return self.fallback.choose(obs)
        if self.learned and not self.constraint_failed:
            choice=self.constraint_action(obs)
            if choice is not None:return choice
        if self.current is None:
            options=[p for p in proposals(obs.grid) if p['key'] not in self.tried]
            if options:
                self.current=max(options,key=lambda p:p['priority'])
                self.counts['hypotheses']+=1
        if self.current:
            p=self.current;b=p['board'];center=b['n']**2//2
            for k,(x,y) in enumerate(b['coords']):
                if k==center:continue
                point=(x+b['size']//2,y+b['size']//2); color=obs.grid[point[1]][point[0]]
                desired=p['desired'][k]
                if color==desired:continue
                attempt=(p['key'],k,color)
                if self.attempts[attempt]:
                    self.tried.add(p['key']);self.current=None;self.counts['rejected_cycles']+=1
                    return self.fallback.choose(obs)
                self.attempts[attempt]+=1
                self.pending={'point':point,'before':color,'predicted':desired,'hypothesis':p['key']}
                self.counts['pattern_clicks']+=1
                return Action(6,*point),{'reason':'test_miniature_palette_rule','goal':list(p['desired']),
                    'mapping':p['mapping'],'transform':p['transform'],'demonstration_origins':p['support'],
                    'board_origin':b['origin'],'transfer_evidence':p.get('transfer_evidence'),'prediction':self.pending}
            # Pattern equality is not success. One central click tests explicit confirmation.
            self.tried.add(p['key']);self.current=None
            x,y=b['coords'][center]
            self.counts['confirmation_tests']+=1
            return Action(6,x+b['size']//2,y+b['size']//2),{'reason':'pattern_equal_confirmation_test'}
        return self.fallback.choose(obs)

    def observe(self,before,action,after):
        self.steps+=1;self.fallback.observe(before,action,after)
        self.latest_feedback={'pattern':None}
        if self.pending:
            p=self.pending;x,y=p['point']; actual=after.grid[y][x]
            progress=after.levels>before.levels or after.state=='WIN'
            matched=actual==p['predicted'] if not progress and p['predicted'] is not None else None
            if not progress:
                self.cycles[before.levels,p['point'],p['before']]=actual
                if self.variables and 'allowed' in p:
                    changed=tuple(q for q in self.variables if before.grid[q[1]][q[0]]!=after.grid[q[1]][q[0]])
                    prior=self.action_effects.get(p['point'])
                    if prior is not None and prior!=changed:
                        self.constraint_failed=True;self.counts['operator_contradictions']+=1
                    self.action_effects[p['point']]=changed
                    self.coupled|=len(changed)>1
            self.latest_feedback={'pattern':{'step':self.steps,'predicted':p['predicted'],
                'actual':actual,'prediction_match':matched,'level_progress':progress,
                'goal_predicate_satisfied':actual in p['allowed'] if 'allowed' in p and not progress else None}}
            self.counts['progress']+=int(progress)
            self.counts['prediction_matches']+=int(matched is True)
            self.counts['prediction_mismatches']+=int(matched is False)
            if progress:
                self.successful.append({'level':before.levels,'step':self.steps,'hypothesis':p['hypothesis']})
                if self.current:
                    chosen=self.current;b=chosen['board'];n=b['n'];paint=b['mini'][n//2][n//2]
                    rule={'n':n,'transform':chosen['transform'],'mapping':chosen['mapping'],
                          'roles':{token:'center' if color==paint else 'not_center' for token,color in chosen['mapping'].items()},
                          'evidence':{'level':before.levels,'progress_step':self.steps}}
                    if 'center' in rule['roles'].values():self.learned=[rule]

            elif actual==p['before']:
                self.tried.add(p['hypothesis']);self.current=None;self.counts['no_effect']+=1
                if 'allowed' in p:self.constraint_failed=True
        if after.state=='GAME_OVER':self.current=None
        self.pending=None

    def diagnostics(self):
        return {'steps':self.steps,'counts':dict(self.counts),'successful_hypotheses':self.successful,
                'learned_rules':self.learned,'scope':'Public miniature/board correlations; level progress confirms individual tests only.'}
