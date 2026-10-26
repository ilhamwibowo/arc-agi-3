"""Bounded multi-piece goal tests using public pixels and within-run control evidence."""
from collections import Counter, defaultdict, deque

from .anchor_agent import AnchorSelectiveAgent
from .assembly_geometry import solve, validate, turn
from .assembly_route import route
from .core import Action
from .object_memory import components
from .visual_alignment import foreground_groups, paint, pixels_match


def normalized_paint(colored, op=0):
    pixels=[(c,*turn(x,y,op)) for c,x,y in colored]
    x0=min(x for c,x,y in pixels);y0=min(y for c,x,y in pixels)
    return frozenset((c,x-x0,y-y0) for c,x,y in pixels)


def role_paint(colored, marker):
    return frozenset((int(c==marker),x,y) for c,x,y in colored)


def body_color(colored, marker):
    colors={c for c,x,y in colored if c!=marker}
    return next(iter(colors)) if len(colors)==1 else None


def recolor(colored, marker, color):
    return frozenset((marker if c==marker else color,x,y) for c,x,y in colored)


def effect(before, after):
    """Translations include measured displacement; rotations specify type, not pivot."""
    modes=[op for op in range(4) if normalized_paint(before,op)==normalized_paint(after)]
    if 0 in modes:
        return ('shift',min(x for c,x,y in after)-min(x for c,x,y in before),
                        min(y for c,x,y in after)-min(y for c,x,y in before))
    if len(modes)==1:return ('turn',modes[0])
    return None


class CompositeEffects:
    def __init__(self):
        self.samples=Counter();self.modes=defaultdict(Counter);self.steps=defaultdict(lambda:deque(maxlen=8))

    def add(self,kind,before,after,step):
        mode=effect(before,after)
        self.samples[kind]+=1
        if mode is not None:self.modes[kind][mode]+=1
        self.steps[kind].append(step)
        return mode

    def model(self,kind):
        if self.samples[kind]<3:return None
        supported=[m for m,n in self.modes[kind].items() if n/self.samples[kind]>=.8]
        return supported[0] if len(supported)==1 else None

    def translations(self):
        return {k:m[1:] for k in self.samples if (m:=self.model(k)) and m[0]=='shift' and m[1:]!=(0,0)}

    def rotations(self):
        return {k:m[1] for k in self.samples if (m:=self.model(k)) and m[0]=='turn'}

    def record(self):
        return {str(k):{'samples':self.samples[k],'model':self.model(k),'recent_steps':list(self.steps[k]),
                        'modes':[{'mode':m,'count':n} for m,n in self.modes[k].items()]} for k in self.samples}


def scene_match(grid,pieces,marker,contact_color=None,learn=False):
    """Check latent piece ownership, allowing one measured marker-contact display."""
    expected=defaultdict(list)
    for piece in pieces:
        for c,x,y in piece:expected[x,y].append(c)
    changed=set();background=Counter(c for row in grid for c in row).most_common(1)[0][0]
    body_colors={c for p in pieces for c,x,y in p if c!=marker}
    for (x,y),owners in expected.items():
        if not(0<=y<len(grid) and 0<=x<len(grid[0])):return False,None
        actual=grid[y][x]
        if len(owners)==1:
            if actual!=owners[0]:return False,None
        elif len(owners)==2 and owners==[marker,marker]:
            if actual!=marker and actual!=contact_color:changed.add(actual)
        else:return False,None
    if not changed:return True,contact_color
    if not learn or contact_color is not None or len(changed)!=1:return False,None
    inferred=next(iter(changed))
    if inferred==background or inferred in body_colors:return False,None
    return True,inferred


def locate_piece(grid,old,marker,focus,allowed_ops,others=(),contact_color=None,learn=False):
    """Find one complete rigid template through its uniquely matching body region."""
    answers=set()
    for op in allowed_ops:
        template=normalized_paint(old,op)
        body={(x,y) for c,x,y in template if c!=marker}
        bx=min(x for x,y in body);by=min(y for x,y in body)
        normalized_body={(x-bx,y-by) for x,y in body}
        for region in components(grid):
            if region['color']!=focus or region['pixels']!=len(body):continue
            rx,ry,_,_=region['bounds']
            if {(x-rx,y-ry) for x,y in region['coords']}!=normalized_body:continue
            candidate=frozenset((c,x+rx-bx,y+ry-by) for c,x,y in template)
            if scene_match(grid,[candidate,*others],marker,contact_color,learn)[0]:answers.add(candidate)
    return next(iter(answers)) if len(answers)==1 else None


class AssemblyAttempt:
    action_limit=128

    def __init__(self,obs,marker,focus,effects):
        self.marker=marker;self.focus=focus;self.effects=effects;self.level=obs.levels
        self.actions=0;self.pending=None;self.work=None;self.probes=Counter();self.feedback=None
        self.contact_color=None
        self.status='active';self.counts=Counter();self.geometry=None;self.targets=[];self.pieces=[]
        groups=[g for g in foreground_groups(obs.grid) if marker in g['colors']]
        self.scene={'marker_groups':len(groups),'palette_areas':[
            {str(c):len(coords) for c,coords in g['colors'].items()} for g in groups]}
        if not 2<=len(groups)<=8 or any(len(g['colors'])!=2 for g in groups):
            self.status='ambiguous_scene';return
        self.pieces=[paint(g) for g in groups]
        selected=[i for i,p in enumerate(self.pieces) if body_color(p,marker)==focus]
        if len(selected)!=1:self.status='ambiguous_focus';return
        self.selected=selected[0]
        allowed={0}
        while True:
            new=allowed|{(a+b)%4 for a in allowed for b in effects.rotations().values()}
            if new==allowed:break
            allowed=new
        self.geometry=solve(self.pieces,marker,len(obs.grid[0]),len(obs.grid),allowed_ops=sorted(allowed),
                            max_nodes=20000,max_seconds=1)
        if self.geometry['status']!='candidate':self.status='geometry_'+self.geometry['status'];return
        check=validate(self.pieces,marker,len(obs.grid[0]),len(obs.grid),self.geometry['solution'],sorted(allowed))
        if not check['valid']:self.status='invalid_geometry';return
        self.targets=[frozenset(map(tuple,p['colored_pixels'])) for p in check['rendered']]
        self.validation=check['checks']

    def done(self,i):
        return role_paint(self.pieces[i],self.marker)==role_paint(self.targets[i],self.marker)

    def stop(self,reason):
        self.status=reason;self.pending=None;self.counts[reason]+=1

    def choose(self,obs):
        if self.status!='active':return None
        if self.actions>=self.action_limit:self.stop('action_budget');return None
        if obs.levels!=self.level or obs.state!='NOT_FINISHED':self.stop('context_changed');return None
        if not scene_match(obs.grid,self.pieces,self.marker,self.contact_color)[0]:self.stop('scene_mismatch');return None
        unfinished=[i for i in range(len(self.pieces)) if not self.done(i)]
        if not unfinished:self.stop('layout_without_progress');return None
        if self.work not in unfinished:
            self.work=self.selected if self.selected in unfinished else unfinished[0]
        i=self.work;old=self.pieces[i];target=recolor(self.targets[i],self.marker,body_color(old,self.marker))
        detail={'piece':i,'level':self.level,'attempt_action':self.actions+1,'marker_color':self.marker,
                'focus_color':self.focus,'warning':'Selection, transferred controls and complete arrangement are hypotheses.'}
        if self.actions==0:
            detail['assembly_entry']={'solution':self.geometry['solution'],'checks':self.validation,
                'scene':self.scene,'controls':self.effects.record()}
        if i!=self.selected:
            if 6 not in obs.actions:self.stop('selection_unavailable');return None
            body={(x,y) for c,x,y in old if c!=self.marker}
            cx=sum(x for x,y in body)/len(body);cy=sum(y for x,y in body)/len(body)
            x,y=min(body,key=lambda p:(-sum(q in body for q in ((p[0]-1,p[1]),(p[0]+1,p[1]),(p[0],p[1]-1),(p[0],p[1]+1))),abs(p[0]-cx)+abs(p[1]-cy),p))
            self.pending={'phase':'select','piece':i}
            return Action(6,x,y),{'reason':'assembly_select',**detail}
        if normalized_paint(old)!=normalized_paint(target):
            options=[]
            for kind,op in self.effects.rotations().items():
                if kind not in obs.actions:continue
                for times in range(1,4):
                    if normalized_paint(old,(op*times)%4)==normalized_paint(target):options.append((times,kind,op))
            if not options:self.stop('rotation_unsupported');return None
            _,kind,op=min(options)
            self.pending={'phase':'rotate','piece':i,'op':op}
            return Action(kind),{'reason':'assembly_rotation_probe','expected_quarter_turn':op,**detail}
        controls={k:v for k,v in self.effects.translations().items() if k in obs.actions}
        path=route(old,target,[p for j,p in enumerate(self.pieces) if i!=j],self.marker,controls,
                   len(obs.grid[0]),len(obs.grid),max_nodes=4096,max_steps=64,max_seconds=1)
        self.counts['route_expansions']+=path.get('expanded',0)
        if path['status']=='candidate' and path['route']:
            move=path['route'][0];self.pending={'phase':'move','piece':i,'expected':move['paint']}
            return Action(move['action']),{'reason':'assembly_translation','vector':controls[move['action']],
                                           'route_actions':len(path['route']),**detail}
        # Unknown buttons are tested directly; their meanings are never guessed from IDs.
        unknown=[k for k in obs.actions if k not in (0,6) and self.effects.model(k) is None and self.probes[k]<6]
        if unknown:
            kind=min(unknown,key=lambda k:(self.probes[k],k));self.pending={'phase':'probe','piece':i}
            return Action(kind),{'reason':'assembly_missing_control_probe','route_status':path['status'],**detail}
        self.stop('route_'+path['status']);return None

    def observe(self,before,action,after,step):
        pending=self.pending
        if pending is None:return
        self.actions+=1;phase=pending['phase'];i=pending['piece'];old=self.pieces[i]
        self.counts[phase+'_actions']+=1
        feedback={'phase':phase,'piece':i,'step':step}
        if after.levels>before.levels or after.state=='WIN':
            self.stop('progress');feedback['outcome']='progress';self.feedback=feedback;return
        if after.state!='NOT_FINISHED' or before.actions!=after.actions or len(before.grid)!=len(after.grid) or len(before.grid[0])!=len(after.grid[0]):
            self.stop('context_changed');feedback['outcome']=self.status;self.feedback=feedback;return
        if phase=='select':
            updated=[];valid=True
            background=Counter(c for row in before.grid for c in row).most_common(1)[0][0]
            for j,piece in enumerate(self.pieces):
                body={(x,y) for c,x,y in piece if c!=self.marker}
                colors={after.grid[y][x] for x,y in body}
                if len(colors)!=1:valid=False;break
                color=next(iter(colors));candidate=recolor(piece,self.marker,color)
                if color in (self.marker,background):valid=False;break
                if j==i:valid&=color==self.focus
                elif j==self.selected:valid&=color!=self.focus
                else:valid&=candidate==piece
                updated.append(candidate)
            if valid and len(updated)==len(self.pieces):
                valid=locate_piece(after.grid,updated[i],self.marker,self.focus,(0,),
                    [p for j,p in enumerate(updated) if j!=i],self.contact_color)==updated[i]
            if valid and len(updated)==len(self.pieces):
                self.pieces=updated;self.selected=i;feedback['outcome']='focus_transfer_matched'
            else:self.stop('selection_mismatch');feedback['outcome']=self.status
        else:
            others=[p for j,p in enumerate(self.pieces) if j!=i]
            if phase=='move':
                found=locate_piece(after.grid,old,self.marker,self.focus,(0,),others,self.contact_color,learn=True)
                candidate=found if found==pending['expected'] else None
            else:
                ops=(pending['op'],) if phase=='rotate' else range(4)
                candidate=locate_piece(after.grid,old,self.marker,self.focus,ops,others,self.contact_color)
            if candidate is None:self.stop('piece_or_reference_mismatch')
            else:
                _,contact=scene_match(after.grid,[candidate,*others],self.marker,self.contact_color,learn=phase=='move')
                if contact is not None and contact!=self.contact_color:
                    feedback['contact_display_inferred']={'color':contact,'step':step}
                    self.contact_color=contact;self.counts['contact_display_inferences']+=1
                measured=self.effects.add(action.id,old,candidate,step)
                self.pieces[i]=candidate;feedback['measured_effect']=measured
                if phase=='probe':self.probes[action.id]+=1
                feedback['outcome']='piece_prediction_matched' if phase=='move' else 'piece_observed'
                if 'contact_display_inferred' in feedback:feedback['outcome']='contact_display_hypothesis'
            if self.status!='active':feedback['outcome']=self.status
        self.pending=None;self.feedback=feedback

    def record(self):
        return {'status':self.status,'actions':self.actions,'counts':dict(self.counts),
                'level':self.level,'geometry':self.geometry,'validation':getattr(self,'validation',None),
                'contact_color':self.contact_color,'scene':self.scene,'controls':self.effects.record()}


class AssemblyControlAgent(AnchorSelectiveAgent):
    """One finite arrangement test per level, entered only after observed goal progress."""
    def __init__(self,seed):
        super().__init__(seed)
        self.history=deque(maxlen=64);self.role=None;self.composite_effects=None
        self.assembly=None;self.attempted_levels=set();self.assembly_counts=Counter();self.assembly_pending=False

    def choose(self,obs):
        self.assembly_pending=False
        if self.role and obs.state=='NOT_FINISHED' and obs.levels not in self.attempted_levels:
            # A bounded count is sufficient for the public maximum-sized observations;
            # overflow disables additional tests rather than forgetting failed attempts.
            if len(self.attempted_levels)<64:
                self.attempted_levels.add(obs.levels)
                self.assembly=AssemblyAttempt(obs,*self.role,self.composite_effects)
                self.assembly_counts['attempts']+=1
        if self.assembly and self.assembly.status=='active':
            proposed=self.assembly.choose(obs)
            if proposed:
                self.assembly_pending=True;self.pending={'phase':'assembly'}
                return proposed
        return super().choose(obs)

    def observe(self,before,action,after):
        if self.pending and self.pending.get('phase')=='goal' and self.plan and (after.levels>before.levels or after.state=='WIN'):
            body=next((b for b in self.memory.tracks if b['id']==self.plan['body_id']),None)
            if body and body['color']!=self.plan['marker_color']:
                marker=self.plan['marker_color'];focus=body['color'];learned=CompositeEffects()
                for step,old,control,new in self.history:
                    if old.levels!=before.levels or old.state!='NOT_FINISHED' or new.levels!=old.levels or control.id in (0,6):continue
                    a=[paint(g) for g in foreground_groups(old.grid) if marker in g['colors'] and len(g['colors'].get(focus,()))==body['pixels']]
                    b=[paint(g) for g in foreground_groups(new.grid) if marker in g['colors'] and len(g['colors'].get(focus,()))==body['pixels']]
                    if len(a)==len(b)==1:learned.add(control.id,a[0],b[0],step)
                self.role=(marker,focus);self.composite_effects=learned
        assembling=self.assembly_pending
        super().observe(before,action,after)
        if assembling:
            self.assembly.observe(before,action,after,self.steps)
            self.latest_feedback['assembly']=self.assembly.feedback
            outcome=self.assembly.feedback.get('outcome')
            self.assembly_counts[outcome]+=1
        self.history.append((self.steps,before,action,after))
        self.assembly_pending=False

    def diagnostics(self):
        return {**super().diagnostics(),'assembly_counts':dict(self.assembly_counts),
                'assembly_attempted_levels':sorted(self.attempted_levels),
                'assembly':self.assembly.record() if self.assembly else None}
