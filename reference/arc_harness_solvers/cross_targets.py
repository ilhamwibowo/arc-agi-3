"""Measure selectable pixel stencils and plan joint colored-target coverage."""
from collections import Counter, defaultdict, deque
from statistics import median_low
from arc_harness.agents import BalancedDirectedAgent
from arc_harness.core import Action


def read_scene(grid):
    height,width=len(grid),len(grid[0]);by_color=defaultdict(list)
    for y,row in enumerate(grid):
        for x,color in enumerate(row):by_color[color].append((x,y))
    markers=defaultdict(list)
    for y in range(1,height-1):
        for x in range(1,width-1):
            color=grid[y][x];ring={grid[yy][xx] for yy in range(y-1,y+2) for xx in range(x-1,x+2) if (xx,yy)!=(x,y)}
            if len(ring)==1 and color not in ring:markers[color].append((x,y))
    crosses={};frequencies=Counter(v for row in grid for v in row);background=frequencies.most_common(1)[0][0]
    for color,points in by_color.items():
        if len(points)>256 or len(points)<10:continue
        remaining=set(points)-set(markers.get(color,[]));bodies=[]
        while remaining:
            seed=remaining.pop();body={seed};queue=[seed]
            while queue:
                px,py=queue.pop()
                for dy in range(-2,3):
                    for dx in range(-2,3):
                        nxt=(px+dx,py+dy)
                        if nxt in remaining:remaining.remove(nxt);body.add(nxt);queue.append(nxt)
            bodies.append(body)
        body=max(bodies,key=len,default=set())
        if len(body)<10:continue
        xs=[x for x,y in body];ys=[y for x,y in body]
        if min(max(xs)-min(xs),max(ys)-min(ys))<6:continue
        cx,cy=median_low(xs),median_low(ys)
        rare=[(x,y) for y in range(max(0,cy-3),min(height,cy+4)) for x in range(max(0,cx-3),min(width,cx+4)) if grid[y][x] not in (color,background) and frequencies[grid[y][x]]<=4]
        selected=len(rare)==1
        if selected:cx,cy=rare[0]
        offsets={(x-cx,y-cy) for x,y in body};offsets.add((0,0))
        crosses[color]={'position':(cx,cy),'offsets':tuple(sorted(offsets)),'selected':selected}
    return crosses,{c:ps for c,ps in markers.items() if c in crosses}


def framed_markers(grid):
    background=Counter(v for row in grid for v in row).most_common(1)[0][0]
    markers=defaultdict(set)
    for y in range(1,len(grid)-1):
        for x in range(1,len(grid[0])-1):
            color=grid[y][x]
            ring={grid[yy][xx] for yy in range(y-1,y+2) for xx in range(x-1,x+2) if (xx,yy)!=(x,y)}
            if len(ring)==1 and color not in ring and background not in ring and color!=background:markers[color].add((x,y))
    return markers


def selected_cursor(grid, marker_color=None):
    counts=Counter(v for row in grid for v in row)
    if marker_color is None:
        shapes,_=read_scene(grid)
        selected=[s for s in shapes.values() if s['selected']]
        if len(selected)!=1:return None
        x,y=selected[0]['position'];marker_color=grid[y][x]
    positions=[(x,y) for y,row in enumerate(grid) for x,v in enumerate(row) if v==marker_color]
    if len(positions)!=1:return None
    x,y=positions[0];background=counts.most_common(1)[0][0];scores=Counter()
    for dy in range(-24,25):
        for dx in range(-24,25):
            if not (0<=x+dx<len(grid[0]) and 0<=x-dx<len(grid[0]) and 0<=y+dy<len(grid) and 0<=y-dy<len(grid)):continue
            c=grid[y+dy][x+dx]
            if c not in (background,marker_color) and grid[y-dy][x-dx]==c:scores[c]+=1
    if not scores:return None
    color,score=scores.most_common(1)[0]
    if score<10:return None
    return {'position':(x,y),'color':color,'marker_color':marker_color}


def color_pads(grid):
    """Uniform colored squares enclosed by a one-pixel contrasting frame."""
    result=defaultdict(set);height,width=len(grid),len(grid[0])
    for size in range(3,7):
        for y in range(1,height-size):
            for x in range(1,width-size):
                color=grid[y][x];frame=grid[y-1][x-1]
                if color==frame:continue
                if any(grid[yy][xx]!=color for yy in range(y,y+size) for xx in range(x,x+size)):continue
                ring=[grid[y-1][xx] for xx in range(x-1,x+size+1)]+[grid[y+size][xx] for xx in range(x-1,x+size+1)]+[grid[yy][x-1] for yy in range(y,y+size)]+[grid[yy][x+size] for yy in range(y,y+size)]
                if all(c==frame for c in ring):result[color].update((xx,yy) for yy in range(y,y+size) for xx in range(x,x+size))
    return result


def plan_contacts(objects,targets,moves,pads,width,height):
    points=[(c,p) for c,ps in sorted(targets.items()) for p in sorted(ps)]
    if not points or len(points)>16:return None
    options={}
    for i,obj in objects.items():
        masks=defaultdict(int);contacts=defaultdict(set);offsets=obj['offsets']
        for j,(color,(tx,ty)) in enumerate(points):
            for dx,dy in offsets:masks[(tx-dx,ty-dy,color)]|=1<<j
        for color,tiles in pads.items():
            for tx,ty in tiles:
                for dx,dy in offsets:contacts[(tx-dx,ty-dy)].add(color)
        start=(*obj['position'],obj['color']);queue=deque([start]);parents={start:None};options[i]={0:(0,start)};distances={start:0}
        while queue and len(parents)<=32768:
            state=queue.popleft();x,y,color=state;distance=distances[state];mask=masks[state]
            if mask not in options[i]:options[i][mask]=(distance,state)
            for action,(dx,dy) in sorted(moves.items()):
                nx,ny=x+dx,y+dy
                if not (0<=nx<width and 0<=ny<height):continue
                touched=contacts[(nx,ny)]
                if len(touched)>1:continue
                nc=next(iter(touched)) if touched else color;nxt=(nx,ny,nc)
                if nxt in parents:continue
                parents[nxt]=(state,action);distances[nxt]=distance+1;queue.append(nxt)
        choices={}
        for mask,(distance,state) in options[i].items():
            path=[];node=state
            while parents[node] is not None:
                prior,action=parents[node];path.append((action,node));node=prior
            choices[mask]=(distance,list(reversed(path)),state)
        options[i]=choices
    dp={0:(0,{})}
    for i,choices in options.items():
        nxt={}
        for prior,(cost,plans) in dp.items():
            for mask,(distance,path,state) in choices.items():
                merged=prior|mask;total=cost+distance
                if merged not in nxt or total<nxt[merged][0]:nxt[merged]=(total,{**plans,i:{'path':path,'goal':state}})
        dp=nxt
    target=(1<<len(points))-1
    return dp[target][1] if target in dp else None


class CrossTargetAgent:
    def __init__(self,seed):
        self.fallback=BalancedDirectedAgent(seed);self.level=None;self.steps=0;self.latest_feedback=None
        self.marker_color=None;self.moves={};self.switch=None;self.counts=Counter();self.pending=None
        self.objects={};self.active=None;self.targets=defaultdict(set);self.closed=False;self.plan=None
        self.calibration=Counter();self.pads={};self.unsupported=False

    def _object(self,cursor):
        position=cursor['position']
        found=[i for i,o in self.objects.items() if o['position']==position]
        if found:return found[0],False
        i=len(self.objects);self.objects[i]={'position':position,'color':cursor['color'],'offsets':{(0,0)},'probes':0,'confirmed':set()}
        return i,True

    def choose(self,obs):
        self.pending=None
        if obs.state!='NOT_FINISHED':return self.fallback.choose(obs)
        if obs.levels!=self.level:
            self.level=obs.levels;self.objects={};self.active=None;self.targets=defaultdict(set);self.closed=False;self.plan=None;self.pads=color_pads(obs.grid);self.unsupported=False
        if self.unsupported:return self.fallback.choose(obs)
        cursor=selected_cursor(obs.grid,self.marker_color)
        if cursor is None and self.active in self.objects:
            obj=self.objects[self.active];cursor={'position':obj['position'],'color':obj['color'],'marker_color':self.marker_color}
        if cursor is None:return self.fallback.choose(obs)
        self.marker_color=cursor['marker_color']
        if self.active is None:self.active,_=self._object(cursor)
        obj=self.objects[self.active];obj['position']=cursor['position'];obj['color']=cursor['color']
        for c,points in framed_markers(obs.grid).items():self.targets[c].update(points)
        controls=[a for a in obs.actions if a not in (0,6,7)]
        if len(self.moves)<4 or self.switch is None:
            untested=[a for a in controls if not self.calibration[a]]
            action=untested[0] if untested else min(controls,key=lambda a:self.calibration[a]);reason='learn_cursor_controls'
        elif obj['probes']<4:
            action=sorted(self.moves)[obj['probes']];obj['probes']+=1;reason='measure_selected_stencil'
        elif not self.closed:
            action=self.switch;reason='discover_selectable_objects'
        else:
            if self.plan is None:self.plan=plan_contacts(self.objects,self.targets,self.moves,self.pads,len(obs.grid[0]),len(obs.grid))
            if not self.plan or self.active not in self.plan:
                self.unsupported=True;self.counts['unsupported_geometry_levels']+=1
                return self.fallback.choose(obs)
            remaining=self.plan[self.active]['path']
            if not remaining:action=self.switch;reason='select_unplaced_object'
            else:action=remaining[0][0];reason='align_measured_stencil'
        self.pending={'id':self.active,'position':cursor['position'],'color':obj['color'],'reason':reason}
        self.counts[reason]+=1
        return Action(action),{'reason':reason,'object':self.active,'position':cursor['position'],'placements':{i:p['goal'] for i,p in (self.plan or {}).items()},'stencil_pixels':len(obj['offsets'])}

    def observe(self,before,action,after):
        self.steps+=1;self.fallback.observe(before,action,after);self.latest_feedback={}
        if after.levels>before.levels:self.counts['levels_completed']+=after.levels-before.levels
        if not self.pending or before.levels!=after.levels or after.state!='NOT_FINISHED':return
        cursor=selected_cursor(after.grid,self.marker_color)
        if cursor is None and action.id in self.moves:
            ox,oy=self.pending['position'];mx,my=self.moves[action.id]
            cursor={'position':(ox+mx,oy+my),'color':self.pending['color'],'marker_color':self.marker_color}
            self.counts['occluded_cursor_predictions']+=1
        if cursor is None:return
        old=self.pending;dx=cursor['position'][0]-old['position'][0];dy=cursor['position'][1]-old['position'][1]
        self.calibration[action.id]+=1
        if self.switch==action.id or (action.id not in self.moves and len(self.moves)>=4):
            self.switch=action.id;self.active,created=self._object(cursor)
            if not created and all(o['probes']>=4 for o in self.objects.values()):self.closed=True
        elif (dx==0)!=(dy==0) and max(abs(dx),abs(dy))<=6:
            self.moves[action.id]=(dx,dy);obj=self.objects[old['id']];obj['position']=cursor['position'];obj['color']=cursor['color']
            if self.plan and self.active in self.plan and self.plan[self.active]['path']:
                expected_action,expected=self.plan[self.active]['path'].pop(0)
                if expected_action!=action.id or tuple(expected)!=(*cursor['position'],cursor['color']):
                    self.plan=None;self.counts['plan_prediction_mismatch']+=1
            if cursor['color']!=old['color']:
                self.counts['observed_recolorings']+=1;self.pending=None;return
            b={(x,y) for y,row in enumerate(before.grid) for x,v in enumerate(row) if v==old['color']}
            a={(x,y) for y,row in enumerate(after.grid) for x,v in enumerate(row) if v==old['color']}
            ox,oy=old['position'];nx,ny=cursor['position']
            observed={(x-ox,y-oy) for x,y in b-a}|{(x-nx,y-ny) for x,y in a-b}
            observed={(x,y) for x,y in observed if max(abs(x),abs(y))<=24}
            obj['confirmed'].update(observed)
            obj['offsets']=obj['confirmed']|{(-x,-y) for x,y in obj['confirmed']}|{(0,0)}
        self.latest_feedback={'cursor_motion':(dx,dy),'objects':len(self.objects),'cycle_closed':self.closed}
        self.pending=None

    def diagnostics(self):return {'steps':self.steps,'counts':dict(self.counts),'moves':self.moves,'switch':self.switch,'objects':{i:{k:(sorted(v) if isinstance(v,set) else v) for k,v in o.items()} for i,o in self.objects.items()},'plan':self.plan}
