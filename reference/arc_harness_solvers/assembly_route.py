"""Finite translation routes for observed colored pieces and measured control vectors."""
import heapq
import math
import time


def translate(colored, dx, dy):
    return frozenset((c, x+dx, y+dy) for c, x, y in colored)


def route(colored, target, others, marker_color, controls, width, height,
          max_nodes=4096, max_steps=64, max_seconds=1):
    """Plan with observed endpoint occupancy; swept motion remains unmodeled.

    ``controls`` maps action identifiers to measured integral pixel translations.
    Only overlapping pixels of the marker role are allowed. A route is a hypothesis:
    execution must check each actual returned piece and all stationary references.
    """
    start_time=time.monotonic();deadline=start_time+max_seconds
    colored=frozenset(colored);target=frozenset(target)
    if not colored or not target:return {'status':'empty_piece'}
    tx=min(x for c,x,y in target)-min(x for c,x,y in colored)
    ty=min(y for c,x,y in target)-min(y for c,x,y in colored)
    if translate(colored,tx,ty)!=target:return {'status':'not_translation'}
    if not controls:return {'status':'no_controls'}
    if any(not isinstance(v,int) for delta in controls.values() for v in delta):
        return {'status':'noninteger_control'}
    controls={a:tuple(d) for a,d in controls.items() if tuple(d)!=(0,0)}
    if not controls:return {'status':'no_controls'}
    occupancy={}
    for piece in others:
        for c,x,y in piece:occupancy.setdefault((x,y),set()).add(c)
    def valid(dx,dy):
        for c,x,y in colored:
            xx,yy=x+dx,y+dy
            if not (0<=xx<width and 0<=yy<height):return False
            old=occupancy.get((xx,yy))
            if old is not None and (c!=marker_color or old!={marker_color}):return False
        return True
    if not valid(0,0):return {'status':'invalid_start'}
    if not valid(tx,ty):return {'status':'invalid_target'}
    step=max(max(abs(x),abs(y)) for x,y in controls.values())
    def heuristic(x,y):return math.ceil(max(abs(tx-x),abs(ty-y))/step)
    queue=[(heuristic(0,0),0,0,0)];distance={(0,0):0};parents={};expanded=0
    status='unreachable'
    while queue:
        if time.monotonic()>=deadline:status='time_budget';break
        if expanded>=max_nodes:status='node_budget';break
        _,depth,x,y=heapq.heappop(queue)
        if distance[x,y]!=depth:continue
        expanded+=1
        if (x,y)==(tx,ty):
            steps=[];at=(x,y)
            while at!=(0,0):
                previous,action=parents[at]
                steps.append({'action':action,'offset':at,'paint':translate(colored,*at)})
                at=previous
            return {'status':'candidate','route':list(reversed(steps)),'expanded':expanded,
                    'elapsed_seconds':time.monotonic()-start_time}
        if depth>=max_steps:continue
        for action,(dx,dy) in sorted(controls.items()):
            nxt=(x+dx,y+dy)
            if distance.get(nxt,max_steps+1)<=depth+1 or not valid(*nxt):continue
            distance[nxt]=depth+1;parents[nxt]=((x,y),action)
            heapq.heappush(queue,(depth+1+heuristic(*nxt),depth+1,*nxt))
    return {'status':status,'expanded':expanded,'elapsed_seconds':time.monotonic()-start_time}
