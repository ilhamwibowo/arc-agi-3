"""Tentative marker constellations and bounded rigid-effect route search."""
from collections import Counter, deque
from functools import lru_cache
import hashlib

from .rigid_memory import canonical, normalize, project

def foreground_groups(grid):
    palette=Counter(v for row in grid for v in row);background,count=palette.most_common(1)[0]
    if count*2<=len(grid)*len(grid[0]):return []
    seen=set();groups=[];height=len(grid);width=len(grid[0])
    for y,row in enumerate(grid):
        for x,value in enumerate(row):
            if value==background or (x,y) in seen:continue
            todo=deque([(x,y)]);seen.add((x,y));colors={}
            while todo:
                a,b=todo.popleft();colors.setdefault(grid[b][a],set()).add((a,b))
                for c,d in ((a-1,b),(a+1,b),(a,b-1),(a,b+1)):
                    if 0<=c<width and 0<=d<height and (c,d) not in seen and grid[d][c]!=background:
                        seen.add((c,d));todo.append((c,d))
            coords=set.union(*colors.values())
            if 2<=len(colors)<=8 and 4<=len(coords)<=512:
                groups.append({'colors':colors,'coords':coords})
    return groups


def islands(coords):
    remaining=set(coords);n=0
    while remaining:
        todo=[remaining.pop()];n+=1
        while todo:
            x,y=todo.pop()
            for p in ((x-1,y),(x+1,y),(x,y-1),(x,y+1)):
                if p in remaining:remaining.remove(p);todo.append(p)
    return n


def candidates(grid):
    groups=foreground_groups(grid);found=[]
    for i,a in enumerate(groups):
        for j,b in enumerate(groups):
            if i>=j:continue
            for color in a['colors'].keys()&b['colors'].keys():
                left,right=a['colors'][color],b['colors'][color]
                if len(left)!=len(right) or islands(left)<2 or islands(right)<2:continue
                if canonical(normalize(left),'d4')!=canonical(normalize(right),'d4'):continue
                found.append({'group_indices':[i,j],'shared_color':color,'marker_pixels':len(left),
                              'marker_islands':islands(left),'markers':[sorted(left),sorted(right)],
                              'group_pixels':[len(a['coords']),len(b['coords'])]})
    return groups,found



def paint(group):
    return frozenset((color,x,y) for color,coords in group['colors'].items() for x,y in coords)


def body_at(coords,color):
    xs,ys=zip(*coords)
    return {'color':color,'coords':coords,'bounds':(min(xs),min(ys),max(xs),max(ys)),
            'normalized':normalize(coords)}


def operation(x,y,op):
    if op>=4:x=-x
    for _ in range(op%4):x,y=-y,x
    return x,y


def move_paint(body,colored,mode):
    """Extend a body-supported rigid transform to its attached colored pixels."""
    op,dx2,dy2=mode;destination=project(body,op,dx2,dy2)
    if destination is None:return None
    x0,y0,_,_=body['bounds']
    rotated=[operation(x-x0,y-y0,op) for x,y in body['coords']]
    rx=min(x for x,y in rotated);ry=min(y for x,y in rotated)
    tx=min(x for x,y in destination);ty=min(y for x,y in destination)
    result=[]
    for color,x,y in colored:
        a,b=operation(x-x0,y-y0,op)
        result.append((color,a-rx+tx,b-ry+ty))
    return destination,frozenset(result)


def agreed_move(body,colored,modes,width,height):
    outcomes={move_paint(body,colored,mode) for mode in modes}
    if None in outcomes or len(outcomes)!=1:return None
    new_body,new_paint=outcomes.pop()
    if any(not(0<=x<width and 0<=y<height) for color,x,y in new_paint):return None
    return new_body,new_paint


def goal_key(colored,marker_color,reference):
    return hashlib.sha256(repr((sorted(colored),marker_color,sorted(reference))).encode()).hexdigest()


def search_alignment(body,actor,marker_color,reference,controls,width,height,tested,
                     node_limit=2048,depth_limit=24):
    """Search only calibrated rigid operations; every real step still needs checking."""
    start=(frozenset(body['coords']),paint(actor))
    states=[start];parents=[None];depths=[0];visited={start};queue=deque([0])
    while queue:
        index=queue.popleft();coords,colored=states[index];depth=depths[index]
        markers=frozenset((x,y) for c,x,y in colored if c==marker_color)
        key=goal_key(colored,marker_color,reference)
        if depth and markers==reference and key not in tested:
            route=[];cursor=index
            while parents[cursor] is not None:
                parent,kind=parents[cursor]
                route.append({'action':kind,'body':states[cursor][0],'paint':states[cursor][1]})
                cursor=parent
            return {'route':list(reversed(route)),'key':key,'nodes':len(states)},len(states)
        if depth>=depth_limit:continue
        current=body_at(coords,body['color'])
        for kind,model in sorted(controls.items()):
            outcome=agreed_move(current,colored,model['modes'],width,height)
            if outcome is None or outcome in visited:continue
            if any((x,y) in reference and c!=marker_color for c,x,y in outcome[1]):continue
            if len(states)>=node_limit:continue
            visited.add(outcome);states.append(outcome);parents.append((index,kind))
            depths.append(depth+1);queue.append(len(states)-1)
    return None,len(states)


def pixels_match(grid,colored):
    return all(0<=y<len(grid) and 0<=x<len(grid[0]) and grid[y][x]==color for color,x,y in colored)
