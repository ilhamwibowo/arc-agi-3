"""Finite image-only marker-pair arrangement hypothesis; no environment interaction."""
from collections import Counter
from dataclasses import dataclass
import time

def split_islands(coords):
    remaining=set(coords);result=[]
    while remaining:
        group={min(remaining)};remaining-=group;todo=list(group)
        while todo:
            x,y=todo.pop()
            for p in ((x-1,y),(x+1,y),(x,y-1),(x,y+1)):
                if p in remaining:remaining.remove(p);group.add(p);todo.append(p)
        result.append(frozenset(group))
    return result



def turn(x,y,op):
    for _ in range(op):x,y=-y,x
    return x,y


@dataclass(frozen=True)
class Variant:
    piece:int
    op:int
    colored:frozenset
    solid:frozenset
    markers:frozenset
    islands:tuple


def variants(piece,index,marker_color,allowed_ops):
    result=[];seen=set()
    for op in allowed_ops:
        raw=[(c,*turn(x,y,op)) for c,x,y in piece]
        x0=min(x for c,x,y in raw);y0=min(y for c,x,y in raw)
        colored=frozenset((c,x-x0,y-y0) for c,x,y in raw)
        if colored in seen:continue
        seen.add(colored)
        solid=frozenset((x,y) for c,x,y in colored)
        markers=frozenset((x,y) for c,x,y in colored if c==marker_color)
        result.append(Variant(index,op,colored,solid,markers,tuple(split_islands(markers))))
    return result


def place(v,dx,dy):
    return {'piece':v.piece,'op':v.op,'offset':(dx,dy),
            'colored':frozenset((c,x+dx,y+dy) for c,x,y in v.colored),
            'solid':frozenset((x+dx,y+dy) for x,y in v.solid),
            'markers':frozenset((x+dx,y+dy) for x,y in v.markers),
            'islands':tuple(frozenset((x+dx,y+dy) for x,y in island) for island in v.islands)}


def bounds(coords):
    xs,ys=zip(*coords);return min(xs),min(ys),max(xs),max(ys)


def normalized(coords):
    x0,y0,_,_=bounds(coords)
    return frozenset((x-x0,y-y0) for x,y in coords)


class SearchBudget(Exception):
    pass


def solve(pieces,marker_color,width,height,allowed_ops=(0,1,2,3),max_nodes=20000,max_seconds=10):
    """Anchor the largest piece; attach every other piece through complete marker pairs.

    The node budget counts candidate placements checked, including rejected ones.
    Every root orientation is considered within one shared budget. Coordinates are
    relative until a complete arrangement is translated to fit the public frame.
    """
    started=time.monotonic();deadline=started+max_seconds
    checks=states=0;budget_reason=None;visited=set()
    if not (1<len(pieces)<=8):return {'status':'unsupported_piece_count','pieces':len(pieces)}
    if any(not p or len({(x,y) for c,x,y in p})!=len(p) for p in pieces):
        return {'status':'invalid_piece_pixels'}
    choices=[variants(p,i,marker_color,allowed_ops) for i,p in enumerate(pieces)]
    if any(not vs or not vs[0].markers for vs in choices):return {'status':'missing_marker_or_operation'}
    if sum(len(vs[0].markers) for vs in choices)%2:return {'status':'unsatisfied_marker_parity'}
    anchor=max(range(len(pieces)),key=lambda i:(len(pieces[i]),-i))
    anchor_x=min(x for c,x,y in pieces[anchor]);anchor_y=min(y for c,x,y in pieces[anchor])

    def check_budget():
        nonlocal budget_reason
        if time.monotonic()>=deadline:budget_reason='time_budget';raise SearchBudget()
        if checks>=max_nodes:budget_reason='node_budget';raise SearchBudget()

    def accepted(candidate,placed,open_markers):
        for prior in placed.values():
            overlap=candidate['solid']&prior['solid']
            if overlap-(candidate['markers']&prior['markers']):return False
        all_prior_markers=Counter(p for prior in placed.values() for p in prior['markers'])
        if any(all_prior_markers[p]>=2 for p in candidate['markers']):return False
        touched=set(all_prior_markers)&candidate['markers']
        for island in candidate['islands']:
            if island&touched and island not in open_markers:return False
        union=set(candidate['solid']).union(*(p['solid'] for p in placed.values()))
        x0,y0,x1,y1=bounds(union)
        return x1-x0+1<=width and y1-y0+1<=height

    def successors(target,placed,open_markers):
        nonlocal checks
        result=[];dedup=set();tx,ty,_,_=bounds(target);shape=normalized(target)
        for i,vs in enumerate(choices):
            if i in placed:continue
            for v in vs:
                for island in v.islands:
                    check_budget()
                    if normalized(island)!=shape:continue
                    x,y,_,_=bounds(island);key=(i,v.op,tx-x,ty-y)
                    if key in dedup:continue
                    dedup.add(key);checks+=1
                    candidate=place(v,tx-x,ty-y)
                    if accepted(candidate,placed,open_markers):
                        contacts=sum(island in open_markers for island in candidate['islands'])
                        candidate['contacts']=contacts;result.append(candidate)
        return sorted(result,key=lambda p:(-p['contacts'],p['piece'],p['op'],p['offset']))

    def search(placed,open_markers):
        nonlocal states
        check_budget();states+=1
        state=tuple(sorted((i,p['op'],*p['offset']) for i,p in placed.items()))
        if state in visited:return None
        visited.add(state)
        if len(placed)==len(pieces):
            if open_markers:return None
            occupied=set().union(*(p['solid'] for p in placed.values()))
            x0,y0,x1,y1=bounds(occupied)
            tx=max(-x0,min(anchor_x,width-1-x1));ty=max(-y0,min(anchor_y,height-1-y1))
            return {'placements':[{'piece':i,'op':p['op'],'offset':[p['offset'][0]+tx,p['offset'][1]+ty]} for i,p in sorted(placed.items())],
                    'anchor_piece':anchor,'global_translation':[tx,ty]}
        if not open_markers:return None
        best=None
        for target in sorted(open_markers,key=lambda s:tuple(sorted(s))):
            options=successors(target,placed,open_markers)
            if not options:return None
            if best is None or len(options)<len(best):best=options
        for candidate in best:
            remaining=set(open_markers)
            for island in candidate['islands']:
                if island in remaining:remaining.remove(island)
                else:remaining.add(island)
            found=search({**placed,candidate['piece']:candidate},remaining)
            if found:return found
        return None

    result=None
    try:
        for v in choices[anchor]:
            root=place(v,0,0)
            if bounds(root['solid'])[2]>=width or bounds(root['solid'])[3]>=height:continue
            result=search({anchor:root},set(root['islands']))
            if result:break
    except SearchBudget:pass
    return {'status':'candidate' if result else budget_reason or 'no_arrangement_under_constraints',
            'candidate_checks':checks,'expanded_states':states,'elapsed_seconds':time.monotonic()-started,
            'allowed_ops':list(allowed_ops),'marker_color':marker_color,'frame':[width,height],
            'solution':result}


def validate(pieces,marker_color,width,height,solution,allowed_ops=(0,1,2,3)):
    """Independent reconstruction and pixel ownership/connected-component checks."""
    from arc_harness.object_memory import components
    transforms={0:(1,0,0,1),1:(0,-1,1,0),2:(-1,0,0,-1),3:(0,1,-1,0)}
    placements=solution.get('placements',[]);checks={}
    indices=[p['piece'] for p in placements]
    checks['each_piece_once']=sorted(indices)==list(range(len(pieces)))
    checks['supported_operations']=all(p['op'] in allowed_ops and p['op'] in transforms for p in placements)
    if not all(checks.values()):return {'valid':False,'checks':checks}
    owned={};island_sets=[];rendered=[];checks['within_frame']=True;checks['pixel_count_preserved']=True
    for p in placements:
        i=p['piece'];a,b,c,d=transforms[p['op']]
        transformed=[(color,a*x+b*y,c*x+d*y) for color,x,y in pieces[i]]
        x0=min(x for color,x,y in transformed);y0=min(y for color,x,y in transformed)
        pixels={(color,x-x0+p['offset'][0],y-y0+p['offset'][1]) for color,x,y in transformed}
        checks['pixel_count_preserved']&=len(pixels)==len(pieces[i])
        marker_mask=[[0]*width for _ in range(height)]
        for color,x,y in pixels:
            inside=0<=x<width and 0<=y<height;checks['within_frame']&=inside
            owned.setdefault((x,y),[]).append((i,color))
            if inside and color==marker_color:marker_mask[y][x]=1
        groups=[frozenset(q['coords']) for q in components(tuple(map(tuple,marker_mask))) if q['color']==1]
        island_sets.extend((i,g) for g in groups)
        rendered.append({'piece':i,'colored_pixels':sorted(pixels)})
    checks['only_marker_overlap']=all(len(owners)==1 or len(owners)==2 and all(c==marker_color for i,c in owners) for owners in owned.values())
    checks['every_marker_paired']=all(len(owners)==2 and all(c==marker_color for i,c in owners) for owners in owned.values() if any(c==marker_color for i,c in owners))
    checks['whole_island_pairs']=all(sum(j!=i and other==island for j,other in island_sets)==1 for i,island in island_sets)
    graph={i:set() for i in range(len(pieces))}
    for owners in owned.values():
        if len(owners)==2 and all(c==marker_color for i,c in owners):
            a,b=[i for i,c in owners];graph[a].add(b);graph[b].add(a)
    visited={0};todo=[0]
    while todo:
        for nxt in graph[todo.pop()]-visited:visited.add(nxt);todo.append(nxt)
    checks['connected_assembly']=len(visited)==len(pieces)
    return {'valid':all(checks.values()),'checks':checks,'rendered':rendered,'contact_graph':{str(k):sorted(v) for k,v in graph.items()}}
