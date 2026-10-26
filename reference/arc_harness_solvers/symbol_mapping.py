"""Infer paired glyph translations and learn controls from public screen changes."""
from collections import Counter, defaultdict, deque
from itertools import product, islice
from pattern_copy import transforms
from arc_harness.agents import BalancedDirectedAgent
from arc_harness.core import Action
from arc_harness.scene import regions


def glyph_panels(grid):
    found=[]
    for color,x,y,x1,y1,area in regions(grid)[0]:
        h=y1-y+1;w=x1-x+1
        if not (5<=h<=12 and w%h==0 and w//h<=12):continue
        cells=[]
        for left in range(x,x+w,h):
            if not all(grid[yy][xx]==color for yy in range(y,y+h) for xx in range(left,left+h)
                       if yy in (y,y+h-1) or xx in (left,left+h-1)):break
            patch=tuple(tuple(int(v!=color) for v in row[left+1:left+h-1]) for row in grid[y+1:y+h-1])
            if not any(v for row in patch for v in row):break
            cells.append(patch)
        if len(cells)==w//h:found.append({'x':x,'y':y,'size':h,'color':color,'glyphs':tuple(cells)})
    return found


def canonical(glyph):
    return min(transforms(glyph))


def translate(sequence,table,limit=128):
    """Parse a visible symbol sequence using bounded demonstrated substitutions."""
    cache={len(sequence):[()]}
    for i in range(len(sequence)-1,-1,-1):
        answers=[]
        for left,right in sorted(table.items(),key=lambda pair:-len(pair[0])):
            if sequence[i:i+len(left)]==left:
                for suffix in cache.get(i+len(left),[]):
                    answer=right+suffix
                    if answer not in answers:answers.append(answer)
                    if len(answers)>=limit:break
            if len(answers)>=limit:break
        cache[i]=answers
    return cache[0]


def mapping_problem(grid):
    panels=glyph_panels(grid);tables=defaultdict(dict);support=Counter();bad=set();example_bottom={}
    for a in panels:
        end=a['x']+a['size']*len(a['glyphs'])
        peers=[b for b in panels if b['y']==a['y'] and b['size']==a['size'] and b['color']!=a['color'] and 0<b['x']-end<2*a['size']]
        if not peers:continue
        b=min(peers,key=lambda b:b['x']-end)
        key=(a['color'],b['color'],a['size']);source=tuple(map(canonical,a['glyphs']));target=tuple(map(canonical,b['glyphs']))
        if source in tables[key] and tables[key][source]!=target:bad.add(key)
        tables[key][source]=target;support[key]+=1;example_bottom[key]=max(example_bottom.get(key,0),a['y'])
    usable={k:t for k,t in tables.items() if k not in bad and len(t)>=3}
    options=[]
    for source in panels:
        if source['y']<=max(example_bottom.values(),default=64):continue
        initial=(source['color'],tuple(map(canonical,source['glyphs'])))
        queue=deque([(initial,())]);visited={initial}
        while queue and len(visited)<=256:
            (color,sequence),path=queue.popleft()
            if path:
                for target in panels:
                    if target['color']!=color or target['size']!=source['size'] or target['y']<=source['y'] or len(sequence)!=len(target['glyphs']):continue
                    goal=tuple(current if canonical(current)==wanted else wanted for current,wanted in zip(target['glyphs'],sequence))
                    options.append({'source':source,'target':target,'desired':goal,'goal_sequences':[goal],
                        'example_count':sum(support[k] for k in path),'table_size':sum(len(usable[k]) for k in path),
                        'rule_path':path})
            if len(path)>=4:continue
            for key,table in usable.items():
                if key[0]!=color or key[2]!=source['size']:continue
                for output in translate(sequence,table):
                    node=(key[1],output)
                    if len(output)<=12 and node not in visited:
                        visited.add(node);queue.append((node,path+(key,)))
    return max(options,key=lambda p:(-len(p['rule_path']),p['example_count'],len(p['desired']))) if options else None


def cursor(grid,target):
    x,y,size,n=target['x'],target['y'],target['size'],len(target['glyphs'])
    scores=[0]*n
    for yy in list(range(max(0,y-3),y))+list(range(y+size,min(len(grid),y+size+3))):
        background=Counter(grid[yy]).most_common(1)[0][0]
        for i in range(n):scores[i]+=sum(v!=background for v in grid[yy][x+i*size:x+(i+1)*size])
    maximum=max(scores)
    return scores.index(maximum) if maximum>0 and scores.count(maximum)==1 else None


class SymbolMappingAgent:
    def __init__(self,seed,problem_reader=None,cursor_reader=None,equivalence=None):
        self.read_problem=problem_reader or mapping_problem
        self.read_cursor=cursor_reader or cursor
        self.equivalent=equivalence or (lambda a,b:canonical(a)==canonical(b))
        self.fallback=BalancedDirectedAgent(seed);self.level=None;self.controls=Counter()
        self.moves={};self.edits=defaultdict(dict);self.edit_actions=set();self.counts=Counter();self.steps=0
        self.pending=None;self.latest_feedback=None;self.rejected_goals=set()

    def choose(self,obs):
        self.pending=None
        if self.level!=obs.levels:
            self.level=obs.levels;self.controls.clear();self.moves.clear();self.edits.clear();self.edit_actions.clear();self.rejected_goals.clear()
        if obs.state!='NOT_FINISHED':return self.fallback.choose(obs)
        problem=self.read_problem(obs.grid)
        if problem is None:return self.fallback.choose(obs)
        current_glyphs=problem['target']['glyphs']
        if current_glyphs in problem['goal_sequences']:
            self.rejected_goals.add(current_glyphs)
        goals=[g for g in problem['goal_sequences'] if g not in self.rejected_goals]
        if not goals:return self.fallback.choose(obs)
        problem['desired']=min(goals,key=lambda g:sum(a!=b for a,b in zip(g,current_glyphs)))
        current=self.read_cursor(obs.grid,problem['target'])
        if current is None:return self.fallback.choose(obs)
        choices=[a for a in obs.actions if a not in (0,6)]
        unknown=[a for a in choices if not self.controls[a]]
        reason='calibrate_symbol_controls'
        if unknown:kind=unknown[0]
        else:
            glyphs=problem['target']['glyphs'];desired=problem['desired'];n=len(glyphs)
            wrong=[i for i in range(n) if glyphs[i]!=desired[i]]
            if not wrong:return self.fallback.choose(obs)
            if current in wrong:
                queue=deque([(glyphs[current],None)]);seen={glyphs[current]};kind=None
                while queue:
                    shape,first=queue.popleft()
                    if self.equivalent(shape,desired[current]):kind=first;break
                    for action,nxt in sorted(self.edits.get(shape,{}).items()):
                        if nxt not in seen:seen.add(nxt);queue.append((nxt,action if first is None else first))
                if kind is None:
                    if not self.edit_actions:return self.fallback.choose(obs)
                    frontier=deque([(glyphs[current],None)]);visited={glyphs[current]}
                    while frontier:
                        shape,first=frontier.popleft()
                        untried=sorted(self.edit_actions-self.edits.get(shape,{}).keys())
                        if untried:
                            kind=first if first is not None else untried[0];break
                        for action,nxt in sorted(self.edits.get(shape,{}).items()):
                            if nxt not in visited:visited.add(nxt);frontier.append((nxt,action if first is None else first))
                    if kind is None:
                        self.rejected_goals.add(desired);return self.fallback.choose(obs)
                    reason='learn_symbol_cycle'
                else:reason='follow_observed_symbol_cycle'
            else:
                queue=deque([(current,None)]);seen={current};kind=None
                while queue:
                    position,first=queue.popleft()
                    if position in wrong:kind=first;break
                    for action,delta in sorted(self.moves.items()):
                        nxt=(position+delta)%n
                        if nxt not in seen:seen.add(nxt);queue.append((nxt,action if first is None else first))
                if kind is None:return self.fallback.choose(obs)
                reason='move_to_unmatched_symbol'
        self.pending={'problem':problem,'cursor':current,'action':kind}
        self.counts[reason]+=1
        return Action(kind),{'reason':reason,'selected_slot':current,'examples':problem['example_count'],
                            'input':problem['source']['glyphs'],'desired':problem['desired'],'current':problem['target']['glyphs']}

    def observe(self,before,action,after):
        self.steps+=1;self.fallback.observe(before,action,after);self.latest_feedback={}
        if self.pending and before.levels==after.levels and after.state=='NOT_FINISHED':
            p=self.pending;new=self.read_problem(after.grid)
            if new:
                old=p['problem']['target']['glyphs'];nxt=new['target']['glyphs'];selected=self.read_cursor(after.grid,new['target'])
                if len(old)==len(nxt):
                    changed=[i for i in range(len(old)) if old[i]!=nxt[i]]
                    if len(changed)==1 and changed[0]==p['cursor']:
                        i=changed[0];self.edit_actions.add(action.id);self.edits[old[i]][action.id]=nxt[i]
                    if not changed and selected is not None and selected!=p['cursor']:
                        self.moves[action.id]=(selected-p['cursor'])%len(old)
                    self.latest_feedback={'changed_slots':changed,'cursor_before':p['cursor'],'cursor_after':selected}
                    self.controls[action.id]+=1
        if after.levels>before.levels:self.counts['levels_completed']+=after.levels-before.levels
        self.pending=None

    def diagnostics(self):return {'counts':dict(self.counts),'steps':self.steps,'cursor_moves':self.moves,'edit_controls':sorted(self.edit_actions)}
