"""Infer a bounded substitution program from visible input/output and rule slots."""
from collections import defaultdict
from itertools import permutations, product
from symbol_mapping import canonical, glyph_panels, translate


def program_candidates(grid,limit=128):
    panels=glyph_panels(grid);groups=defaultdict(list)
    for a in panels:
        end=a['x']+a['size']*len(a['glyphs'])
        peers=[b for b in panels if b['y']==a['y'] and b['size']==a['size'] and b['color']!=a['color'] and 0<b['x']-end<2*a['size']]
        if peers:
            b=min(peers,key=lambda b:b['x'])
            groups[a['color'],b['color'],a['size']].append((a,b))
    results=[]
    for (left_color,right_color,size),rules in groups.items():
        if not 2<=len(rules)<=8:continue
        bottom=max(a['y'] for a,b in rules)
        inputs=[p for p in panels if p['color']==left_color and p['size']==size and p['y']>bottom]
        outputs=[p for p in panels if p['color']==right_color and p['size']==size and p['y']>bottom]
        for source in inputs:
            for target in outputs:
                if target['y']<=source['y']:continue
                given=tuple(map(canonical,source['glyphs']));wanted=tuple(map(canonical,target['glyphs']))
                current=tuple((tuple(map(canonical,a['glyphs'])),tuple(map(canonical,b['glyphs']))) for a,b in rules)
                bounds=[(len(a),len(b)) for a,b in current];visited=set();answers=set();work=[(0,0,(None,)*len(rules))]
                while work and len(visited)<8192 and len(answers)<limit:
                    i,j,binding=work.pop();key=(i,j,binding)
                    if key in visited:continue
                    visited.add(key)
                    if i==len(given) and j==len(wanted):
                        program=tuple(pair or old for pair,old in zip(binding,current));table={a:b for a,b in program}
                        if len(table)!=len(program):continue
                        predictions=translate(given,table)
                        if predictions and predictions[0]==wanted:answers.add(program)
                        continue
                    for index,(ln,rn) in enumerate(bounds):
                        if i+ln>len(given) or j+rn>len(wanted):continue
                        pair=(given[i:i+ln],wanted[j:j+rn])
                        if binding[index] is not None and binding[index]!=pair:continue
                        assigned=list(binding);assigned[index]=pair
                        work.append((i+ln,j+rn,tuple(assigned)))
                for program in sorted(answers):
                    cost=sum(sum(x!=y for x,y in zip(a,c))+sum(x!=y for x,y in zip(b,d))
                             for (a,b),(c,d) in zip(program,current))
                    results.append({'source':source,'target':target,'rules':rules,'program':program,
                                    'changed_glyphs':cost,'search_states':len(visited),
                                    'scope':'Candidate program consistent with displayed input/output under longest-first substitutions; actual progress must validate it.'})
    if not results:results=composed_candidates(panels,groups,limit)
    return sorted(results,key=lambda p:p['changed_glyphs'])[:limit]


def composed_candidates(panels,groups,limit):
    """Factor a fixed-width expansion through an observed intermediate alphabet."""
    answers=[];states=0
    for (ca,cb,size),first in groups.items():
        for (middle,cc,other_size),second in groups.items():
            if middle!=cb or other_size!=size or cc==ca:continue
            if not (1<len(first)<=4 and 1<len(second)<=4):continue
            expansion=len(first[0][1]['glyphs'])
            if not all(len(a['glyphs'])==1 and len(b['glyphs'])==expansion for a,b in first):continue
            if not all(len(a['glyphs'])==len(b['glyphs'])==1 for a,b in second):continue
            bottom=max(a['y'] for a,b in first+second)
            alphabet=sorted({canonical(g) for a,b in first for g in b['glyphs']}|{canonical(g) for a,b in second for g in a['glyphs']})
            for source in panels:
                if source['color']!=ca or source['y']<=bottom:continue
                given=tuple(map(canonical,source['glyphs']))
                for target in panels:
                    if target['color']!=cc or target['y']<=source['y']:continue
                    wanted=tuple(map(canonical,target['glyphs']))
                    if len(wanted)!=len(given)*expansion:continue
                    direct={};conflict=False
                    for i,g in enumerate(given):
                        chunk=wanted[i*expansion:(i+1)*expansion]
                        if g in direct and direct[g]!=chunk:conflict=True
                        direct[g]=chunk
                    output_symbols=sorted(set(wanted))
                    if conflict or len(direct)!=len(first) or len(output_symbols)!=len(second):continue
                    rules=first+second;current=tuple((tuple(map(canonical,a['glyphs'])),tuple(map(canonical,b['glyphs']))) for a,b in rules)
                    order=sorted(range(len(rules)),key=lambda i:(rules[i][0]['y'],rules[i][0]['x']))
                    for symbols in permutations(alphabet,len(output_symbols)):
                        encoding=dict(zip(output_symbols,symbols))
                        first_rules=[((g,),tuple(encoding[v] for v in chunk)) for g,chunk in sorted(direct.items())]
                        second_rules=[((encoding[v],),(v,)) for v in output_symbols]
                        for left_order,right_order in product(permutations(first_rules),permutations(second_rules)):
                            states+=1
                            if states>65536:return sorted(answers,key=lambda p:p['changed_glyphs'])[:limit]
                            program=left_order+right_order
                            cost=sum(sum(x!=y for x,y in zip(a,c))+sum(x!=y for x,y in zip(b,d)) for (a,b),(c,d) in zip(program,current))
                            answers.append({'source':source,'target':target,'rules':[rules[i] for i in order],
                                            'program':tuple(program[i] for i in order),'changed_glyphs':cost,'search_states':states,
                                            'scope':'Candidate two-stage substitution program using observed intermediate symbols; online control feasibility and progress remain unverified.'})
    return sorted(answers,key=lambda p:p['changed_glyphs'])[:limit]


def rule_cursor(grid,target):
    from collections import Counter
    scores=[]
    for panel in target['panels']:
        x,y,size=panel['x'],panel['y'],panel['size'];width=size*len(panel['glyphs']);score=0
        for yy in list(range(max(0,y-3),y))+list(range(y+size,min(len(grid),y+size+3))):
            background=Counter(grid[yy]).most_common(1)[0][0]
            score+=sum(v!=background for v in grid[yy][x:x+width])
        scores.append(score)
    maximum=max(scores)
    return scores.index(maximum) if maximum and scores.count(maximum)==1 else None


from symbol_mapping import SymbolMappingAgent, mapping_problem, cursor


class InverseSymbolAgent(SymbolMappingAgent):
    def __init__(self,seed):
        self.program_signature=None;self.goal_program=None;self.tried_programs=set()
        super().__init__(seed,problem_reader=self.read_problem,cursor_reader=rule_cursor,
                         equivalence=lambda a,b:tuple(map(canonical,a))==tuple(map(canonical,b)))

    def reachable(self,current,goal):
        if not self.edit_actions:return True
        queue=[current];seen={current};open_edge=False
        while queue and len(seen)<=256:
            state=queue.pop()
            if self.equivalent(state,goal):return True
            edges=self.edits.get(state,{})
            open_edge|=bool(self.edit_actions-edges.keys())
            for nxt in edges.values():
                if nxt not in seen:seen.add(nxt);queue.append(nxt)
        return open_edge or len(seen)>256

    def read_problem(self,grid):
        candidates=program_candidates(grid)
        if not candidates:return None
        first=candidates[0]
        signature=(tuple(map(canonical,first['source']['glyphs'])),tuple(map(canonical,first['target']['glyphs'])),
                   tuple((a['x'],a['y'],len(a['glyphs']),b['x'],b['y'],len(b['glyphs'])) for a,b in first['rules']))
        if signature!=self.program_signature:
            self.program_signature=signature;self.goal_program=None;self.tried_programs.clear()
        values=tuple(panel['glyphs'] for pair in first['rules'] for panel in pair)
        def feasible(program):
            return all(self.reachable(old,goal) for old,goal in zip(values,(part for pair in program for part in pair)))
        if self.goal_program is not None and not feasible(self.goal_program):
            self.tried_programs.add(self.goal_program);self.goal_program=None
        current=tuple((tuple(map(canonical,a['glyphs'])),tuple(map(canonical,b['glyphs']))) for a,b in first['rules'])
        if self.goal_program==current:
            self.tried_programs.add(self.goal_program);self.goal_program=None
        if self.goal_program is None:
            candidate=next((c for c in candidates if c['program'] not in self.tried_programs and feasible(c['program'])),None)
            if candidate is None:return None
            self.goal_program=candidate['program']
        panels=[panel for pair in first['rules'] for panel in pair]
        values=tuple(panel['glyphs'] for panel in panels)
        desired=tuple(old if tuple(map(canonical,old))==goal else goal
                      for old,goal in zip(values,(part for pair in self.goal_program for part in pair)))
        return {'source':first['source'],'target':{'panels':panels,'glyphs':values},'desired':desired,
                'goal_sequences':[desired],'example_count':0,'program_search_states':first['search_states']}


class GrammarAgent:
    """Apply demonstrated rules or infer editable rules according to the visible cursor."""
    def __init__(self,seed):
        self.forward=SymbolMappingAgent(seed);self.inverse=InverseSymbolAgent(seed)
        self.owner=self.forward;self.latest_feedback=None;self.steps=0

    def choose(self,obs):
        if obs.state=='NOT_FINISHED':
            problem=mapping_problem(obs.grid)
            if problem and cursor(obs.grid,problem['target']) is not None:self.owner=self.forward
            else:
                inverse=self.inverse.read_problem(obs.grid)
                self.owner=self.inverse if inverse and rule_cursor(obs.grid,inverse['target']) is not None else self.forward
        action,decision=self.owner.choose(obs)
        return action,{**decision,'grammar_mode':'inverse' if self.owner is self.inverse else 'forward'}

    def observe(self,before,action,after):
        self.steps+=1;self.owner.observe(before,action,after);self.latest_feedback=self.owner.latest_feedback

    def diagnostics(self):
        return {'steps':self.steps,'forward':self.forward.diagnostics(),'inverse':self.inverse.diagnostics(),
                'rejected_programs':len(self.inverse.tried_programs)}
