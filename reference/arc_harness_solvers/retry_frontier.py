"""Bounded tests of retired frontiers after ordinary observed-route exploration stalls."""
from collections import OrderedDict, deque

from .core import Action
from .contextual_frontier import neighborhood
from .episode_frontier import EpisodeFrontierAgent


class RetryingFrontierAgent(EpisodeFrontierAgent):
    """Revisit a retired control only in a locally unfailed neighborhood.

    Ordinary navigation retains broad terminal retirement. Sixteen executed
    fallback actions permit an observed route toward a retired frontier. Routes
    last at most eight actions; each actual retry or route exhaustion is followed
    by sixteen ordinary actions. A retired control is tested at most once in each
    retained local neighborhood, and existing local/history risk filters apply.
    Retry records are evicted oldest-first at 4096 entries and clear with the map.
    """
    stall_limit = 16
    route_limit = 8
    retry_memory_limit = 4096

    def __init__(self, seed):
        super().__init__(seed)
        self.fallback_streak = 0
        self.probe_remaining = self.route_limit
        self.retry_cooldown = 0
        self.retry_attempts = OrderedDict()
        self.retry_pending = None
        self.parent_fallback = False

    def _new_context(self, obs):
        super()._new_context(obs)
        self.fallback_streak = 0
        self.probe_remaining = self.route_limit
        self.retry_cooldown = 0
        self.retry_attempts.clear()
        self.retry_pending = None

    def retry_key(self, key, region, obs, kind):
        return (key, (*region['position'],region['shape']), neighborhood(obs.grid,region), kind)

    def fresh_context(self, key, region, obs, kind):
        retry_key = self.retry_key(key,region,obs,kind)
        risk = self.context_risks.get(self.risk_key(key,region,obs.grid,kind))
        return retry_key not in self.retry_attempts and (risk is None or risk.failures == 0)

    def permitted_first(self, obs, kind, exception=None):
        node = self.history.get(self.prefix)
        if node and self.edge_blocked(node.edges.get(kind)):
            return False
        for key,region in self.regions(obs).items():
            graph = self.graphs[key]
            if graph.moves < 2:
                continue
            pose = (*region['position'],region['shape'])
            risk = self.context_risks.get(self.risk_key(key,region,obs.grid,kind))
            if risk and risk.avoid():
                return False
            if graph.retired(pose,kind) and (key,pose,kind) != exception:
                return False
        return True

    def retired_route(self, obs, key, region, controls):
        graph = self.graphs[key]
        start = (*region['position'],region['shape'])
        queue = deque([(start,None,0)])
        visited = {start}
        while queue and len(visited) <= 1024:
            pose,first,distance = queue.popleft()
            for kind in controls:
                if not graph.retired(pose,kind):
                    continue
                if distance == 0:
                    if not self.fresh_context(key,region,obs,kind):
                        continue
                    if not self.permitted_first(obs,kind,(key,pose,kind)):
                        continue
                    return {'track':key,'target_pose':pose,'target_action':kind,
                            'action':kind,'distance':0}
                return {'track':key,'target_pose':pose,'target_action':kind,
                        'action':first,'distance':distance}
            if distance >= self.route_limit-1:
                continue
            for kind in controls:
                if first is None and not self.permitted_first(obs,kind):
                    continue
                nxt = graph.prediction(pose,kind)
                if nxt is None or nxt == pose or nxt in visited:
                    continue
                visited.add(nxt)
                queue.append((nxt,kind if first is None else first,distance+1))
        return None

    def choose(self, obs):
        before_count = self.counts['balanced_fallback_actions']
        action,decision = super().choose(obs)
        self.parent_fallback = self.counts['balanced_fallback_actions'] > before_count
        self.retry_pending = None
        if (obs.state != 'NOT_FINISHED' or not self.parent_fallback or
                self.fallback_streak < self.stall_limit or self.retry_cooldown):
            return action,decision
        controls = [kind for kind in obs.actions if kind not in (0,6)]
        routes=[]
        visible = self.regions(obs)
        for key,region in visible.items():
            if self.graphs[key].moves >= 2:
                route = self.retired_route(obs,key,region,controls)
                if route:
                    routes.append(route)
        if not routes:
            return action,decision
        route = min(routes,key=lambda r:(r['distance'],r['track'],r['target_pose'],r['target_action']))
        key = route['track'];region=visible[key];graph=self.graphs[key]
        pose=(*region['position'],region['shape'])
        kind=route['action']
        self.pending={'track':key,'from_pose':pose,'expected_pose':graph.prediction(pose,kind),
                      'action':kind,'evidence_steps':list(graph.steps.get((pose,kind),[]))}
        self.focus=key
        self.retry_pending={**route,'context_key':self.retry_key(key,region,obs,kind) if route['distance']==0 else None}
        self.counts['retired_probe_choices'] += 1
        return Action(kind),{'reason':'bounded_revisit_of_retired_frontier','navigation':self.pending,
            'retry':{**route,'fallback_streak':self.fallback_streak,'route_remaining':self.probe_remaining,
                     'current_neighborhood':neighborhood(obs.grid,region),
                     'terminal_examples':list(graph.terminal_examples.get((route['target_pose'],route['target_action']),[]))},
            'parent_action':action.record(),
            'warning':'A changed neighborhood is not proof of safety; this is a bounded one-time experiment after observed-route exploration stalls.'}

    def observe(self, before, action, after):
        pending=self.retry_pending
        parent_fallback=self.parent_fallback
        executed=before.state=='NOT_FINISHED' and action.id!=0
        boundary=(before.levels != after.levels or
                  (before.state == after.state == 'NOT_FINISHED' and
                   (before.actions != after.actions or
                    (len(before.grid),len(before.grid[0])) != (len(after.grid),len(after.grid[0])))))
        feedback=None
        if executed and pending and pending['distance']==0:
            self.retry_attempts[pending['context_key']]=self.steps+1
            self.retry_attempts.move_to_end(pending['context_key'])
            while len(self.retry_attempts)>self.retry_memory_limit:
                self.retry_attempts.popitem(last=False)
            self.counts['retired_frontier_tests'] += 1
            outcome='progress' if after.levels>before.levels or after.state=='WIN' else 'terminal' if after.state=='GAME_OVER' else 'nonterminal'
            self.counts['retired_retry_'+outcome] += 1
            feedback={'step':self.steps+1,'track':pending['track'],'pose':pending['target_pose'],
                      'action':action.id,'neighborhood':pending['context_key'][2],'outcome':outcome}
        super().observe(before,action,after)
        if executed and not boundary:
            self.fallback_streak=self.fallback_streak+1 if parent_fallback else 0
            if self.retry_cooldown:
                self.retry_cooldown-=1
                if not self.retry_cooldown:
                    self.probe_remaining=self.route_limit
            if pending:
                self.probe_remaining-=1
                if pending['distance']==0 or self.probe_remaining==0 or after.state=='GAME_OVER':
                    self.retry_cooldown=self.stall_limit
                    self.probe_remaining=self.route_limit
                    self.counts['retired_probe_recoveries'] += 1
        if feedback:
            actual=self.regions(after).get(pending['track']) if before.levels==after.levels and after.state=='NOT_FINISHED' else None
            translated=actual['position'] != pending['target_pose'][:2] if actual else None
            feedback['translation_observed']=translated
            self.counts['retired_retry_translations'] += int(translated is True)
            self.latest_feedback['retired_frontier_test']=feedback
        self.retry_pending=None
        self.parent_fallback=False

    def diagnostics(self):
        result=super().diagnostics()
        result['retired_frontier_probes']={'fallback_streak':self.fallback_streak,
            'route_remaining':self.probe_remaining,'cooldown':self.retry_cooldown,
            'remembered_contexts':len(self.retry_attempts)}
        return result
