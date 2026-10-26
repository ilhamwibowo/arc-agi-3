from collections import Counter, defaultdict, deque
import random
import math
from .core import Action
from .scene import SceneMemory, EffectLearner
from .controller import ExperimentController
from .frontier import FrontierAgent, TerminalAwareFrontierAgent
from .contextual_frontier import ContextualFrontierAgent
from .goal_frontier import GoalFrontierAgent
from .bounded_goal import BoundedGoalFrontierAgent
from .episode_frontier import EpisodeFrontierAgent
from .local_risk_frontier import LocalRiskFrontierAgent
from .retry_frontier import RetryingFrontierAgent
from .anchor_agent import AnchorCalibrationAgent, AnchorAlignmentAgent, AnchorSelectiveAgent
from .assembly_control import AssemblyControlAgent


class RandomAgent:
    def __init__(self, seed):
        self.rng = random.Random(seed)

    def choose(self, obs):
        if obs.state in ("GAME_OVER", "NOT_PLAYED"):
            return Action(0), {"reason": "required_reset"}
        action = self.rng.choice([a for a in obs.actions if a != 0])
        a = Action(action, self.rng.randrange(64), self.rng.randrange(64)) if action == 6 else Action(action)
        return a, {"reason": "uniform_action_type"}

    def observe(self, before, action, after):
        pass


def salient_points(grid):
    """Represent every 4-connected color region by a member near its centroid."""
    height, width = len(grid), len(grid[0])
    seen, points = set(), []
    for y in range(height):
        for x in range(width):
            if (x, y) in seen:
                continue
            seen.add((x, y))
            pending, cells = [(x, y)], []
            while pending:
                px, py = pending.pop()
                cells.append((px, py))
                for nx, ny in ((px-1, py), (px+1, py), (px, py-1), (px, py+1)):
                    if (0 <= nx < width and 0 <= ny < height and
                            (nx, ny) not in seen and grid[ny][nx] == grid[y][x]):
                        seen.add((nx, ny))
                        pending.append((nx, ny))
            cx = sum(p[0] for p in cells) / len(cells)
            cy = sum(p[1] for p in cells) / len(cells)
            center = min(cells, key=lambda p: (p[0]-cx)**2 + (p[1]-cy)**2)
            points.append((len(cells), center))
    return [point for _, point in sorted(points)]


class DirectedAgent(RandomAgent):
    """Observation novelty with action memory; no engine access or forward model."""
    def __init__(self, seed):
        super().__init__(seed)
        self.tried = Counter()
        self.unchanged = Counter()
        self.visits = Counter()
        self.successors = defaultdict(Counter)
        self.recent = deque(maxlen=12)
        self.point_cache = {}

    def candidates(self, obs):
        key = obs.key()
        candidates = [Action(a) for a in obs.actions if a not in (0, 6)]
        if 6 in obs.actions:
            if key not in self.point_cache:
                if len(self.point_cache) >= 256:
                    self.point_cache.clear()
                self.point_cache[key] = salient_points(obs.grid)
            # Salience is a proposal, not an action mask. Uniform exploration
            # retains support over all 4096 coordinates.
            points = self.point_cache[key][:48]
            points += [(x, y) for y in range(4, 64, 8) for x in range(4, 64, 8)]
            points += [(self.rng.randrange(64), self.rng.randrange(64)) for _ in range(16)]
            candidates += [Action(6, x, y) for x, y in sorted(set(points))]
        if not candidates:
            raise ValueError("No playable action")
        return candidates

    def choose(self, obs):
        if obs.state in ("GAME_OVER", "NOT_PLAYED"):
            return Action(0), {"reason": "required_reset"}
        key = obs.key()
        candidates = self.candidates(obs)
        # Uniform exploration preserves action support; candidate ranking below
        # still weights action types by their number of tied candidates.
        if self.rng.random() < 0.15:
            a, _ = super().choose(obs)
            return a, {"reason": "coverage_exploration"}
        ranked = []
        for a in candidates:
            pair = (key, a)
            successors = self.successors[pair]
            revisit = sum(self.visits[s] * n for s, n in successors.items()) / max(1, sum(successors.values()))
            cost = self.tried[pair] + 2 * self.unchanged[pair] + 0.15 * revisit
            if any(s in self.recent for s in successors):
                cost += 1
            ranked.append((cost, self.rng.random(), a))
        cost, _, a = self.select_ranked(ranked)
        return a, {"reason": "least_explored_transition", "cost": cost,
                   "candidates": len(candidates), "observation_visits": self.visits[key]}

    def select_ranked(self, ranked):
        return min(ranked, key=lambda item: item[:2])

    def observe(self, before, action, after):
        key, nxt = before.key(), after.key()
        pair = (key, action)
        self.tried[pair] += 1
        self.unchanged[pair] += int(before.grid == after.grid)
        self.visits[nxt] += 1
        self.successors[pair][nxt] += 1
        self.recent.append(nxt)


class BalancedDirectedAgent(DirectedAgent):
    """Diagnostic ablation: choose an action type uniformly, then rank within it."""
    def select_ranked(self, ranked):
        types = sorted({item[2].id for item in ranked})
        if len(types) == 1:
            return super().select_ranked(ranked)
        selected = self.rng.choice(types)
        return super().select_ranked([item for item in ranked if item[2].id == selected])


class SceneAgent(BalancedDirectedAgent):
    """Balanced exploration with bounded sharing across similar visible scenes."""
    use_effects = False

    def __init__(self, seed):
        super().__init__(seed)
        self.memory = SceneMemory()
        self.effects = EffectLearner()
        self.scene_tried = Counter()
        self.scene_unchanged = Counter()
        self.retrieval_actions = 0
        self.family_actions = 0

    def choose(self, obs):
        if obs.state in ("GAME_OVER", "NOT_PLAYED"):
            return Action(0), {"reason": "required_reset"}
        scene = self.memory.get(obs)
        candidates = self.candidates(obs)
        if self.rng.random() < .15:
            action, _ = RandomAgent.choose(self, obs)
            decision = {"reason": "coverage_exploration"}
        else:
            ranked = []
            for action in candidates:
                pair = (scene.exact, action)
                group = (scene.key, scene.action_keys(action)[0])
                tried = self.scene_tried[group]
                successors = self.successors[pair]
                revisit = sum(self.visits[s] * n for s, n in successors.items()) / max(1, sum(successors.values()))
                cost = self.tried[pair] + 2 * self.unchanged[pair] + .15 * revisit
                cost += int(any(s in self.recent for s in successors))
                cost += .5 * math.log1p(tried) + self.scene_unchanged[group] / max(1, tried)
                if self.use_effects:
                    prediction = self.effects.predict(obs, scene, action)
                    p = prediction["probabilities"]
                    # Visible change is a weak exploration signal; only public
                    # level progress is a strong positive outcome. Animation is
                    # recorded separately and carries no assumed reward.
                    utility = 4*p["progress"] + .3*p["change"] - 2*p["terminal"] - .5*p["unchanged"]
                    novelty = .75 / math.sqrt(1 + self.tried[pair] + .25*tried)
                    cost = .1 * min(cost, 10) - utility - novelty
                ranked.append((cost, self.rng.random(), action))
            if self.use_effects:
                # One offer per type prevents a large click pool from receiving
                # probability mass merely because it has more candidates.
                offers = [min((r for r in ranked if r[2].id == kind), key=lambda r: r[:2])
                          for kind in sorted({a.id for a in candidates})]
                best = min(r[0] for r in offers)
                weights = [math.exp(-(r[0]-best)/.5) for r in offers]
                chosen = self.rng.choices(offers, weights=weights)[0]
                type_probabilities = {str(r[2].id): w/sum(weights) for r, w in zip(offers, weights)}
            else:
                chosen = self.select_ranked(ranked)
                type_probabilities = {str(k): 1/len(obs.actions) for k in obs.actions}
            cost, _, action = chosen
            decision = {"reason": "learned_effect_exploration" if self.use_effects else "scene_exploration",
                        "cost": cost, "candidates": len(candidates),
                        "type_probabilities": type_probabilities}
        group = (scene.key, scene.action_keys(action)[0])
        prediction = self.effects.predict(obs, scene, action)
        self.retrieval_actions += int(self.scene_tried[group] > self.tried[(scene.exact, action)])
        self.family_actions += int(prediction["scope"] == "family")
        decision.update(scene=scene.key, exact_observation=scene.exact,
                        residual_pixels=len(scene.residual),
                        region_action=list(scene.action_keys(action)[0]),
                        scene_action_samples=self.scene_tried[group],
                        prediction=prediction)
        return action, decision

    def observe(self, before, action, after):
        if action.id != 0:
            scene = self.memory.get(before)
            prediction = self.effects.predict(before, scene, action)
            self.effects.update(before, scene, action, after, prediction)
            group = (scene.key, scene.action_keys(action)[0])
            self.scene_tried[group] += 1
            self.scene_unchanged[group] += int(before.grid == after.grid)
        else:
            self.recent.clear()
        super().observe(before, action, after)

    def diagnostics(self):
        return {"scene_unique_queries": self.memory.queries,
                "similar_scene_hits": self.memory.hits,
                "actions_with_shared_experience": self.retrieval_actions,
                "family_prediction_actions": self.family_actions,
                "effects": self.effects.diagnostics()}


class EffectAgent(SceneAgent):
    """Scene memory plus empirical action effects that steer coordinates and types."""
    use_effects = True


AGENTS = {"random": RandomAgent, "directed": DirectedAgent,
          "balanced": BalancedDirectedAgent, "scene": SceneAgent, "effects": EffectAgent,
          "controller": ExperimentController, "frontier": FrontierAgent,
          "frontier_terminal": TerminalAwareFrontierAgent,
          "frontier_context": ContextualFrontierAgent, "frontier_goal": GoalFrontierAgent,
          "frontier_goal_bounded": BoundedGoalFrontierAgent,
          "frontier_episode": EpisodeFrontierAgent,
          "frontier_local_risk": LocalRiskFrontierAgent,
          "frontier_retry": RetryingFrontierAgent,
          "anchor_calibration": AnchorCalibrationAgent, "anchor_alignment": AnchorAlignmentAgent,
          "anchor_selective": AnchorSelectiveAgent, "assembly_control": AssemblyControlAgent}
