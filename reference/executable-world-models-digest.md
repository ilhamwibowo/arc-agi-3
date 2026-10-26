# Executable World Models for ARC-AGI-3: Implementation Digest

Sources (all read in full unless noted):
- Tycho paper: https://arxiv.org/abs/2607.28287 (HTML: https://arxiv.org/html/2607.28287; the fetched text was 195k chars, I read all of it except possibly the appendix tail)
- Tycho repo: https://github.com/NIMI-research/Tycho (prompts and templates pulled raw from `tycho/prompts/*.j2`, `tycho/workspace/templates/*.tmpl`, `tycho/workspace/wmlib_template.py`)
- Rodionov 1: https://arxiv.org/abs/2605.05138 (v2 HTML) - code https://github.com/astroseger/arc-3-agents-baseline1
- Rodionov 2 (ablation): https://arxiv.org/abs/2607.15439 (v2 HTML; the per-game tables in Sec. 11 were not fully visible) - agents in `papers/paper02/agents/{ewma_sv_v1.6,twma_v1.6}` of the same repo
- AERA: https://arxiv.org/abs/2605.25931 - code https://github.com/farmountain/aera-arc3-paper

Caveat: the WebFetch tool summarises, so quotes below are verbatim only where I pulled raw files with curl (Tycho prompts, Rodionov v1.6 prompts). Paper-derived quotes are short fragments. Model names (Opus 4.8, Opus 5, GPT-5.6 Sol, gpt-5.5) are as the papers write them.

---

## 0. One-paragraph takeaway

Every approach that scores well uses the same core: a coding agent writes a deterministic Python simulator of the game (state, transition, render, outcome), a verifier replays recorded real observations through it, and a search (A* or BFS) over the simulator finds a level-completing action sequence that is then executed in the real game. The expensive part is the LLM writing and repairing the simulator, not the search. Public-set scores of 83-100 RHAE were obtained only with frontier models (Opus 4.8/5, GPT-5.5/5.6) at xhigh or max effort and thousands of dollars of compute per policy. With gpt-5.4 the same pipeline gets 39-54 RHAE, and a 0.5B Qwen agent (AERA) gets about 0.2. A 27B local model will land far below the headline numbers. The pieces worth borrowing are the verifier, the planner and the outcome/frame-separation conventions. These are mechanical and need no LLM.

---

## 1. Agent loops

### 1.1 Tycho (NIMI-research)

Architecture (docs/ARCHITECTURE.md): `agent` (actor, builder, orchestration modes), `prompts` (jinja templates), `workspace` (evidence store, tools, world-model interface, verifier, planner), `harness` (ARC engine interaction, scoring, resume), `serving` (Anthropic Messages, OpenAI Responses, OpenAI-compatible Chat Completions). All agent-written Python runs in a network-disabled container.

**What the actor sees per turn.** The current decision frame plus its action mask. The actor can submit one control per turn through `take_action`; only the actor changes the environment. Per-turn inlining is configurable (`set_verbosity`): full grid text (rows labelled y00..y63, `|` every 8 columns), the cell-change diff (`on` = exact cells, `summary` = count and bounding box, `off`), and a rendered PNG of the grid. The Opus 5 config uses `vision: true`, `text_grid: full`, `grid_keep: 3`, `image_keep: 3` (older grids and images are evicted from context). The actor also has a Python tool with `grid` preloaded as a numpy array and a `wmlib` helper library. Each frame is saved to disk as `level_L/turn_NNN.{txt,png,json}` with `diff_<a>_<b>.txt` files.

**Tool steps per turn.** Many tool calls are free (reading files, running Python, planner calls); only `take_action` costs a scored action and ends the turn. The paper's budgets are 40 tool steps per turn, 24k answer tokens per LM call, 3,500 LM calls per game, and $750 per game ($1,500 for the selected-policy runs).

**Context management.** The conversation is cleared at each level boundary. Before clearing, a consolidation ("scribe") pass writes level summaries. The workspace persists across levels: actor beliefs (`notes/actor_beliefs.md`), builder notes (`notes/world_model.md`), helper programs, `world_model.py`, and validated plans. Resets archive abandoned attempts instead of deleting them. `history: tail_evict` drops old turns; prompt caching is on.

**Four policies share all of the above** and differ only in who edits the world model:
- `no_world_model`: the actor reasons from evidence and notes only.
- `single`: the actor owns and edits `world_model.py` itself.
- `orchestrator` (best mean RHAE): the actor calls `invoke_builder` when it wants a model. The builder is a separate focused subagent that edits `world_model.py` and returns an advisory report. It cannot act in the environment.
- `trigger`: the harness invokes the builder automatically when verification fails (unusable model, low coverage, transition mismatch, outcome inconsistency) and at new-level and fatal-reset boundaries. The actor does not write models.

**When the actor delegates (orchestrator).** From `wm_orch.j2`, verbatim:

> World modeling is delegated to a focused subagent; you do not write `world_model.py` yourself. Once you have a mechanics hypothesis, or enough evidence that the dynamics look learnable, call `invoke_builder` and put your beliefs in the `reason`: state representation, action effects, objective/outcome/subgoals, and any probe you want modeled. The builder constructs/refines `world_model.py`, verifies it against observed frames, and returns a compact advisory report: scoped dynamics/outcome evidence, plan start and canonical-validation status, remaining assumptions, recommended action + subgoal, or `no model yet`. You decide the action: rely on a plan only from its stated start and when its remaining assumptions are acceptable; keep exploring when current-level evidence is absent or material assumptions remain. Planner advice only searches goals/subgoals grounded in the current model; if the target, route, or off-screen terrain is still unseen or unencoded, no-plan is not proof that the level is impossible. When the report lists open outcome hypotheses, choose an action that distinguishes them and feed the result into the next builder call. Re-invoke after new evidence or a failed prediction - a few times per level, not every turn; reuse the model once built.

**Builder loop.** The builder kickoff (`builder.user.j2`) is short:

> You were invoked at level {{level}}. Current legal actions for the actor are: ... First read notes/actor_beliefs.md (the actor's current hypotheses - a starting point, not ground truth; verify before encoding), notes/world_model.md (your scratchpad from earlier invocations), and notes/animation_evidence.md if it exists. Then inspect the observed transitions, build/refine world_model.py, update notes/world_model.md with what you learned this pass, write your report to {{report_file}}, and reply 'done'.

A `[verify state]` block is attached automatically: initial-render check, current `simulation_accuracy`, per-level breakdown, first divergence, and two compact diffs at that transition (observed delta vs. predicted delta, as run-length text like `9->8 row 12, cols 4-9`). Two PNGs (prev frame and observed next frame) are attached when there is a divergence. Every semantic save of `world_model.py` auto-runs verification on three channels, so the builder iterates edit, then feedback, then edit.

Builder report format (the actor reads this every turn):

```
confidence: none|low|medium|high
model: <one line: what transition you encoded, or 'none - insufficient evidence'>
dynamics: simulation_accuracy=X; matched=M/evaluated=E/observed=O current-attempt transitions; current_level=L matched=m/evaluated=e/observed=o; ... coverage=Y
outcome: <committed outcome() rule> | hypotheses still open: <h1 vs h2> | unknown
outcome_verified: current-level|prior-level|no|no-terminal-yet
plan: start=current Lx turn y|clean Lx; validated=yes|no; length=N|none; assumptions=none|<unobserved facts>
recommended_action: <one legal action, or ACTION6(row,col)> subgoal: <one grounded subgoal>/none
note: <one line for the actor - esp. what action would disambiguate open outcome hypotheses>
```

### 1.2 Rodionov's baseline (Codex-based)

Different architecture: a scripted outer controller wraps the Codex CLI agent (v0.128.0 in the main study, 0.144.1 in v1.6).
- The controller starts the game, passes observations (settled 64x64 ASCII frames plus PNGs) to the agent, tracks RUNNING / LEVEL_COMPLETED / GAME_OVER, and sends fixed prompts. It does not solve anything.
- The agent works until the level completes or hits GAME_OVER, then must stop and report. The agent is forbidden to call RESET itself ("You are forbidden to call RESET by yourself"). On GAME_OVER the controller sends a death-analysis prompt (plus optional simplification prompts), issues RESET, and sends a continuation prompt ("The level has been reset. You have another attempt.").
- Stuck handling: if the agent returns without acting, the controller sends a reminder ("If you do not move, you will lose"). After a second consecutive no-action return it stops. v1.5/v1.6 add a "trouble" prompt when more than 50 extra actions accumulate since the last intervention.
- Recovery: if the process dies or is idle for 30+ minutes, a new conversation is started from the main prompt plus a recovery prompt (reconstruct state from the workspace). Up to 10 recoveries per game.
- Caps: 1,500 environment actions per unsolved level.
- Plan execution: `plan_executor.py` simulates an action list in the model, runs the same actions in the real game, and after each non-terminal step compares the predicted settled ASCII frame with the observed one. It stops on any mismatch and saves mismatch artifacts (real/simulated PNGs, mismatch region, magenta overlay). In paper 1 the agent is told to use it. In paper 2 it exists but no prompt mentions it, and enforcement is left to the agent.

### 1.3 AERA (0.5B Qwen, for contrast)

Three phases driven by a small LLM emitting structured fields: EXPLORE (HYPOTHESIS / UNCERTAIN / NEXT_ACTION / REASON; budget `Bmax = max(5, min(30, floor(0.4 * human_median_L1)))`; stops when the UNCERTAIN field is short), VERIFY (1-3 falsifying actions), PLAN+EXECUTE (PLAN / CONFIDENCE / FALLBACK; return to EXPLORE on any unexpected observation). Prefers ACTION7 (undo) after exploratory moves. No world model code. See sec. 2 for results and why it matters.

---

## 2. World-model interfaces and verification

### 2.1 Tycho `world_model.py` (seed template, `seed_world_model.py.tmpl`)

State is a dataclass the builder owns. Contract:

```python
UNKNOWN = wmlib.UNKNOWN            # -1
TRANSITION_RETURNS_NEW_STATE = True  # transition clones before mutating

@dataclass
class State: grid: np.ndarray      # add fields: avatar_xy, steps_left, parts, ...

def init_state(grid0: np.ndarray, level: int) -> State
def transition(state, action: dict) -> State
    # action = {"action": "ACTION1", "row": None|int, "col": None|int}; read action["action"]
def render(state) -> np.ndarray     # 64x64 int16, colors 0-15, -1 = not determined
def outcome(state) -> str           # "ongoing" | "level_complete" | "game_over"
# optional
def observation_variants(state) -> list   # <=5 alt grids for HUD quantization only
def actions(state) -> list[dict]          # candidate actions for planner (focused clicks for ACTION6)
def subgoals(state) -> list[callable]     # ordered targets
def heuristic(state) -> float             # A* distance estimate
def planner_key(state)                    # hash only dynamics-relevant fields
```

The builder prompt warns that `action == "ACTION1"` "silently scores ~0 on every move" because actions are dicts. State may be a grid, an object list, a counter, or a finite-state controller. Hidden state (counters, phase, off-screen geometry, viewport offset) must live in State fields and be advanced in `transition()`. `wmlib` helpers include `clone`, `composite(parts,h,w)`, `segment` (connected-component object extraction), `diff_text`, `frames()`, `transitions()`, `attempts()`, `death_events()`, `terminal_events()`, `animation_index/grids`, `current_state(model)` (replays history to carry hidden state), `verify`, `verify_outcome`, `plan_bfs`, `plan_astar`, `plan_subgoals`, `validate_plan`. The file is ~85 KB; it is the mechanical core and is reusable as-is (Apache 2.0).

### 2.2 Rodionov fixed interface (`ewma_sv`)

Four files:
- `world_model_engine.py`: `world_model_engine(state: dict, action: dict) -> (new_state, status)` with status in `RUNNING | LEVEL_COMPLETED | GAME_OVER`. State is a dict with an obligatory `level` field. Action is `{"name": ..., "x": ..., "y": ...}`. Models dynamics within one attempt only.
- `world_model_state_io.py`: `initial_state_reconstruction(level_index, initial_frame)` and `state_renderer(state) -> 64x64 int16`. For partially visible worlds, the full initial frame is stored as a level artifact built from all attempts, and the reconstruction loads only that artifact.
- `world_model_main_planner.py`: `planner(state)`, plus optional `level_N_planner_i.py`.
- Hard rule: "It is strictly forbidden to load real game observations into the world model engine in any way." A `apply_render_overrides` hook exists but is flagged as "modeling debt" with a warning when used.
- Also required: `world_model.md` (textual model), `level_N_reasoning_log.md`, `level_N_report.md`.
- Budget rule: "Actions taken in the real game are costly... Simulation inside the world model is free."

### 2.3 Verification

**Tycho (verification runs on every save, three channels):**
1. *Initial render*: `render(init_state(observed_start, level))` must reproduce every observed level start, independent of `transition`.
2. *Dynamics*: replay the most recent attempt of each level from its first frame, threading `transition()`. A transition is graded if the recorded successor differs from its predecessor, or if the recorded grid is unchanged but the model predicts a change (penalises false motion). Visually unchanged no-ops advance the state but are not scored. Reports `simulation_accuracy`, `strict_simulation_accuracy` (exact canonical render), `prediction_coverage` (share of cells not `-1`), `known_cell_accuracy`. Archived pre-RESET attempts are not replay-scored.
3. *Outcome*: `outcome()` must return `level_complete` on every recorded winning terminal, `game_over` on recorded API GAME_OVER states, and `ongoing` on ordinary decision states. It catches missed wins, phantom wins, missed game-overs and false game-overs.

Pass bar: the builder aims for `simulation_accuracy = 1.00`. The harness repair threshold (Appendix C) is: repair when accepted transition match is below 0.999, coverage below 0.75, or the outcome classifier contradicts observations. The 0.75 coverage threshold was "fixed in advance and not swept or ablated."

**Rodionov:** the verifier replays every recorded attempt from level 1 through the current level. It requires the renderer to reproduce the initial settled frame exactly, advances with recorded actions, and requires the model's status to equal the recorded status at every step. Non-terminal states need a cell-exact settled frame; animation frames are excluded from the test. A second planner verifier asks the planner for a plan from the reconstructed initial state of each solved level and requires it to reach LEVEL_COMPLETED in the model. There is a 180 s timeout per verify run ("Your world model is too slow, please consider refactoring it.").

### 2.4 Frame separation (Tycho)

Outcome set {ongoing, level_complete, game_over}. The harness tags frames as decision, transient (animation), completion terminal, fatal terminal, reset initialisation, and next-level initialisation. The actor cannot act from transient frames. Each action is attached to the decision frame it was chosen from. Level completion closes the level, preserves the terminal evidence in `terminal.txt/json`, and advances at no action cost. Game over closes only the current attempt. RESET costs one scored action (the initial unscored RESET aside). Animations are indexed as keyframes (`animation_index`) and loaded on demand rather than put in context. Rodionov uses "final settled" frames for modeling and treats animation frames only as visual clues. For a local pipeline this is the key rule: only (settled frame before, action, settled frame after) triples enter the model's training and verification data.

### 2.5 Planning

Tycho (`seed_plan.py.tmpl`, `python plan.py [auto|astar|bfs|subgoals]`):
- Start state is `wmlib.current_state(world_model)`, i.e. the model replayed through the current level's real history so hidden variables are carried.
- Target is `outcome(state) == "level_complete"`. Search is A* with the model's optional `heuristic` (BFS-like if absent), node cap `TYCHO_PLAN_MAX_NODES` default 20,000, with game-over states pruned. `subgoals` mode chains A* legs through ordered subgoal predicates. If a `planner.py` exists, `auto` uses its `plan(world_model, start_state)` instead.
- Every plan is replayed through the canonical model; only a plan that truly reaches `level_complete` is written to `notes/validated_plan.json` along with a SHA-256 of the expected render at each step, the start-frame hash and the model hash.
- Execution: the next planned action is surfaced only while the model is unchanged, the observed frame hash matches, and the action is legal. The actor re-observes after each action and re-plans on mismatch. There is also an "automatic probe" (a short advisory prefix) that does not write the guarded artifact.
- Click games: `actions(state)` must return a focused set (object centres, switches, targets), never all 4096 cells.
- Example: tu93 level 6 used a 9x13 room graph; A* expanded 189 nodes for a 13-action plan.

Rodionov: free-form. The agent writes `world_model_main_planner.py` itself (typically BFS/A*/custom search). The planner prompt says "Prefer a main planner that is easy to understand, easy to verify, and reusable across levels over one that is heavily tuned to individual situations."

### 2.6 Goal inference

There is no dedicated goal-inference algorithm in either system. It is delegated to LLM reasoning, with scaffolding:
- Builder prompt: "Outcome is a first-class inference, not an afterthought. The objective is never told to you; infer it from what changed at the moments a level ended, not from whatever changes most every turn. A counter draining as the agent acts... is a spent cost, not success by itself." It must keep competing outcome hypotheses with distinct predicted terminals plus the probe action that separates them in `notes/world_model.md`, commit the likeliest as `outcome()`, and report the open ones so the actor can probe.
- Actor prompt (orch): HUD and interface cells are "observations of game state, not decoration" - infer their role from transitions and terminal evidence.
- The outcome verification channel is the only hard feedback: after a win it falsifies bad `outcome()` rules. Before the first win in a game there is no ground truth.
- Rodionov simplification step 3: "The overall goal of the game is the same across all levels and should be expressible in one short sentence."
- Failure mode (Tycho sk48): the mechanics were accurate but the objective wrong, so across 13 builder calls and 314 scored actions no level was completed. "An accurate model can amplify a wrong goal."

### 2.7 Simplification / refactoring (Rodionov)

Run as separate prompted turns after a level is solved, after a reset, or when stuck (skipped for twma and plain ewma). Light prompt on level 1 and early level 2; later, four sequential prompts. Verbatim from the repo:

Step 1:
> The underlying game mechanics is very simple. The current model is likely explaining the observed data with machinery that is too specific, too descriptive, and too close to the exact seen trajectories. Assume, by default, that the real game mechanics are simpler than your current explanation. Your job is to compress the ontology and dynamics, not preserve them - to find the simplest possible world model that explains the previous observations. Refactoring priority: 1. Find where the current world model explains observations too literally instead of modeling a small underlying rule. 2. Remove distinctions that are not forced by evidence. 3. Replace case-by-case behavior with parameterized shared mechanics. 4. Push level-specific facts out of core world-model logic and into compact data/configuration when they are truly necessary

Step 2 asks for a full review listing every overly specific mechanic and, for each: "can this be expressed as a simpler shared rule? can this be reduced to a parameter of an existing mechanic? is this distinction actually forced by evidence?" It ends: "Do not defend the current model... keep simplifying until this pass yields no further meaningful reduction." Step 3 is conceptual: state the game goal in one sentence and the mechanics in a few sentences, revisit earlier levels' frames. Then a planner-refactor prompt (quoted above). Tycho has no separate simplification stage; it relies on the builder prompt: "There is one game engine across levels, so transition() must generalize - do not hardcode per-level branches there."

Other verbatim prompts worth copying:

Death prompt (Rodionov):
> You have reached the `GAME_OVER` state in the current attempt. Use this outcome to improve your world model. Look at the last step and any available visual clues, including animations when they are useful. Consider whether the failure has a simple explanation that can generalize across levels.

Trouble prompt (v1.5/1.6):
> You are having trouble with this level. This is a strong signal that you are probably missing something important, because every level should be solvable in a single attempt. ... Analyze what you have already tried. Remember that every attempt at the same level starts from the same initial state, and the game is deterministic. ... Identify what you may have missed. Formulate a new hypothesis about how to solve the level. Explain what you are going to try in the next attempt and why.

Tycho actor system prompt core (verbatim):
> You are playing an unfamiliar turn-based game on a 64x64 grid of colored cells (colors 0-15). Its objects, mechanics, and goal are not given - you need to discover them by exploring and observing, like a scientist. Each game has multiple levels of usually increasing complexity; a level ends when you reach its (unknown) goal. Your objective: complete each level in as few actions as possible. Benchmark prior: every level is solvable by human players through a manageable amount of interaction and reasoning. RESET is always available to restart the current level/attempt; it is an API recovery control, not a learned game mechanic. ... ACTION1-ACTION4 ... often act like directional moves (ACTION1=up, ACTION2=down, ACTION3=left, ACTION4=right), ACTION5 ... interact/select, ACTION6=click a cell at (row,col) in 0-63, and ACTION7 ... commonly intended as undo. ... Tool calls are free for scoring; take_action commits one environment control and ends the turn. Every take_action call, including RESET, spends one scored environment action ... When the current attempt is unwinnable or clearly more expensive than restarting, prefer RESET over deliberately wasting actions.

Tycho appendix prompt fragments (from the paper): "Treat HUD and interface cells as possible observations of game state, not decoration"; "A draining move or life counter is a cost, not success."; "A plan is validated only when replay through world_model.py reaches level_complete."

Tycho builder key passages (verbatim from `builder.system.j2`): "The grid you see is only part of the true state: counters, palettes, selected tools, a mover's phase, off-screen geometry, and other hidden variables can accumulate as the agent acts. A single frame does not determine the next frame. Hold hidden state in `State` fields and advance it in `transition()`". And: "Planning stays simple by default. Use the generic planner first. Only when an exact pixel-heavy model makes search the bottleneck, add `planner_key(state)` ... or write `planner.py`".

---

## 3. Reported numbers

### 3.1 Tycho (25 public games, 183 levels; one stochastic run per cell)

Matched policy comparison, Claude Opus 4.8, xhigh effort, 3,500 LM-call cap, $750 cap:

| Policy | RHAE | Games solved | Levels | Scored actions | Builder calls | Model exec. rate | Accepted transition match | Level-complete recall | Cost (list) |
|---|---|---|---|---|---|---|---|---|---|
| No world model | 79.07 | 19 | 157 | - | - | - | - | - | ~$5.66k |
| Single | 85.36 | 19 | 162 | 11,576 | 0 | 89.3% | 13.9% | 21.1% | ~$7.27k |
| Orchestrator | **88.49** | 21 | 166 | 10,354 | 147 | 83.5% | 16.2% | 20.0% | ~$5.78k |
| Trigger | 83.07 | 18 | 162 | 9,442 | 1,192 | 52.0% | 88.1% | 77.7% | ~$8.03k |

Selected-policy runs (orchestrator, $1,500 cap):
- GPT-5.6 Sol (max): 100.00 RHAE, 25/25 games, 183/183 levels, 7,766 actions, 98.5 game-balanced human midrank, 660 builder calls, ~$4.47k.
- Opus 5 (xhigh, thinking off in API config): 100.00 RHAE, 183/183, 6,641 actions (14.5% fewer than GPT-5.6; 61% fewer than aggregate human baselines), 100.0 midrank, 130 builder calls, ~15.1k model calls, ~$2.99k (about $120 per game).
- Cost-limit sensitivity: Opus 5 hit 98.62 RHAE at a $250 total cap and 100 by $500; GPT-5.6 Sol hit 89.75 at $250 and 100 by $750 (read as per-game cap per the paper's wording; unclear in the fetched summary).
- Reference: the same official scorecard has plain Opus 4.8 at 1.5 RHAE, versus 88.49 with the Tycho orchestrator (the paper says this is not a controlled component estimate).
- Advice-following (builder recommendation matches the next action exactly): orchestrator 35/68, trigger 552/970, GPT-5.6 644-recommendation run 634/644, Opus 5 109/129.
- Repair recovery (accepted prediction restored after a rejection): orchestrator 29/73 evaluable (39.7%), trigger 354/531 (66.7%).
- Trigger-policy operational: 44.4k LM calls, peak prompt 852k tokens. Context emergency threshold in config: 850k tokens.
- Case studies: cd82 L4 planned 13 actions after matching 38 graded transitions. tu93 L6 98% coverage, 100% accepted match, 189 A* nodes, 13-action plan. tu93 L8 168 graded transitions matched after a repair. tr87 L6 24 actions vs 146 human. ka59 L6 45 vs 132 human; ka59 final level: a long builder phase and then a validated route with 88 remaining actions. ls20 (appendix): 7/7 levels, 478 scored actions, simulator omitted life-loss and respawn.
- Wall time per game is not reported. The paper notes that action-based scores "do not capture this pre-action computation."
- Limitations: one run per config, five of the 25 games informed development, contamination not ruled out, the harness encodes ARC-specific priors (64x64, 16 colours, RESET convention).

### 3.2 Rodionov paper 1 (Codex, GPT-5.4/5.5 high)

- GPT-5.5 high: 15/25 games fully solved, mean per-game RHAE 58.12%.
- GPT-5.4 high: 8/25 fully solved, mean 41.29%.
- GPT-5.5 scored 100% on ar25, ka59, lp85, tr87, tu93 and 0% on dc22. GPT-5.4 scored 100% on cd82, ft09, lp85, tr87, tu93.
- Time/cost: one ChatGPT Pro subscription covered roughly 2-8 games per run. Hard games took up to about 48 h. A full 25-game evaluation took several days. No token counts, no ablations (earlier ones were done on a leaky harness and were dropped).
- Leakage audit: earlier harnesses leaked via env vars, game ID in process args, open internet, and web search; one run downloaded external scorecard material. Results from those harnesses were excluded.

### 3.3 Rodionov paper 2 (nested ablation, 4 variants x 2 models x 2 efforts x 25 games)

RHAE (unsolved games / unsolved levels in brackets):

| Model, effort | Textual (twma) | Executable (ewma) | + Simplification | + Verification (ewma_sv) |
|---|---|---|---|---|
| gpt-5.4 high | 34.16 (19/85) | 33.54 (17/92) | 30.60 (20/88) | **39.16** (18/78) |
| gpt-5.4 xhigh | 40.67 (18/71) | 44.72 (16/73) | 53.10 (14/59) | **53.72** (13/45) |
| gpt-5.5 high | 58.85 (13/36) | 51.16 (13/53) | 58.35 (11/34) | **65.64** (7/22) |
| gpt-5.5 xhigh | 72.51 (6/17) | 69.70 (9/17) | 73.09 (6/14) | **74.78** (5/12) |

Weighted token proxy (millions): textual 63-98, executable 54-104, simplification 127-177, verification 148-222. Verification costs 1.82-3.26x the textual variant and 1.16-1.49x the simplification variant.

Follow-ups (not controlled against the main study): ewma_sv_v1.5 gpt-5.5 xhigh 82.01; twma_v1.6 gpt-5.6-sol xhigh 92.34 (32.3M tokens); ewma_sv_v1.6 gpt-5.6-sol xhigh **98.97**, 0 unsolved, 8,347 actions, 90.75M; twma_v1.6 gpt-5.6-sol max 95.97, 10,111 actions, 30.6M; ewma_sv_v1.6 max 98.77, 7,758 actions, 103.5M. Human baseline total for all 183 levels is 17,135 actions; completed blocks used 41-55% fewer. v1.6 adds the trouble prompt, an accelerated client, and a rule rejecting all actions except RESET after GAME_OVER.

Wall clock (campaign): median 4 h 26 min per playthrough, mean 6 h 8 min, max 53 h 8 min, up to 18 concurrent playthroughs, 32 vCPU / 64 GiB machine.

### 3.4 AERA (0.5B Qwen2.5, Kaggle P100, CPU FP32)

All 25 games: RHAE 0.2116, 4/25 solved (VC33, FT09, LP85, S5I5 or R11L depending on the run). Random and no-explore: 0. Eight repeats: mean 0.164 +/- 0.059. A ReAct baseline with a 50-step budget gets 0.388 (8/25). The paper's real contribution is a diagnosis of the public set: 10 games are solvable in one blind step, 5 more with ACTION6 after one probe, 8 with one action repeated 50-200 times, and ACTION6 with null coordinates reportedly skips 18 games on local arc_agi v0.9.8 (not confirmed on the Kaggle server, and the author does not use it). The author says the 55-game private set is the only real test (its code track entry reached 0.30 RHAE there). This matters for you: public-set results overstate generality, and cheap probing heuristics can pick up a few games for free. Do not rely on the null-coordinate trick; it is a library bug.

---

## 4. Ablations: what hurt and what helped

Tycho (Opus 4.8):
- Orchestrator (actor decides when to build) > single (85.36) > trigger (83.07) > no world model (79.07). The world model itself added 6-9 RHAE.
- Forcing repair after every verification failure (trigger) gives much better simulation fidelity (accepted match 88% vs 14-16%; level-complete recall 78% vs 20%) but lower RHAE (83.07 vs 88.49). It burned 1,192 builder calls and ~$8k vs $5.8k, and was "perfecting the model" instead of playing. Fidelity is not the same as having the goal right.
- The actor following the builder's advice matters: orchestrator followed only 35/68 recommendations, while the stronger models followed 90%+.
- A world model can mislead: sk48 had the right physics and the wrong goal.
- Stronger model > any policy choice: Opus 5 and GPT-5.6 Sol reach 100 with the orchestrator, using 130 vs 660 builder calls.

Rodionov paper 2:
- Capability and effort dominate. All 16 matched comparisons go the same direction (xhigh > high, 5.5 > 5.4), and the gains often exceed variant differences.
- Executable-only is not uniformly better than textual. Textual beat flexible-executable at both gpt-5.5 efforts (58.85 vs 51.16; 72.51 vs 69.70).
- Simplification helps in 3 of 4 settings (+3.4 to +8.4), hurts at gpt-5.4 high (-2.9).
- Fixed-interface + exact-replay verification ranks first in all 4 settings. Margin over simplification: +0.6 to +8.6. Cost is 1.2-1.5x more.
- At gpt-5.6-sol max the plain textual model-maintainer completes everything at 41% fewer actions than human at one-third the tokens of ewma_sv. So the machinery is not necessary once the base model is strong enough.
- Paper 1 caveat: the "verification" bundle includes fixed templates, verifiers and planners, so the gain cannot be attributed to replay alone.

---

## 5. Assessment for a 27B local model (RTX PRO 6000, ~600 tok/s aggregate, 9 h, 110 games, no internet)

### 5.1 Budget arithmetic

9 h x 3,600 s x 600 tok/s = ~19M generated tokens total, or about 175k tokens per game if you split evenly. Opus 5 used ~$120 per game, ~140 LM calls per game on average (15.1k calls / 110? no: 15.1k calls over 25 games = ~600 calls per game), and its prompts regularly reach hundreds of thousands of tokens of context. Rodionov's pipelines used 30-220M weighted tokens for 25 games, i.e. 1.2-9M tokens per game. You have roughly 2-5% of that. Prefill is a second cost: a 64x64 grid as text is ~4k-8k tokens per frame, so you cannot paste many frames. Use compressed diffs.

### 5.2 What depends on frontier reasoning (do not expect from a 27B)

- Open-ended goal inference from a handful of frames (sk48-type failures happen even to Opus 4.8). Every system here leaves this to the LLM, and gpt-5.4 vs gpt-5.5 shows steep capability scaling (39 to 65 RHAE at identical scaffolding).
- Writing a 200-500 line exact simulator with hidden state and iteratively repairing it over dozens of tool calls with long context.
- Multi-level knowledge transfer ("later levels extend earlier mechanics") and the conceptual simplification pass.
- Long-horizon agentic stability: Tycho allows 40 tool steps per turn and 600+ LM calls per game; small models derail on tool-call formatting and long contexts.
- Honest estimate: the 0.5B AERA gets 0.2 RHAE; gpt-5.4 with the full stack gets 39-54. A 27B, quantised, with a heavy token cap, plausibly lands somewhere between, and probably in the low single digits to low teens if you only run an LLM-driven loop. I would not budget for the paper numbers.

### 5.3 What is mechanical and feasible (borrow these)

1. **Frame bookkeeping.** Settled-frame extraction, animation separation, per-level/attempt history, hash-anchored plan execution. All zero-LLM. Reuse Tycho's `wmlib_template.py` structure (segment, diff_text, verify, verify_outcome, plan_bfs/astar, validate_plan) and the status tags. This is the highest-value borrow.
2. **Exact replay verifier** with the three channels (initial render, dynamics, outcome) and first-divergence reporting as a compact run-length diff (`9->8 row 12, cols 4-9`). The diff format keeps LLM prompts short (tens of tokens rather than a full grid) and is exactly what a 27B can act on.
3. **Search over the model.** BFS/A* over a Python `transition` with a node cap (20k default in Tycho) and canonical replay to validate. Cheap on CPU. Planning is not where models fail.
4. **Focused click candidates** from connected-component segmentation (object centres, distinct colours, HUD-excluded) rather than 4,096 cells.
5. **Cheap probing before any LLM call.** Per AERA's taxonomy, many public games fall to one action or a repeated action. A scripted opening probe (each legal action once, then each repeated N times, plus ACTION6 on a few salient components) costs few actions and produces the transition data the model needs. Compare this against the human-relative scoring: action efficiency penalises waste, so cap it.
6. **Light-weight state, not full simulators.** The builder prompt's insight that most state is `grid` plus a few hidden fields is right: start with `State.grid` and a `transition` that only moves/recolors a small number of components.

### 5.4 Concrete simplifications for your regime

- **Model movement only.** Restrict the LLM's modeling job to: (a) which colour/component is the avatar, (b) what each ACTION1-4 does (translation vector, blocked-by which colours, push/pull), (c) which colour cells are walls/hazards (cause game_over) and which are goal/pickup. Encode as a small *parameterised template* the LLM fills in as JSON/a few lines of Python instead of a free-form simulator. You can even induce (a) and (b) without an LLM: for each (action, recorded transition) compute component-level motion via `segment` on prev/next and find a consistent translation. Reserve the LLM for naming the goal.
- **Goal as a ranked hypothesis list, tested by BFS.** Enumerate a small fixed menu of goal predicates ("avatar overlaps colour c", "all cells of colour c removed", "pattern A matches pattern B", "counter reaches 0"), run the planner for each against the movement model, and execute the shortest plan first. A win confirms the predicate; a failure (no terminal) rules it out and the plan prefix still counts as exploration. This replaces open-ended inference with search, which is what Tycho's "open hypotheses plus separating probe" does in prose.
- **Use the LLM as a hypothesis proposer, not a code author.** Prompt it with: compressed object table (from `segment_summary`: colour, bbox, size), the run-length diffs for the last 5-10 transitions, the legal actions, and terminal evidence. Ask for a short JSON (avatar colour, per-action effect, suspected goal predicate from the menu). Short outputs fit the 600 tok/s budget and a 27B's reliability.
- **Per-game token and wall caps** with a global scheduler. For example ~150k generated tokens and ~4 min wall per game on average, with more for games where the model verifies at >=0.99 and a plan exists, and an early cutoff to scripted exploration (or abandon) when verification fails twice. Abandoned levels cost nothing except RHAE for that level, which the metric scores per level.
- **Skip simplification and the trigger loop.** Both cost 1.2-3x tokens for single-digit RHAE gains on strong models and showed no gain (even a loss) on weaker ones (gpt-5.4 high: simplification 30.60 vs 33.54). The one thing worth keeping from simplification is the instruction "do not hardcode per-level branches in `transition()`", a single line in the prompt.
- **Textual fallback.** Rodionov's textual variant (no executable simulator) is nearly as good as the executable ones at gpt-5.5 level and 2-3x cheaper. For a 27B, a concise textual "mechanics + goal + next probe" note carried across levels is the cheap path when code generation fails.
- **Reset discipline.** RESET is a scored action. Replay verified action prefixes to the interesting state rather than exploring fresh. Make the harness reject non-RESET actions after GAME_OVER (v1.6 rule).
- **Click games.** Candidate set = one click per segmented component plus a few grid-lattice probes; avoid per-pixel clicking.
- **Do not reproduce Tycho's context size.** Keep only the last 1-3 grids, always as diffs plus the object table, never full 64x64 text unless the model asks (the `set_verbosity` idea).

### 5.5 Suggested minimal pipeline

1. Scripted probe: RESET, each legal action once (and ACTION6 on top-K components), record settled-frame diffs.
2. Mechanical induction of movement: avatar candidate = component whose position changes under directional actions; build a component-level `transition`.
3. LLM call (single, short): fill in the goal predicate from the menu, using the object table plus any terminal/level-complete frame. Retry with the last failed predicate excluded.
4. A* / BFS in the model; validate by replay; execute with hash-anchored per-step check; re-plan on mismatch (cap re-plans).
5. On level-complete: store (frames, actions, predicate) as the prior for the next level; first try the same predicate and movement model.
6. On repeated failure: fall back to a bounded scripted/random-with-memory explorer and move on.

Verification target: the Tycho three-channel check. Even with a coarse model, "accepted transition match >= 0.999 on graded transitions" is a reliable gate for trusting a plan.

### 5.6 Risks

- The papers' results are from the public set; three of the four papers say contamination or development on those games cannot be ruled out, and AERA shows the public set is partly trivial. The Kaggle private set (55 games in AERA's account) is harder and the AERA 0.5B agent got only 0.30 there.
- RHAE is action-relative to human baselines, and none of the papers charge for thinking time. Your constraint is thinking time, so the papers' trade-off (spend more compute to save actions) is inverted for you: every 1k tokens of reasoning has a real wall-clock price, and scripted exploration actions have a real score price. Tune against both.
- Tycho's orchestrator delegation depends on a capable actor to decide when a model is "learnable". With a weak actor, a harness trigger (cheap checks: no verified model yet and >=N transitions recorded) is safer than model discretion, but note trigger lost to orchestrator at the frontier-model scale because it over-invested in repair; cap builder calls per level (for example 2-3).
