# Trace review: run #3 (arc3-dev9b/1), weak games wa30, g50t, bp35 (+ re86 L4-6)

Source: `runs/kaggle/arc3-dev9b/1/output/runs/kaggle/<game>/trace.jsonl`. Model Qwen3.8-Flash-Next with
thinking, 9000 s per game. Tool-call numbers (`#n`) count `tool_call` events per game from 1; step
numbers are the game step of the action. Model `content` is empty in all 620 model calls (all text is
in reasoning, which the trace does not store), so intent is read from code comments, `memory` writes
and what the code checks.

Pace remarks ("at reference", "~1.7x") compare with the environment metadata. They are for this
analysis only and must not go into any prompt.

## 0. Headline

| game | levels | actions per level (step boundaries) | deaths | end |
|---|---|---|---|---|
| wa30 | 4/9 | 49, 207, 180, 162, L5 99 unfinished | 4 GAME OVER (all budget) | deadline |
| g50t | 2/7 | 109, 196, L3 10 unfinished | 2 GAME OVER, 3 LEVEL RESET | deadline |
| bp35 | 4/9 | 21, 44, 89, 68, L5 44 unfinished | 7 GAME OVER | deadline |
| re86 | 6/8 | 23, 37, 48, 124, 163, 150, L7 107 | 0 | deadline |

Where the actions went, in one list:

- **Deliberate budget burning: 93 actions.** RESET is not exposed, so the model burned the move
  budget to get a fresh attempt three times: g50t L2 (58), wa30 L3 (27), wa30 L2 (8). Code comment
  g50t #84: `# Box A is NOT pushable -> level needs 2 shadow deaths (~59 moves); only 28 left. Burn
  budget to force a fresh attempt.` wa30 #59: `# ... Prepare planner, and burn remaining actions to reset.`
- **First attempts that ran out of budget: wa30 L2-L4, 270 actions.** The successful second or third
  attempts were 67, 80 and 62 actions, each well under reference pace. The failed first attempts are
  the whole gap.
- **Unpredicted deaths in bp35: about 115 actions** lost to 7 deaths plus prefix replays. The model
  had written the fatal rules into `memory['model']` before 2 of them.
- **`wm` was never called** (0 of 604 tool calls). Models wrote their own BFS or solvers in wa30 and
  re86, and never checked them against observed transitions. That is where wa30 L2 attempt 2 failed.
- **Time, not actions, ended all four games.** 203 of 620 model calls took over 60 s and used
  25,100 s of 35,800 s. 20 calls hit `finish_reason=length` with no tool call and wasted 4,470 s,
  2,456 s of it in re86.
- re86 L4-6 (124, 163, 150) are at or under reference pace. Little to save there in actions.

## 1. Per game

### wa30 (keyboard, 4/9)

**Mechanic.** A 4x4 player block with a white "nose" for facing moves one 4-px cell per arrow.
SPACE picks up the box in front and SPACE again drops it. Goal: cover every cell of a blue-framed
target area with boxes. A pink bar on row 63 is a move budget of 64 cells. From L2 on, an orange
autonomous agent moves every action, picks up boxes and also delivers them. From L3 the player is
walled off and only that agent can deliver. L4 adds an ejection mechanic.

**When understood.**

- L1 core rules by step ~31; written at #25 right after L1: `memory['mech']="player=green 4x4
  block, white nose edge=facing ... SPACE picks up box in front ... Goal: place boxes on all
  blue-framed silver bar cells ... Timer=pink bar bottom row, decrements slowly."`
- Budget kills: learned at step 119 (`pink 0 ... game_over True`). It was in the L1 note as
  "Timer", but L2 attempt 1 hand-played 70 actions without counting it.
- Orange agent: first tracked at #27 (`'orange': (36, 24, 39, 27)` in `snap`), immediately after
  a batch whose per-action diffs listed `O (36,24)->(36,28)`. The delivery role was understood in
  L3 (#105 `'snake':'grabs wall-cell box by bumping ... carries to bar, SPACE drops'`).
- L4 ejection: #128 `memory['l4']='MECHANIC: player sealed in silver room ... box is EJECTED
  outward by orange piston and slides to blue marker'` (step 466, 30 actions into the level).

**Action attribution (L1-L4, 598 actions).**

| cause | actions | where |
|---|---|---|
| exploration / learning by hand-play | 226 | L1 1-31; L2 att1 50-119 (died on budget); L3 att1 257-315; L4 att1 learning part |
| executing a wrong self-written model | 62 | L2 att2 120-181: carry model wrong, agent unmodeled; covered 5 of 6, budget ran out |
| deliberate budget burn | 35 | L2 182-189, L3 330-356 (`['LEFT','RIGHT']*8+['LEFT']` while writing `memory['l3plan']` for the restart) |
| waiting for the autonomous agent | 82 | oscillations `['UP','DOWN']`, `['LEFT','UP','RIGHT','DOWN']*3` in L3 and L4 |
| executing the winning plan | 193 | L1 32-49, L2 att3 190-256, L3 att2, L4 att2 |

The carry-model error was found only after attempt 2 was spent. #59 code comment: `# corrected
carry BFS: head may move in offset direction (box leads).`

L4 attempt 2 was hand-played after the model's own BFS failed 5 times in a row (#140-#144:
`KeyError: 'SPACE'` x3, `TypeError ... 'int' and 'str'` x2) and then hit the 30 s timeout (#145).

Waiting is a real cost of this game. Doing useful work while the agent moves is the only saving
there, and that is game-specific advice.

### g50t (keyboard, 2/7)

**Mechanic.** An 8x8 lattice maze with a 5x5 player moving one cell (6 px) per arrow. SPACE ends
the current "life": the player returns to start and a silver ghost replays the previous path. HUD
icons count the lives. Standing on a cap moves a red block (gate). The intended solution parks a
ghost on a cap so the gate stays open while the player walks to the goal ring. A 64-cell budget bar
is on row 63.

**When understood.**

- Ghost tracking from #35 (step ~46): `print(i,str(h.action)[:6],'player',pc,'ghost',s)`.
- Full rule only after the L1 win (#58, step 109): `memory['l1']='die on cap->ghost parks on
  cap->gate opens->walk to ring'`. L1 alone took 5,754 s of the 9,000 s.

**Action attribution (305 actions).**

| cause | actions | where |
|---|---|---|
| probes | 9 | L1 1-9 |
| exploration with wrong timing hypotheses | 76 | L1 10-85: died at 63, apparently because the gate closed on the player when the ghost left the cap; SPACE reset at 85 |
| executing the winning L1 plan | 24 | L1 86-109, step-verified loop #57 with `'abort: box back at gate'` check |
| exploration of L2 | 57 | L2 110-166 |
| wrong hypothesis "box A is pushable" | 13 | L2 169-181, ended `BAD (13, 'RIGHT', 'got', (2, 0), 'exp', (2, 1)...` |
| deliberate budget burn | 58 | L2 182-239 (`['UP','DOWN']*14` etc.), ending in GAME OVER |
| executing the winning L2 plan | 66 | L2 240-305, with expected-position checks per step |
| other | 2 | 167 SPACE (LEVEL RESET), 168 SPACE no-op |

Thinking was the binding constraint. 49 calls over 60 s used 7,208 s. 6 calls ended on `length`
with no tool call (1,238 s), for example turn 41: two consecutive 195 s and 190 s calls, then a nudge.

### bp35 (keyboard + click, 4/9, 7 deaths)

**Mechanic.** A vertically scrolling level on a 6-px cell lattice. LEFT/RIGHT moves the player one
cell, then the player slides along the gravity direction until blocked. Clicking a green block
destroys it. Clicking a red block flips gravity. Purple "pots" kill. Reaching the pink cross ends
the level. UNDO is available.

**When understood.** Piecewise, by dying. The full rule set appeared at #113 (step 173, L4), right
after the 5th death: `memory['model']="Level4: player moves 1 cell horiz per LEFT/RIGHT, then slides
vertically along GRAVITY dir as far as possible (walls/blocks stop; pots kill). Green blocks: click
to destroy ... Red blocks: click FLIPS GRAVITY (dangerous: player then slides the other way)."`

**Action attribution (266 actions).**

| cause | actions | where |
|---|---|---|
| probes (incl. most of the 17 HUD-only actions, mostly misclicks) | ~23 | L1 clicks on (55,21), (57,21); L4 #121, #126, #128 `only border/HUD cells changed` |
| lost to deaths: the fatal attempt incl. replayed prefixes | ~115 | L2 22-23 (double click); L3 66-72, 73-94, 95-120; L4 155-172; L5 223, 224-262 |
| UNDO experiments | 5 | L4 192-195, 202; the 5-UNDO loop hit `StaleStateError` |
| executing winning attempts | ~123 | L1, L2 24-65, L3 121-154, L4 173-222 |

Deaths that a simulator of already-known rules would have predicted:

- **L5 step 262:** a LEFT after destroying a green block slid the player into pots, with the
  "pots kill" and slide rules already in `memory['model']`. 39 actions of attempt 2 were lost.
- **L5 step 223:** the first click on the red block flipped gravity onto pots, with "Red blocks:
  click FLIPS GRAVITY (dangerous ...)" already noted.
- **L3 steps 94 and 120:** after the first L3 death, about 48 actions. Partly new information.

The model replayed known-safe prefixes correctly after each death. #86 `prefix ok`, then the same
8-action opening `MOUSE(9,21), MOUSE(9,27), MOUSE(9,33), MOUSE(39,33), R, R, R, MOUSE(33,39)` was
used 3 times. With no checkpointing, that replay is unavoidable once a death happens.

The scrolling view also cost perception effort. #63 `frame_diff(history[16],history[17])` returned
`disappeared: C 952px, C 433px, ...`, which is useless for a scroll. The model then wrote its own
shift matcher (#65: `shift 0 match 0.7998 ... shift 6 match 0.7077`) and a world-row map (#76).

### re86 L4-6 (keyboard, for comparison)

**Mechanic.** Several outline shapes (cross, X, diamond, plus). SPACE cycles the active shape and
arrows move it 3 px. A shape passing over a colour box takes that colour. Goal: every dot covered by
a shape of its colour. From L6, sockets block shapes.

**How it played.** This is the only game where the model routinely planned with code:

- **L4 #29:** a placement solver, `((30, 39), [...], (30, 15), [...], 23)`.
- **L5 #43-#54:** coverage solvers; several returned `combos 0` before one worked.
- **L6 #93-#95:** a real simulator plus BFS, `goal (9, 9, 3, 12) len 19`. Its simulated trace
  matched execution exactly (#96 `H span 9 33`, `V span 3 27`).

L4-6 cost 124, 163 and 150 actions, at or under reference pace. Most extra actions came from
discovering the recolour and blocking rules while executing. The real cost was time: about
800-2,200 s per level and 11 length-truncated calls (2,456 s, 27% of the game).

## 2. Simulators, search and `wm`

`wm.` appears in 0 of 604 tool calls across the four games. In run #1 it appeared in 4 calls, all in
tu93. The 40-line WORLD_MODEL prompt section is not being acted on.

Self-written planners:

| game | what it wrote | verified against observed transitions? | outcome |
|---|---|---|---|
| wa30 L2 | `read(g)` abstract state, `bfs`, `bfs_carry`, `legs3`, `two_leg`, permutation search over box-target assignments (#40-#75) | no | attempt 2 executed a wrong carry model (62 actions), fixed afterwards (#59) |
| wa30 L4 | BFS over (head, facing, carry, boxes) (#140-#145) | never ran | 5 crashes + timeout, abandoned, hand-played |
| g50t | BFS reachability on the walkable cell map (#41, #64, #98) | partially (per-step position checks during execution) | did not model ghosts, so no use for the hard part |
| bp35 | none; cell maps only | no | 7 deaths |
| re86 | placement / coverage solvers (L4-5), plus simulator + BFS (L6) | L6 yes: `sim2` replay matched | L6 path executed exactly |

Actions a verified model plus planner would have saved:

- **wa30 L2: 40-70.** A replay of attempt-1 transitions through `read` and the carry step would
  have shown the carry divergence before #50. The autonomous agent would still have made the model
  imperfect.
- **bp35 L3: 25-48, L5: about 40.** These deaths came from rules the model had already observed
  or written down.
- **g50t: little.** The failed hypothesis cost 13 actions. The 58-action burn is a RESET problem,
  not a model problem.
- **re86: under 10 per level.** It already had one.

The pattern is consistent. The model will write an abstraction and a search over it when the level
calls for it. It does not use `wm` because `transition(state, action)` over the full grid plus
`verify` on interior cells does not fit how it thinks: it reads abstract state from the grid. What
is missing is a cheap check of its own abstraction against history.

## 3. New features

**`[effects]` diff lines.** They cannot be verified directly, because the trace stores raw stdout,
not the model-visible tool result.

- Models still printed `per_action[...]['diff']` themselves in 25 wa30 and 19 re86 calls, which
  duplicates the output.
- One likely help: wa30 started tracking the orange agent (#27) right after diffs listed
  `O (36,24)->(36,28)`.
- Diffs were noise in g50t (animated HUD changes) and in bp35 (the view scrolls).
- They can also be odd. wa30 #13 shows `moved: 1 (W (32,16)->(32,16)); rotated: 1 (G 90deg...)`
  for a turn-in-place.

**`memory` dict.** It was used as write-only notes.

- wa30 wrote 25 keys and bp35 and re86 wrote rule notes.
- Rule notes were never read back: `mech`, `l3plan`, `l4`, `l4b`, `l5`, `l5b`, `model`, `notes`.
- Only plan keys were read, to execute a stored action list (`memory['L3'][0]`,
  `memory['be_leg']`, `memory['legE']`, `memory['plus_path']`).
- No rule note was read back after any of the 30 trims.

**LEVEL RESET diagnosis.** It fired only in g50t, 3 times, every time after SPACE, which is that
game's deliberate "end this life" mechanic.

- The flag was correct: the interior went back to the start.
- The attached explanation ("a life was probably lost: the previous move was fatal or the move
  budget ran out ... Do NOT replay the same sequence") was wrong here. The budget had not refilled:
  #80 `refills: ['63:33->64 REFILL', '109:42->64 REFILL']`, none at 167.
- The model did not act on the wrong part.
- API gap: `last_transition` has no `level_reset`. g50t #52 printed `LEVEL RESET? None` right after
  one.

**Stop reasons.** They are visible only if the model prints `r['stop_reason']`, which it often did
(wa30 54 calls, re86 32). The automatic text omits them:

- The sandbox sets `summary["stop_reason"]` (no_op, level_completed, game_over, level_reset,
  max_actions_per_call) but never `ActionResult.stop_reason`.
- So `describe_result` and `summarize_sequence` never say "then stopped (no_op)" or how many
  actions were skipped.
- The trace's `stops` field is empty in all 604 tool calls, for the same reason.
- Example: wa30 #133 requested `['DOWN','DOWN','SPACE','DOWN','SPACE']`, 2 executed, and the
  model's printout showed no stop. It recovered next turn from the step count.

**New errors and API friction** (35 tool errors in total):

- **Imports do not persist; only functions do.**
  - wa30 #64 `NameError: name 'itertools' is not defined`
  - bp35 #152 `NameError: name 'collections' is not defined`
  - wa30 #59 tried `exec(src)` to persist a planner: `NameError: name 'exec' is not defined`
- **A retained function's source cannot be seen.** bp35 #73/#74 hit `AttributeError: 'numpy.ndarray'
  object has no attribute 'grid'` twice when calling its own `cmap`. #75 `import inspect` was
  refused.
- **Missing fields.** `TransitionView` lacks `level_reset`, `stop_reason` and `step`; wa30 #6 used
  `hasattr(h,'step')`. `history[i].result` lacks `level_reset`.
- **30 s timeout on pure-Python BFS:** wa30 #145, re86 once.
- **Blocked modules tried:** `sys`, `inspect`, `builtins`, `time`. Harmless.
- **StaleStateError:** 3 times, including bp35's 5-UNDO loop (#127).

## 4. Context trims

There were 30 trims, 7-8 per game. Each trim fired at 111-121k prompt tokens and cut to 47-59k.
The token estimate tracks the server count within about 1%.

The main driver is retained reasoning. Each game generated 1.2-1.3M reasoning characters, about
350k tokens, and `keep_reasoning=True` sends all of it back. Each trim drops roughly the last 20
turns of tool calls and results with no summary. `memory` is not re-shown.

Evidence that knowledge was lost and re-derived:

- **wa30, trim at L3 step 315.** The next 4 calls were inspect-only and took 383 s of model time
  before the next action.
  - #99 re-derives what happened in L2 (`# level2: track bar boxes and resets`).
  - #100 re-locates the row-63 budget bar it had been printing as `pink` for 300 steps
    (`---row63 runs---`).
  - #101 re-checks the agent's motion at steps 259-263.
- **g50t, trim at L1 step 38.** #28 recomputes the full maze run map, identical to #18 printed
  before the trim. 206 s passed before the next action. After the L2 trim at step 167, #80
  recomputes budget history (175 s).
- **bp35, trim right after the L3 death at step 72.** 11 consecutive inspect-only calls (#57-#67,
  754 s) reopen L1/L2 frames (`--- history 18 RIGHT level 1`, `frame_diff(history[16],
  history[17])`) to re-learn what the cross is and what killed it at step 23. Part of that is new
  death analysis, but the L1/L2 facts had been established before the trim.
- **re86** was least affected: 25-256 s stalls, and the model mostly re-measured geometry.

Not every trim hurt. 25 of 30 trims were followed by an action within the first two calls. The
cost is time and occasional re-derivation, not wrong actions. No action sequence was clearly
replayed because of a trim.

## 5. Recommendations, ranked

Expected savings are per affected level and grounded in the cases above. Everything here is
game-agnostic unless flagged.

1. **Expose RESET and say when to use it.** It saves 7-57 actions per level where the budget
   would otherwise be burned: 93 actions over 3 levels in this run.
   - Set `expose_reset=True`.
   - Prompt addition to ACTIONS/RESET: "If the move budget left cannot cover your plan, or the level
     is in a state you cannot win from, use RESET now instead of spending the remaining moves.
     Never burn moves to force a reset."
   - Verify locally what a RESET right after a RESET does in these environments; in some ARC-AGI-3
     builds it restarts the whole game. If so, add a sandbox guard that refuses a RESET when no
     action was taken since the last one.
2. **Make the move budget a first-class number.** It saves the hopeless tails of failed attempts,
   about 15-30 actions per level in budgeted games.
   - The wa30 failed attempts each ended with 15-40 actions of play that could not finish: L2
     100-119, L3 316-356 (27 of them burned, see item 1), L4 516-536.
   - Harness: detect an edge bar whose cell count shrinks monotonically with actions. Report
     `budget: ~N actions left (bar -1 per action)` in each opener and in the `[actions]` line, and
     reset the estimate on refill.
   - Prompt: "Before executing a plan, check that its length fits the budget left."
   - Generic: wa30, g50t, ls20 and others use such bars.
3. **Give the model a check for its own abstraction instead of pushing `wm`.** It saves 40-70
   actions in wa30 L2-like cases and 25-48 per level in bp35 L3/L5-like cases.
   - Add `check_model(read, step)`, or `wm.check`. It runs the model's `read(grid) -> state` on
     each observed before/after pair of this level (`all_levels=True` for earlier ones), applies
     `step(state, action)`, and reports the first mismatch as `level, index, action, expected vs
     observed (abstract)`.
   - The model already writes `read` and `step` (wa30 `read`/`legcost3`, re86 `sim2`, bp35
     `cellmap`).
   - Prompt, replacing most of the WORLD_MODEL section: "Before you execute more than about 8
     actions from your own simulator, run check_model on this level's history and fix the first
     mismatch. After a death, add the cause to your step function and re-check."
   - Keep `wm.plan` as an optional search over the same `step`.
4. **Trim reasoning before trimming turns, and re-show notes.** Mostly a time saving: 175-754 s
   stalls after 8 of 30 trims. It also removes re-derivation, and every game ended on the deadline.
   - Change `ContextManager.trim`: first drop `reasoning` from all assistant messages except the
     last 2-3 blocks, then drop blocks only if still over budget. Reasoning is about three quarters of
     generated tokens, so trims should become much rarer while tool code and results stay.
   - After a trim and at each level start, insert one user line with `memory` keys and the full
     `memory['rules']`.
   - Prompt in TOOL: "Keep `memory['rules']` as one line per confirmed rule. It is shown back to
     you after context trims and at level starts."
5. **Bound per-call thinking.** Time only, but it buys levels: 4,470 s across 4 games went to
   20 calls that ended on `length` with no tool call.
   - Use the server's reasoning-budget option if available, at about 6-8k tokens.
   - Otherwise, on `finish_reason=length`, send a one-line nudge right away: "Stop analysing. Write
     the python call now." The current nudge already works; the later call is fast, 5-15 s.
6. **Make stop reasons and skipped actions visible.** Small action savings, higher correctness.
   - In `PythonTool._action`, set `res.stop_reason` on the last executed result when the sandbox
     stops a batch.
   - Make `[actions]` end with `requested N, executed M, stopped: no_op`.
   - The trace `stops` field then works too.
7. **Small API fixes.** They cut errors and wasted calls (35 errors; the 5-crash BFS in wa30 L4).
   - Add `level_reset`, `stop_reason` and `step` to `TransitionView` and its `result`.
   - Preload `collections`, `itertools`, `math` and `heapq` like `np`, or keep top-level imports
     from earlier calls.
   - Add `source('name')` to print a retained function.
   - Allow 60 s for calls that do not act.
8. **Scroll-aware diffs.** Small, but it removes perception work in scrolling games. When most of
   the interior shifts by a constant offset, `diff_line` should report `view scrolled by (dr,dc)`
   and diff objects after compensating. This is generic: camera scroll is a common pattern.
9. **Fix the LEVEL RESET text.** Drop "a life was probably lost" when the edge bar did not refill
   and the last action was SPACE or a click. Say "the board returned to the level start; check
   whether that was the effect of your action" instead. g50t is the only evidence.
10. **Logging for future reviews.** Trace the model-visible tool result text (or at least the
    `[actions]`/`[effects]` suffix) and the last 500 characters of reasoning per call. The trace
    currently cannot show what the model saw or why it acted.

Do not encode game-specific advice: waiting for autonomous agents, carrying boxes, ghost or clone
mechanics, or gravity. Items 2 and 8 rest on generic HUD and camera conventions and need detectors
that fail safe when no bar or scroll is present.

Rough combined effect on these traces:

| game and levels | saving from items 1-3 |
|---|---|
| wa30 L2-L4 | about 150-200 of 549 actions |
| g50t L2 | about 60 of 196 |
| bp35 L3-L5 | about 60-100 |
| re86 | little |

Items 4-5 recover about 15-25% of wall time, which matters because every game ended on the deadline
with levels left.
