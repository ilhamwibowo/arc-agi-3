# Trace review: Kaggle run #1 (arc3-plumbing/1), 4 games

Source: `runs/kaggle/arc3-plumbing/1/output/runs/kaggle/<game>/trace.jsonl`. Model Qwen3.8-Flash-Next
with thinking. Turn numbers below are the trace's `turn` counter, tool-call numbers (`#n`) count
`tool_call` events per game from 1. Note: the sandbox in `src/arc3/sandbox.py` has been patched
since this run (history views with `.frame`, `before_frame` aliases, `TransitionView.level`,
dict-style access, `np` preloaded). Errors below that hit those are from the run's older build.

## 0. Headline numbers

| game | levels | actions per level | end |
|---|---|---|---|
| ft09 | 4/6 | L1 13, L2 7, L3 16, L4 16, L5 0 | clock, mid-analysis of L5 |
| ls20 | 1/7 | L1 13, L2 57 (unfinished) | clock |
| tu93 | 3/9 | L1 24, L2 26 (1 game over), L3 19, L4 0 | clock, just reached L4 |
| vc33 | 4/7 | L1 11, L2 10, L3 24, L4 37, L5 8 | clock |

Time, not actions, was the binding constraint in every game:

- Every game's deadline started at `run_start`, but the first model call came ~745 s later
  (server boot). Each game had ~1050 s of model time, not 1800 s. All four ended with
  `model_error: LLMTimeout` -> `abandon: deadline reached` while still making progress.
- Long thinking calls dominate: 14 calls over 60 s used ~1230 s across games. Examples: ft09
  turn 8 (108 s, 29k reasoning chars) and turn 9 (110 s) were spent decoding a mini-map that had
  been mis-sampled; tu93 turn 13 took 121 s (35k chars) before a 5-move batch.
- Prefix cache misses at every turn that follows an action (see section 5): 73 of 78 such calls
  re-prefilled ~48k tokens each (cached ~4-5%), while calls after a yield or inside a turn hit
  97-99%.

## 1. Per game: what it learned, how fast, where it stalled

### ft09 (click puzzle, 4/6)

- Rule: tiles in a lattice around a 3x3 "mini-map"; the mini-map says which neighbour tiles must
  be which colour; a click cycles a tile's colour. Correctly found by L1 end, generalised well.
- L1 (13 actions, 5 wasted): probes on the mini-map icons did nothing
  (`MOUSE(5,14)`, `(13,15)`, `(47,47)`, `(33,33)`, `(34,46)` all "nothing changed"). The real
  stall was a perception bug: the model sampled icons with `g[r0+2*i, c0+2*j]` (#2/#4) and
  got the wrong pattern, then spent turns 7-9 (~4.5 min, two 100+ s calls) trying rotations,
  reflections, XOR/OR/AND of the icons (#13) to explain it. Only after printing an 8x8 crop
  (#15) did it say: *"I misread the mini-map earlier (string indexing). Recomputing the target"*.
  Then 3 clicks solved it with a verify-each-click loop (#16).
- L2: 7 clicks, one call, no waste. Visible text: *"two 3x3 mini-maps ... Target reds:
  (0,0),(1,0),(1,2),(2,0),(2,2),(4,0),(4,1)"*.
- L3 (16): 3 probe clicks on one tile to test the colour cycle, then *"Rule confirmed (all 23
  cells covered, zero conflicts). Executing the 13 mismatch clicks"* -> solved.
- L4 (16): three colours. Refined rule twice (*"white block = map's center color; silver =
  any color != center"*), solved in 4 batches.
- L5: its lattice scanner (`scan4`) broke on new colours (`maps 250` false positives), it was
  rebuilding detection when the clock ran out. It rewrote the scanner 4 times across levels
  (`scan`, `scan3`, `scan4`, `solve`..`solve4`).

### ls20 (maze + key/lock + move budget, 1/7)

- L1 (13 actions): learned quickly that the 5x5 O/B block moves 5 px per action (#4 frame_diff
  `delta [0,-5]`), built a 5-px macro map (#7), walked over the W/V "plus" (which changes the
  key icon in the bottom-left HUD box) and into the goal box. It did not state why the level
  ended. The animation it inspected (#10) showed the goal box flashing `('W', 24, (9,33,15,39))`
  when the plus was touched; this match signal was not picked up.
- **L2, 57 actions, unfinished.** Breakdown:
  1. Steps 14-30 (17 actions): BFS straight to the goal box, treating it as walkable (#16-17).
     Last move blocked: `16 DOWN (35, 14) PRED (40, 14)`. The goal is locked because the key
     shape does not match yet.
  2. Steps 31-35 (5): probes UP x4, RIGHT. These happened to collect a budget refill
     (transition 33 diff shows `(3, 11) 76` cells refilling the bar). Not noticed.
  3. Step 36-41 (6): tried `UNDO` (not valid in this game: `ValueError: UNDO is not a valid
     action now`), then **replanned the same route to the same locked goal**, blocked again at
     `DOWN (35, 14) PRED (40, 14) nothing changed`. Direct violation of the prompt's "a
     sequence that failed once will fail again".
  4. Turns 16-18 (0 actions, 4 failed calls): built a TSP over refill A (15,14), refill B
     (50,39), plus C (45,49) and goal G. The idea was right (`best (43, ('B','C','A'))`), but
     calls failed on `IndexError` (hand-typed map), `history[i].frame` missing, `KeyError:
     'frame'`, and `open('/tmp/plan.json','w')` (trying to persist the plan).
  5. Steps 42-47 (6): spent real actions measuring the bar (`before 26 ... after 18`). The same
     data was free in `transitions`; it computed exactly that later at #44 with zero actions.
  6. Steps 48-59 (12): executed a 37-move plan S->C->G with ~9 moves of budget left. The bar hit
     0 at step 56 and the game **silently reset the block to the level start and took a life**
     (`56 DOWN (10, 44) -> (40, 29) diff 146 ... animated (3996 transient cells)`; a red life
     square turned grey). The harness reported this as a normal "board changed, animated"
     action. No `game_over`, no GAME OVER diagnosis prompt. The batch continued two more actions
     before the no-op guard stopped it.
  7. Steps 60-70 (11): wandered back toward the plus, blocked at (15,44).
  8. Turns 26-30 (0 actions): finally reconstructed the bar history from transitions
     (`(55, 'RIGHT', 0), (56, 'DOWN', 42)`) and at #52 was diffing the L1 plus pickup, which
     shows the HUD key changing (`(9, 5) 4 [(57, 3)...]`). It was about to find the key/lock rule
     when the clock ran out.
- Root causes: no explicit write-down of L1 mechanics (what ended L1?); a blocked move into the
  goal was read as a wall, not as "locked"; the move budget was never part of planning; the
  harness hid a life loss.

### tu93 (lattice maze with facing enemies, 3/9)

- L1: found the 3-px coarse lattice and the "jump 2 cells over a connector" rule by turn 4.
  First hand plan ignored connectors: 8 moves, 6 of them HUD-only no-ops, and the batch kept
  going because the no-op guard is off on level 1 (`guards_from_level`). Then BFS with the
  connector rule (14 moves), then the only `wm` use (section 3) -> solved at step 24.
- L2: a red enemy with a purple "eye". Walked into it: `RIGHT, RIGHT -> GAME OVER` (step 28).
  Then 8 actions of `UP, DOWN` oscillation to test whether the enemy moves (it did not),
  then single-step probing in 4 turns, 60-120 s of thinking each, eventually killing the enemy
  from behind (`UP B (33,36) -> (27,36) R (27,36) -> None`) and finishing at step 50.
- L3: built a facing-aware BFS (`facing: 19 [...]`, `strict: None`), a renderer and a
  step-verified executor of its own (`runplan`, `runplan2`, `runplan3`, with two `KeyError:
  slice(None, 60, None)` bugs), and finished L3 with a 16-move batch. It re-implemented
  `wm.execute` by hand instead of using it.

### vc33 (click "water level" puzzle, MOUSE only, 4/7)

- L1 (11): first 3 clicks hit inert objects (HUD-only: they only advance the step bar in row
  0). Found that the blue buttons on the right shift a boundary between bands, then clicked one
  button repeatedly until the level ended.
- L2 (10), L3 (24): on L3 it abstracted the state to 5 column heights and searched with its own
  BFS (`start=(6,0,6,6,26); target=(16,0,0,18,10)`, `20 BABABACBACBADDDDDDDD`), then executed
  20 clicks. This is a hand-written world model over an abstract state and it worked.
- L4 (37): gates and a lift; mostly single clicks with heavy thinking (turns 24-25: 83 s and
  104 s for one click each). 32 of 53 tool calls acted, median batch size 1.

## 2. Tool usage

| | ft09 | ls20 | tu93 | vc33 |
|---|---|---|---|---|
| tool calls | 39 | 52 | 52 | 53 |
| calls that acted | 14 | 15 | 17 | 32 |
| inspect-only calls | 25 | 37 | 35 | 21 |
| actions | 52 | 70 | 69 | 90 |
| tool errors | 3 | 11 | 8 | 5 |

Batch sizes of `action([...])`:

```
ft09 [1,3,1,1,1,3,3,7,3,13,2,8,4,2]
ls20 [1,3,2,6,1,17,4,1,6,2,4,12,6,4,1]
tu93 [1,1,8,14,1,1,2,8,5,3,2,1,3,1,1,1,16]
vc33 [1,1,1,3,1,2,2,1,5,1,1,2,1,1,2,12,7,1,1,1,6,8,1,1,1,3,1,8,1,5,1,7]
```

Large batches always came from code-computed plans (BFS, solver), and the model usually wrapped
them in a per-step check loop (ls20 #17, ft09 #16, tu93 `runplan`). That habit is good.

Feature use (number of tool calls mentioning it):

| | ft09 | ls20 | tu93 | vc33 |
|---|---|---|---|---|
| raw `.grid` / numpy | 31 | 28 | 43 | 41 |
| `segmentation` | 2 | 22 | 0 | 9 |
| `.ascii` / `.crop` | 0 / 0 | 13 / 13 | 2 / 1 | 6 / 6 |
| `frame_diff` | 0 | 4 | 1 | 0 |
| `last_animation_frames` | 3 | 9 | 1 | 0 |
| `transitions` / `history` | 0 / 0 | 13 / 4 | 8 / 3 | 13 / 3 |
| `wm.` | 0 | 0 | 4 | 0 |
| `last_action_call_result` | 0 | 0 | 0 | 1 |

- The model prefers raw numpy on `.grid` over every structured helper. `frame_diff` was used 5
  times in 196 calls; instead each game hand-wrote a per-action tracker (`bp`/`blockpos`/`bar`
  in ls20, `info`/`ib`/`ia` in tu93, `desc`/`st`/`H`/`HW`/`boat` in vc33, `tiles`/`icon` in ft09).
- `last_animation_frames` was used where it mattered (ls20 #10, #25, #43; ft09 #8-10).
- Images: no visible evidence either way. Visible text was empty (`"\n\n"`) in ls20, tu93 and
  vc33. ft09's 13 visible notes never mention an image. The trace only stores
  `reasoning_chars`, so image use cannot be judged.

Things it tried that did not exist or behaved unexpectedly, beyond the known list (`np` not
preloaded, `.frame`/`.before_frame` missing, `res.frame`):

- `globals()['blockpos']` -> `NameError` (ls20 #19). It was trying to reach a retained function.
- `open('/tmp/plan.json','w')` -> `NameError` (ls20 #29). It wanted to persist a computed plan
  across calls. Variables reset between calls; only functions are kept.
- `action(['UNDO'])` -> `ValueError` (ls20 #23). The system prompt lists UNDO because
  `expose_undo=True`, but ls20 does not offer it.
- `r.get('board_changed')`, `r.get('gameplay_changed')`, `r.get('transient_pixels')` on the
  `action()` result -> `None` (ls20 #4, tu93 #4). These keys exist only inside `per_action` and
  `animation`.
- `transitions[i]['before']` -> `TypeError: 'TransitionView' object is not subscriptable`
  (ls20 #37); `t.level` -> `AttributeError` (ls20 #46, tu93 #27). Both fixed in the current build.
- `history` entries were dicts with keys `index, action, level, gameplay_changed,
  level_completed, game_over` and no frame (ls20 #28, tu93 #7, vc33 #11), contrary to the
  prompt. Fixed in the current build.
- `wm.plan()` result read as `p.get('length')` -> `None` (tu93 #19). The keys are `found,
  actions, plan, nodes, reason`.
- `wm.register(transition, outcome)` with a top-level `def init(...)` not passed in ->
  `KeyError: 'p'` (tu93 #17). The prompt says "add ... an `init=lambda grid, level: {...}`",
  which the model read as "define init".
- `ndarray.tostring()` (removed in numpy 2) in ls20 #45. Minor.
- The interior no-op guard is off on level 1, so tu93's 8-move batch ran 6 HUD-only moves and
  vc33 L1 ran HUD-only clicks without stopping.

## 3. The `wm` world model: all 4 uses (tu93, L1)

1. #17, `wm.register(transition, outcome)` plus `wm.verify()` with a state carrying `p`, `goal`
   and `base`, built by a top-level `init` that was never passed to register:
   ```
   {'ok': False, ... 'dynamics': {'accuracy': 0.0, 'graded': 10, 'correct': 0,
    'first_divergence': {'level': 1, 'index': 0, 'action': 'RIGHT', 'error': "KeyError: 'p'"}}
   ```
2. #18, rewrote everything as a grid-only state (player found from pixels each step), passing
   no init. Verify succeeded:
   ```
   {'ok': True, ... 'summary': 'init ok; dynamics 4/4; outcome ok'}
   ```
3. #19, `p = wm.plan(); print(p.get('length'), p.get('actions'))` -> `(None, [{'action': 'UP',
   'row': None, 'col': None}, ...])`. The plan was the same 14 moves as its own BFS from #15.
4. #20, `wm.execute([{'action':a} for a in acts])` with the list re-typed by hand ->
   `{'executed': 14, 'mismatch_index': None, ..., 'stop_reason': 'level_completed'}`. L1 solved.

So the API worked end to end on the second try, and the plan was optimal. It was never used
again, and never in the other three games. Likely reasons, from the evidence:

- **Pixel-exact dynamics is too expensive for this model.** `verify` grades the whole board
  outside the HUD border. On tu93 L2/L3 that means rendering the player's eye, the enemies'
  eyes and their movement. The model instead wrote abstract simulators (coarse positions plus
  facing in tu93, five column heights in vc33 L3, tile colours in ft09) and checked only what
  it cared about. Its own `runplan` compared a rendered grid and kept failing on details
  (`px_mismatch 18`, `mismatch 9`).
- **Verify replays all levels by default.** After L1 a new mechanic means the model must still
  reproduce L1, which raises the cost of every later registration.
- **The first attempt failed on an API detail** (`init` not passed), which costs a turn and
  gives no feedback about the right fix (the error was just `KeyError: 'p'`).
- **The prompt frames it as optional and gives no example.** WORLD_MODEL is the last section
  and says "(optional, powerful)". There is no worked example of a 10-line transition function,
  no mention of what `plan()` returns, and no rule for when to switch from hand play to the
  model. The free helpers (`induce_motion`, `untried_actions`, `transition_graph`) were never
  called in any game.
- In ft09 and vc33 L1-2 a planner adds little: the solver is a direct computation, not a search.
  ls20 was the game where it would have mattered most (maze, budget, key state), and there the
  model never got the mechanics right.

## 4. Improvements, ranked by expected impact

1. **Detect level resets and life loss in the harness, and treat them like game over.**
   ls20 lost a life and was teleported to the start with no signal (`diff 146 ... 3996
   transient cells`). Flag an action as `level_reset` when the settled board matches the level
   start grid (ignoring the HUD band) after a non-trivial move, or when transient cells cover
   most of the board. Then stop the batch and send the GAME OVER diagnosis prompt with the death
   frame. Expected: avoids silent budget deaths in every budget game.

2. **Add a move-budget line to every opener and to the action result.** The model measured the
   bar three different ways and with 6 real actions (ls20), counted magenta pixels in row 63
   (tu93) and orange pixels in row 63 (ft09). The harness can track the edge bar it already
   classifies as HUD: "Budget bar: 18/42 cells, -2 per action, about 9 actions left; the last
   refill was at step 47." Add to STRATEGY: "Before executing a plan, check its length against
   the budget; if it does not fit, the plan must pass a refill."

3. **Make the per-action effect automatic.** Every game hand-wrote a tracker of "what moved".
   Append to each `per_action` entry (and the `[actions]` line of the tool result) a compact
   diff: "moved O+B 5x5 (40,29)->(35,29); HUD bar -2". Use `frame_diff` internally. Also put
   top-level `board_changed`, `gameplay_changed`, `transient_pixels` on the `action()` result,
   since the model reached for them twice.

4. **Ship a lattice and abstract-map helper.** ls20 (5-px cells), tu93 (3-px cells) and ft09
   (8-px tiles) all re-derived the cell pitch and a coarse map, ls20 about 8 times. Add
   `current_frame.lattice()` returning pitch, origin and a coarse letter map, plus a
   `bfs(passable, start, goal)` helper. This removes the largest share of hand-written code and
   the bugs in it (the `IndexError` on a hand-typed map, ft09's mis-sampled icons).

5. **Persist data across calls.** The model tried `open('/tmp/plan.json')` and `globals()`. Add a
   `memory` dict that survives calls and say so in TOOL: "Variables reset between calls; store
   anything you want to keep in `memory['name']`." Expected: fewer repeated rebuilds and fewer
   errors after yields.

6. **Fix the prefix cache miss on every turn that follows an action.** Calls that start a turn
   after an action re-prefilled ~48k tokens (cached 4-5%, 73 of 78 calls), while calls after a
   yield (text-only resume) hit 97-99%. The visible difference is the new image-bearing opener,
   which makes the image window strip an older image. Test with `send_images=False` or a
   window that never strips (replace old images by a fixed placeholder at insertion time, not
   later), and check the cached tokens. This is wall-clock time, which was the binding limit.

7. **Cap thinking per call and start the game clock when the model is up.** 14 calls over 60 s
   used ~20 min across games, and ~745 s per game passed before the first model call. Lower
   `max_tokens` for reasoning (or use a thinking budget) and push the model to "run code rather
   than reason longer", which the prompt already says but the model ignored on ft09 turns 7-9.
   Start each game's deadline at model-ready, or account for boot time in the scheduler.

8. **Add a "locked/blocked target" rule and a mechanics ledger to the prompt.** ls20 walked into
   the same locked goal twice. Add to STRATEGY: "If a move into a target does nothing, the
   target is gated. Look for what changed when you last succeeded (HUD icons, keys, colours) and
   satisfy that first; never resend a move that was blocked from the same state." To
   LEVEL_START add: "First write down in one line each: what ended the previous level, which
   HUD elements changed during it, and what the goal looked like." ls20 never stated why L1
   ended, and its L1 animation already showed the goal flash on pickup.

9. **Rework WORLD_MODEL so a 27B model can use it.**
   - Show the exact call: `wm.register(transition, outcome, init=init)`, and say `plan()` returns
     `{'found', 'actions', 'reason'}` and that `wm.execute(plan['actions'])` takes it directly.
   - Allow verification on an abstract projection: `wm.register(..., observe=fn)` where
     `fn(grid)` extracts the state the model cares about (positions, heights, tile colours) and
     verify compares `observe(after)` to the predicted state. That matches how the model
     actually modelled tu93 L3 and vc33 L3.
   - Default verify to the current level only, with all levels opt-in.
   - Add a 10-line worked example (a grid walker) and a trigger: "Once a level has a goal you
     can reach by search, register a model before playing it by hand."
   - Advertise `induce_motion()` in the first-turn opener; it answers the probe questions that
     cost ls20 and tu93 several actions.

10. **Make the action list in the prompt dynamic.** Only describe UNDO and RESET when they are
    in `valid_actions` (ls20's `UNDO` error). Turn the level-1 no-op guard on for directional
    games, or at least stop a batch on the first interior no-op whose move was planned.

11. **Log more of the model's reasoning.** Store the first ~1500 chars of `reasoning` in
    `model_call`. Three of four games have no visible text at all, which makes image use and
    hypothesis quality impossible to judge.

## 5. Hidden info and context trouble

- **Hidden info: none seen.** No tool call or output mentions game ids, scorecards or baselines.
  The opener gives the number of levels ("level 2 of 7"), which is the public `win_levels`. One
  tu93 call contained `__import__('sys')._getframe` inside a branch guarded by `if False`, so it
  never ran. No call inspected `wm` internals or the session.
- **Context size:** largest prompt was 109k tokens (ft09), `trims: 0` in all games. Longer games
  will hit trimming, which this run did not test.
- **Repeated prompts:** `nudges: 0` in all games; no repeated openers.
- **State after yields:** the model generally continued where it was (yields were 9, 15, 9 and 2).
  The visible cost is rebuilding data, not forgetting conclusions. ls20 rebuilt the macro map
  and BFS in #15, #16, #22, #26, #27-#29 and #32, and tu93 redefined `extract`/`render`/`sim`
  three times, because variables reset between calls.
- **Cache behaviour** is the real context issue: see item 6 above.
