# Trace review: run #5 (arc3-full25/1), games cd82, vc33, tn36, dc22, ka59, ft09

Source: `runs/kaggle/arc3-full25/1/output/runs/kaggle/<game>/trace.jsonl`. 25 games, 3 h per game,
10 decode slots shared by 25 games. `MCn` numbers model calls per game from 1, `#n` tool calls.
Model `content` is empty in 317 of 319 calls; intent is read from code, comments and `memory`
writes. Reference per level is the human baseline in `reference/dfranzen-m2-pergame.md`.

## 0. Headline

| game | levels (ref) | actions per level (human) | model s | slot wait s | calls | cuts (s) | errors | trims | RESET |
|---|---|---|---|---|---|---|---|---|---|
| cd82 | 1/6 (6.0) | 58 (55); L2 7 unfinished | 2905 | 7847 | 45 | 2 (356) | 4 | 1 | 0 |
| vc33 | 3/7 (6.0) | 11 (7), 8 (18), 23 (44); L4 76 (61) | 5214 | 5580 | 72 | 6 (1370) | 7 | 3 | 5 |
| tn36 | 1/7 (4.2) | 13 (32); L2 97 (72) | 3126 | 7698 | 60 | 2 (421) | 2 | 2 | 2 |
| dc22 | 0/6 (3.8) | L1 104 (59) | 1910 | 8952 | 37 | 2 (305) | 5 | 1 | 3 |
| ka59 | 1/7 (2.8) | 102 (28); L2 13 (109) | 2747 | 7879 | 45 | 7 (1218) | 3 | 1 | 1 |
| ft09 | 4/6 (5.5) | 3, 8, 14, 16 (43, 12, 23, 28); L5 53 (65) | 3157 | 6677 | 60 | 4 (760) | 8 | 2 | 0 |

"cuts" = calls that ended on `finish_reason=length` with no tool call. Totals over the six games:
319 model calls, 299 tool calls, 605 actions, 19,059 s of model time, 44,633 s of slot waiting
(69% of the 64,900 s wall time). All six ended on the deadline.

Where the model time went:

- **117 calls over 60 s used 14,791 s (78% of model time)** and were followed by 197 actions, 26
  of them single-action probes. Average decode speed was 25 s per 1k tokens (19.5 in run #3), so a
  cut at the 8,000-token cap costs 150-285 s.
- **23 cuts wasted 4,430 s (23%).** The nudge call after a cut thinks for 64-1,166 characters,
  takes 2-63 s and then acts on a small batch. The cut reasoning is kept in context, so the plan
  is not lost, but the whole call's output is. Run #3 had 24 cuts in 1,248 calls (1.9%) at a 12k
  cap; run #5 has 79 in 2,249 (3.5%) at 8k. Lowering the cap raised the cut rate without lowering
  the cost of a cut much.
- **The slot wait after a trim.** Every one of these games trimmed once at 1,730-2,220 s, released
  its slot (agent.py:330) and then waited 1,792-8,952 s (cd82 7,847 s, dc22 8,952 s, ka59 7,879
  s, ft09 6,677 s). The priority formula (scheduler.py:83) decays with tokens spent on the current
  level, so a game that has thought hard on a level is exactly the one that does not get the slot
  back. cd82 had its L2 plan found in #43 when the deadline hit. Known topic; noted once.
- `wm` was used in 8 of 299 tool calls (vc33 6, dc22 1, ka59 1); `wm.check` and
  `wm.plan_abstract` never. RESET was used 11 times, mostly well (budget nearly out, or lost
  position). 6 of those RESET calls then raised `TerminalStateError` (see fix 3).

## 1. Per game

### cd82 (keyboard + click, 1/6)

**Mechanic.** A cup orbits a 10x10 square in 8 positions (arrows move it one position; a move
into the same position is a no-op). SPACE pours: it paints the half-plane of the square on the
cup's side with the swatch selected in the HUD (clicks at (4,37) white, (4,43) purple, (4,46)
orange from L2). The target pattern is the 10x10 icon at rows 3-12, cols 3-12. A row-63 bar
shrinks about 1 per action.

**What it believed and when.** Rules correct by MC38 (step 50), stated in the only non-empty
`content` of the game: "the cup orbits the square; SPACE pours and paints the half-plane on the
cup's side ... Target = top half white / bottom half purple". L1 was won at step 58 (human 55).
L2 (3 colours, diagonal target) was analysed with a depth-first search over 8 masks x 3 colours
(#41 wrong, `none`; #43 right, `depth 2 [('SE','O'),('N','U')]`) - and the deadline hit before
the 2-pour plan could run.

**Where it went wrong.** Not actions: time. L1 took 2,507 s of model time over 35 calls; 18 calls
over 60 s took 2,472 s and produced 29 actions, 5 of them single probes (MC17-MC19, MC24, MC26:
128-165 s each for one `UP`/`RIGHT`). Two cuts (MC15 136 s, MC22 220 s). #14 and #16 crashed on
numpy shape errors after 145 s and 128 s of thinking. Then the trim at 1,979 s released the slot
and the game waited 7,847 s.

**Single most useful change.** A thinking budget per call (fix 1): with the same 2,905 s at 40 s
per call cd82 gets about 70 calls instead of 45, enough to execute L2 and probably L3.

### vc33 (click, 3/7; 7/7 in run #3)

**Mechanic.** Containers of fluid with pour buttons on the bottom row; a click moves fluid
between neighbouring containers, raising one level and lowering another. Coloured plugs float on
the fluid and must reach the gap of the same colour in the dividing walls. From L3 the plugs can
be clicked. A top-row pink bar is a move budget (64) and some clicks cause a `LEVEL RESET`
(steps 5, 60, 94).

**What it believed and when.** L1-L3 were understood from the diffs: `memory['l3_init']` (#26)
records fluid edges per container and the plug/gap pairs; L3 was solved in #28 with an 18-click
batch built from the edge model (23 actions, human 44). L4 (5 containers, 2 plugs) was never
modelled: 76 actions of alternating `MOUSE(62,40)`/`MOUSE(62,52)` and `(62,46)`, three RESETs
and two level resets, each batch checking `edges4` afterwards but never predicting them.

**Where it went wrong.**

- Breadth of probing. Run #3 clicked the plugs at (40,28), (46,5), (50,35), (48,13) within the
  first 23 actions of L4 and won in 42. Run #5 clicked (48,13) and (44,5) once (steps 46-47, both
  no-ops at that fluid level) and never again.
- Time: 34 calls over 60 s (4,450 s); 6 cuts (1,370 s, 26% of model time), four of them in L4
  (MC40, MC45, MC57, MC60). L4 alone took 3,924 s of model time.
- A `wm` rabbit hole: MC51-MC55 (5 calls, ~400 s) poked at `wm.goal_menu()` predicates with
  grids and FrameViews, which they do not accept (wm.py:968 expects `state['grid']`), then tried
  `inspect` and `__code__`. `wm.induce_motion()` returned `{'actions': {}}` for a click game.
- `action([RESET, click, click...])` twice hit `TerminalStateError` (#56, #61), the second after
  199 s of thinking.

**Single most useful change.** An untried-objects list in the opener (fix 2) would have put the
two unclicked plugs in front of the model at the start of L4; run #3 shows that is the level.

### tn36 (click, 1/7)

**Mechanic.** A panel of toggles (bar and stem per column, 4-5 columns, from L2 3 rows) and a
launch button (L1 (55,36), L2 (58,46)). Launching plays an 8-frame animation: a ball runs along
the toggle row and a yellow plug drops; the toggle pattern decides where the plug lands. L1: set
all 5 columns on, launch, done (13 actions, human 32). L2: two panels; the right one has 3 rows
x 4 columns, a left panel is silver (inactive), and two target boxes at rows 53-62, cols 3-25
show yellow shapes with white dots.

**What it believed and when.** L1 rule by #12: "slot bar/stem" states read from rows 42-46; all
on, launch, level complete. In L2 it guessed "pairs per row" (`pairs(st)` #44) and tried 15
toggle configurations, each followed by a launch: 15 launches, all `only border/HUD cells
changed`. It printed the target boxes twice (#27, #56) and never used them. Budget ran to 19 and
it RESET (step 56, right call), then repeated the same style of probing; a second RESET at 73.

**Where it went wrong.** 97 actions on L2 with no hypothesis that linked the target boxes to the
toggles. The animation readouts (#23-#25, #29-#33) showed the plug dropping rows 24-27 -> 28-31 and
vanishing, i.e. the first row's result, but the model did not ask which toggle row controls
which plug. Two cuts (422 s); 19 calls over 60 s (2,189 s). Slot wait 3,511 s + 2,199 s.

**Single most useful change.** A novelty counter in the opener (fix 4: "last 20 actions reached
k new interior states, the launch button has been a no-op 15 times") plus the unclicked-objects
list (the target boxes, the left panel's silver toggles) would have broken the loop by step ~40.

### dc22 (keyboard + click, 0/6; 4/6 in run #3)

**Mechanic (from run #3's `memory['rule']`, #26 there).** "green 2x2 knob moves 2 cells/press;
can occupy cells of piece colors (silver, blue, red, maroon) not dark gray; clicking the red hat
(19,48) toggles the red piece between (30,12) and (24,18); clicking the blue hat (36,48) toggles
the blue piece between (20,18) and (34,8); goal: reach the yellow 2x2 at (20,24) in the silver
box". Run #3 solved L1 in 60 actions by toggling pieces to build a path.

**What it believed in run #5.** By step 26 it knew the knob moves 2 px and both hat toggles
(`snap`, `fullsnap`, `redpos`, `bluepos`). It never formed the passability rule. From step 27 on
it ran blind sweeps: two identical 8-move "tours" (#19, #22) with a blue click between, a 9-move
sweep (#24), six clicks on HUD/dividers (#25), five more on pieces (#33), an 18-step random walk
(#34, crashed on `changed_cell_count`), and `sys`/`globals` introspection (#35, #27-equivalent).
104 actions, 0 levels, 3 RESETs.

**Where it went wrong.** The evidence for the rule was on the screen by step 12: `LEFT no change`
at col 8, `UP` blocked at row 38, while moves inside the checker and over pieces worked.
`wm.induce_motion()` (#32) even listed `blocked_cases: [6, 7, 8, 18, 42, 51]` for LEFT, but
nothing summarised what the blocked target cells looked like. `wm.untried_actions()` said every
action was untried "here" (see bug 2). Time: 13 calls over 60 s (1,348 s), 2 cuts (305 s); then
the trim at 1,836 s and an 8,952 s wait, so only 37 calls in 3 h.

**Single most useful change.** A blocked-move summary (fix 2b): "LEFT was refused 4 times; the
cells it would have entered were colour D; moves that succeeded entered colours B, S, R". That is
the rule run #3 wrote by hand.

### ka59 (keyboard + click, 1/7)

**Mechanic.** Two 3x3 rings (G) with a dot inside (W = controlled, K = idle) move 3 px per arrow
inside an ivory region; a ring pushes the other when they touch. Clicking a ring makes it the
controlled one; the old position keeps a dark dot that can be clicked to switch back. Each ring
has to end inside a 5x5 dark frame (D). L1 won at step 102 (human 28).

**What it believed and when.** Movement and pushing by step 7 (`gd` diffs, animation frames
#6). The switch-by-click rule only at step 93 (#30: `click block: [(31,43,5,0),(40,22,0,4)]`),
after which L1 took 9 more actions. In between: a 23-move and an 18-move serpentine "visit every
cell" sweep (#21, #23, `path_to`, 41 actions), 7 `RIGHT` into a wall (#28), and a click on the
controlled ring itself (step 27, no-op) which was read as "clicks do nothing".

**Where it went wrong.** The other ring was never clicked for 92 actions although it was one of 4
non-background objects. Time: 7 cuts = 1,218 s, 44% of ka59's model time (MC13, MC17, MC23,
MC26, MC28, MC33, MC44, each 145-217 s); after each cut the nudge call acted within 3-19 s with
a 146-794 character thought, which is where the blind sweeps came from (MC24 after MC23, MC27
after MC26). Trim at 1,867 s, then 7,879 s of waiting. L2 was going well (13 actions, ring
selection used correctly at #36 and #38) when time ran out.

**Single most useful change.** Unclicked-objects list (fix 2): "4 objects never clicked: G ring
at (30,27), ..." at the top of the opener. Second: the thinking budget (fix 1).

### ft09 (click, 4/6)

**Mechanic.** A grid of 6x6 cells, each one of two or three colours, toggled by clicking. Some
cells hold a 3x3-block sprite whose centre colour and marks encode the colours the 8 neighbouring
cells must have. When all constraints hold the level completes. L1-L4 were solved in 3, 8, 14 and
16 actions (human 43, 12, 23, 28): the constraint solver (#12) found a consistent assignment and
one batch executed it.

**What it believed and when.** L5 (7x7, 11 sprites, two colours plus a legend sprite at (0,6))
broke the rule. The model did the right things: checked the L4 win frame against its rule (#46:
"W = sprite centre colour, S = not centre"), enumerated 665 then 256 mark-meaning variants (#31,
#39), found the maximal consistent sprite subset (#54: 5 of 11), and a shifted reading that
covers all 27 cells (#57, shift (-1,-1)). Its two full assignments (18 clicks #34, 20 clicks
#47) did not complete the level, so a rule element is still missing (perhaps the legend sprite or
the 'a' blocks). 53 actions on L5 against a human 65; not wasteful.

**Where it went wrong.** Time and friction, not actions. 19 of 38 turns ended on `yield` (no
action after 2,048 generated tokens), 4 cuts (760 s), 16 calls over 60 s (2,133 s). The board
parser (`for i in range(7): for j in range(7): b=g[R0+8*i...]`) was re-typed in 12 calls because
every helper that referenced it was refused: 10 `[not retained] ... uses spr, cells which is not
available later` messages, plus `NameError: name 'spr'/'pat'/'opts' is not defined` in 3 calls
that assumed retention. Trim at 1,733 s, then 6,677 s of waiting; the last 1,000 s of model time
after the wait re-derived the sprite table from scratch (#50-#57).

**Single most useful change.** Retain helper functions by snapshotting the module-level values
they reference (fix 5), or at least tell the model to park parsed tables in `memory`. Second: the
thinking budget.

## 2. Ranked generic fixes

1. **Bound thinking per call; make cuts cheap.** 23 cuts = 4,430 s (23% of model time) across
   these games; 117 calls over 60 s = 78%. Options, in order of preference: (a) a server-side
   thinking budget if the SGLang/Qwen template supports one (verify; then set about 4-5k tokens
   and keep `max_tokens` 8k for the code), (b) otherwise `output_tokens` 5,000 so a cut costs
   about 125 s, (c) add one line to TOOL: "Think in short bursts: when a hypothesis can be checked
   with code, call the tool instead of reasoning further." Expected: 20-30% more calls per game;
   cd82 (L2 solved in code at the deadline), ka59 (44% cut), vc33 (26%) are the direct wins.
2. **Opener line with untried objects and blocked moves (harness, free).** From the level's
   record: (a) "Objects never clicked on this level: G 8px at (30,27), D 16px at (26,44), ..."
   (by colour+shape hash, not position, so toggled copies count as tried), (b) "Blocked moves:
   LEFT refused 4x, target cells were colour D; successful moves entered B, S, R". Both derive
   from data `wm` already holds (`click_candidates`, `induce_motion.blocked_cases`). Prompt:
   "Before 20 actions of trial and error on a level, click each unclicked object once." Games:
   ka59 (second ring, 70+ actions), vc33 L4 (plugs), dc22 (passability rule), tn36 (target boxes).
3. **Let a batch continue after RESET.** sandbox.py:493-499 sets `run.terminal` for a model
   RESET, so `action([RESET] + replay)` raises `TerminalStateError`: 6 times in 4 games (vc33 #56
   #61, tn36 #50, dc22 #28 #30, ka59 #14), each a lost model call (dc22: 117 s + 84 s). The start
   board after RESET is deterministic and `describe_result` already reports it. Keep the latch for
   game_over/level_reset/level_completed only, and say in the RESET prompt line that actions may
   follow it in the same call.
4. **Loop visibility: novelty counter in the opener.** "This level: 97 actions, 31 distinct
   interior states, last 20 actions reached 2 new states; MOUSE(58,46) was a no-op 15 times." Use
   `wm.view_hash` (HUD blanked). Prompt: "If the last 20 actions reached fewer than 5 new states,
   stop probing variants: write down what you have not examined (unclicked objects, legend panels,
   animation frames) and test one of those." Games: tn36 (15 no-op launches), vc33 L4 (alternating
   two buttons for 76 actions), dc22 (tours), ka59 (sweeps).
5. **Retain functions that reference module-level data.** sandbox.py:651-664 refuses any helper
   that reads a snippet-level name. Snapshot the referenced values at definition (deep copy, size
   capped) into the function's globals, or auto-store them in `memory['_bound'][fn]`. ft09 lost 10
   helpers and re-parsed the board 12 times; vc33, cd82, ka59 one each. Token and time saving
   mainly; for ft09 the post-trim re-derivation (#50-#57, ~1,000 s) would have been one call.
6. **Fix `wm` API friction so the free helpers are usable from the prompt as written.**
   `click_candidates` needs a grid argument (wm.py:426) while the prompt lists helpers as
   no-arg; `goal_menu` predicates only accept `state['grid']` (wm.py:968), which cost vc33 five
   calls; `induce_motion` says nothing for click games. Accept a grid, a FrameView or nothing
   (current grid) everywhere, and make `untried_actions` use `view_hash` (bug 2). Small, but it
   removes the 400-800 s rabbit holes seen in vc33 and dc22.

## 3. Bugs seen in our code

1. **sandbox.py:493-499** - a model RESET ends the snippet (`restarted = a.name == "RESET"`), so
   the natural `action([RESET, ...prefix])` raises `TerminalStateError`. 6 occurrences, see fix 3.
2. **wm.py:837-843** - `untried_actions` compares `grid_hash(before) == grid_hash(current)` on
   the full grid. With a per-action HUD bar no two grids match, so `here` is always every action
   (ka59 #18, dc22 #32: `'here': ['UP','DOWN','LEFT','RIGHT','MOUSE','RESET']`). Use `view_hash`.
3. **wm.py:426 / wm.py:968** - `click_candidates(grid)` and `goal_menu` predicates take arguments
   the prompt does not mention; vc33 #48-#52 spent 5 calls on it.
4. **agent.py:330** - the slot is released at every trim. With the priority decay on
   `tokens_on_level` (scheduler.py:83) the four games that trimmed once mid-level waited 6,677-
   8,952 s for it back. If the scheduler stays, at least do not release when the game has acted
   in the last two turns.
5. **sandbox.py:651-664** - helper retention refuses functions with free module-level names and
   non-literal defaults; see fix 5. Not a crash, but the dominant source of repeated code in ft09.
6. **Trace gaps** (same as run #3): user-side text (opener, budget line, trim notice) and the
   reasoning text are not traced, so whether the budget/trim notices were read cannot be checked.
