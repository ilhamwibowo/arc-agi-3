# Trace review: run #10 (arc3-full25f/1), all 25 games, failure attribution for the 65 lost levels

Source: `runs/kaggle/arc3-full25f/1/output/runs/kaggle/<game>/{trace.jsonl,summary.json}` and the
code copy that ran (`.../output/arc3/*.py`). 25 games, 3 h per game, 10 decode slots, context
116k/58k, output cap 6,000 tokens, yield after 2,048 generated tokens per turn, slot rotation after
900 s if outranked by 1.2x, slot released at a trim if outranked, idle notice at 3 idle turns, park
at 8. `MCn` = n-th model call of the game, `Tn` = n-th turn, `#n` = n-th tool call.
Reasoning text and opener text are not traced (only `reasoning_chars`); beliefs are read from code,
comments, `memory` writes and printed results. Reference levels = dfranzen M2 mean of 4 passes.

## 0. Headline

- **118/183 levels.** 8 games won, 17 hit the deadline. Of the 65 lost levels, 17 were attempted
  and 48 were never reached.
- **Root causes of the 65:** rules or goal never learned **35** (9 stuck + 26 behind them), out of
  time **22** (6 + 16), rules known but execution wasted **8** (2 + 6). Lost knowledge at a trim,
  idling, and tool errors are secondary only; none is the primary cause of any lost level.
- **The GPU was saturated the whole run.** Exactly 10 games held a slot at every sample point while
  anything waited; on average 11.1 games were waiting. Wait time is therefore a reallocation between
  games, not lost compute. The only levers are (1) what the 1,800 slot-minutes are spent on and
  (2) how many turns a slot-minute buys.
- **36% of model time (≈635 of 1,782 min) went to the 17 levels that were not completed**, ≈460
  min of it to 9 levels whose rules or goal were never found. Six games that ran out of time waited
  64-116 min each.
- **Planner tools were not used.** In 4,581 tool calls, `wm.plan_abstract`/`wm.plan`/
  `wm.register`/`wm.execute` were called 0 times and `wm.check` 3 times (s5i5, never `ok`).
  Hand-written search (deque/BFS) appears in 293 calls on 50 levels, with no win-rate difference.

## 1. Per-game table

"min in slot" = game wall time minus slot wait minus park sleep. "cuts" = calls ending on the
6,000-token cap with no tool call. Trims in parentheses = trims that released the slot.

| game | levels | ref | actions | calls | out tok | cuts | trims (rel.) | slot wait min | parks | end | min in slot | slot % of wall |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ar25 | 8/8 | 7.2 | 305 | 140 | 178k | 2 | 3 (0) | 48 | 0 | win at 90 min | 42 | 47% |
| bp35 | 3/9 | 1.0 | 466 | 286 | 451k | 11 | 11 (3) | 65 | 0 | deadline | 115 | 64% |
| cd82 | 6/6 | 6.0 | 112 | 85 | 146k | 5 | 2 (0) | 65 | 0 | win at 100 min | 35 | 35% |
| cn04 | 5/6 | 2.5 | 409 | 206 | 350k | 7 | 7 (3) | 100 | 0 | deadline | 80 | 45% |
| dc22 | 4/6 | 3.8 | 551 | 330 | 476k | 9 | 10 (1) | 56 | 0 | deadline | 124 | 69% |
| ft09 | 6/6 | 5.5 | 269 | 116 | 169k | 3 | 3 (1) | 70 | 0 | win at 108 min | 39 | 36% |
| g50t | 5/7 | 1.8 | 629 | 183 | 333k | 13 | 6 (0) | 101 | 0 | deadline | 79 | 44% |
| ka59 | 5/7 | 2.8 | 1309 | 329 | 429k | 17 | 10 (3) | 72 | 0 | deadline | 108 | 60% |
| lf52 | 2/10 | 1.0 | 405 | 257 | 332k | 6 | 7 (5) | 99 | 0 | deadline | 81 | 45% |
| lp85 | 8/8 | 8.0 | 189 | 125 | 186k | 5 | 4 (0) | 31 | 0 | win at 72 min | 40 | 56% |
| ls20 | 6/7 | 4.0 | 924 | 269 | 451k | 12 | 9 (2) | 65 | 1 | deadline | 105 | 58% |
| m0r0 | 6/6 | 3.2 | 411 | 207 | 294k | 4 | 6 (1) | 60 | 0 | win at 129 min | 68 | 53% |
| r11l | 5/6 | 5.2 | 367 | 300 | 373k | 5 | 9 (3) | 83 | 1 | deadline | 88 | 49% |
| re86 | 4/8 | 6.2 | 490 | 146 | 262k | 6 | 5 (5) | 120 | 0 | deadline | 61 | 34% |
| s5i5 | 2/8 | 3.2 | 447 | 304 | 404k | 23 | 9 (2) | 84 | 0 | deadline | 96 | 53% |
| sb26 | 8/8 | 8.0 | 173 | 59 | 86k | 1 | 1 (1) | 18 | 0 | win at 37 min | 19 | 50% |
| sc25 | 4/6 | 2.8 | 504 | 192 | 267k | 8 | 6 (3) | 117 | 0 | deadline | 63 | 35% |
| sk48 | 1/8 | 0.0 | 308 | 126 | 264k | 16 | 5 (1) | 116 | 0 | deadline | 64 | 36% |
| sp80 | 1/6 | 1.8 | 303 | 123 | 243k | 11 | 5 (5) | 123 | 0 | deadline | 57 | 32% |
| su15 | 4/9 | 4.2 | 293 | 193 | 341k | 14 | 7 (5) | 100 | 0 | deadline | 80 | 44% |
| tn36 | 4/7 | 4.2 | 566 | 191 | 402k | 13 | 9 (2) | 83 | 0 | deadline | 97 | 54% |
| tr87 | 6/6 | 5.8 | 232 | 66 | 97k | 2 | 1 (0) | 50 | 0 | win at 73 min | 23 | 32% |
| tu93 | 9/9 | 8.8 | 292 | 169 | 282k | 9 | 5 (1) | 90 | 0 | win at 155 min | 65 | 42% |
| vc33 | 3/7 | 6.0 | 446 | 206 | 380k | 14 | 7 (2) | 88 | 0 | deadline | 92 | 51% |
| wa30 | 3/9 | 1.2 | 806 | 203 | 375k | 11 | 8 (2) | 85 | 0 | deadline | 95 | 53% |
| **total** | **118/183** | 104.2 | 11,206 | 4,811 | 7.58M | 227 | 155 (51) | 1,989 | 2 | 8 win, 17 deadline | 1,817 | |

No `model_error`, `evidence_error`, `render_error` or `abandon` events in any game. Against the
reference we are +14 levels in total; below reference only on re86 (-2.2), vc33 (-3.0), s5i5
(-1.2), sp80 (-0.8), tn36/su15 (-0.2).

On **won** levels the agent is already at human pace: median actions/human-baseline 0.81 over 118
levels. Level count, not action efficiency, is where the score is lost.

## 2. The 65 lost levels

### 2.1 The 17 attempted levels

| game level | category | min on level (model / wait) | actions (human) | key evidence |
|---|---|---|---|---|
| dc22 L5 | **a** (+c, d) | 66 / 32 | 261 (324) | All panel buttons probed once MC135-143; combinatorial gate/piece sweeps from MC166; MC310 memory "goal unreachable in every tested state". Re-clicked the same buttons after 4 trims (MC187-195, 210-219, 239-260, 262-279), ≈60-80 actions, 20-25 min, because no L5 finding was put in `memory` before MC310. 6 no-tool "final status" calls MC300-324. |
| ka59 L6 | **a** (+g) | 41 / 15 | 965 (132) | Contradictory notes: MC205-206 "plus at socket != win", MC241 "top half = static recap, all MOUSE no-op". "Exhausted hypothesis space" MC238 (161 min). Then ≈45 single DOWNs to "hold", and 137 DOWNs at MC297-299 to burn the budget (forbidden by the prompt). |
| lf52 L3 | **a** (+d, e) | 63 / 49 | 295 (60) | Jump/capture rules in memory MC68-116; 5 hand-written BFS runs solved the wrong goal (reduce to 3 tokens, no completion). MC241 memory `L3_STOP` "PROVEN UNWINNABLE". MC222, MC244 end with no tool call. |
| s5i5 L3 | **a** (+d) | 73 / 23 | 322 (106) | `plan3` (MC86) failed at step 1. `wm.check` 62/97, 45/97, 56/97 at MC145-147, never `ok`. "Exhaustively mapped" MC135; "unwinnable" in memory ≈MC170; then ≈100 one-click grow/shrink turns MC201-304: 223 of 322 actions after the first give-up. 17 cuts = 24.5 min. |
| vc33 L4 | **a** (+d) | 60 / 10 | 286 (61) | Hand-written pump simulator + BFS over 546 states (MC80-119) found no goal; MC128 memory "appears unsolvable ... only 6 buttons are clickable". Plugs clicked 3 times (steps 164, 186, 215), all while closed. Runs #3 and #9 won this level by clicking the plugs. 160 actions after the first give-up, 27 RESETs. |
| re86 L5 | **a** (+e) | 46 / 105 | 272 (189) | Recolour rule in memory MC70; own simulator + BFS plans of 39/64/46/16 actions (#61, #117, #130, #139) executed exactly ("fail: None, completed: False"): the goal predicate was wrong and no alternative was formed. 36 of 55 turns ended on yield. |
| sp80 L2 | **a** (+e, c) | 50 / 107 | 260 (58) | 30 batches stopped as no-op, mostly `[..., 'SPACE']`. First click on a gameplay object (red bar, selects it) at MC85 #78, 121.6 min, ≈70 actions after the prompt's "click each unclicked object before 20 actions" threshold. After the trim at 96.8 min, MC81 fired the known-fatal SPACE again (death #3). |
| tn36 L5 | **a** (+b) | 38 / 43 | 290 (30) | L4 rules correct (won at #115). L5's new left strip never examined; 37 batches of `[toggles, MOUSE(60,25), MOUSE(58,57)]` ended as no-op (#123-#169); 10 RESET+re-toggle batches. |
| r11l L6 | **a** (+e) | 21 / 27 | 217 (49) | MC185 (140 min) "win condition unknown", MC188 "hold here". Notes `final`..`final7` record wrong goals. Payload-merge mechanic found MC276 at 168.7 min ("too late to exploit"). |
| bp35 L4 | **b** (+a) | 65 / 0 | 266 (38) | L3 rules in memory at MC122. World map with the goal 2 rows from the fish rebuilt at MC135; then ≈120 calls of 1-4 action probes, 8 RESETs, 5 game overs, no search. L4's moving element never written down. |
| sc25 L5 | **b** (+e) | 40 / 87 | 319 (143) | Full spell rules in memory MC88 (52.9 min). Same macro "RESET + L-teleport + diamond + column" at MC105, 113, 123, 138, 156, 162, 175, 189; no search over spell sequences. 87 min wait = 68% of the level's wall time. |
| ls20 L7 | **e** (+d) | 29 / 20 + 10 park | 216 (186) | Started at 119 min. Rules stored by L4; L7 exploration graph + BFS (#209-#242), routes executing at the deadline. 8 idle turns T131-138 triggered the park. Earlier levels slow: L1 82 (22), L3 139 (73), L5 188 (96). |
| sk48 L2 | **e** | 24 / 0 | 69 (177) | 116 of the first 155 min waiting: 15.6 min at start, 58.6 min after a rotation mid-L1 (31.6 min), 41.7 min after a trim release (102.6 min). L1 won at 155 min; L2 got 25 min. |
| cn04 L6 | **e** | 0.5 / 0 | 0 (113) | Started at 179.5 min (MC204-206). 100 min of waits, 75 of them after trim releases. |
| g50t L6 | **e** | 5 / 0 | 7 (54) | Started at 174 min (MC167-183). L1 spent 101 of 132 min waiting (rotations at 15 and 101 min), then L2-L5 took 42 min. |
| su15 L5 | **e** | 9 / 0 | 45 (36) | Started at 171 min (MC171-193). L4 took 57 model-min + 84 min wait after trim releases. |
| wa30 L4 | **e** | 5 / 0 | 89 (98) | Started at 174 min (MC184-203). L3 took 501 actions (human 183) and 41 model-min. |

### 2.2 The 48 unreached levels

Proximately, all 48 are "out of time". The root cause is the level that blocked them:

| blocked by | games (unreached levels) | count |
|---|---|---|
| a, rules/goal never learned | lf52 (7), s5i5 (5), sp80 (4), re86 (3), vc33 (3), tn36 (2), dc22 (1), ka59 (1) | 26 |
| b, execution wasted | bp35 (5), sc25 (1) | 6 |
| e, out of time | sk48 (6), wa30 (5), su15 (4), g50t (1) | 16 |

### 2.3 Category counts

| category | primary, attempted | + unreached behind it | total | secondary mentions |
|---|---|---|---|---|
| (a) rules or goal never learned | 9 | 26 | **35** | bp35 |
| (b) rules learned, actions wasted | 2 | 6 | **8** | tn36 |
| (c) knowledge lost at a trim | 0 | 0 | 0 | dc22 (≈20-25 min), sp80 (1 death) |
| (d) idle / over-thinking | 0 | 0 | 0 | dc22, lf52, s5i5, vc33, ls20 |
| (e) out of time / slot starvation | 6 | 16 | **22** | lf52, r11l, re86, sp80, sc25 |
| (f) tool / harness errors | 0 | 0 | 0 | none decisive (see 4) |
| (g) other | 0 | 0 | 0 | ka59 budget burning |

**Caveat on the 26 behind (a) levels.** The upper bound assumes the later levels are as playable as
the earlier ones. The reference also scores lf52 1.0/10 and sp80 1.8/6, so most of those are not
recoverable by any fix in this list.

**The (e) group splits further.** Slot starvation dominates in sk48 (116 min of waiting), g50t
(101), cn04 (100) and su15 (100). Slow earlier levels dominate in wa30 (L1 119, L3 501 actions) and
ls20 (L1/L3/L5 at 2-4x human). Several (a) levels are also starved: re86 L5, sp80 L2 and sc25 L5
each waited 87-107 min on that level.

### 2.4 Trims (category c) are not a measurable cause

155 trims, 140 with 10 calls of context on each side. The 10 calls after a trim versus the 10
before show no change:

| | 10 calls before | 10 calls after | run overall |
|---|---|---|---|
| fraction of calls with no action | 0.45 | 0.48 | 0.47 |
| calls running structural-parse code, per 10 | 2.79 | 2.74 | 2.56 |

Per-level reading agrees. Re-derivation after trims was ≈5 calls or ≈2-8 min per level in bp35,
lf52, re86, ka59 and s5i5, mostly re-typing parsers that the sandbox refused to retain ("[not
retained] TBL: uses cnt"). The one real loss is dc22 L5: 4 rounds of re-probing the panel, because
no finding was written to `memory`. **The cost of a trim is the slot release (section 5), not
forgetting.**

## 3. Planner and world-model tool use

| tool | calls (games) |
|---|---|
| `wm.plan_abstract`, `wm.plan`, `wm.plan_to`, `wm.register`, `wm.verify`, `wm.execute` | 0 |
| `wm.check` | 3 (s5i5 L3, MC145-147: 62/97, 45/97, 56/97 matched, never `ok`) |
| `wm.goal_menu` | 19 (11 games; outputs generic "all colour X removed" predicates, never acted on) |
| `wm.click_candidates` / `induce_motion` / `untried_actions` | 9 / 9 / 9 |
| hand-written search code (deque, heapq, BFS) | 293 calls on 50 levels in 18 games |

Hand-written search does not correlate with winning or with efficiency. The search levels are
probably harder levels, so this is not evidence that search hurts.

| | levels | won | win rate | median actions/human on won levels |
|---|---|---|---|---|
| with search code | 50 | 43 | 0.86 | 0.84 |
| without | 85 | 75 | 0.88 | 0.81 |

Where search ran on a stuck level it executed faithfully and solved the wrong problem: lf52 L3
(wrong goal), vc33 L4 (incomplete click model, "proved" unsolvable), re86 L5 (wrong goal predicate,
64-action plan executed exactly). The bottleneck on lost levels is **hypothesis formation**
(what is clickable, what the goal is), not search. A better planner would have changed none of the
17 stuck levels, except possibly bp35 L4 and sc25 L5, where the rules were known and no search was
written.

## 4. Where the output tokens went

7.58M output tokens over 4,811 calls, at ≈71 tok/s per stream. Prefill was 95% cached, so model time
is proportional to output tokens.

| tokens per call | calls | share of tokens |
|---|---|---|
| < 500 | 32% | 5% |
| 500-2,000 | 41% | 28% |
| 2,000-4,000 | 17% | 31% |
| 4,000-6,000 (incl. cuts) | 11% | 36% |

Percentiles: p50 949, p90 4,134, p95 5,799. Per turn: median 2,023, p90 5,372.

- **Calls that executed no action: 2,244 (47%), holding 55% of tokens and time.** Most of these are
  intended: 2,012 inspection calls ran code without acting (median 999 tokens, against 755 for
  acting calls).
- **The waste is in the 232 calls with no tool call.** 227 of them were cuts at the 6,000 cap:
  1.36M tokens and 320 model-min, **18% of all model time**. Worst games: s5i5 32 min, ka59 23,
  sk48 23, su15 21, vc33 21, tn36 20.
- **715 of 3,284 turns (22%) ended on yield**, meaning 2,048 tokens without an action.
- **Longest thinking streaks without an action:**
  - r11l L2, MC66-84: 19 calls, 34.6k tokens, 7.7 min.
  - re86 L5, MC96-111: 29.8k tokens.
  - ls20 L7, MC226-236: 25.8k tokens.
  - tu93 L9, MC148-157: 25.3k tokens.
  - Overall, 94 streaks of 10k tokens or more hold 18% of all tokens.

**What a 30-40% cut in thinking per turn could buy.** If quality held, a slot-minute would buy
≈1.5x the turns. That only converts to levels where time binds:

- **Likely +1 each: cn04 L6, g50t L6, su15 L5 and ls20 L7.** Their won levels ran at 6-16
  model-min each, and ≈25-35 slot-min would be freed per game.
- **Maybe:**
  - wa30 L4 (+0.5).
  - sk48 L2 (+0.3; it needs about 177 actions of new-level play).
  - r11l L6 (+0.5; the key was found at 168 min).
  - sc25, re86, sp80 and bp35 (+0.2 each).
- **Zero for the 9 (a) levels.** The model had the turns and spent them in loops.

The gross total is **≈+5 to +7 levels**. The risk is quality: run #7 showed that prompt changes move
thinking length. The rule-discovery calls are the long ones, and the run #5 review found that a
lower cap raised the cut rate without making cuts cheaper. **Net estimate: +2 to +5.** It largely
overlaps with the scheduler change below, because both feed the same 6-8 time-bound levels, so the
gains do not add.

## 5. Slot economics

- **Saturation.** While any game waited, 10 games held slots; the minimum was 10 at every 30 s
  sample. Mean waiting was 11.1 games, with a maximum of 15 for the first 35 min. It fell to 4-6 in
  the last 20 min as 8 games won. Total slot time was 1,817 min against a capacity of
  10 x 180 = 1,800.
- **Total wait: 1,989 min.** Park sleep adds 20 min.

  | wait cause | waits | minutes |
  |---|---|---|
  | slot release at a trim | 49 | 986 (median 19 min each, max 53) |
  | rotation after 900 s held | 39 | 706 |
  | start-up queue (games 11-25) | 15 | 297 (13-30 min each) |
  | after an idle park | 2 | 1 |

- **Share of each game's wall clock spent holding a slot: 32-69%.** The model had the slot least in
  sp80 (32%), tr87 (32%), re86 (34%), sc25 (35%) and sk48 (36%), and most in dc22 (69%), bp35 (64%)
  and ka59 (60%). The three most-served games are three of the dead (a)/(b) levels.
- **Rotation evicts mid-level by design.** All 40 rotations happened inside a level, at a median of
  14.5 min after the level started, waiting included. 27 of those levels were later completed.
  The costly ones were all mid-L1:
  - g50t lost 70.8 + 30.2 min.
  - cd82 lost 65.5 min, then solved L2-L6 in 12 min.
  - sk48 lost 58.6 min.
  - s5i5 lost 46.0 min in L2.
- **The priority formula starves long first levels.** The C term decays with tokens on the level,
  which is high on L1, where the game's rules are being learned. g50t, sk48, cd82 and wa30 each
  waited 65-116 min during L1, then ran L2+ at 3-15 min per level.
- **Dead levels were well served.** Of the ≈635 model-min spent on the 17 levels that were not
  completed, ≈460 went to the 9 (a) levels. About 120 of those came after the model had itself
  declared the level impossible: s5i5 ≈42, vc33 ≈30, ka59 ≈19, lf52 ≈14, dc22 ≈14.

## 6. Ranked changes

Expected gains are levels on this 25-game set, against a run-to-run noise of about ±5 levels
(runs #6/#7/#10 scored 114/113/118; vc33 was 7/7 in run #9 and 3/7 here). Any single change below
+5 is not measurable with one run.

1. **Scheduler: stop starving L1 and time-bound games, demote dead levels.** Ties to the 22 (e)
   levels and the 1,692 min of rotation and trim-release waits.
   - (i) Do not release the slot at a trim. Rotation alone redistributes, and a release costs a
     median 19 min.
   - (ii) Exempt level 1 from the C-term token decay, or start the decay at 2x the 62k half-life.
   - (iii) Add a stall term that lowers priority when the level shows no progress:
     - fewer than 5 new interior states in the last 40 actions (`wm.novelty`);
     - actions on the level above 3x the game's median for completed levels;
     - `memory` or code comments containing "unsolvable", "unwinnable", "impossible" or
       "exhausted".
   - (iv) Raise priority for games that completed a level within the last 20 slot-min.
   - Expected **+3 to +5** (cn04, g50t, su15, ls20 likely; wa30, sk48 possible), taken from ≈120-250
     slot-min of dead-level time.
   - Risk: r11l L6 broke open at 168 min, and demotion would have cost it that chance.
2. **Bound thinking per call and make cuts cost nothing.** Ties to 227 cuts = 320 min (18%) and the
   time-bound levels.
   - First check whether SGLang/Qwen supports a thinking budget, and set it to ≈3,000 tokens while
     `max_tokens` stays at 6,000 for code.
   - If not, add a stop at the end of the thinking block plus a forced tool call.
   - Expected **+2 to +4** on its own. Mostly the same games as change 1, so the two together are
     ≈+4 to +6, not the sum.
3. **Break false-impossibility loops by forcing new hypotheses, not just actions.** Ties to the 9
   (a) levels and the 26 levels behind them.
   - Trigger on the stall signals from change 1, or on the model's own "impossible" note.
   - Inject a checklist:
     - (1) Objects whose clicks were only tried in one board state: re-list an object once the board
       around it has changed since its last click. vc33 plugs were clicked 3 times, always while
       closed, so the current shape-hash "unclicked" evidence counted them as tried.
     - (2) Elements that are new compared with the previous level's start board: tn36 left strip,
       bp35 mover, ka59 top half.
     - (3) Two alternative goal predicates to test before any repeat: re86, lf52, r11l.
   - Ban budget-burning and "hold" moves in the notice text (ka59).
   - Expected **+1 to +3** with high variance. vc33 L4 alone carries 4 levels, and the same model
     won it in runs #3 and #9; sp80 L2 and tn36 L5 are the other candidates.
4. **Harness-written per-level probe log in `memory`.** Ties to (c), secondary only. Record every
   distinct action and its interior effect per level automatically, and show it in the trim notice.
   Also let retained helpers capture module-level names, which bp35, ka59 and lf52 re-typed after
   trims. Expected **+0 to +1** (dc22 L5 time back; the level was still (a)).
5. **Do not invest further in `wm` planner tools for level count.** 0 planner calls; search is
   used where needed and does not separate wins from losses. Consider shortening the WORLD_MODEL
   prompt section, since the KV pool peaked at 99%. Expected **0 levels**; a small throughput gain
   at most. Revisit only if the score, rather than the level count, becomes the target. Won levels
   are already at 0.81x human actions.

## 7. What could not be determined

- **Reasoning and openers are not traced.** The model's reasoning and the opener text are absent,
  including the unclicked/blocked/novelty evidence. Whether vc33's opener listed the plugs, or
  sp80's the red bars, is inferred from how the evidence code counts objects, not observed.
- **"Rules learned" is a judgement.** It is read from code comments and `memory` notes. A model
  may have known a rule without writing it down.
- **Counterfactual allocations are estimates.** Freed slot-minutes would also go to other games,
  and the win/loss of a reallocated level is a judgement based on that game's per-level pace.
- **One run.** Category counts for (a) are stable in kind across runs #5/#9/#10, but which games
  land in (a) varies; vc33 is the clearest case.
- **Thinking-cut quality loss is unknown.** No run has measured it; the +2 to +5 net is a guess
  bounded by the time-bound level count.
