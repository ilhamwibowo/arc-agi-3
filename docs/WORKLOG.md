# Work log

## 2026-10-27 05:00–06:30 WIB
- Project created; PLAN.md written after reading arc-harness, decker, dfranzen write-up, Duck brief, Tycho digest.
- Core modules built in parallel (env, percept, sandbox, agent/llm/context/prompts, session, scheduler, runner, wm, kaggle serve); 63+ tests pass.
- Smoke: runs/local/smoke-ollama-1 (ls20, qwen3.5:4b local): loop works end to end — 14 turns, 19 tool calls, 40 actions, 0 model failures, 0 levels (model too weak to matter). Found: segmentation API mismatch with prompt (3 wasted calls) and numpy scalar repr noise → fixes requested.
- Pending: wm wired into sandbox/prompts; Kaggle notebook builder; first private Kaggle plumbing run.
- 05:30 Pushed private Kaggle notebook `wibowoilham/arc3-plumbing` v1 (RTX Pro 6000, offline mode, games ls20/ft09/tu93/vc33, 30 min cap each). Purpose: validate SGLang launch from public datasets + our agent loop on the real model. Record: runs/kaggle/arc3-plumbing/1.

## 2026-10-27 07:30 WIB — plumbing run #1 result (runs/kaggle/arc3-plumbing/1)
- Kaggle run COMPLETE (1h40m wall). SGLang from public wheelhouse + W4A16 model READY 753 s after
  notebook start (prefetch 23 s wheels + 475 s model; capture 2 min). VRAM 93.5/95.6 G, decode
  ~350–500 tok/s with 4 concurrent, MTP accept rate ~0.6. 0 errors, no hangs, keeper keep-alive 3 resets.
- Score at a 30-min cap, 4 concurrent: ft09 4/6, ls20 1/7, tu93 3/9, vc33 4/7 → 12/29 levels,
  RHAE 0.27 (dfranzen full-budget reference: 5.5, 4.0, 8.8, 6.0). Per-call ~20 s, ~80 k generated
  tokens/game in 30 min, prompt cache 30–75 %, no context trims.
- Bugs: (1) deadline mid-request was counted as llm failure → end_reason "llm_failures" (cosmetic;
  fixed: DeadlineReached). (2) API friction, 27 tool errors: `np` not preloaded, `history[-1].frame` /
  `transitions[-1].before_frame` / `res.frame` missing (prompt promised them) → fixed in sandbox.
- wm tools nearly unused (4 of 196 tool calls, tu93 only). To investigate after a longer baseline.
- 08:10 Run #2 pushed: `wibowoilham/arc3-dev9` v1 — 9 games (tu93 sb26 ls20 vc33 re86 dc22 bp35 g50t wa30), 2.5 h cap, old code (pre-review fixes) → baseline. Record: runs/kaggle/arc3-dev9/1.
- Trace review (docs/trace-review-run1.md) + cache analysis: every model call that opened a turn after an
  action cached exactly 2304 tokens (= system prompt, page-aligned), even before any image-window
  stripping; calls with an identical image set cached 97–99 %. ⇒ the server hashes a request's images as
  one unit, so any new image breaks the prefix at the first image. Fix: images_window=1 (only the newest
  user message carries images → only the last turn re-prefills). To be confirmed in run #3.
- Also: per-game deadline now starts at model-ready (runner.wait_model_up); UNDO listed only when the
  game offers it; review fixes C–F (silent level reset, auto object diff, `memory` dict, WORLD_MODEL
  prompt rework, verify current level by default) delegated.
- 08:50 Run #3 pushed: `wibowoilham/arc3-dev9b` v1 — same 9 games/cap as run #2, with all review fixes (commit 86c38b6). Record: runs/kaggle/arc3-dev9b/1.
- 08:10 Cancelled run #2 (deleted queued kernel arc3-dev9): old-code baseline not worth 3 h of GPU; run #1 + dfranzen table serve as reference. Run #3 (dev9b) running since 07:53.

## 2026-10-27 10:50 WIB — run #3 result (runs/kaggle/arc3-dev9b/1)
- 9 games, 2.5 h cap, 9 concurrent: 49/70 levels, mean RHAE 0.551 (dfranzen full-budget reference on the
  same 9 games: 0.448). Full wins sb26, tu93, vc33; levels ≥ reference on 8/9 (bp35 4 vs 1.0, wa30 4 vs 1.2,
  ls20 5 vs 4.0). 0 errors; trims 1–8/game (max prompt 120.7k, mean 72k); ~51 s/call at 9 concurrent.
- Boot: SGLang READY at 1741 s (model prefetch 1455 s vs 475 s in run #1 — Kaggle disk variance).
- NOT fixed: after-action calls still cache 11 % (images_window=1 did not help); yield/in-turn 93–94 %.
  wm tools used 0/1233 calls (reworked prompt made usage worse, not better). level_reset never fired.
- Next: run #4 = images A/B (4 games with images, 4 text-only, 1 h cap) to settle the cache question.
- 11:00 Run #4 pushed: `wibowoilham/arc3-ab-images` v1 (images on: sb26 dc22 ls20 ft09; off: vc33 re86 tu93 bp35; 1 h cap).
- 11:25 Review of run #3 weak games (docs/trace-review-run3-weak.md): 93 actions burned on purpose to exhaust
  budgets (RESET not exposed), 270 actions lost to budget-exhausted first attempts in wa30, wm unused 0/604.
  Fixes (docs/fix-review-3.md, commit d5181f1): RESET exposed + never-burn rule, budget-bar
  readout, wm.check/plan_abstract with a 1600-char WORLD_MODEL, trims strip old reasoning first + trim notice
  with memory summary, stop_reason visible, thinking cap 8k, NUDGE_CUT, tool_result traced. 126 tests.
- 11:30 Run #5 pushed: `wibowoilham/arc3-full25` v1 — all 25 public games, 3 h cap (queues behind run #4).

## 2026-10-27 15:40 WIB — run #4 result (runs/kaggle/arc3-ab-images/1) — CRASHED at 60 min of play
- Boot 793 s (READY 07:13:37 UTC). Played 60 min, then SGLang scheduler killed with exit -9 (host OOM):
  RAM flat at 106 GB until 08:00, 161.8 GB at 08:10, dead 08:13. Cause: ls20 snippet at 08:02 with an
  infinite loop (`while seen[cur]: path.append(cur[0]), cur` never advances `cur`); the sandbox's 30 s
  timeout only stopped waiting — the daemon thread kept appending until the kernel OOM-killed the
  biggest process (the model server). Run #3 was simply lucky. 6 games hung, notebook aborted.
- Fix (this commit): sandbox polls the worker every 0.25 s; on timeout or >3 GB RSS growth it cancels,
  then raises SnippetKilled asynchronously into the thread (PyThreadState_SetAsyncExc) until it dies
  (5 s grace); kaggle_main caps the agent process address space at 40 GB (ARC3_AS_CAP_GB) as backstop.
  3 new tests (130 total).
- Images A/B (what the run was for) — decisive, partial data (8 games, 780 calls):
  | setting | after-action cache | in-turn cache | s/call |
  | images on (sb26 dc22 ls20 ft09) | 43 % | 99 % | 34.1 |
  | text only (vc33 re86 tu93 bp35) | 96 % | 98 % | 27.6 |
  Text-only games progressed at least as well per hour (tu93 8/9, re86 5/8, vc33 4/7, bp35 3/9 in 60 min;
  run #3 with images reached tu93 9/9 / re86 6/8 / vc33 7/7 / bp35 4/9 in 150 min). Images on: ft09 6/6
  win, sb26 8/8 win, dc22 4/6, ls20 3/7. ⇒ default send_images=False. Option for later: one image at each
  level start only (one cache miss per level).
- Run #5 (full25, images on, pre-fix sandbox) still running; it carries the same OOM risk.
- 15:50 Run #6 pushed: `wibowoilham/arc3-full25b` v1 — all 25 games, 3 h cap, text-only default + sandbox kill fix (commit fb7ea9b). Queues behind run #5.
- 16:40 Run #6 v1 FAILED at SGLang boot in 70 s: "CUDA driver error: out of memory" at context creation. Cause: my new
  40 GB RLIMIT_AS cap was set in the notebook process before SGLang was spawned, so the child inherited it.
  Fix: launch_sglang resets RLIMIT_AS in the child (preexec_fn). Pushed `arc3-full25b` v2 (16:50).

## 2026-10-27 18:00 WIB — run #5 result (runs/kaggle/arc3-full25/1): all 25 games, 3 h cap, images on, old sandbox
- Clean: boot 761 s, 25 ok, 0 errors, 0 hung, RAM flat 107 GB (no runaway snippet this time).
- 86/183 levels, mean score 0.378 vs dfranzen reference 0.465 on the same 25 games. Wins: ar25 8/8, sb26 8/8.
  Above ref: ar25, m0r0 5/6 (3.2), g50t 5/7 (1.8), sc25 4/6 (2.8), lf52 2/10 (1.0), bp35 2/9 (1.0).
  Far below ref: cd82 1/6 (6.0), vc33 3/7 (6.0; 7/7 in run #3), tn36 1/7 (4.2), dc22 0/6 (3.8; 4/6 in run #3),
  ka59 1/7 (2.8), ft09 4/6 (5.5). Zero levels: dc22, sk48, wa30. → trace review delegated (docs/trace-review-run5.md).
- Throughput is the binding constraint: 10 decode slots for 25 games → each game waited 40–80 % of its 3 h
  (e.g. dc22: 1910 s in model calls out of 10864 s). Server ran 8–10 requests 97 % of the time, queue>0 in
  30 % of batches, KV pool 68 % mean / 99 % max, 636 gen tok/s aggregate, 40 tok/s per stream, accept len 2.6.
  2149 calls, 4.2M gen tokens: calls ≥3k output tokens are 22 % of calls but 56 % of model time; 79 hit the
  8k cap. Uncached prompt 7.3k tok/call median (images on; cache 54 %).
- Competition is 1.5× tighter (110 games / 9 h = 0.82 slot-hours per game vs 1.2 here) → expect < 0.378.
  Levers: (1) text-only (run #6); (2) more slots via smaller context (trim target ~60k, --max-running-requests
  14–16); (3) scheduler: allocate slot time by progress rate, starve games at 0 levels after a probe budget;
  (4) shorter thinking on routine turns.
- 18:15 Trace review of run #5 weak games (docs/trace-review-run5.md) + fix round (docs/fix-review-5.md, commits
  f3cd11b..25dfd4b, 146 tests): RESET no longer ends a batch; wm helpers take grid/frame/nothing and
  untried_actions uses view_hash; opener evidence lines (unclicked objects, blocked moves, novelty); output cap
  6000 + "think in short bursts"; retained helpers freeze ≤64 KB of referenced data; slot released at trim only
  when outranked (should_yield, 1.2×) and rotated after 15 min held.
- 18:20 Run #7 pushed: `wibowoilham/arc3-full25c` v1 — all 25 games, 3 h cap, text-only + fix round (commit b9399d0).

## 2026-10-27 20:30 WIB — run #6 result (runs/kaggle/arc3-full25b/2): all 25 games, 3 h cap, TEXT-ONLY, sandbox fix (commit fb7ea9b)
- 114/183 levels, mean score 0.500 (run #5 images-on: 86/183, 0.378; dfranzen reference 0.465). 25 ok, 0 errors,
  RAM flat. Wins: ar25 8/8, cd82 6/6, ft09 6/6, lp85 8/8, m0r0 6/6, tr87 6/6. Cache 97 %, 18 s/call (50 in run #5),
  5882 calls (2149), 7.9M gen tokens. Slot wait still 55 % of wall. Cuts 128/5882 (2.2 %), 453 calls > 60 s.
- Regressions: g50t 0/7 (5/7 in #5), bp35 1/9, vc33 3/7 again, sb26 4/8 (8/8 in every earlier run).
- sb26 pathology: the model declared level 5 "unsolvable" (miscounted palette) and then PARKED: 1,598 calls of
  `action(['RESET'])` (refused as reset_noop, 1 s each, 68 tokens) across 130 tool-cap turns for 2.7 h, holding a
  decode slot the whole time. No harness guard caught a turn with zero effective actions. Fix next: idle-turn
  detection → release slot + "idling earns nothing" notice, sleep the game after repeated idle turns; sandbox
  refuses a second consecutive no-op RESET.
- 20:50 Idle guard committed: after 3 consecutive turns with no executed action the opener carries an "idling earns
  nothing" notice and the decode slot is released; after 6 the game parks 10 min (then counter back to 3). 147 tests.

### 2026-10-27 21:35 Run #7 result (arc3-full25c v1, all 25 games, 3 h cap, text-only + fix round, no idle guard)
- 113/183 levels, mean 0.476 (run #6: 114/183, 0.500). Flat within per-game noise: g50t 0→4, vc33 3→6, bp35 1→3 up; cn04 4→1, s5i5 4→2, m0r0 6→5 down. 6 wins (ar25 cd82 ft09 lp85 tr87 + none new). No errors, server healthy, 25 ok.
- Cost side: 4626 calls at 23.0 s (run #6: 5882 at 18.2 s). Prompt tokens/call equal (~77k); generated tokens/call 1634 vs 1339 (+22 %), so the fix-round prompt lines made the model think longer, not shorter.
- Slot wait sum 1998 min (run #6: 2209 min); per game 15–132 min of the 180 min wall. KV pool 75 % mean / 99 % max with ~9.2 running requests: the KV pool, not the slot count, is the ceiling.
- sb26 again 4/8 but no RESET loop this time (the idle guard was not in this build anyway).
- Decision: run #8 = idle guard + 12 slots (MAX_ACTIVE, MAXREQ) + context 80k/drain 30k so 12 concurrent prompts fit the pool (12 × ~55k ≈ 660k). Risk: more trims. Compare vs run #6.

### 2026-10-28 01:20 Run #8 result (arc3-full25d v1: idle guard, 12 slots, context 80k/drain 30k)
- 95/183 levels, mean 0.431 (run #6 114/0.500, run #7 113/0.476). Regression entirely on throughput: 2610 calls at 49 s/call (run #7: 4626 at 23 s). Levels per call actually rose (0.036 vs 0.024), so the 80k context did not visibly hurt.
- Cause: serve.log shows `cuda graph: False` whenever #running-req is 11 or 12; GRAPH_BS stopped at 10. Decode throughput by batch: ≤10 → 630–780 tok/s, 11–12 → 412–423 tok/s. KV 64 % mean / 84 % max, queue ~0, so the 12 prompts fit as planned.
- Idle guard worked: 92 idle_release, 5 idle_park; sb26 back to 8/8 (win). New wins sc25 6/6, sb26; losses m0r0 0/6, ls20 1/7, g50t 0/7 (throughput-starved).
- Fix: GRAPH_BS += (11, 12). Run #9 = same config with graphs captured. Quota after run #9 ≈ 25.5/30 h.

### 2026-10-28 04:30 Run #9 result (arc3-full25e v1: run #8 config + cuda graphs for bs 11-12)
- Throughput fixed: 829 tok/s mean, graphs captured at bs 11-12, 4961 calls at 25.7 s, KV 68 % mean / 85 % max, slot wait 1742 min. But 98/183 levels, mean 0.377 (run #7 113/0.476 with fewer calls). Wins tu93 9/9, vc33 7/7, m0r0, lp85, tr87, ft09, cd82 (slow: 431 actions → 0.226).
- Two causes visible in traces: (1) context 80k/drain 30k → 309 trims vs 154, model re-explores after each trim (10753 actions vs 9157 for fewer levels); (2) idle guard released the slot at the notice threshold 169 times; each release re-waited for a slot (≈364 min total, tr87 95 min, g50t 52 min, re86 40 min) and dropped the warm prefix, mostly on legitimate analysis turns.
- Run #10 (arc3-full25f): back to run #6/#7 serving config (10 slots, 116k/58k context; GRAPH_BS keeps 11-12, harmless), idle notice stays but no slot release until the park at 8 idle turns. Expected ≈ run #7. Quota after run #10 ≈ 28.8/30 h → last run this week.

### 2026-10-28 08:05 Run #10 result (arc3-full25f v1: 10 slots, 116k/58k context, idle notice w/o release, park at 8)
- **118/183 levels, mean 0.510 — best so far** (run #6 114/0.500, run #7 113/0.476, run #9 98/0.377). 8 full wins: ar25 cd82 ft09 lp85 m0r0 sb26 tr87 tu93. No errors; 766 tok/s, KV 75 % mean / 99 % max, 4811 calls at 22.2 s, 95 % cache, 155 trims, 2 idle parks, slot wait 1989 min.
- Confirms run #9's diagnosis: the 80k context and the slot release at the idle notice were the regressions; the idle guard as a pure prompt notice + late park is harmless (sb26 8/8 again, tu93 9/9).
- Weak spots unchanged: sk48 1/8, sp80 1/6, lf52 2/10, s5i5 2/8, wa30 3/9, vc33 3/7 (vc33 was 7/7 in run #9: high variance). Slot wait still ~55 % of wall.
- Weekly Kaggle GPU quota exhausted (~29/30 h). Next: this config is the submission candidate; ask the user for the first competition submission (dress rehearsal).

### 2026-10-28 08:50 Submission sub-001 (user-approved: "build the notebook and submit")
- Notebook `arc3-submission` v1 built from HEAD 8011bc7 (run #10 config), smoke-saved with offline subset 1 game / 3 min cap (18 min of GPU: boot + 1 game, 2 levels, no errors).
- Submitted via CLI: `kaggle competitions submit -k wibowoilham/arc3-submission -v 1 -f submission.parquet`. Submission ref 57718940, PENDING at 01:18 UTC. 0 submissions remaining today. Rerun: 110 hidden games, 9 h.

### 2026-10-28 09:30 sub-001 scored 0.00 (COMPLETE ~35 min after submission) — early exit, no rerun logs
- Leaderboard research: unchanged dfranzen M2 notebook scores 25–34 in healthy reruns (best public fork 35.35); the team's 22.23 and the many 0.00s this week are a Kaggle platform incident (slow dataset cache, gateway DNS failures, scorecards closing). Top 5 = 50–75 with private harnesses; the 36–39 cluster merges the three M2 winners' published tricks (host-tier KV pool 1.4M tokens / 14–16 streams, solved-level memory pinning, UCB time scheduler, 950–1150 tok/s decode).
- Our competition path matches the official sample (gateway:8001, test-key-123, /api/games wait); the teammate harness validated the same path on Oct 26. Likeliest early-exit causes: (a) gateway not up within our 600 s wait → RuntimeError → run ends; (b) SGLang died at boot on the rerun host (slow reads / memory) → SERVER_DEAD aborted the run.
- Hardening for sub-002 (all committed): gateway wait bounded by the run (run_end − 1 h) instead of 10 min; SGLang install+launch retried up to 3× with MEMFRAC 0.96 → 0.94 → 0.92, only the third failure aborts, and a server that dies mid-run is relaunched; agent max_llm_failures 10 → 180 so games survive a 30 min relaunch. Tests 147 green.
- 09:50 New tests/test_competition_gateway.py: toolkit's own HTTP gateway (competition_mode=True) over the public games + our Arena in competition mode + runner keeper/scorecard + wait_for_gateway. 3 tests pass: game listing, single scorecard open/close, keeper RESETs while the model is down. So the API path is sound; sub-001's early exit was the platform (gateway/dataset) or server boot, which the hardening above now survives.

- 2026-10-29 00:27 UTC: **sub-002 submitted** (ref 57753205, notebook arc3-submission v2, build 2171329bd002). The scheduled 00:00:40 script never fired: the laptop slept and macOS pauses `sleep`, so it was still "sleeping" at 00:26. Submitted manually; lesson: use an absolute-time poll loop, not one long sleep.

- 2026-10-29 01:00 UTC: **sub-002 scored 0.19 after 30 min. ROOT CAUSE FOUND, OURS, NOT THE PLATFORM:** the notebook smoke settings (`PER_GAME_CAP_SECONDS = 180`) were passed into competition mode: 110 games x 180 s / 10 slots = 33 min, matching both sub-001 (35 min, 0.00) and sub-002 (30 min, 0.19). The games were played, each for 3 minutes. Fix: `kaggle_main.main` replaces any cap under 4 h with 9 h when competition=True; regression test `test_main_competition_ignores_offline_smoke_cap` (151 tests pass). Notebook v3 built (build 1fd5bb07106a) and pushed for its save run. Next submission needs user approval; quota resets 00:00 UTC Oct 30.
- 2026-10-29 01:20 UTC: notebook v3 (build 1fd5bb07106a) save run COMPLETE: SGLang READY 995 s, 1 game, 1 level, 0 errors, 19.5 min total. Ready to submit as sub-003 at the Oct 30 00:00 UTC reset pending user approval.
- 2026-10-29 02:10 UTC: **competition-path audit** (user-ordered after sub-002). Found and fixed, with tests (156 pass): (1) toolkit returns None on any gateway HTTP failure (10 s timeout) -> Env.step raised -> game marked error for the rest of the run; now retries 24x15 s. (2) Arena.make at thread start (110 at once) same failure mode; now retries 40x15 s. (3) Arcade swallows a failed /api/games -> zero games -> RUN FAILED; main() now rebuilds/retries until run_end-1h. (4) max_llm_failures 180 (30 min) < possible 60 min server relaunch -> every game abandons; now effectively until deadline (runner abort covers a dead server). (5) gateway auto-closes a scorecard idle 15 min (toolkit STALE_MINUTES default); added runner idle watchdog: RESET on a game that cannot lose levels after 10 min with no gateway activity. (6) HARD_WALL shaved 5 min for container start-up. Independent reviewer agent still running.
- 2026-10-29 03:00 UTC: **audit round 2** (independent reviewer agent, 14 findings; 8 fixed, 161 tests pass): keeper promotion (first game with a working env keeps the scorecard alive, not a fixed index); open/close scorecard retried (40x/20x 15 s); wrapper whose constructor RESET failed gets retried RESETs; agent_overrides ignored in competition; wait_for_gateway treats 4xx as up and sends the key; RLIMIT_AS 40->120 GB (110 sessions x 8.5 h history); STOPPING flag stops the relaunch watchdog after the final stop; gateway retries are competition-only (offline None is deterministic). Not done (tuning, needs GPU): unstarted-game sweep sized for 25 games (110 games x 15 min first slice = ~2 h of round-robin); hung-but-alive server detection. Notebook v4 (build 3e71bf94a062) pushed for its save run.
- 2026-10-29 02:20 UTC: notebook v4 (build 3e71bf94a062) save run COMPLETE: AS cap 120 GB applied, SGLang READY 775 s, 1 game, 1 level, 0 errors, ~16 min. Candidate for sub-003 at Oct 30 00:00 UTC pending user approval.
- 2026-10-29 ~04:30 UTC: **v5 stack started (no GPU until the weekly reset).** Item 1 done: pinned notes. The model keeps memory['rules'] (prompted); PythonTool.pinned_block() puts it plus per-level snapshots (pin_level on each new completed level, from Agent._pin_if_level_completed) into every opener under [pinned notes]; trim notice points at it. 163 tests. Two analysts running: run #10 trace failure review (Opus) -> docs/trace-review-run10.md; public writeup research (Sonnet) -> reference/top-writeups-2026-10-29.md. Plan: Thu-Fri research+build full stack, Sat reset: save run + one 25-game offline run, Nov 1/2 submissions tested. Repo now mirrored privately at github.com/ilhamwibowo/arc-agi-3.
- 2026-10-29 ~06:00 UTC: **v5 stack built and tested (169 tests), not yet run on GPU.** Inputs: docs/trace-review-run10.md (65 lost levels: 35 rules never learned, 22 time/slot starvation, 8 waste; trims and planner not causes; 227 cut replies = 18% of model time) and reference/top-writeups-2026-10-29.md (36-39 cluster = selection noise on Franzen; stream/pool/thinking-cap changes measured negative; D-prime and pinning unresolved +0..3). Changes: (1) pinned notes [done earlier]; (2) scheduler: L1 token/action half-lives x2, stall demotion x(1-0.5*stall), x1.5 for 20 min after a level; (3) no slot release at trims (release_at_trim=False); (4) a reply cut at 6000 without a tool call is re-asked at 2500; (5) stall notice every 6 turns while stalled (signals: novelty <3 new states in last 20 after 40+ actions, actions >3x median of solved levels, impossibility words in memory) with facts: clicks whose 9x9 surroundings changed since, elements new vs previous level start, untested goal predicates. Expected +4..8 levels on the 25-game set per the trace review; single-run noise ~+-5. Test notebook arc3-full25g built locally (25 games, 3 h cap), to be pushed when GPU quota exists (weekly reset). Submission plan unchanged: v4 Oct 30 (armed), v5 Nov 1/2 only if the Saturday run is not worse.
- 2026-10-30 00:17 UTC: **sub-003 submitted** (ref 57786640, notebook v4, build 3e71bf94a062). The clock-poll script woke with the laptop at 00:15 but the first attempt failed on DNS (network not yet up); submitted by hand at 00:17 and stopped the script. Score expected ~09:30 UTC.
- 2026-10-30 02:30 UTC: **Cloud test rig.** Ollama Cloud (user key in ~/.config/ollama/cloud.key, never in the repo) serves glm-5.3-flash etc. via an OpenAI-compatible API that our client handles unchanged (tool_calls + reasoning field). scripts/play_cloud.py runs the real offline runner (PriorityGate, keeper, per-game cap) against public games. Smoke: ls20 7 min, 23 turns, 1 level. A/B launched 02:26 UTC: v5 (main) vs v4 (worktree at afed053) on vc33 cn04 g50t su15 ls20 tn36 sp80 ar25, 1 h cap, 4 concurrent each -> runs/cloud/ab1-v5, ab1-v4. Caveat: different model from Kaggle (Qwen3.8); use for agent-logic validation and relative comparison only. Note: the keeper loop needs model_ready; without it an offline run with no acting game deadlocks until run_end (play_cloud passes model_ready=True).
- 2026-10-30 03:30 UTC: **Cloud A/B #1 (glm-5.3-flash, 8 games, 1 h cap, 4 concurrent per arm):** v4 14 levels, v5 15; 0 errors, 0 hung both arms. v5 exercised all new paths: 5 stall notices on 3 games (all single-signal, stall=0.5), no pin/evidence errors, slot waits spread 15-39 min (v4: 0-44). Cut-at-cap rate identical (18%): the 2500 re-ask only cheapens the second cut. Per-game swings of +-2..3 levels in both directions (g50t 0->3, ar25 3->5, su15 4->2, vc33 4->2) = noise at this size. Verdict: v5 is functionally sound; level effect unresolved. Next: A/B #2 overnight, all 25 games, 1.5 h cap, 8 concurrent per arm.
- 2026-10-30 05:05 UTC: Cloud A/B #2 (25 games, but the window was only 1.5 h total with 8 slots = ~0.5 slot-h/game, too short: ~20 calls/game): v4 16 levels, v5 15, 0 errors both, 0 stall notices (levels too short to stall). Noise again. A/B #3 launched 05:03 UTC: 25 games, 4 h window, 8 slots per arm (1.3 slot-h/game, above Kaggle 0.82) -> runs/cloud/ab3-v4, ab3-v5; done ~09:05 UTC. Note: play_cloud --hours is the total window AND the per-game cap, so per-game time = window x slots / games.
