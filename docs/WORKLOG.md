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
