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
