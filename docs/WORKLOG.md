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
