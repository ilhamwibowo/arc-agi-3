# ARC-AGI-3 — plan (written 2026-10-27 05:10 WIB)

Deadline: Nov 2, 23:59 UTC (Nov 3 06:59 WIB). ~6.5 days. One submission/day, two finals.

## Situation
- Public LB today: #1 71.5, #2 68.9 (Tufa), #3 56.0, #5 50.4, #12 41.0. Thousands of teams sit at
  22.x running the public dfranzen Milestone-2 notebook (27.89 when he ran it; 22.23 for our team's
  copy — the gap is run-to-run variance + his later tuning). Rank 5 needs >50 and rising.
- Everything that scores is: a ~27B open VLM (Qwen3.8-Flash-Next, W4A16, MTP spec-decode) on one
  RTX PRO 6000, a Python REPL as the actuator, structured perception (segmentation, frame diffs,
  animation transients, 10x images), HUD-aware no-op guards, 128k sliding context with thinking kept,
  and score-aligned GPU scheduling over 110 games. Model-free agents score ~0.
- Fresh perspective with real evidence: executable world models (Tycho, arXiv 2607.28287):
  the agent writes a per-game simulator + goal predicate, the harness verifies it against recorded
  transitions and plans with search → 88–100 RHAE with frontier models, 61% fewer actions than
  humans. Score is quadratic in action efficiency, so planning over a verified model is the
  single largest lever nobody at 22–28 is using. Unknown: how well a 27B model writes simulators.
- Constraint that dominates everything: no local GPU. Every real measurement is a Kaggle RTX run
  (quota unknown; a full 110-game-equivalent run is 9 h). Expect 4–6 real runs total.

## Decisions
1. Own harness (`src/arc3`), written from scratch to a design informed by the public solutions.
   Reuse only public *infrastructure*: the arc-agi toolkit, dfranzen's public Kaggle datasets/models
   for serving (SGLang "Pennyroyal" wheelhouse, Intel W4A16 AutoRound checkpoint, Albucino MTP
   draft). No Duck/dfranzen harness code is copied.
2. Honest target: substantially beat 27.89 on the public half (35–45 = top ~20); top-5 is not a
   realistic promise in 6 days with ~5 GPU runs. Every design choice is judged by expected LB gain
   per GPU-hour.
3. Validate the harness locally without a GPU using (a) unit tests, (b) a scripted "fake model"
   that replays canned tool calls through the real loop against the 25 public games (arcengine runs
   offline), (c) ollama qwen3.5:4b for protocol smoke only. GPU runs are reserved for measuring score.

## Architecture (arc3)
- `env/`     arc-agi adapter: offline + competition gateway modes, one scorecard, animation frames,
             level-only reset, keep-alive RESET until the model is up, per-game clocks.
- `percept/` segmentation (4-conn components, shape/pose hashes, containment, adjacency), ascii,
             10x PNG + diff PNG + fatal-frame PNG, frame_diff (move/rotate/appear/disappear/recolor/
             resize), animation collapse + transient-pixel detection, HUD border classifier.
- `sandbox/` single `python` tool in a subprocess: globals (current/previous/history/transitions/
             valid_actions/frame_diff/last_result), `action([...])` batching with guards
             (interior no-op stop, stale-state, terminal latch), retained functions, 30 s / 3k-token
             output caps, stdlib allowlist.
- `wm/`      NEW: executable world-model tools exposed to the model —
             `register_model(step, goal)`, `verify()` (replay every recorded transition; report
             first mismatch with before/after crops), `plan(max_depth, max_nodes)` (BFS/A*/IDDFS
             over the model from current state to goal), `execute(plan)` (step-verified; stops at the
             first prediction mismatch and reports it). Also a harness-side generic transition graph
             (state hash → action → state hash) for loop detection and "untried action" hints.
- `agent/`   prompts (system, opener, game-over diagnosis with BAR rule, level transfer, resume),
             context manager (calibrated token estimate incl. images, block trimming with hysteresis,
             thinking kept), turn loop with yield after N generated tokens, failure retry/abandon.
- `llm/`     OpenAI-compatible client (SGLang), tool-call parsing, reasoning passthrough.
- `sched/`   priority gate over 10 decode slots: P=(A+B)·C with the published formula, slot kept
             across tool calls, handover at context trim, tail fade in the last 20%.
- `kaggle/`  notebook builder: offline wheel install, SGLang launch from public datasets with
             prefetch, health wait, keep-alive, competition-rerun detection, diagnostics off.
- `scripts/` local runner (fake model / ollama / remote endpoint), per-game report, trace viewer,
             kaggle push/pull/status wrappers.

## Schedule (WIB)
- D1 Oct 27: interfaces + env + percept + sandbox + agent loop + fake-model tests; serving notebook
  that boots SGLang from public datasets and runs our agent on 3 public games. Kaggle run #1 (plumbing).
- D2 Oct 28: prompts/guards/context/scheduler complete; wm tools v1; 25-game offline fake-model pass;
  Kaggle run #2: 8-game dev subset, compare per-game vs dfranzen's public numbers.
- D3 Oct 29: fix what run #2 shows; wm/planner tuning on the games it should win (navigation,
  puzzles); Kaggle run #3: 25 games.
- D4 Oct 30: packaging hardening (deadline clamps, abandon logic, memory); ask user for submission #1.
- D5–6 Oct 31–Nov 1: iterate on run logs; submission #2; choose the two finals with the user by Nov 2.

## Delegation
Parallel opus agents for env+percept, sandbox+guards, agent+llm+context, kaggle builder; sonnet for
tests/tooling/docs. I hold the interfaces (`src/arc3/types.py`) and integration. Each agent gets a
narrow brief with the reference files (`reference/duck-architecture-brief.md`,
`reference/dfranzen-writeup-notes.md`, `reference/executable-world-models-digest.md`).

## Open questions for the user
1. GPU beyond Kaggle? Renting one RTX PRO 6000 (~$2–3/h) for ~100 h would turn 5 measurements into 30.
2. Remaining Kaggle GPU quota this week (shown in the notebook editor sidebar).
3. Reusing dfranzen's public serving datasets/models (infrastructure only) — confirm acceptable.
