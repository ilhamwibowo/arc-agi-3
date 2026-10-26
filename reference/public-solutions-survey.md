# ARC Prize 2026 ARC-AGI-3: public solutions survey (as of 2026-10-27)

Caveat: Kaggle leaderboard/discussion pages render client-side and were not directly readable. Many facts come from secondary summaries and are flagged. I did not read the Kaggle notebooks, the write-ups or the rules page.

## 1. Leaderboard

Public Kaggle board, Oct 26 2026 17:43 EDT (arc.markbarney.net tracker; 4,865 teams; public half of test data only; final ranking uses the other half):

| Rank | Team | Score |
|---|---|---|
| 1 | Yi-Chia Chen @ Fields Model | 71.48 |
| 2 | Tufa Labs | 68.91 |
| 3 | Team Turducken | 56.02 |
| 4 | the last dance | 50.69 |
| 5 | mtg | 50.37 |
| 6 | NVARC3 | 48.84 |
| 7 | Grigoriy Zatravkin & co | 47.19 |
| 8 | 復活の混テキスト | 46.82 |
| 9 | artificialagencylab.com | 45.41 |
| 10 | Majkel1337 | 45.03 |

- Top-5 on the public board needs about 50.5 or higher now; expect the cutoff to rise by Nov 2 (Turducken went 52.82 on Oct 20 to 56.02 on Oct 26).
- History: Yi-Chia Chen 28.34, then 48.59, then 67.49 (Oct 19), then 71.48. Tufa 55.89 (Oct 4), then 68.91 (about Oct 19-20; ARC Prize on X: "jumped from 56% > 68%").
- Early-October articles (metirai, TechTimes) had Tufa first at 55.89 and Chen second at 48.59, with a cluster at 33-38. Older snapshots, superseded.
- Prizes: ARC-AGI-3 pool $850K. Grand prize $700K for 100% (unclaimed). Top-score award $75K split $40K/$15K/$10K/$5K/$5K across five places. Milestones $75K total.
- Non-Kaggle context: verified leaderboard (standardized harness, API models) GPT-6 Astra 62.7%, Claude Opus 5 30.2%. Not comparable (Kaggle forbids API models). OpenAI experiment (Jul 29): GPT-5.6 Sol 7.8% to 38.3% by keeping CoT between actions and using compaction instead of rolling truncation.

## 2. Public solutions

### Milestone 1 (Jun 30; $37.5K)
- **Tufa Labs "The Duck" (1st, $25K)**
  - Repo: github.com/Tufalabs/duck-harness. Write-up: tufalabs.ai/research/duck-harness. Notebook: kaggle.com/code/jeroencottaar/tufa-labs-duck-harness-june-30-milestone-winner.
  - Score: 1.21% on Kaggle (initial 1.30% retracted); 1.6002 +/- 0.4475 mean over 25 public games x 20 passes.
  - Model: Qwen 3.6 27B FP8 (from the Tufa blog and milestone blog, not the README).
  - Architecture: agent writes Python in a live REPL; observations are variables. Three perception channels: rendered image, ASCII grid, segmentation/zoom tools. "Infinite play via eviction": oldest messages dropped, system prompt and recent history kept.
  - Serving: local vLLM or OpenRouter; optional `server` extra installs vLLM and Torch (GPU-only). No version or flags in README.
  - License: not stated in README; LICENSE at main/LICENSE returned 404. Unconfirmed.
  - Wins: about 1/10 the cost per game vs GPT 5.4. Hand-built tools hurt; improvisation worked better.
  - Failures: uneven across games (some >40% of levels, some never clear level 1). Authors name context management and perception as weak points.
  - Repo layout: ARC3-Inference/ (harness), tufa-arc-agi-framework/ (TAAF), example-run/.
- **Reki (2nd, $10K):** Gemma-4-31B VLM policy; rendered recent frames in, one JSON action out; reflection memory refreshed about every 10 steps; numpy click heuristic needing no GPU.
- **forge, Md Boktiar Mahbub Murad (3rd, $2.5K):** similar image + JSON design on Gemma-4-31B; generator and arbiter over candidate actions; best run turned off the extra machinery.
- The official blog publishes no numeric scores for winners.

### Milestone 2 (Sep 30), per TechTimes
- Daniel Franzen (Kaggle: dfranzen) 27.9% ($25K); Lord Han Solo 23.8% ($10K); Lohit Siriki 22.5% ($2.5K).
- Notebook kaggle.com/code/dfranzen/arc-agi-3-milestone-2-solution shows public score 33.68 in one summary; conflicts with 27.9%. Unresolved (page unreadable). Guess only: 33.68 is the notebook's public-board score, 27.9% the milestone-eval score.

### Franzen's repo: github.com/da-fr/arc-agi-3-solution (copy: sabbaresef/arc-agi-3-solution)
- Fork of Duck harness at upstream commit 7652836. Credit to Jeroen Cottaar / Tufa Labs.
- Model: "Qwen3.8-Flash-Next" per README (name uncertain). Quantizations from RadixArk and Intel; exact checkpoints in the competition notebook and build scripts.
- Serving: optimized offline SGLang runtime built as a wheelhouse; patches for Mamba prefix-cache retention and checkpoint prefetching; see serving/README.md. No SGLang version in README.
- GPU: RTX Pro 6000, selected manually in the notebook.
- License: README silent; an earlier repo-page fetch said Apache-2.0. Verify LICENSE.
- Changes vs Duck:
  - Context: blockwise history trimming, calibrated token estimates, image-aware token accounting.
  - Compute: bounded concurrent admission; priority scheduling by progress, actions, tokens spent and continuation value; priority for cached continuations; tail-fade scheduling favoring level completions near deadline.
  - Perception: larger board images, board-diff and death images, animation frames accessible from Python, clearer action history.
  - Tools: executable undo, optional reset, cross-level knowledge-transfer guidance.
  - Safeguards: stale-state and no-op action checks, level-boundary protection.
- Lessons: evicting larger blocks at once improves prefix reuse; bounded admission reduces cache pressure; "enabling every option is not the intended setup." Negative/inconclusive results are in WRITEUP.md (not fetched; read it).
- Setup: use the Kaggle notebook; local viewer: `cd ARC3-Inference; uv sync --locked; make view VIEW_RUN_DIR=... VIEW_PORT=8011`. Docs: ARC3-Inference/README.md, CONFIGURATION.md.

Not found: other high-scoring public notebooks.

## 3. Discussion-thread lessons
Nothing found on the 15-minute no-action timeout, gateway quirks, RTX quota problems or common submission failures. WebFetch returned only titles for discussion/717133 and discussion/725002. Check the discussion tab and Discord manually. Confirmed from docs:
- Code competition; notebook runs twice. Phase A checks it runs; Phase B (on "Submit to Competition") plays the hidden games. Each submit uses a daily submission.
- Accelerators: CPU, T4, P100, RTX 6000 (ARC-AGI-3 notebooks only, burns GPU quota faster). Internet disabled.
- Repo lists a 6-hour runtime limit; the brief says 9h. Reconcile with the Kaggle rules.
- Game count not given in docs (`make list-games`); the brief says 110.
- GPT-OSS template docs warn the model is slow on the RTX hardware.
- Vendor model card: gateway exposes only the action API and rendered frame (not official).
- Repo lists team merge deadline 2026-10-26 (passed).

## 4. Official methodology (docs.arcprize.org/methodology)
- Metric: Relative Human Action Efficiency (RHAE).
- Per-level score = (human baseline actions / AI actions)^2. Baseline = 1.0; 2x actions = 0.25; 10x = 0.01.
- Per-level cap: 1.15x the human-baseline value.
- Game score: weighted average of level scores, weight = 1-indexed level number (later levels count more).
- Completion cap: in a 5-level game, finishing levels 1-4 caps at 10/15 = 66.7%. 100% requires completing the last level.
- Total score = mean of game scores.
- Human baseline per level = upper median of first-time players ranked by fewest actions (even count: upper of the two middle values).
- Only inputs that change game state count as actions; tool calls, reasoning and retries do not.
- Not on the page: per-game baseline counts, competition_mode rules. Try docs.arcprize.org/llms.txt and the Kaggle rules page.
- From secondary sources: solutions must be open source under a permissive license (CC0 / MIT-0) before private scores are released; no API models; no internet during evaluation. Dates: submissions close Nov 2, paper deadline Nov 8, winners Dec 4.

## 5. Papers and blogs
- Tufa Labs Duck write-up (tufalabs.ai/research/duck-harness); linked MLST episode and YouTube video (not opened).
- **Tycho** (arXiv 2607.28287, Jul 30, Lehmann, Aioanei, Vahdati): coding agent builds executable per-game world models during play and decides when to build, repair, use or skip them. Best policy (actor-requested delegation to a model builder) 88.49 mean RHAE on 25 public games with Opus 4.8; GPT-5.6 Sol and Opus 5 hit 100 RHAE (all 183 levels); Opus 5 used 61% fewer scored actions than aggregate human baselines. Auto-repair after verification failures lowered RHAE to 83.07. Single runs per policy. API models, so not usable offline.
- arXiv 2605.05138 "Executable World Models for ARC-AGI-3 in the Era of Coding Agents"; 2607.15439 "Do Coding Agents Need Executable World Models, Simplification, and Verification to Solve ARC-AGI-3?"; 2605.25931 "Explore Before You Solve: The Speed-Depth Trade-off in Epistemic Agents for ARC-AGI-3" (titles/abstract only; Tycho abstract read).
- Articles: TechTimes (Oct 6), metirai.

## 6. Top-team methods (follow-up)
- **Tufa Labs (current):** per the veeda241 tracking repo (verified 2026-10-19; second hand), Duck harness with Qwen3.8-Flash-Next, AutoRound INT4 quantized, with a speculative-decoding draft model. No evidence of RL/fine-tuning found. Franzen's fork uses the same model name.
- **Yi-Chia Chen (Fields Model Initiative; NTU and Waseda):** method unknown. Kaggle profile says Chen will not open-source the solution for the milestone prize. Also 84.72 on ARC-AGI-2 (second behind Tufa 88.06), suggesting a strong general scaffold (inference only).
- **Team Turducken:** no method found. The veeda241 repo is a competitor's tracker whose current target is Duck + Qwen3.8-Flash-Next, not Turducken's code.
- No public info on RL, fine-tuning, world models or action-efficiency tricks for the top teams. The jump from about 28 to 55-70 over October is unexplained publicly; model/stack upgrades are the best guess.

## Sources
- https://docs.arcprize.org/methodology
- https://docs.arcprize.org/arc-prize-2026
- https://arcprize.org/blog/arc-prize-2026-milestone-1
- https://github.com/Tufalabs/duck-harness
- https://raw.githubusercontent.com/Tufalabs/duck-harness/main/README.md
- https://tufalabs.ai/research/duck-harness/
- https://www.kaggle.com/code/jeroencottaar/tufa-labs-duck-harness-june-30-milestone-winner
- https://github.com/da-fr/arc-agi-3-solution
- https://raw.githubusercontent.com/da-fr/arc-agi-3-solution/main/README.md
- https://github.com/sabbaresef/arc-agi-3-solution
- https://www.kaggle.com/code/dfranzen/arc-agi-3-milestone-2-solution
- https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/leaderboard
- https://arc.markbarney.net/arc-leaderboards
- https://github.com/veeda241/ARC-Prize-2026---ARC-AGI-3
- https://www.kaggle.com/threerabbits/discussion
- https://x.com/arcprize/status/2110031207601979715
- https://x.com/tufalabs/status/2072336849465417747
- https://www.metirai.com/blog/arc-agi-3-kaggle-scores-surge-harnesses-arc-prize-2026
- https://www.techtimes.com/articles/328542/20261006/arc-agi-3-scaffolding-beats-model-upgrades-same-ai-two-settings-49x-score-gain.htm
- https://arxiv.org/abs/2607.28287
- https://arxiv.org/pdf/2605.05138
- https://arxiv.org/pdf/2607.15439
- https://arxiv.org/pdf/2605.25931
