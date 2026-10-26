# Notes from Daniel Franzen's Milestone-2 write-up (public LB 27.89, Oct 1 2026)
Source: https://www.kaggle.com/writeups/dfranzen/arc-agi-3-milestone-2 (read 2026-10-27)
Repo: https://github.com/da-fr/arc-agi-3-solution (Duck harness fork, base commit 7652836)

## Ranked impact (author's TL;DR)
1. Better model: Qwen3.6-27B -> Qwen3.8-27B -> Qwen3.8-Flash-Next; each a substantial gain.
2. Expose animation frames (intermediate frames) as REPL variables; hint to inspect only when
   "transient pixels" exist (pixels changing more than once start->end). tn36 essentially needs this.
3. Perception: image upscale 4x -> 10x (biggest image win); difference images (changed pixels since
   last analyzer turn); show the fatal frame after game over.
4. UNDO (ACTION7) exposed and explained; large gain on competition set (su15 locally).
5. Guards: stop remaining actions in a snippet when interior board unchanged (4-cell border excluded);
   stale-state guard from level 2; terminal-state guard after game over / level complete.
6. Context: 128Ki window, 12Ki output reserve; trim in large blocks (hysteresis) down to ~57Ki;
   keep model thinking in history (removing it = large regression); disable the structured
   text world-model mechanism; retain model-defined Python functions across calls/levels.
7. Throughput: W4A16 AutoRound quant, FP8 KV, MTP speculative decoding (3 steps), 10 parallel
   streams, prefix cache 93% reuse, Mamba checkpoint retention patch. ~609 tok/s generation.
8. Scheduler: all games started, 10 admitted at a time; priority P=(A+B)*C with
   A = l*(25/(25+a))^2 * 55/(N(N+1)/2), B = {>=3 left:8, 2:7, 1:5, 0:0},
   C = 0.25*2^-(a/115)^2 + 0.75*2^-(t/62000)^2 (a=actions on level, t=tokens on level);
   B fades linearly to 0 in last 20% of runtime. Slot kept across tool calls; handover at trim.

## Prompt snippets worth reusing (paraphrase, author's wording in write-up)
- Game-over semantics: game_over = attempt FAILED, level auto-resets, completed levels kept.
  Diagnose: history[-2] fatal action, history[-2].frame death frame, history[-3].frame before.
  Bar rule: shrinking border bar is normal; budget death only if bar (almost) empty.
- "Do NOT re-submit the same series of moves: the game is deterministic."
- Level transfer: carry mechanics forward as hypothesis; new elements usually matter; reassess goal.

## Did NOT help
- Structured world-model updates (parsing brittle) -> disabled entirely.
- Summarization (inserted or replacing) -> no clear gain.
- Composite animation images -> hurt. Auto timeline insertion -> no gain.
- Extra verification hints, repeated-state guards, death ledger -> unclear.
- Fancier priority rules -> no clear gain over constant tail.

## Serving settings
Single RTX PRO 6000; 10 streams; server ctx 136Ki; chunked prefill 8192; temp 0.7 top-p 0.95 top-k 20;
static mem fraction 0.96; Mamba cache 60 slots. Model shards resharded + background prefetch for startup.
Kaggle datasets: dfranzen/pennyroyal-v253 (sglang wheelhouse 6.6GB), models
dfranzen/intel-qwen3.8-flash-next-w4a16-autoround, dfranzen/albucino-qwen3-8-flash-next-drafter.
Dev compute: Kaggle + ~150 rented RTX PRO 6000 hours.
