# Top ARC-AGI-3 writeups and settings (as of 2026-10-29, ~04:00 UTC)

Method: Kaggle pages were read through a real browser (WebFetch only returns the page title). Primary code was pulled from GitHub raw URLs. Anything I did not read myself is marked "second hand". Speculation is marked "SPECULATION". Numbers are quoted as published.

## 0. Bottom line (read this first)

1. **The "36-39 cluster" is not a different technology.** Every public notebook above 31 is Daniel Franzen's Milestone 2 notebook plus small add-ons (second hand, from the mayankraj4728 playbook, which pulled about 30 notebooks). Same-code score spread is large: forum thread 745062 (second hand) gives mean 25.77, SD 3.93. Franzen unchanged has drawn 21.01, 25.67, 27.62, 27.80, 27.83, 27.89, 31.47, 35.35. Best-of-100 draws from that distribution is about mean + 2.5 SD, which is about 35.6. That matches the cluster. DERIVED, not published. Evidence in favour: rellik13 shows 62 entries and Lord Han Solo 105 entries on the board.
2. **The lordhansolo "host-tier KV pool, 1.4M tokens, 14-16 streams" pointer is a blend of two different entries.** Lord Han Solo runs vLLM (not SGLang), has no host KV tier, and gets 1,417,100 tokens from `gpu_memory_utilization` 0.98 with 14 streams. The host tier belongs to rellik13 (Lohit Siriki), on SGLang, with only a 1,004,288-token GPU pool and 16 streams. Details in section 1.
3. **No team above 40 has published a method.** Chen (75.56), Tufa (68.91), Turducken (56.02) and the next six are silent. Tufa has said it will release after the competition ends. Chen said he will not open-source for the milestone.
4. **Our current settings (10 streams, mem-fraction 0.96, pool about 1.01M, 116k trimmed to 58k) are identical to Franzen's.** Nothing I found contradicts them. Three published A/B results say moving away from them hurts (section 5).
5. **The highest-value finding is platform risk, not tuning.** Zero-score and error reruns are still being reported as of Oct 28. Staff have announced no extension and no rescore as of the last page I read (section 4).

## 1. Server settings, verbatim, per source

### 1.1 Side-by-side table
Source: Kaggle discussion 744792 "Comparing the Milestone 2 winning solutions", author Tong Hui Kang (25th), 65 votes, posted about a month ago (early Oct 2026). Cells copied from the page. Tong Hui Kang links each cell to the line of code it came from.

| | dfranzen (public LB 27.89) | lordhansolo (23.84) | sirikilohit = rellik13 (22.53) |
|---|---|---|---|
| Model and quant | Intel/Qwen3.8-Flash-Next-W4A16-AutoRound (AutoRound W4A16 + BF16 PLE) | primitive-ai/Qwen3.8-Flash-Next-mixed-NVFP4-FP8 (NVFP4 + FP8 mixed + BF16 PLE, NVFP4 MTP head built in) | Intel W4A16-AutoRound + RadixArk FP8 PLE |
| Speculative | 3 draft tokens, Albucino INT4 MTP draft, 64k FR-Spec hot tokens | 3 draft tokens, built-in MTP head, 32k draft vocab | 3 draft tokens, NVFP4 MTP draft (RadixArk) |
| Weights on GPU | 73.64 GB (69.85 target + 3.79 draft) | 71.94 GB (target incl. MTP head) | 73.57 GB (69.78 target + 3.79 draft) |
| KV + Mamba on GPU | 15.94 GB (11.58 target KV, 0.96 draft KV, 3.40 Mamba) | 19.69 GB | 16.97 GB (11.50 target KV, 0.96 draft KV, 4.51 Mamba) |
| CUDA graphs | 1.14 GB (0.48 target verify, 0.37 draft decode, 0.29 draft extend) | 0.46 GB | 1.59 GB (0.65, 0.61, 0.33) |
| GPU memory left free | 4.25 GB | 1.90 GB | 3.17 GB |
| Offloaded to host | 95.37 GB (BF16 PLE) | 96.55 GB (95.37 BF16 PLE + 1.18 BF16 embed_tokens) | 95.68 GB (47.68 FP8 PLE + 48 GB hierarchical KV + Mamba cache) |
| Server | SGLang Pennyroyal v2.5.3 | vLLM 0.29.1rc1 nightly e975732 | SGLang Pennyroyal v2.5.0 |
| MoE backend | auto (Marlin for W4A16) | auto (flashinfer_cutlass for NVFP4) | auto (Marlin), draft on flashinfer_cutlass |
| Linear-attention backend | flashinfer | flashinfer prefill (auto), triton decode | flashinfer |
| KV dtype | fp8_e4m3 | fp8_e4m3 | fp8_e4m3 |
| Pool sizing | mem fraction 0.96 | gpu util 0.98, profiled | mem fraction 0.97 + 48 GB host tier |
| KV pool (tokens) | 1,011,264 | 1,417,100 | 1,004,288 |
| Context (server / harness) | 139,264 / 131,072 | 147,072 / 127,488 | 69,632 / 69,632 |
| Games streaming | 10 (all 110 games live) | 14 (14 live) | 16 (28 live) |
| Notebook start to ready | 531 s | 544 s | 615 s |
| Peak decode tok/s | 946 | 1,135 | 1,159 |
| Temperature | 0.7 | 0.6 | 0.6 |

Reading of the pool sizes (DERIVED):
- Franzen and rellik13 both hold about 11.5 GB of target KV for about 1.0M tokens (about 11.5 KB per FP8 token).
- Lord Han Solo's 19.69 GB covers KV plus Mamba. If Mamba is about 3.4 GB as in Franzen's run, KV is about 16.3 GB, which matches 1.417M tokens at the same bytes per token.
- Where his extra memory comes from, from the table: no separate draft model KV (Franzen 0.96 GB), 1.7 GB smaller weights, 2.35 GB less free headroom, 0.68 GB less CUDA graph. The 0.98 utilisation figure is the profiler's budget, not a measured gain.
- There is no published recipe for 1.4M tokens on SGLang with the W4A16 checkpoint. The nearest evidence is rellik13's host tier, which does not enlarge the GPU pool (his pool is 1,004,288).

### 1.2 Franzen, SGLang launch (verbatim from the D′ notebook cell, which carries Franzen's launcher unchanged)
Source: `ref_kernels/kawh1111_dprime/arc-agi-3-dprime-3154.ipynb` in github.com/mayankraj4728/ARC-AGI-3-2026, cell 13. I checked that it matches Franzen's CFG as described in the 744792 table and his README. I did not open Franzen's own notebook.

```
CFG = dict(
    CTX=(116+12+8)*1024,   # 139,264
    MEMFRAC=0.96,
    MAXREQ=10,
    CUDAGRAPH_MAXBS=10,
    MAMBA_CACHE=60,
    CHUNK=8192,
    MAX_PREFILL=16384,
    SPEC=True, FRSPEC=True, AUTOTUNE=True,
    PREFETCH_CHECKPOINTS=True, DROP_CACHE_AFTER_LOAD=True,
    LINEAR_BACKEND="flashinfer",
    KVDTYPE="fp8_e4m3", SSM_DTYPE="bfloat16",
    MAMBA_RADIX="extra_buffer", GDN_MTP_CACHE_MODE="none",
    SPEC_ACCEPT_SINGLE=1.0, SPEC_ACCEPT_ACC=1.0, SPEC_STEPS=3,
)
sglang serve --model-path MODEL_DIR --load-format safetensors
  --model-loader-extra-config '{"enable_multithread_load":false}'
  --tensor-parallel-size 1 --dtype bfloat16 --quantization auto-round
  --kv-cache-dtype fp8_e4m3 --mem-fraction-static 0.96
  --context-length 139264 --page-size 64
  --max-running-requests 10 --chunked-prefill-size 8192 --max-prefill-tokens 16384
  --cuda-graph-max-bs-decode 10 --cuda-graph-bs-decode 1 2 4 7 8 9 10
  --mamba-ssm-dtype bfloat16 --max-mamba-cache-size 60
  --mamba-radix-cache-strategy extra_buffer --mamba-track-interval 64
  --mamba-backend flashinfer --linear-attn-decode-backend flashinfer --linear-attn-prefill-backend flashinfer
  --moe-runner-backend auto --ple-offload-embedding --trust-remote-code
  --mm-feature-transport cpu --image-processor-backend pil
  --reasoning-parser qwen3 --tool-call-parser qwen3_coder
  --default-chat-template-kwargs '{"preserve_thinking":true}'
  --watchdog-timeout 1800 --schedule-policy lpm --warmups structured_output
  --enable-cache-report --enable-metrics --enable-request-time-stats-logging
  --weight-loader-prefetch-checkpoints --weight-loader-drop-cache-after-load
  --gdn-mtp-cache-mode none
  --speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1
  --speculative-num-draft-tokens 4 --speculative-draft-model-path DRAFT_VIEW
  --speculative-draft-model-quantization compressed-tensors --speculative-moe-runner-backend auto
  --speculative-draft-kv-cache-dtype fp8_e4m3
  --speculative-accept-threshold-single 1.0 --speculative-accept-threshold-acc 1.0
  --speculative-token-map <64k FR-Spec map>
```
Also in Franzen's README (already in our notes): Pennyroyal v2.5.3, commit d00d88efc8d6281b12be4f4073126aec95038c55; five patches (low-M BF16 GEMM, speculative-state memory budget, Marlin scale dtype, sparse Mamba prefix-cache checkpoints with LRU refresh, bounded checkpoint prefetch). Prefetch: `--weight-loader-prefetch-num-threads` default 4, 16 MiB blocks.

### 1.3 rellik13 / sirikilohit, SGLang launch (verbatim, cell 11_4 of his submitted notebook)
Source: github.com/LohitSiriki/arc-agi-3-milestone2-solution at commit af105202327dcf35bd670a50b2558adab12e8dd3, `cells/11_4_import_the_bundled_source_and_boot_sglang.py`.

```
--tp 1 --dtype bfloat16 --quantization auto-round --kv-cache-dtype fp8_e4m3
--mem-fraction-static 0.97 --context-length 69632 --page-size 64
--max-running-requests 16 --chunked-prefill-size 4096
--mamba-radix-cache-strategy extra_buffer --mamba-ssm-dtype bfloat16
--max-mamba-cache-size 80 --gdn-mtp-cache-mode none
--linear-attn-decode-backend flashinfer --linear-attn-prefill-backend flashinfer
--mamba-track-interval 64 --ple-offload-embedding --trust-remote-code
--reasoning-parser qwen3 --tool-call-parser qwen3_coder
--speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1
--speculative-num-draft-tokens 4 --speculative-draft-model-quantization modelopt_fp4
--speculative-moe-runner-backend flashinfer_cutlass
--watchdog-timeout 1800
--enable-hierarchical-cache --hicache-size 48 --hicache-write-policy write_through --hicache-io-backend kernel
--weight-loader-prefetch-checkpoints --weight-loader-prefetch-num-threads 16
--model-loader-extra-config {"enable_multithread_load": true, "num_threads": 8}
```
His own comments in that file:
- The draft MoE must run on `flashinfer_cutlass`. The auto choice picked FLASHINFER_TRTLLM, which is SM100 only, and the server died at draft CUDA-graph capture.
- He patches the SGLang KV configurator so the speculative SSM scratch reservation (comment says 2.9 GB at 12 requests) is not subtracted from the KV budget when `gdn_mtp_cache_mode` is "none". His first attempt removed it entirely and ran 15 requests, and "the server died ~10-15 min into play". The comment says a later attempt kept half and ran 13 requests, and the final line reads "v14.64 v3: reservation removed". Franzen's patch 2 is the same idea. The shipped notebook ran 16 requests.
- Draft-only checkpoint folder: the RadixArk checkpoint has 206 shards, but only 3 files hold the 31 `mtp.*` tensors. Loading only those cut the draft load from 365-459 s to about 135-153 s ("boot -5 min").
- Parallel PLE row copy on 16 threads, and prefetch run once instead of per checkpoint. Both cut load time.

His writeup explains the 0.97 and host-tier choices only as "memory moved off the GPU". He never isolates the host tier. It is not shown to have helped.

### 1.4 Lord Han Solo, vLLM config (verbatim, `configs/inference.json` in his TAAF source bundle)
Source: github.com/tonghuikang/daniel-franzen-arc-agi-3, `kaggle/lordhansolo/dataset-taaf-kaggle-source/src/ARC3-Inference/configs/inference.json`. This is a third-party mirror of his dataset. His Kaggle writeup was read directly (kaggle.com/writeups/lordhansolo/arc-agi-3-milestone-2, dated Oct 2, 2026).

```
"shared":   {"model_name": "primitive-ai/Qwen3.8-Flash-Next-mixed-NVFP4-FP8", "provider": "vllm", "context_window": 127488}
"server":   {"backend": "vllm", "tensor_parallel_size": 1, "max_model_len": 147072, "dtype": "bfloat16",
             "quantization": "compressed-tensors", "gdn_decode_kernel": "triton",
             "gpu_memory_utilization": 0.98, "kv_cache_dtype": "fp8_e4m3", "indexer_kv_dtype": "fp8",
             "mamba_ssm_cache_dtype": "bfloat16", "max_num_seqs": 14, "max_num_batched_tokens": 2048,
             "kv_cache_gib": 0, "mtp_tokens": 3, "draft_vocab": "configs/draft_vocab_32k.json",
             "enable_prefix_caching": true, "prefix_match_unit": 128, "prefix_cache_retention_interval": 0,
             "tool_call_parser": "qwen3_coder", "reasoning_parser": "qwen3",
             "default_chat_template_kwargs": {"preserve_thinking": true, "reasoning_effort": "xhigh"}}
"environment": {"concurrent_jobs": 14, "n_passes": 1}
"analyzer": {"target_context": 81536, "max_output": 0, "thinking": true, "reasoning_effort": "xhigh",
             "temperature": 0.6, "top_p": 0.95, "top_k": 20, "timeout": 120, "tool_steps": 0,
             "tool_timeout": 30, "tool_output_tokens": 1024, "yield_seconds": 150}
"multimodal": {"context": "current_grid", "upscale": 4}
```
Kaggle-run values from his Makefile (target `kaggle-duck`):
```
# 14 workers make 8 waves, 522.4 minutes at 65.3 minutes (65 min 18 s) per game.
# This leaves 17.6 minutes within the 9 h kernel limit for setup and shutdown.
MAX_RUNTIME_MINUTES = 65.3        # = 3,918 s per game
MAX_EXPERIMENT_RUNTIME_MINUTES = 540
N_PASSES = 1   CONCURRENT_JOBS = 14   ANALYZER_TIMEOUT = 900
```
His writeup says the "yield seconds" per-turn limit "should grow along with" concurrency "so a turn still gets about as many tokens as before", and that moving game state from disk to memory removed most tool latency.

Not found: the actual vLLM command line, speculative method string, or his patch list beyond the names in table 1.1 and PATCH_IDENTITY.json. I did not read the vLLM runtime dataset itself.

## 2. Agent-side techniques

### 2.1 Solved-level memory pinning (rellik13, cell 42 "M85", shipped in his 22.53 notebook)
Read in full. What is kept and where:
- Trigger: the first prompt after a level transition.
- Per cleared level it records: level number, action count, last `M85_ACTIONS_KEPT = 30` actions, the last `M85_REASONING_CHARS = 900` characters of the model's reasoning from the clearing turn, and a rule.
- The rule: the same prompt appends this text. "LEVEL N SOLVED. In your reply this turn, before acting, write one short paragraph that starts exactly with 'Rule for level N:' -- the goal, what each control/action did, and the plan that won (at most 5 lines). It is pinned for every later level." Asked for up to `M85_ASK_TURNS = 2` prompts, then recorded as "not stated". Rule capped at `M85_RULE_CHARS = 600`.
- Placement: one block at the END of the system prompt, between `=== SOLVED LEVELS (facts from this game + your own conclusions; kept for the whole game) ===` and `=== END SOLVED LEVELS ===`. It is replaced in place on updates.
- Newest `M85_FULL_LEVELS = 3` levels keep actions plus reasoning. Older levels keep the rule only.
- Closing line in the block: "Reuse these where the new level has the same controls; verify with one probe before relying on them."
- Why he built it: in 4 lost levels the agent re-derived the previous rule for 60-100+ minutes. The harness's own "Cross-level notes" field was filled in 0 of 989 and 0 of 1,189 level>=2 prompts.
- Measured by him: +2.8 (7.14 to 9.96), one run each. His writeup calls it "the only text addition that helped".
- Cache cost (DERIVED): the system prompt sits at the start of the prompt, so every update forces a full re-prefill of that game's history. lalalia9's port avoids this by rewriting only during trims (second hand, playbook).
- Contradicting evidence: the mayankraj4728 team's five leaderboard draws with memory average about 25.2 against 27.9 for D′ without it (n=5 vs n=3, noise SD about 4). Not significant either way.

### 2.2 Context trimming
| Source | Setting |
|---|---|
| Franzen | window 128 Ki, 12 Ki output reserve, trim at 118k down to about 59k in one go; reasoning kept in history; 150-assistant-message cap drained to 120 (table); token counts calibrated against backend usage |
| rellik13 | trim at 57,344 back to 45,056 real server tokens (cell 35, M84); context 69,632 (12,288 above the trim point); history turn cap lifted to 1,000,000,000; the original 30-assistant-turn cap deleted because it could wipe a long turn. Earlier values were 36,864 back to 26,624, and 49,152 context. |
| Lord Han Solo | trim at about 126k down to a target of 81,536; measured with a local token estimate; no turn cap |
| rellik13 (cell 33) | uses the server tokenizer to measure prompt length before trimming (the exact-count "context guard") |

Both Franzen and rellik13 say the point of big trims is to keep the prefix cache hot (Franzen: 93% reuse).

### 2.3 Time allocation across games
| Source | Rule |
|---|---|
| Franzen | priority scheduler, 10 admitted of 110 live. Constants already in `dfranzen-writeup-notes.md`. Slot kept across tool calls, handed over at context trim. |
| D′ (shiiin9 / kawh1111 / Affectify team, byte-identical public notebooks, 31.54 draw) | exact formula below |
| rellik13 | one shared pool, UCB, 2,400 s slices (cell 44) |
| Lord Han Solo | fixed 3,918 s cap per game, 14 workers, 8 waves, freed slot goes to the next game at once |
| Tufa, Chen, others | unknown |

**D′ formula** (verbatim from the code comments in the D′ notebook; tuned in simulation, the author claims "+1.5 approx over v1"):
```
priority = A*M*C + B*phi
A  = (1 + 0.5*(L-1)) * norm(N) * (300/(300+a))^2.5          # L=level (1-based), a=actions on this level
norm(N) = 55/(n*(n+1)/2), n = clamp(N,6,10), N=total levels (10 if unknown)
M  = clip((30000/pace)^0.4, 0.25, 4)   # pace = mean tokens per cleared level so far; M=1 if none cleared
C  = 0.1*max(0.1, 1 - a/115) + 0.9*max(0.1, 1 - t/T)         # t = tokens spent on this level
T  = 225000 * clip(pace/30000, 0.5, 2)^0.5                   # T x1 if no level cleared yet
B  = {>=3 levels left: 16, 2 left: 14, 1 left: 10, 0 left: 0}
phi = 1 for the first 60% of runtime, then fades linearly to 0 over the last 40% (Franzen's tail_fraction)
Unstarted games are ranked by the same formula (L=1, a=t=0) instead of "everyone starts once".
```
Evidence for it: leaderboard draws 31.54, 30.74, 29.05, 24.77, 30.01 (mean about 29.0; three D′-only draws by the playbook team average 27.94). A third-party log of 23 changes (charleneleong-ai/tgaer PR 61, second hand) gives plain Franzen mean 26.52 (24.60-28.39) against slot-priority mean 27.10 (22.40-29.71). So the real gain is somewhere between 0 and +3 and not resolved.

**rellik13 UCB** (verbatim from cell 44; ported from `Wang-Zhongwei/duck-harness` `inference/framework/allocation.py`, which is now private):
```
_M72_QUANTUM_S = 2400.0      # was 300 s; "resume re-prefill costs ~10% of the GPU on 110 games"
_M72_UCB_C = 0.0002
_M72_RECENT_TOKENS = 32_000
budget = min(ceil(games / concurrency) * max_runtime_s_per_game, soft time left)
warm-up: every game plays one slice, in order
then each free slot goes to the playable game with the highest
  UCB = priority + c * sqrt(log(1 + minutes_elapsed) / actions_taken)    # unvisited games = infinity
  priority = 0.5 * (levels_per_token over life + levels_per_token over last 32k generated tokens)
  levels weighted L*(L+1)/2 (same as the score)
A slice ends at the first request boundary after its end. Won, game-over and out-of-actions games are never resumed.
```
Warnings in the same cell: "no score gain was measured for the scheduler in isolation" (Wang's note); rellik13's notebook README table says the shared pool is "~31k s" with every game getting a 2,400 s slice first. His writeup text says a priority scheduler was "held back" and mentions only "40-minute time slices"; the shipped code is the UCB cell, so trust the code.

### 2.4 Other agent settings
- **Parallel games.** Franzen 10 (110 live). Lord Han Solo 14 (vLLM, pool 1,417,100). rellik13 16 (28 live, pool 1,004,288, history only 57k).
- **Thinking budget per turn.** No team caps thinking tokens. Franzen reserves 12 Ki output. rellik13 keeps 12,288 above the trim point. Lord Han Solo sets `max_output` 0 (unbounded), `reasoning_effort` "xhigh", `yield_seconds` 150 and `timeout` 120. A ThinkingCap model with about 38% fewer thinking tokens was tested and rejected (section 3).
- **Action batching / no-op stop.** Franzen stops the remaining actions in a snippet when the interior board is unchanged (4-cell border excluded). A second guard (no-op action check) gave no gain in the playbook team's A/B (+3.3 +/- 5.4 practice, 23.57 on the board).
- **Strategy audit** (Lord Han Solo). When a level stalls the turn prompt asks the model to separate observed facts from assumptions and run one discriminating experiment. rellik13 measured a version at 25% of game time. The playbook team A/B: +1.9 +/- 6.3, 85 levels vs 87 and 89 for the controls, so inconclusive.
- **Saved modules with regression tests** (Lord Han Solo `save_module`). Tests replay against observations captured when the module was added, so a later change that breaks an earlier test is rejected. He says he "cannot confirm that any of these changes improved the score".
- **Helper retention.** The playbook team found that a model-call timeout wiped all Python helper functions (119 lost in one run). Fix gave +2.8 +/- 5.7 offline, below noise. Franzen keeps functions for the whole game.
- **Harness wording fixes (rellik13, reported 13.40 to 14.49).** Rename ACTION7 to UNDO. Say "Level restarted." instead of "The game is over." after a reset. Delete the prompt line that told the agent to ignore the edge bar, since that bar is the action budget. Expose intermediate animation frames. Franzen made the same changes.
- **Image upscale.** Franzen 10x (his biggest perception gain). Lord Han Solo 4x. rellik13 says higher resolution did not help.

### 2.5 How teams decide a game is hopeless
Nobody ships a validated stop rule. The data that exist:
- Francise Liang (thread 750885, 25 public games x 2 passes, about 9 days ago): 19.3% of generated tokens were spent after a game's last level-up. Median gap between level-ups 19.8k tokens. The largest gap that still ended in a level-up was 104k tokens. Ending a game once a level used 110k tokens gave mean 38.97 against 48.11 for the two baselines, a loss of 9.1 (10 wins, 11 losses, 4 ties). Games it cut, such as r11l at level 1/6, were fully cleared in both baselines.
- rellik13: restarting a stuck level with fresh context did not help; "the chance of clearing a level didn't fall with time spent on it".
- Rakha Abid Bangsawan (forum 739938, shadow mode only, 25-game run): a progress-weighted allocator proposed 4 early stops, and 0 of them would have missed later progress. Tiny sample.
- Franzen's soft version is the decay terms in his priority (token and action decay), not a cutoff.

## 3. What separates 50+ from 39 (mostly unknown)

Facts:
- Public board today, read from the Kaggle leaderboard page at about 03:57 UTC Oct 29: 1 Yi-Chia Chen 75.56 (44 entries), 2 Tufa 68.91 (174), 3 Turducken 56.02 (213), 4 the last dance 51.51, 5 yktori 51.04, 6 mtg 50.37, 7 NVARC3 48.84. Rank 24 rellik13 37.97 (62 entries). Rank 25 Tong Hui Kang 37.92. Rank 29 Md Boktiar Mahbub Murad 37.12. Rank 30 Lord Han Solo 37.07 (105 entries). The 36-39 band runs from rank 18 to about rank 46.
- Chen's 75.56 comes from only 44 entries and a jump 28.34 to 48.59 to 67.49 to 71.48 to 75.56 (history from `public-solutions-survey.md`). Selection from a mean-26 distribution cannot produce that, so he runs a different system.
- Tufa: the thread 742801 comments around Sep 30 show a jump to 27.29 then 45.33 (the commenter does not name the team, but the preceding remark says Tufa "just found a big improvement"). ARC Prize on X: Tufa 11.04, then Chen 59.17 on Oct 9, then Tufa "jumped from 56% > 68%" (the search result implies Oct 13; our survey says Oct 19-20, not reconciled).
- Tufa wrote on Sep 30 (thread 742801): "all high-scoring public notebooks now building on this harness" and that "We remain fully committed to sharing our final solution when the competition ends". CPMP (7th) said his team will open-source only if they end in gold.
- Son Pham, ranked 3rd when he posted in thread 744792: "I did not thing of potentially just pruning of changing the serving until I came a cross a REAP Qwen3.8 model ... It feels like an entire untapped area to explore." and "So many token wizardry ... we only figured out how important these are late in the game." I cannot confirm his team is Turducken.
- Counter-evidence on pruning: rellik13 tried "pruning a quarter of the model's experts to free memory. More room, worse play."

SPECULATION (nothing published supports these):
- The top three may serve the same model but run it far more efficiently (pruned experts, more tokens per game), as Son Pham's comments hint.
- Chen's ARC-AGI-2 result (84.72, second to Tufa 88.06, from our survey) suggests a strong general scaffold rather than a fine-tuned Qwen.
- No evidence of RL or fine-tuning for any top team. Failed public attempts: rellik13's LoRA on Qwen3.8-27B trained on open-pool games "scored badly"; the author of thread 739938 (addressed as phuongncn in the replies) trained on his own full playthroughs and it "wasn't really better".

Measured negatives worth not repeating (all second hand unless noted; where I read the source I say so):
| Change | Result | Source |
|---|---|---|
| 14 streams instead of 10 on SGLang (`ARC3_MAX_ACTIVE_STREAMS`, MAXREQ, CUDAGRAPH_MAXBS) | 29.90 vs 47.30 mean, 5.07M vs 8.74M generated tokens; the 14-stream machine also had slow disk (837 s vs 219 s preload) | Francise Liang, read directly |
| Stop a game after 110k tokens on one level | -9.1 | Francise Liang, read directly |
| 12 streams, or compacting per-turn prompts | "neutral on public but cost about 7-8 points on hidden" | lalalia9 writeup via playbook |
| Extra prompt text | 6.42 to 4.16; 13.40 to 9.98 | rellik13, read directly |
| ThinkingCap model (about 38% fewer thinking tokens) | 62.24 vs 74.47 offline, 57 vs 62 of 72 levels | playbook team, A/B n=1 per game |
| Temperature 1.0 | no gain (rellik13); 0.3 worse (playbook) | |
| MXFP8 re-quant of non-expert layers | about 25% slower decoding | rellik13, read directly |
| Swift 1.5 (public fine-tune) | about 35% slower, no better per token | rellik13, read directly |
| Go above 16 concurrent requests | saturation, sometimes slower | thread 739938 author, read directly |
| Structured world-model updates, summarisation, composite animation images, death ledger, repeated-state guards | no gain | Franzen (in our notes) |

## 4. Platform notes (state as of the last page read, 2026-10-29 ~04:00 UTC)

- **Competition status.** The page header says "5 days to go". Official end per our notes: submissions close Nov 2.
- **Slow dataset cache.** Staff María Cruz (Kaggle Staff, thread 751598, posted Oct 23): "Over the past few days, the dataset cache was unstable in some regions. This caused slow model/dataset reads and startup timeouts. Dustin (@herbison) has deployed fixes ... He also suggested storing model weights in Kaggle Models rather than Datasets." Dustin (thread 751756, same week): "There was some leftover poisoned cache entries in one other region, I cleared them, and I'm running a job to defrag that cache a bit."
- **Still unresolved after those fixes.** Reports dated Oct 25-28:
  - Thread 753556 (Thalcor, Oct 28): 0.00 "still ongoing".
  - Thread 752855 (Masum Huq): six consecutive "A system error" failures Oct 23-26 across two kernels, including a new slug.
  - Thread 752485 (xz, updated Oct 28): server start takes 10 to 60 minutes for the same notebook, and slow-starting instances ran about 10% slower afterwards.
  - Thread 753192 (Michael Poluektov, Oct 27-28): "model load times increased from roughly 300 seconds ... to 800 seconds"; some runs sat 8+ hours with no output, then began printing.
  - Thread 751598, Sebastian Brodehl: "Filesystem performance still varies by daytime", and new cases where the first cell never starts.
  - A Model-mount error persists with Kaggle Models too: `ERRORED_MOUNTING_DATASET ... retry budget exhausted (30 attempts)` (CPMP, thread 752147; Tsitsino Dzotsenidze).
- **Gateway DNS.** dustyhorizon (thread 751598, submission from Oct 25): the run crashed after 22m35s with `RuntimeError: Kaggle gateway did not become ready: URLError(gaierror(-3, 'Temporary failure in name resolution')) after 600 s of retries`, while model weights had loaded normally (185.40 GB, 754.5 s). A rerun on a different machine scored 8.61 against the usual 33.68, which they read as a partial scorecard. Their standard parameters were "1-pass / concurrency 120 / 31920s". The playbook team widened the gateway wait from 10 to 30 minutes (taking the extra wait off each game's time).
- **Zero-score signature.** Many teams report `COMPLETE`, 0.00, no error text, and a 3,411-byte `submission.parquet`, against 3,775 to 4,100 bytes for healthy runs. One report puts the one-row placeholder parquet as the likely output (Hamachi). Runs often end after about 27 to 40 minutes or 1.5 to 4.5 hours instead of about 9.
- **Inactivity rule.** Jeroen Cottaar (Tufa, thread 750173): "If you don't do any move for 15 minutes, the run will end with the score so far". His fix: a thread that issues a RESET move to one game every 10 minutes until startup finishes. The playbook team's keepalive sends a free RESET every 8 minutes. Caveat (Ya Xu): this keeps the session alive but the 20-40 minutes of load time still comes out of game time.
- **Other fixes teams report.** Retry game start up to 4 times at 5 s spacing (one failed RESET closes the shared scorecard, reproduced on CPU). cosmorin (thread 753489, Oct 28): stock Franzen with a 90-minute server wait scored 26.16 on the board, up from 23.39. Several people suggest submitting early in the UTC day so an error can be retried the same day (unverified).
- **Extension.** Requests only. CPMP (7th), Benedicte Helfer, Grigoriy Zatravkin (8th), Son Pham and ivan (7th) asked in thread 751598 for an extension or a freeze. Thread 753379 "Extension for the competition deadline" has 14 votes and 0 comments. I found no staff announcement of an extension, rescore or restored slots. Latest staff text I could read (María Cruz, 1 to 2 days before I read it): "We are actively looking into the broader platform issues ... due to capacity constraints, we aren't able to look into or troubleshoot individual submissions right now ... we will share updates in Topic 751598 and Topic 752147". Zakinosuke wrote "thank you for the rescores" to the staff, which hints some rescores happened. I could not find staff confirming it.
- **Rerun rules from the host.** Greg Kamradt (Competition Host, thread 743723): private scores are "calc'd at original run time. They aren't rerun"; the submission plays both datasets and only the public 50% is shown; the limit is 9 hours for v3. Tong Hui Kang (unverified): a Kaggle Error does not cost a submission if the games were not queried.
- **Scorecard/timing.** Brodehl (unverified): networking and filesystem setup is probably not counted against play time, but copying model and code files from datasets does count.
- **Not read:** the ARC-AGI-2 forum posts quoted by staff (#3543186, #3543199), forum threads 745062, 746010, 743723 (body), 746295, 751198, 751344, 751427.

## 5. Ranked findings

Ranking = expected gain for us, then implementation risk (lower is safer). Our baseline is Franzen's M2 (best 22.23 for us; the public notebook draws 21 to 31).

| # | Finding | Expected gain | Risk | Notes |
|---|---|---|---|---|
| 1 | Startup safety set: gateway-ready wait of 30 min, game-start retry (4 x 5 s), a keepalive RESET every 8-10 min while the model loads, model-server wait up to 90 min, and a per-game budget derived from the wall-clock deadline so a slow load comes off the shared budget | Prevents a 0.00. One 0.00 run costs the whole score, and Francise Liang reports 2 of 12 submissions scored 0.00. The playbook team reports six 0.00 runs among 27. | Low. Code is public in the playbook repo (`extra_cells/startup_keepalive.py`, `startup_retry.py`) and weitaoguo's gateway probe. | A smoke run showed the keepalive held through a 25-minute load. Do not use a 55-minute hold that delays all games (playbook rejects it). |
| 2 | Put model weights in Kaggle Models, not Datasets (staff suggestion) | Fewer slow mounts | Low-medium. Mixed reports: still slow with Models for some users. | Cannot be validated from here. |
| 3 | Keep settings as they are: 10 streams, 0.96, 116k trimmed to 58k, FP8 KV | Protects the baseline | None | Three published A/B or hidden-set results say changing them loses 7-17 points. |
| 4 | D′ priority formula (section 2.3) in place of Franzen's A,B,C | +0 to +3, unresolved | Low. Pure Python, drop-in for `priority_value`, already run in about 5 public draws. | Franzen's tail-fade and slot-at-trim behaviour must stay (D′ keeps them). Needs `ARC3_PRIORITY_PACE=1` and a pace tracker. |
| 5 | Solved-level rule memory (rellik13 M85, lalalia9 cache-safe port) | -2 to +3, unresolved | Medium. System-prompt rewrites invalidate the cache unless done at trims. | Franzen's harness already keeps history and level-transfer guidance. Playbook draws are lower with it. |
| 6 | Reduce load time: draft-only checkpoint folder (3 of 206 shards), parallel PLE copy, prefetch once | Saves about 3-5 minutes of game time per run (his figure for the draft folder: "boot -5 min") | Low-medium. Anchored source patches fail loudly if the build differs. | Only applies if we use the RadixArk NVFP4 draft. Franzen's Albucino INT4 draft is a different, smaller checkpoint. |
| 7 | rellik13 UCB with 2,400 s slices | About 0, unmeasured in isolation | Medium. Resume costs about 10% GPU, and it competes with Franzen's slot-at-trim cache design. | Skip. |
| 8 | Strategy-audit prompt after 25% of game time | +1.9 +/- 6.3, not significant | Low-medium (adds prompt text) | Only after items 1 to 4. |
| 9 | Host KV tier (`--enable-hierarchical-cache --hicache-size 48 ...`) | Unproven | Medium-high. Needs about 48 GB more pinned host RAM. With the BF16 PLE (95.37 GB) that is about 143 GB of the 180 GB the machine has, versus about 96 GB for rellik13 who used FP8 PLE. | A diagnostic by the playbook team shows no benefit to gain: 0 retractions at 10 streams, KV use median 0.41 and p95 0.76. Skip. |
| 10 | Reaching a 1.4M pool or 14-16 streams | Negative or zero on SGLang | High | DERIVED: 10 streams x average context between 59k and 118k is about 0.88M, under the 1.011M pool, so the pool is not binding at 10. 14 streams x 88k is about 1.23M, over the pool, which is consistent with the preemption xz suspects. More streams only makes sense together with a bigger pool and a lower trim, as Lord Han Solo (14 streams, target 81,536, pool 1.417M) and rellik13 (16 streams, trim 57k, pool 1.004M) did. Both finished below Franzen at Milestone 2. |
| 11 | Switch to vLLM + NVFP4 (Lord Han Solo's stack) | Negative | High | Playbook note: NVFP4 variants about 22-24 on the board. |

## 6. Contradictions with our current settings
None of the above contradicts what you listed. Three places to double-check:
- **Throughput comparison.** Our "about 760 tok/s across 10 games" is above Franzen's average (609 in his writeup, 600-620 in Francise Liang's baseline run). The 946 / 1,135 / 1,159 figures in table 1.1 are peaks, not averages. Do not read them as targets.
- **Memory fraction.** The playbook says "never MEMFRAC > 0.96" (OOM risk, a greenalgeria notebook), yet rellik13 ships 0.97 with 16 streams. His setup also removes the speculative SSM scratch reservation and has a smaller context (69,632), so the 0.97 is not transferable to a 136k-context server.
- **Context.** Franzen 139,264 server and 131,072 harness. Lord Han Solo 147,072 and 127,488. rellik13 69,632. All three score within noise of each other. No published evidence ranks them.

## 7. Source list
Read directly:
- Kaggle discussion 744792 (M2 comparison), 751598 (zero scores), 751756, 752147, 752485, 752727, 752855, 753192, 753379, 753489, 753556, 750173, 750885, 742801, 732854, 739938, 743723 (host reply)
- Kaggle code listing sorted by score, and the leaderboard page
- Kaggle writeup https://www.kaggle.com/writeups/lordhansolo/arc-agi-3-milestone-2
- https://github.com/LohitSiriki/arc-agi-3-milestone2-solution (WRITEUP.md, README.md, cells 11_4, 33, 35, 40, 42, 44)
- https://github.com/da-fr/arc-agi-3-solution (WRITEUP.md, serving/README.md)
- https://github.com/tonghuikang/daniel-franzen-arc-agi-3 (Lord Han Solo `inference.json`, `Makefile`, `kaggle.py`)
- https://github.com/mayankraj4728/ARC-AGI-3-2026 (README.md playbook, `ref_kernels/kawh1111_dprime` notebook, cells 13 and 21)

Search results only (not opened): https://x.com/arcprize/status/2110031207601979715 (Tufa 68.91), https://x.com/arcprize/status/2108577347192676612 (Chen 59.17), https://github.com/cataluna84/arc-agi-3 (no relevant content; scores of 0.19 to 0.24 from May).
