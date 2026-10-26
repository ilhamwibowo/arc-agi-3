# Kaggle deployment

Verified 2026-10-27 with `uv tool run --from kaggle==2.2.5 kaggle ...` (user `wibowoilham`) and the
public Milestone-2 notebook (`reference/public_notebooks/dfranzen-m2`, cells 4, 6, 8, 12, 20).

## Files

| File | Role |
|---|---|
| `src/arc3/scheduler.py` | `PriorityGate` (10 decode slots, P=(A+B)·C), `GameProgress` |
| `src/arc3/runner.py` | `run_games`: thread per game, keeper keep-alive, one scorecard, summaries |
| `src/arc3/kaggle_serve.py` | prefetch, offline SGLang install + launch, `/health`, system monitor |
| `src/arc3/kaggle_main.py` | notebook entry: gateway wait, game list, env/session/agent glue |
| `src/arc3/notebook.py` | builds `notebooks/<name>/<name>.ipynb` + `kernel-metadata.json` |
| `scripts/kaggle_push.py` | push/status/output/pull with records in `runs/kaggle/<name>/<version>/` |

Build: `.venv/bin/python -m arc3.notebook --name arc3-submission --offline-subset 3`
(`--offline-subset all|N|ls20,ft09`). The notebook embeds `src/arc3` as a base64 tar.gz (sha256
checked on unpack to `/kaggle/working/arc3`), so no dataset is needed for our code. Rebuild after
every source change.

## Inputs (all verified to exist and be public)

| Kaggle ref (kernel-metadata) | Mount path in the kernel |
|---|---|
| competition `arc-prize-2026-arc-agi-3` | `/kaggle/input/competitions/arc-prize-2026-arc-agi-3` (`arc_agi_3_wheels/`, `environment_files/<game>/<hash>/`, `ARC-AGI-3-Agents/`) |
| dataset `dfranzen/pennyroyal-v253` (6.64 GB, updated 2026-09-29) | `/kaggle/input/datasets/dfranzen/pennyroyal-v253` |
| model `dfranzen/intel-qwen3.8-flash-next-w4a16-autoround/Transformers/default/1` | `/kaggle/input/models/dfranzen/intel-qwen3.8-flash-next-w4a16-autoround/transformers/default/1` |
| model `dfranzen/albucino-qwen3-8-flash-next-drafter/Transformers/default/1` | `/kaggle/input/models/dfranzen/albucino-qwen3-8-flash-next-drafter/transformers/default/1` |

Note the lower-case `transformers` in mount paths. Both models have exactly one version (1), public.

Metadata: private, GPU, internet off, `machine_shape: NvidiaRtxPro6000`, and
`docker_image` pinned to the same image as the public notebook
(`gcr.io/kaggle-private-byod/python@sha256:57e612b4...`). Reason: the competition wheels and the
Pennyroyal wheels are cp312, and the SGLang venv is created from `sys.executable`. The October 26
CPU notebook kernel was Python 3.13 (with `/usr/bin/python3.12` present; see
`../arc-harness/docs/SUBMISSION_REQUIREMENTS.md`). Cell 0 raises if the kernel is not 3.12.
**Unverified**: whether the pinned image is accepted on push for our account; if not, drop
`docker_image` and handle 3.13 (run the agent with `/usr/bin/python3.12` in a subprocess).

## Wheelhouses

- Competition `arc_agi_3_wheels/`: 31 wheels, identical to `reference/kaggle-runtime-lock.json`
  (arc-agi 0.9.8, arcengine 0.9.3, numpy 2.4.4, pillow 12.2.0, requests 2.33.1, pydantic 2.13.2,
  matplotlib 3.10.8, flask 3.1.3, ...). Cell 2 installs `arc-agi` with
  `pip install --no-index --find-links <dir> arc-agi` into the kernel's Python. pandas (for the
  placeholder parquet) is not in it; it comes from the Kaggle image.
- Pennyroyal `pennyroyal-v253/`: `requirements.lock`, `dependencies.lock`, `hot_tokens_64k.pt`
  (FR-Spec map, sha256 `becfa41d...`), and `wheels/` (197 wheels: one `sglang-*.whl`, one
  `uv-*.whl`, torch, flashinfer 0.6.17 + cu130 JIT cache, nvidia CUDA 13 wheels incl. nvcc).

## SGLang launch (kaggle_serve.launch_sglang)

1. Extract the `uv` binary from its wheel; `uv venv --python sys.executable /tmp/sgl-pennyroyal/venv`;
   `uv pip install --no-index --find-links wheels -r requirements.lock`; reinstall the sglang wheel
   `--no-deps`. All offline (`UV_OFFLINE=1`).
2. CUDA from the venv's `nvidia/cu13` (symlink `lib64`, `.so` links, `libcuda.so` from the driver);
   host gcc/g++; `TORCH_CUDA_ARCH_LIST=12.0`; caches under `/tmp/sgl-pennyroyal/cache`.
3. Draft view: symlinked copy of the INT4-g32 MTP checkpoint with the quantization target rules
   rewritten to the regex form SGLang accepts (weights are never copied).
4. Command (port 30000, served name `flashnext`, log `/kaggle/working/serve.log`, detached):

```
sglang serve --model-path $MODEL --load-format safetensors
  --model-loader-extra-config '{"enable_multithread_load":false}'
  --served-model-name flashnext --host 127.0.0.1 --port 30000 --tensor-parallel-size 1
  --dtype bfloat16 --quantization auto-round --kv-cache-dtype fp8_e4m3 --mem-fraction-static 0.96
  --context-length 139264 --page-size 64 --max-running-requests 10 --chunked-prefill-size 8192
  --max-prefill-tokens 16384 --cuda-graph-max-bs-decode 10 --cuda-graph-bs-decode 1 2 4 7 8 9 10
  --mamba-ssm-dtype bfloat16 --max-mamba-cache-size 60 --mamba-radix-cache-strategy extra_buffer
  --mamba-track-interval 64 --mamba-backend flashinfer --linear-attn-decode-backend flashinfer
  --linear-attn-prefill-backend flashinfer --moe-runner-backend auto --ple-offload-embedding
  --trust-remote-code --mm-feature-transport cpu --image-processor-backend pil
  --reasoning-parser qwen3 --tool-call-parser qwen3_coder
  --default-chat-template-kwargs '{"preserve_thinking":true}' --watchdog-timeout 1800
  --schedule-policy lpm --warmups structured_output --enable-cache-report --enable-metrics
  --weight-loader-prefetch-checkpoints --weight-loader-drop-cache-after-load --gdn-mtp-cache-mode none
  --speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1
  --speculative-num-draft-tokens 4 --speculative-draft-model-path $DRAFT_VIEW
  --speculative-draft-model-quantization compressed-tensors --speculative-moe-runner-backend auto
  --speculative-draft-kv-cache-dtype fp8_e4m3 --speculative-accept-threshold-single 1.0
  --speculative-accept-threshold-acc 1.0 --speculative-token-map hot_tokens_64k.pt
  [--chat-template $MODEL/chat_template.jinja]
```

Client: `http://127.0.0.1:30000/v1`, model `flashnext`; readiness `GET /health` == 200.
Public notebook: ~10.5 min to READY. Model shards are prefetched in a background thread (60 s delay,
3 threads); the wheelhouse is read once in the server thread before install.

## Run loop

- Mode: `KAGGLE_IS_COMPETITION_RERUN` in {1,true} → competition; else offline on
  `environment_files` with `OFFLINE_SUBSET` (notebook variable, set at build time).
- Gateway (competition): `ARC_BASE_URL` default `http://gateway:8001/`, `ARC_API_KEY` default
  `test-key-123` (official starter + Duck notebook). Poll `<base>/api/games` until < 500, up to 600 s.
  Games = `Arcade(..., OperationMode.COMPETITION).available_environments` via `Arena.game_ids()`.
- `ONLY_RESET_LEVELS=true`, `MPLBACKEND=Agg`, `RECORDINGS_DIR=/kaggle/working/recordings` set in cell 0.
- Budget: `HARD_WALL = notebook start + 9 h`; the runner ends every game 15 min earlier (8 h 45 m),
  per-game deadline `min(wall - 15 min, game start + PER_GAME_CAP_SECONDS)`. Join is bounded at
  the wall - 13 min, then the single scorecard is closed regardless of hung games.
- Keep-alive: gateway ends the run after 15 min without an action. Game 1 is the keeper: RESET
  immediately, then every 300 s until any game sends a non-RESET action or `/health` is up.
  Games start in cell 6 while SGLang is still installing/loading; the agent waits on `/health`.
- A watchdog thread sets `SERVER_DEAD` if the install/launch raises or the server exits; the runner
  then stops waiting (closes the scorecard, writes summaries) instead of idling for 9 h.
- Outputs: `/kaggle/working/runs/kaggle/<game>/{summary.json,trace.jsonl}`, `run_summary.json`,
  `serve.log`. Offline runs write the starter's placeholder `submission.parquet`
  (`row_id='1_0', game_id='1', end_of_game=True, score=1`); in a rerun the gateway produces it.
- Failures: setup cells raise with explicit messages. Run/server failures are printed with the
  serve.log tail; the last cell raises in offline mode but never in a rerun (what was played is
  already on the scorecard).

## Pushing (not done yet)

`.venv/bin/python scripts/kaggle_push.py push arc3-submission` (asks for confirmation; `--no-run` for
a quick save), then `status`, `output` (logs → `runs/kaggle/arc3-submission/<version>/output/`), `pull`.
