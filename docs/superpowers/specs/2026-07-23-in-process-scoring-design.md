# In-Process Scoring Re-Architecture — Design

**Date:** 2026-07-23
**Status:** Approved (pending spec review)
**Component:** `src/jed_attack/campaign/submission_score.py`, `src/jed_attack/harness/models.py`, green runtime

## Goal

Make green's local `score_submission` faithful and deterministic by replaying through **in-process `llama-cpp-python`** (the exact engine the T4 competition gateway uses) instead of green's HTTP `llama-server`. Retire the llama-servers entirely.

## Motivation (the investigation chain)

The optimizer's local scores did not transfer to the real leaderboard. Root-caused over this session (see memory `jed-t4-replay-time-budget`):

1. The scorer's shape-dedup inflated scores (fixed: no-dedup, commit `ea8bf1c`).
2. Even faithful, green-local overestimated the LB ~2× **and was non-deterministic**: the same submission scored 1.405 / 1.56 / 1.81 with wildly different per-message firing, because green scores via `llama-server` with `-np 8` (continuous batching → non-reproducible greedy), while the gateway uses **in-process `llama-cpp-python`**. Sampler-pinning (`repeat_penalty`) did not converge — it's an engine difference, not a sampler bug.
3. **Spike validated (2026-07-23):** re-scoring the 5 verified candidates in-process on green reproduced **gemma exactly (0.370 vs T4 0.360) and deterministically**, confirming in-process is the fix. gpt_oss showed a residual ~1.4× (1.410 vs 0.990) — gpt_oss-specific (its long reasoning is numerically sensitive, or its GGUF differs from the `llkh0a` upload), but now **consistent/deterministic**, so rankings hold and it is calibratable.

Determinism is the crucial win: noisy scores can't steer the optimizer (the refine loop compares scores to decide "improvement"); consistent scores can.

## Key decisions (locked with the user)

1. **In-process replay.** `score_submission` uses resident in-process `llama-cpp-python` backends via the existing `gguf_agent_factory`, not `llama_server_agent_factory`. `SandboxEnv` + `eval_predicates` + board math are unchanged (already faithful).
2. **Resident, one backend per model.** Each GGUF loads once (module-level singleton) and is reused across every replay — no per-candidate reload.
3. **GPU placement in one process.** gpt_oss on GPU 0 (RTX 3090), gemma on GPU 1 (RTX 6000 Ada), via `main_gpu` + `split_mode=LLAMA_SPLIT_MODE_NONE` threaded through `gguf_agent_factory` → `from_model_path(llama_kwargs=...)` → `Llama`.
4. **Per-model lock.** One `Llama` context is not thread-safe, and the async team scores via `asyncio.to_thread`; a per-model `threading.Lock` serializes replays on each model. The two models run in parallel (separate GPUs), so a submission's gpt_oss and gemma passes overlap.
5. **Retire the llama-servers.** Scoring no longer uses HTTP; the `gptoss`/`gemma` tmux server sessions and serve scripts are gone. `run_optimizer.sh` just launches the optimizer, which loads the models in-process.
6. **`llama-cpp-python` is a real dependency.** Add it to `pyproject` (with the CUDA-12.8 build note) so `uv sync` stops dropping it. Green build: `CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=86;89"`.

## Architecture

### `gguf_agent_factory` (harness/models.py) — add GPU placement

Already loads the GGUF once and returns a factory over a shared backend (matches Kaggle's `gguf_model_server.py`: n_ctx 8192, max_new_tokens 1024, tools on). Add a `main_gpu: int | None = None` parameter that, when set, forwards `llama_kwargs={"main_gpu": main_gpu, "split_mode": llama_cpp.LLAMA_SPLIT_MODE_NONE}` to `from_model_path` so the model loads entirely on that GPU. (`from_model_path` accepts `llama_kwargs` and forwards them to `Llama` — verified.)

### `submission_score.py` — swap the backend, drop HTTP failover

Today: `resolve_endpoints(model)` returns HTTP endpoints, and `agent_factories[(model, endpoint)] = llama_server_agent_factory(...)`, replayed via `replay_trace_failover` across endpoints.

Change to resident in-process backends:

- A module-level cache `_backend_factory(model_key) -> factory`, built once per model via `gguf_agent_factory(model_key, gguf_path, main_gpu=_MAIN_GPU[model_key])`. GGUF path from `gguf_target_path(model_key, MODELS_DIR)`; `MODELS_DIR` and the model→GPU map are config.
- A module-level `_locks: dict[str, threading.Lock]` (one per model).
- `replay_trace(message, model_key, guardrail_factory)` acquires `_locks[model_key]`, builds the env with the resident backend's agent, replays, releases. The multi-endpoint `replay_trace_failover` collapses to the single resident backend (there is no HTTP endpoint to fail over to) — remove the endpoint/failover machinery for the in-process path.
- The message/model loops and board math are otherwise unchanged.

### Concurrency

The async team keeps its shape (one worker per API-key lane; proposer stays concurrent over external APIs). Scoring serializes per model via the locks; the two models replay in parallel. Since the GPU is the bottleneck, serialized per-model scoring is the natural ceiling, not a regression.

### Green runtime

- No `gptoss`/`gemma` llama-server tmux sessions. Remove/retire the serve scripts and the llama-server launch from any runbook.
- `run_optimizer.sh` unchanged in shape (it already just launches the optimizer), but the optimizer process now loads both GGUFs in-process at startup (needs `LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64` and `CUDA_DEVICE_ORDER=PCI_BUS_ID`, which the launch already sets for LD_LIBRARY_PATH; add the device-order export).
- Both GGUFs resident: gpt_oss ~11 GB on the 3090 (24 GB), gemma ~16 GB on the Ada (49 GB). Fits.

## Determinism & the gpt_oss residual

In-process single-context greedy is deterministic (spike-confirmed: gemma per-message `[16,16,0,16,16]` reproduced). The gpt_oss residual (~1.4×) is consistent, so it does not break ranking; it is calibratable if we ever need absolute LB prediction. **Parallel follow-up (not in this plan):** confirm green's `gpt-oss-20b-Q4_K_M.gguf` (11624759488 B) matches the `llkh0a/gpt-oss-20b-gguf` Kaggle model_source; if it differs, aligning the file likely closes the residual.

## Testing

- **Unit (no GPU):** `gguf_agent_factory` forwards `main_gpu`/`split_mode` into `from_model_path`'s `llama_kwargs` (assert on a stubbed `from_model_path`). `score_submission` acquires the per-model lock and uses the resident backend (monkeypatch the backend + a stub replay; assert one backend built per model, lock used, per-message severity from each message's own replay — reuse the existing no-dedup test shape).
- **On-green integration (manual, GPU):** load both models in one process on their GPUs (`main_gpu`), re-score the 5 verified candidates → gemma ≈ 0.37, gpt_oss ≈ 1.41 (matches the spike), and two back-to-back runs are **identical** (determinism). This is the acceptance gate for the deploy.

## Risks

- **Single-process two-GPU placement is untested** (the spike used per-process `CUDA_VISIBLE_DEVICES`; the design uses in-process `main_gpu`). The on-green integration test above validates it before deploy. Fallback if `main_gpu` misbehaves: two scorer subprocesses, one per model/GPU (heavier; avoid unless needed).
- **Throughput drop:** in-process serialized replay is slower than batched llama-server. Acceptable — faithful + deterministic is the requirement, and the ~385s/candidate T4 budget already caps submissions at ~23 candidates.
- **`llama-cpp-python` not in the lockfile** currently drops on `uv sync`; adding it to `pyproject` fixes it, but the CUDA build must be reproduced on green (documented flags).

## Out of scope

- Closing the gpt_oss residual (separate GGUF-parity follow-up).
- Parallelizing the two models inside a single `score_submission` call beyond the natural separate-GPU overlap (optimization, later).
- The time-aware objective (separate, already calibrated at ~385s/gpt_oss-candidate).
