# Serving the target models with `llama-server`

The competition scores against the GGUF models via llama.cpp. We run them as two
persistent `llama-server` endpoints (llama.cpp's OpenAI-compatible HTTP server), which
keeps the models resident and lets the harness — and Codex — hit them over HTTP with no
per-run reload. This is the highest-fidelity server option: same engine, same GGUF chat
template and tool grammar as the in-process path Kaggle uses.

## 1. Get the exact GGUFs

Use the **same Q4_K_M quant** as Kaggle (a different quant changes model behavior and breaks
score fidelity). Either download via the harness or from the Kaggle Models mirror:

```bash
# via the harness (Hugging Face; set HUGGINGFACE_TOKEN, accept the model licenses first)
uv run python -c "from pathlib import Path; from jed_attack.harness.models import download_gguf; \
  download_gguf('gpt_oss', Path('models')); download_gguf('gemma_4', Path('models'))"
```

- gpt-oss: `unsloth/gpt-oss-20b-GGUF` → `gpt-oss-20b-Q4_K_M.gguf`
- gemma:   `unsloth/gemma-4-26B-A4B-it-GGUF` → `gemma-4-26B-A4B-it-UD-Q4_K_M.gguf`
  (also on Kaggle Models as `llkh0a/gemma-4-26b-a4b-it-ud-q4-k-m-gguf` — accept the Google license).

## 2. Launch two servers

`--jinja` is required: it makes llama-server apply the GGUF's chat template and tool
formatting, matching the in-process `create_chat_completion` path.

```bash
llama-server -m models/gpt-oss-20b-Q4_K_M.gguf           -ngl -1 -c 8192 --jinja --port 8080 &
llama-server -m models/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf  -ngl -1 -c 8192 --jinja --port 8081 &
```

`-ngl -1` offloads all layers to GPU; `-c 8192` matches Kaggle's context. Point the harness at
them:

```bash
export GPT_OSS_BASE_URL=http://localhost:8080/v1
export GEMMA_BASE_URL=http://localhost:8081/v1
```

## 3. Run experiments

```bash
uv run python -m jed_attack.scripts.experiment --budget-s 900              # both models
uv run python -m jed_attack.scripts.experiment --model gpt_oss --budget-s 900
```

Scores go to `runs/runs.jsonl` (compare runs there) and `runs/submission.csv` (public scores;
the private guardrail is hidden, so it isn't computable locally).

## 4. Fidelity spot-check (do this once)

Before trusting server scores, confirm the HTTP path fires the same predicates as the
in-process llama.cpp path on the same candidate — build the in-process factory with
`gguf_agent_factory` and the server factory with `llama_server_agent_factory`, run the same
attack under each, and compare `per_predicate`. They should match; if they diverge, check that
you launched with `--jinja` and the exact Q4_K_M GGUF.

## Notes

- Both factories feed the SDK's real `GPTOSSAgent` / `Gemma4Agent`, so tool-call parsing (what
  the scorer's predicates read) is identical to Kaggle regardless of where inference runs.
- The `experiment` runner scores under the **public** `OptimalGuardrail` (the public
  leaderboard). Dev/unit tests use the permissive `GuardrailBase` because the deterministic CPU
  agent scores 0 under OPTIMAL.
