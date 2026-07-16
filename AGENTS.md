# AGENTS.md — jed_attack (AI Agent Security competition)

You are iterating on a Kaggle red-team attack submission to **raise its normalized score**.
Read `docs/strategy.md` first — it is the verified objective and the scoring math that decides
what wins. Read `docs/superpowers/specs/2026-07-16-jed-attack-harness-design.md` for the design.

## Mission

Improve the attack's normalized score against the two target models (**gpt_oss**, **gemma**)
under the public guardrail, without breaking the submission contract. The score comes from the
SDK's own scorer via `run_attack` — it matches the leaderboard scale. Bigger is better.

The dominant lever (see `docs/strategy.md`): **severity-per-candidate × candidate-count**, bounded
by a replay-budget ceiling of ~200–300 candidates. Diversity is a rounding error. Stack multiple
predicates into single candidates; repeat the best candidates to fill the budget.

## Setup: two local model APIs (run once, keep resident)

The real models run as two `llama-server` endpoints (llama.cpp, OpenAI-compatible). Use the exact
`Q4_K_M` GGUFs so behavior matches Kaggle. On the GPU host:

```bash
# gpt-oss on :8080, gemma on :8081  (download the GGUFs first; see docs/serving.md)
llama-server -m models/gpt-oss-20b-Q4_K_M.gguf          -ngl -1 -c 8192 --jinja --port 8080 &
llama-server -m models/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf -ngl -1 -c 8192 --jinja --port 8081 &
export GPT_OSS_BASE_URL=http://localhost:8080/v1
export GEMMA_BASE_URL=http://localhost:8081/v1
```

`docs/serving.md` has the full launch details. The harness reaches these via
`jed_attack.harness.models.llama_server_agent_factory`.

## The experiment loop (your inner loop)

```bash
# Fast mechanics check (CPU, deterministic agent, ALLOW guardrail) — run after EVERY code change:
uv run python -m pytest -q

# Real-model experiment — scores the attack against both served models, records to the archive:
uv run python -m jed_attack.scripts.experiment --budget-s 900          # both models
uv run python -m jed_attack.scripts.experiment --model gpt_oss --budget-s 900   # one model
```

`experiment` prints per-model normalized scores and appends `(config, scores)` to
`runs/runs.jsonl` (via `jed_attack.harness.report.save_run`). Compare successive runs there to
know whether a change helped. Keep changes that raise the score; revert changes that don't.

Iterate: edit `src/jed_attack/submission/{attack,selector,templates}.py` (and `encoders.py` for
exfil payloads) → run pytest → run an experiment → compare → keep or revert.

## Hard guardrails (do NOT violate)

1. **Submission isolation is sacred.** Everything under `src/jed_attack/submission/` imports ONLY
   `aicomp_sdk` + the Python stdlib + its own siblings — never `jed_attack.harness.*`, never a
   network/3rd-party package. The built `dist/attack.py` must import only `aicomp_sdk` + stdlib.
   The test `tests/test_submission_isolated.py` enforces this against the BUILT source — it must
   stay green.
2. **Pre-commit must pass before every commit:** `uv run pre-commit run -a` (ruff, ty, pytest all
   green). Fix type errors properly; never add `# type: ignore` or `# noqa` to silence them.
3. **Respect the replay ceiling.** The attack self-calibrates a candidate cap; keep it. Never
   return an unbounded/huge candidate set — on the real models an over-large set fails the whole
   submission (no score), it does not degrade gracefully. Target ~200–300.
4. **Do not overfit the public guardrail.** Candidates are scored against a stricter hidden
   private guardrail too. Prefer robust attacks.
5. **Rebuild + verify after attack changes:** `uv run python -m jed_attack.scripts.build_submission`
   then confirm `tests/test_build_submission.py` (incl. the end-to-end run test) passes.
6. **Do not change** the SDK vendoring (`vendor/`), the runner's scoring path
   (`harness/runner.py`), or the model-factory fidelity wiring (`harness/models.py`) — those keep
   local scores matching the leaderboard. Extend the attack, not the measurement.

## Out of scope for you

- **Do not submit to Kaggle** — the human does that (it spends a daily slot). Your job is to raise
  the local score, which predicts the leaderboard.
- **Do not reinterpret scoring** — `docs/strategy.md` is verified against the SDK. If you think it's
  wrong, verify against `aicomp_sdk.scoring.score_attack` and note it, don't silently diverge.

## Where things are

- Attack (edit these): `src/jed_attack/submission/{attack,selector,templates,engine,encoders}.py`
- Measurement (don't break): `src/jed_attack/harness/{runner,models,report}.py`, `scripts/experiment.py`
- Build: `src/jed_attack/scripts/build_submission.py` → `dist/attack.py`
- Objective & design: `docs/strategy.md`, `docs/serving.md`, `docs/superpowers/specs/`
