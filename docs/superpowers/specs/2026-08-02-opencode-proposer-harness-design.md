# OpenCode Agentic Proposer Harness — Design

## Goal

**Fully replace** the one-shot proposer (`optimize_prompts.propose_batch_async`) with an
**agentic OpenCode session** that can call tools — fetch a candidate's victim trace and
live-score a candidate — so the proposer models can *systematically debug* their way to better
attack candidates instead of generating a JSON batch blind. The one-shot path is removed. Our
scoring / blackboard / objective / wandb loop is unchanged; only *how candidates are authored*
changes.

## Motivation

The public score has plateaued (`best_objective` ~2.2, `mean_public` ~0.09): most proposed
candidates barely fire, and the optimizer is **proposer-quality-bound, not throughput-bound**
(confirmed by reverting the 2× scoring pool with no effect). A one-shot completion cannot see
*why* a candidate failed to fire or *how many characters* it made the victim generate (the
gen-char objective's cost) — it just emits text. An agent that can call `score_candidate` and
`get_trace`, read the result, and iterate can close that loop. All live proposer models
(deepseek-v4-flash, mimo-v2.5, minimax-m3, kimi-k2.7, glm-5/5.1/5.2/5-turbo) were probed and
**all support OpenAI tool calling**, so the fleet can drive an agent loop today.

## Key decisions (locked)

- **OpenCode is the proposer — a full switch.** `propose_batch_async` (the one-shot path) is
  **replaced and deleted**; no enable flag, no A/B, no one-shot control lane. Our loop still owns
  authoritative scoring (`score_submission`), the blackboard, the gen-char objective, reship, and
  wandb — whatever the agent returns is **re-scored by us** (the agent's self-scores are advisory).
- **Headless, driven from Python.** Run `opencode serve` (HTTP/OpenAPI) and drive it from the
  optimizer via the Python `opencode-agent-sdk` (or raw HTTP). One session per lane.
- **Custom tools = thin TS wrappers → Python CLI.** OpenCode tools are TS files in
  `.opencode/tools/`; each shells to a new module `jed_attack.campaign.opencode_tools`
  (`python -m … <tool> <args>` → JSON). No scoring logic is reimplemented in TS.
- **Two tools to start (YAGNI):**
  - `score_candidate(message_text)` → live in-process score of a single message via the SAME
    `score_submission`: returns `{fires, public, gen_chars, severity_by_model, trace_id,
    trace_summary}`. Deterministic (seed 123), so it matches the outer loop exactly.
  - `get_trace(trace_id)` → the full victim replay trace for a prior `score_candidate` call
    (session-scoped cache), so the agent can inspect the generated tail / tool events in full.
- **Agent output contract:** the session's final message is the batch in the existing
  `SubmissionBatch` JSON schema (same `_salvage_batch` parser). The agent iterates internally
  via tools, then emits the batch.
- **Providers reuse existing config.** OpenCode points at z.ai + cheapestinference
  (OpenAI-compatible) with the existing keys (`ZAI_API_KEY`, `CHEAPEST_API_KEY`); model roster
  mirrors `config.TEAM_PROPOSERS`.
- **System prompt = adapted `prompts.toml`.** The gen-char objective, tool inventory, and
  single-post-EXFIL guidance become the agent's instructions, plus a short "debug loop"
  preamble ("score a draft, read its trace, make it fire in fewer generated chars, repeat").
- **Agents get the superpowers skills.** The whole premise — "systematic debugging → better
  ideas" — *is* a superpowers skill, so the agents must be able to invoke `systematic-debugging`
  (and `brainstorming`, etc.). See Skills below.

## Architecture

```
per lane, per generation:
  optimizer  --(incumbent + objective)-->  OpenCode session (provider = lane model)
                                              |  agent loop (bounded):
                                              |    draft candidate
                                              |    score_candidate(text) --> Python CLI --> score_submission
                                              |    get_trace(trace_id)   --> Python CLI --> cached trace
                                              |    refine; repeat until confident or budget hit
                                              v
  optimizer  <--(final SubmissionBatch JSON)-- session
  optimizer: score_submission (AUTHORITATIVE) -> blackboard.append -> objective / reship / wandb
```

The outer loop (`worker_loop`, `_refine_batch`, blackboard, `_gen_chars_cost` objective, reship)
is untouched. `propose_batch_async` is **replaced** by the OpenCode driver as the sole proposer
and the one-shot code is **deleted** — no flag, no A/B, no control lane. A full switch: if
OpenCode fails, proposing fails loudly (see Error handling), by design.

## Tool backend (`jed_attack.campaign.opencode_tools`)

A small CLI exposing the two tools, reusing existing primitives:

- `score_candidate`: builds a one-message `Submission`, calls `score_submission`
  (`GATE_GUARDRAILS`, `config.MODELS`), returns fire/public/`_gen_chars_cost`/per-model
  severity + a `trace_id` and `trace_summary` (`victim_feedback.trace_summary`). Caches the
  full `MessageScore.trace` under `trace_id` in a session-scoped store (temp file keyed by
  session id) for `get_trace`.
- `get_trace`: returns the cached full trace for a `trace_id` (or an error if unknown).

Determinism: same seed, same in-process backends as the outer loop, so an agent-tested score
equals the loop's re-score — no drift, no double standard.

## Skills (superpowers)

OpenCode natively discovers `SKILL.md` (same frontmatter format as Claude) from, in order,
`.opencode/skills/`, `.claude/skills/`, `.agents/skills/` (project, walking up to the git
worktree) and `~/.config/opencode/skills/`, `~/.claude/skills/`, `~/.agents/skills/` (global).
So the agent can use superpowers skills — but superpowers ships as a **plugin cache**
(`~/.claude/plugins/cache/claude-plugins-official/superpowers/<ver>/skills/`), which is NOT a
native discovery path. We expose the needed skills onto one:

- Symlink (or vendor a pinned copy of) the wanted superpowers skills into the worktree's
  `.opencode/skills/` so they are committed and reproducible per worktree — at minimum
  `systematic-debugging`; `brainstorming` and others as they prove useful.
- Superpowers cross-skill references (`superpowers:<name>`) and its `Skill`-tool invocation are
  plugin-specific; if they don't resolve under OpenCode's skill mechanism, vendor the
  self-contained subset (flatten the `superpowers:` prefixes). Confirmed in M2.

## Throughput & budget

Agentic authoring is slower and costlier per candidate than the old one-shot call. This is an
accepted trade — the bottleneck is *quality*, not throughput — but it is bounded:

- Hard budgets per generation: max tool calls (`OPENCODE_MAX_TOOL_CALLS`) and an idle/session
  timeout, so a flailing agent can't run unbounded.
- Scoring stays single-gpt_oss (the 2× pool was reverted); `score_candidate` calls contend on
  the same resident backend lock as the outer loop — acceptable at agent cadence.
- N lanes = N concurrent OpenCode sessions, matching today's per-lane worker model.

## Migration (full switch — not a trial)

Build in milestones so a blocker surfaces early and loudly; the end state is OpenCode as the
sole proposer with the one-shot path deleted.

- **M1 — Tools + skills.** `opencode_tools` CLI (`score_candidate`, `get_trace`) + the
  `.opencode/tools/` wrappers; expose the superpowers skills (`systematic-debugging`,
  `brainstorming`) on an OpenCode discovery path. Unit-test the CLI against `score_submission`.
- **M2 — Driver + mechanics.** Python OpenCode driver (session per lane, system prompt from
  `prompts.toml`, provider = lane model). Prove on one provider that the agent invokes
  `systematic-debugging`, calls the tools, debugs a non-firing/bloated draft via `get_trace`,
  and returns a valid firing candidate leaner than its first draft. A blocker here is a real
  finding to raise immediately (fail-loud) — not an adoption vote. Runs in a worktree so the
  live optimizer is untouched until M3.
- **M3 — Cut over.** Replace `propose_batch_async` with the driver for the WHOLE team and
  **delete** the one-shot proposer path. Restart the optimizer on OpenCode.
- **M4 — Validate live.** Confirm the loop scores / reships / logs as before and
  `best_objective` moves; watch wandb. Any OpenCode failure is visible (fail-loud), never masked.

## Config

- `OPENCODE_SERVER_URL` / spawn config; `OPENCODE_BIN` path.
- `OPENCODE_MAX_TOOL_CALLS`, session idle timeout — per-generation budget.
- Provider→model map derived from `config.TEAM_PROPOSERS` + `providers.PROVIDERS`.
- No enable flag and no one-shot path — OpenCode IS the proposer.

## Observability

Per generation, log to wandb: tool-call count, debug iterations, and first-draft vs final
candidate quality (fires, gen_chars) — so we can see the agent's debug behavior and confirm the
objective moves after cut-over.

## Error handling

**Fail loud.** OpenCode is the only proposer, so a session/driver failure must surface (error the
generation, log it, tag it in wandb) — never be silently swallowed. A broken OpenCode must look
broken, not quietly produce nothing. This matches the repo philosophy: raise a concise error for
unsupported cases rather than papering over failures.

- OpenCode/session/driver failure → **raise and surface** (visible error + wandb tag); skip that
  generation. There is no one-shot path to hide behind.
- Tool errors return a structured `{"error": …}` to the *agent* (its own retry, not a harness
  workaround) so it can try a different draft.
- Provider 429 / concurrency handled as today (retry/backoff), surfaced per key.

Hardening the failure path (retry/resilience) is **out of scope until OpenCode is proven** —
deliberately not now, so we harden a known-good path later instead of masking a broken one today.

## Testing

- `opencode_tools` CLI: unit tests that `score_candidate` output matches `score_submission`
  for a firing and a blocked message, and that `get_trace` round-trips the cached trace.
- Loop integration: a fake OpenCode driver (returns a canned batch) proves the driver feeds
  scoring / blackboard / objective correctly; a failing fake driver proves the generation errors
  **visibly** (raises, no silent swallow).
- M2 mechanics check is manual/observational.

## Out of scope

- OpenCode owning scoring authority (our loop always re-scores).
- Replacing the outer optimizer, blackboard, objective, or reship.
- MCP servers, multi-repo, or agent-run experiments beyond `score_candidate`/`get_trace`.
- The uncommitted "full traces in-prompt (capped)" change — superseded by `get_trace`; shelve.

## Open questions

- `opencode-agent-sdk` (Python) vs raw HTTP against `opencode serve` — decide in M2.
- Persistent session per lane vs fresh session per generation (context reuse vs isolation).
- Does one combined `score_candidate` (returning `trace_summary` inline) reduce the need for a
  separate `get_trace` round-trip for most candidates?
