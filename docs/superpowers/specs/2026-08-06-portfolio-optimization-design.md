# Fill-on-Dump Submission — Design Spec

**Date:** 2026-08-06 (rev 6 — supersedes the portfolio-objective design)
**Status:** Approved (dialogue), pending written review
**Goal:** Make the shipped submission the *materialized candidate list itself*. The proposer proposes **templates**; the model **fills them on dump** into the full URL-stamped candidate list, serialized to JSON; a **thin, fixed `attack.py`** loads that JSON and does nothing but the grade-time budget trim. Scored == shipped, by construction.

---

## 1. Why (the thread that led here)

The recurring problem: the optimizer scored a *proxy* (a few shapes, a per-gen-char rate) while a separate live-fill algorithm re-derived ~2000 candidates at grade time. Two code paths for "shape → candidates" that could drift, and an objective in units nobody ships. The fix is to make **one** definition of "template → candidate list," run it **at dump**, ship the result verbatim, and score that same result.

Confirmed constraints (verified in code, see §6):
- The shipped `attack.py` runs isolated on Kaggle: import roots ⊆ `{aicomp_sdk, stdlib}`, single self-contained file. So it can carry **data** (a JSON string) but not our model or fill code.
- The grade-time budget is real and unknowable offline (green↔T4 latency gap), so a self-probe **trim** stays in `attack.py`.

`json` is stdlib, so a JSON candidate list embedded as a string literal satisfies isolation. That is the whole trick.

---

## 2. The pipeline

1. **Propose.** The proposer authors a `Submission` of distinct **templates** (message-shapes with a URL + `SECRET_MARKER`). Code fills the URLs — the LLM never enumerates them.
2. **Fill on dump.** `Submission.candidate_chains(cap)` deterministically stamps a unique host per candidate, round-robining across the templates, producing an **ordered** candidate list up to `cap`. `Submission.to_shipped_json(cap)` serializes it.
3. **Score.** `score_submission` replays each **distinct template once** (fire, severity, gen-chars per model), then **projects the board** by walking the same ordered sequence and trimming to a deterministic per-model gen-char budget. Exact (URL-variants are homogeneous) and cheap (k replays, not N).
4. **Champion.** Best projected board wins (`best_objective`).
5. **Ship.** `assemble.build` writes a **fixed** `attack.py` skeleton with the champion's candidate JSON embedded. At grade time `run()` `json.loads` the list, probes each candidate, keeps firing ones, and stops at the real replay budget (`_should_stop`). Returns what fits.

One ordered candidate sequence; the scorer trims it to a green-char budget (predict), the artifact trims it to the real T4 budget (enforce). Same sequence, same trim shape — scored predicts shipped.

---

## 3. What each component becomes

| Component | Today | After |
|---|---|---|
| Fill/stamp logic | inside `assemble.py`'s `_TEMPLATE` (shipped) | `campaign/fill.py` (local; used by the model + scorer) |
| `Submission` model | list of messages | templates + `candidate_chains(cap)` / `to_shipped_json(cap)` — **fill on dump** |
| Scorer | board of ≤30 literal messages | replay distinct templates, **project** the trimmed-fill board |
| Objective | per-gen-char rate proxy | the **projected public board** (LB points) + optional λ·diversity |
| `assemble.py` | renders templates + a live-fill/round-robin/deputy `run()` | **fixed thin skeleton**: embed JSON, `json.loads`, probe/trim, return |
| Ship input | `[m.text for m in champion.messages]` | `champion` → `to_shipped_json()` |

The `_TEMPLATES`/`_POOL`/`_GPT_TEMPLATE_ORDER`/round-robin/deputy machinery in `assemble.py` is deleted. The reusable primitives (`_host`, `_url`, `_message`/render, `_templatize`, `_probe_chain`, `_should_stop`) survive — the stamping ones move to `fill.py`; the probe/trim ones stay in the skeleton.

---

## 4. The objective

`score(submission) = projected_public_board(submission) [+ PORTFOLIO_LAMBDA · diversity]`

- **`projected_public_board`**: per model `m`, walk the ordered candidate sequence, accumulate each candidate's template gen-chars until a per-model char budget `FILL_BUDGET_CHARS[m]`, capped at the candidate cap; `board_m = fired_candidates × (severity + NOVELTY_PER_CELL)/200`, capped at 1000; `public = mean_m board_m`. This is the board the filled+trimmed submission scores, in LB points — not a rate. Deterministic (gen-chars, no wall-clock noise).
- **`FILL_BUDGET_CHARS[m]`**: one calibrated per-model constant — the gen-char budget that reproduces today's ~candidate count on the current champion. Below the cap the ranking matches the old per-gen-char rate; at the cap it **saturates** (correctly telling the search "you're candidate-bound, diversify — don't chase leanness past the cap").
- **`diversity`** (optional, `PORTFOLIO_LAMBDA` default 0): distinct firing templates / cap — a blind private-board hedge. Off by default; env-activated.

Legacy break: scheme tag bumps to `v7`, retiring the v6 pool.

---

## 5. Why not author the full list / skip the trim

- **Author 2000 candidates?** No — they differ only by URL (mechanical) and can't be scored (2000 replays/proposal). The proposer authors templates; code fills.
- **Ship a static list, no trim?** No — the T4 per-candidate latency is unknown offline; a static count is an `INVALID`-if-over gamble. The trim self-calibrates to the real budget, making over-budget structurally impossible. Fill materializes generously (to `cap`); the trim removes the tail.
- **Inline a shared `_fill.py` into `attack.py`?** Unnecessary now — fill runs at dump, so the artifact carries *data*, not fill logic. Its only logic is the generic probe/trim, which is submission-agnostic and fixed.

---

## 6. Verified facts (checked in code)

- Isolation is enforced by a test that ASTs the built `attack.py` and asserts import roots ⊆ `{aicomp_sdk, stdlib}` (the old `test_submission_isolated.py`; live coverage is `test_campaign.py::test_assemble_build_writes_isolated_attack_py_from_literal_messages`).
- The kernel builder (`submit-kernel`) embeds a single file via `%%writefile`, internet off — no bundled sibling modules.
- The retired `submission/*` + `build_submission.py` parallel codebase has been deleted (it shipped nothing).
- Grade-time primitives exist and are reused: `AttackCandidate.from_messages(chain)`, `_probe_chain(env, chain, hops) -> (fired, latency, gen_chars)`, `_should_stop(...)`.

---

## 7. Open decisions

1. **`cap`** (materialized candidates in the JSON). Mirror `assemble._HARD_N_CAP` (2000). Larger = bigger JSON, more headroom for a fast T4.
2. **`FILL_BUDGET_CHARS[m]`** calibration — derive from the current champion's measured fill; tune later.
3. **Fill order** — round-robin across templates (even spread, best for the novelty-cell + private hedge). Recommended.
4. **`PORTFOLIO_LAMBDA`** default 0 (throughput only). Recommended.
