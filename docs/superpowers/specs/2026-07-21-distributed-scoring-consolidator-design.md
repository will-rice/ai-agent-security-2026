# Distributed scoring + consolidator — design

Date: 2026-07-21
Status: approved (pre-implementation)

## Problem

GPU scoring is the campaign's bottleneck, and the write path is contended:

1. **Throughput ceiling.** Both target models are served by exactly two `llama-server`
   instances on green (gpt_oss:8080, gemma:8081). Every optimizer worker and the
   `score_daemon` contend for those two servers, so search throughput (candidates scored
   per wall-second) is capped by one host's GPUs. green ↔ dylan is a 0.19 ms LAN link
   (green 192.168.1.235, dylan 192.168.1.220, same /24), and dylan has two idle 24 GB
   cards (RTX 3090 + TITAN RTX) — a full second copy of both models fits.
2. **Contended, polluted write path.** Workers do a `fcntl`-locked read-modify-write into
   the shared `archive.jsonl` on every scored candidate. Proposer refusals ("I cannot
   generate adversarial prompts…") and duplicate templates get recorded, and identical
   templates with identical gates never Pareto-dominate each other, so duplicates
   accumulate. The `score_daemon` also replays the whole composed pool every cycle purely
   to produce `public_lb` for monitoring — a number we can now compute for free.

## Objective

- **~2× scoring throughput**: dylan serves both models; green's workers distribute replay
  calls across all four servers (green×2 + dylan×2), with failover so a dead dylan never
  stalls scoring.
- **A clean map-reduce write path**: lock-free sharded appends (map) + one **consolidator**
  (reduce) that is the sole writer of the canonical archive and centralizes
  refusal-filtering, template dedup, the Pareto merge, and `total_score_est`.

## Architecture (Approach B: distributed endpoints + consolidation)

dylan is **serving-only** — no repo, no workers, no shard sync. Green owns all workers,
the consolidator, and the composer. Two orthogonal changes: distribute the *scoring calls*
across hosts, and restructure the *write path* through a consolidator.

### 1. dylan serving

Stand up the same stack green runs, LAN-reachable:

- Build `llama.cpp` with CUDA on dylan (its own build — arch differs: 3090 = sm_86,
  TITAN RTX = sm_75, vs green's A100 sm_80).
- Transfer the two GGUF files (`gpt-oss-20b-Q4_K_M.gguf` ~12 GB,
  `gemma-4-26B-A4B-it-UD-Q4_K_M.gguf` ~15 GB) to dylan (rsync over the LAN).
- Serve, one model per card, **bound to `0.0.0.0`** so green can reach it:
  `llama-server -m <gguf> -ngl 999 -c <ctx> -np <slots> --jinja --host 0.0.0.0 --port
  {8080,8081}`, gpt_oss on the 3090, gemma on the TITAN RTX. Greedy + seed identical to
  green. `-np`/`-c` tuned **down** for 24 GB (gemma-26b + 16k-ctx KV at `-np 8` is likely
  too large; start `-np 4 -c 8192` and raise if VRAM allows).
- Ensure green→dylan:8080/8081 is open (dylan's firewall blocked TCP/22 from green in
  probing even though ICMP passed — the serve step must confirm the model ports are
  reachable from green, adjusting ufw/iptables if needed).

### 2. Distributed endpoint scoring (green code)

- New `resolve_endpoints(model_key) -> list[str]` in `harness/models.py`: reads
  `GPT_OSS_BASE_URLS` / `GEMMA_BASE_URLS` (comma-separated), else falls back to the single
  `resolve_base_url` default. green's `.env` sets, e.g.,
  `GPT_OSS_BASE_URLS=http://localhost:8080/v1,http://192.168.1.220:8080/v1`.
  `resolve_base_url` stays (returns endpoints[0]) for single-endpoint callers.
- `prompt_opt.score_prompt` distributes its concurrent `models × trials × guardrails`
  replay tasks across each model's endpoints **round-robin by task index**, so both hosts'
  servers fill. Build one agent factory per `(model, endpoint)` up front and select by
  index in the `cell` closure.
- **Per-endpoint failover.** Replays are idempotent (greedy-deterministic), so on a
  connection failure to the assigned endpoint the task re-dispatches to the next endpoint
  in the model's list (localhost is the last-resort tail). A `score.py` replay path used
  by the on-demand ground-truth scorer distributes the same way.

### 3. Sharded write path (map) — race-free per-entry files

- Each worker writes its scored candidate as **its own file**
  `run/shards/<worker_id>-<counter>.json` (`JED_WORKER_ID` env, else pid; counter is a
  per-worker monotonic int). The write is **temp-file + `os.replace`** into the shards dir,
  so the file appears atomically and complete — no append, no lock, no partial-read or
  lost-write race (a candidate file is either fully there or not there at all). This is
  the "make the bug structurally impossible" choice over append-shards + rename.
- `record_message` splits: the worker still writes its **knowledge note** and now writes a
  **shard file** instead of doing the locked `archive.insert`. The canonical archive is
  built only by the consolidator. (Its CLI caller `prompt_opt.main` writes a shard too.)
- New `shards.py`: `write(entry, shards_dir, worker_id)` (atomic per-entry file) and
  `claim(shards_dir) -> list[tuple[Path, Entry]]` (glob + parse, skipping any half-written
  temp files; the consolidator deletes each path after merging).

**Notes stay shared and instant.** The `knowledge` log is untouched. Every worker writes
its tries under the shared `"prompt_opt"` producer (all appending to one
`prompt_opt.jsonl` — small, atomic per-line appends, lock-free), and each worker's
`_feedback_digest` reads that file (`read_notes` globs + merges every `*.jsonl`,
filtering `producer == "prompt_opt"`). So all workers still see *all* workers' recent
tries in real time — the cross-worker learning that keeps them from re-proposing the same
template is fully preserved (and complements the consolidator's template dedup). The only
thing that gains a small lag is the **archive incumbent** shown to the proposer
(`archive_incumbent` reads the canonical archive, now refreshed on the consolidation cycle
rather than instantly) — harmless, since a propose+score generation takes far longer than
`CONSOLIDATE_INTERVAL_S`.

(Optional, plan-level: give each worker its own note producer id and widen the digest
filter to the `prompt_opt` prefix, mirroring the per-worker shard keying. Not required —
the shared-file appends are already safe — but it makes notes and shards symmetric.)

### 4. Consolidator (reduce) — sole canonical-archive writer

A new `consolidator.py --loop` process (supervised by the watchdog), every
`CONSOLIDATE_INTERVAL_S` (~15 s):

1. **Claim** all `run/shards/*.json` via `shards.claim` (glob + parse; half-written temp
   files are skipped by construction — see §3). Hold each claimed path for deletion after
   the merge.
2. **Filter**: drop entries whose template does not render (refusals / no
   `{urls}`/`{addrs}` placeholder / invariant violations) via the existing
   `prompt_opt.render` / `render_deputy` validity check.
3. **Dedup by template**: group surviving entries by template string; keep one per
   template (they are ~identical under greedy determinism). This fixes the duplicate
   accumulation the per-worker insert allowed (identical gates never dominate).
4. **Pareto merge** each survivor into the canonical `archive.jsonl` via `archive.insert`
   (unchanged Pareto/hop-dominance logic). The consolidator is the sole writer, so the
   `fcntl` lock is now a no-op; `insert`'s atomic temp-file + `os.replace` already gives
   the composer a consistent read. The lock is left in place (no change to `insert`).
5. **Write `total_score_est`** = `compose.predicted_public_score(ARCHIVE_FILE)` to its own
   `run/consolidator_status.json` (`{ts, total_score_est, archive_size, shards_consumed}`)
   — NOT `score.json`, which stays owned by the on-demand ground-truth scorer so the
   estimate and the real replay never clobber each other.
6. Delete the claimed shard files.

Consolidation is **idempotent**: a candidate is a whole file consumed once; a crash before
step 6 just re-claims the same files next cycle, and template dedup + Pareto insert make a
re-merge a no-op.

### 5. Reads unchanged; score_daemon → on-demand

`compose` + `assemble_daemon` read the canonical `archive.jsonl` exactly as today. The
continuous `score_daemon` is **removed from watchdog supervision**; its replay-based
ground-truth `public_lb` runs **on-demand before a ship** (and in the T4 verifier). The
per-cycle monitoring number is now the consolidator's free `total_score_est`.

## Data flow

```
green workers: propose → distribute-score across {green×2, dylan×2} endpoints
             → write Entry file to run/shards/<worker>-<n>.json  (+ knowledge note)
consolidator : claim shard files → filter refusals → dedup by template → Pareto-merge
             → canonical archive.jsonl → write total_score_est to consolidator_status.json
compose/assemble: read canonical archive → build_next/attack.py
score.py (ground truth): on-demand before ship, endpoint-distributed replay → score.json
```

## Score identity across hosts (gating verification)

dylan's 3090/TITAN vs green's A100 run different CUDA kernels; FP non-associativity could
shift greedy token outputs. Severity depends on *which predicates fire* (did it emit K
clean `http.post` with `SECRET_MARKER`), which is robust to minor token differences — but
this must be **verified, not assumed**:

- A `verify_score_identity` check replays a fixed set of N known candidates on green and on
  dylan for both models and asserts equal (or within-tolerance) severity + score-cell.
- **Gate**: if scores match, enable dylan in the endpoint lists. If they diverge
  materially, keep dylan **out** of scoring (green-only) — the consolidation refactor
  (Phase 1) still stands and delivers value without dylan.

## Testing

- `resolve_endpoints`: parses `*_BASE_URLS`, falls back to the localhost default, and
  `resolve_base_url` returns the first endpoint.
- endpoint distribution: `score_prompt` with a monkeypatched two-endpoint list dispatches
  tasks across both; a failing endpoint fails over to the live one (no zero-scores).
- shards: `write` produces one parseable per-entry file (atomic — no `.tmp` left behind);
  `claim` returns all complete entries and skips a half-written temp file.
- consolidator: shard files with a refusal + a duplicate + two non-dominated entries →
  canonical archive has the deduped non-dominated set, the refusal dropped, the shard
  files deleted, and `consolidator_status.json` carries `total_score_est`; re-running with
  the same entries re-claimed is idempotent (no archive change).
- record_message: writes the note + the shard line, and does NOT touch the canonical
  archive.
- No live-model mocks — reuse the local-served-model + rendered-message patterns; the
  cross-host identity check is a runtime step, not a unit test.

## Build phases

- **Phase 1 — consolidation on green alone.** `shards.py`, `consolidator.py`,
  `record_message` split, watchdog swaps score_daemon→consolidator, refusal-filter + dedup.
  Fully testable single-host; delivers the coordination cleanup + score_daemon reclaim
  before dylan exists.
- **Phase 2 — dylan scale-out.** `resolve_endpoints` + distributed `score_prompt`/`score`,
  dylan serving setup (build + model transfer + serve + firewall), the score-identity
  verification gate, and green's `.env`/launch env for the endpoint lists.

## Open items / risks

- **green→dylan port reachability** (8080/8081) — confirm/open in the serve step.
- **green→dylan SSH for the model rsync** — TCP/22 read closed in probing; if SSH is
  blocked, transfer the GGUFs laptop-mediated instead of green→dylan direct.
- **24 GB VRAM tuning** — `-np`/`-c` on the 3090/TITAN; gemma-26b is the tight one.
- **Score identity** — the Phase 2 gate above; green-only fallback if it fails.
- **Shard growth** — per-entry files are deleted each consolidation cycle; a crashed
  consolidator lets them accumulate until the watchdog restarts it (acceptable — they're
  small and re-claimed on restart).

## Out of scope / deferred

- Workers on dylan + cross-host shard sync (Approach A) — B keeps dylan serving-only.
- A third GPU host / autoscaling the endpoint list — the design is N-endpoint-general, but
  only green + dylan are wired now.
- Load-aware endpoint selection (least-outstanding) — round-robin + failover first;
  revisit if one host's cards are much slower and skew the batch.
