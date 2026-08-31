"""Agentic proposer lane: codex `exec` with READ-ONLY repo access -- the SOLE proposer.

This lane runs the codex CLI agentically (``codex exec``) to author a batch by READING
the SDK source (victim parsers, chat templates, scoring) and reasoning over the
optimizer's cross-round scored feedback -- the real per-model cost + severity of every
shape graded so far, fed in via the OPTIMIZER PROMPT. It does NOT run the scorer itself:
the optimizer scores every batch it returns on its own resident models and feeds the
result back next round, so the agent needs no GPU. It is called by
:func:`optimize_prompts._propose_batch_oneshot` -- islands, scoring, archive unchanged.

Sandbox: codex runs ``--sandbox workspace-write`` with its cwd set to the batch OUTPUT
dir (``_OUT_DIR``), so that dir (where it drops its answer) is the ONLY writable root --
the source tree, tests, and ``.git`` are outside the workspace and READ-ONLY. The agent
can read any source but cannot modify the codebase or ``git restore`` the working tree.

Output handoff: the agent writes the final ``SubmissionBatch`` JSON to a unique file
(the agent's stdout carries its reasoning/tool-calls, not the answer); we validate that
file with ``SubmissionBatch.model_validate_json`` so an invalid batch drops WHOLE.
"""

import asyncio
import json
import logging
import os
import signal
import uuid
from pathlib import Path

import pydantic

from jed_attack.campaign import providers
from jed_attack.campaign.submission import Submission, SubmissionBatch

_log = logging.getLogger("codex_agentic")

# Hard wall-clock cap for one agent session (investigate + author). Agentic sessions are
# minutes, not the seconds a streaming call takes, so this lane ignores the streaming
# idle_timeout and uses its own generous budget.
AGENTIC_TIMEOUT_S = float(os.getenv("JED_AGENTIC_TIMEOUT_S", "1200"))
_REPO = Path(__file__).resolve().parents[3]  # src/jed_attack/campaign/ -> repo root
_OUT_DIR = _REPO / "run" / "agentic_batches"
# The committed, versioned mission prompts (domain facts the agent gets on top of
# the optimizer prompt). Which one loads is selected by JED_MISSION_PATH so the
# same code runs both the exfil optimizer (default) and the deputy hedge -- no
# scratchpad file-swap. Default is the packaged exfil mission next to this module.
_MISSIONS_DIR = Path(__file__).resolve().parent / "missions"
_MISSION_PATH = Path(os.getenv("JED_MISSION_PATH", str(_MISSIONS_DIR / "exfil.md")))


def _mission(prompt: str, out_file: Path) -> str:
    """Glue only: the (hot-reloaded) mission .md carries the agentic job + facts.

    Everything static -- the agentic framing, the domain facts, the process --
    lives in the mission file (``_MISSION_PATH``) so it is versioned and hot-reloads.
    The only thing that MUST be code is the per-session output handoff (``out_file`` is
    a fresh path each run) and appending the optimizer's own prompt (schema + history).
    """
    extra = _MISSION_PATH.read_text(encoding="utf-8") if _MISSION_PATH.exists() else ""
    return (
        f"{extra}\n\n"
        f"REPO: the source you may READ is at `{_REPO}` (read-only). Your working "
        "directory is a scratch output dir, NOT the repo, so resolve every source path "
        f"the mission mentions against `{_REPO}` (e.g. read "
        f"`{_REPO}/src/jed_attack/campaign/submission.py`). You may write ONLY inside "
        "your working directory.\n\n"
        f"OUTPUT: write ONLY the final valid SubmissionBatch JSON to `{out_file}` "
        '(an object {"diagnoses":[...],"submissions":[{"gpt_oss":[...],'
        '"gemma_4":[...]},...]}) that matches the JSON SCHEMA below EXACTLY. Nothing '
        "else you do matters except that this file holds a schema-valid batch of "
        "DISTINCT firing shapes (the optimizer scores them for you next round).\n\n"
        "=== OPTIMIZER PROMPT (context + schema) ===\n"
        f"{prompt}"
    )


async def propose_batch_codex_agentic(
    prompt: str, provider: providers.Provider, idle_timeout_s: float
) -> tuple[list[Submission], list[str], str]:
    """Author a batch by running codex agentically against the source + feedback.

    Args:
        prompt: The rendered proposer prompt (PARENTS/OPRO/INCUMBENT/SCHEMA), same text
            the blind lanes get.
        provider: The ``codex_agentic`` lane (``provider.model`` = the codex model id,
            e.g. ``gpt-5.5``).
        idle_timeout_s: Streaming idle cap (ignored here; this lane uses a hard
            wall-clock budget of AGENTIC_TIMEOUT_S (sessions are minutes).

    Returns:
        ``(submissions, diagnoses, reasoning)`` -- empty if the agent produced
        no schema-valid batch file (dropped WHOLE, like the other lanes).
    """
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    sid = uuid.uuid4().hex[:12]
    out_file = _OUT_DIR / f"batch_{sid}.json"
    log_file = _OUT_DIR / f"session_{sid}.log"  # live, tail-able: SEE codex work
    mission = _mission(prompt, out_file)
    worker_env = os.environ.copy()
    _log.info(
        "agentic proposer (%s): session %s -> watch %s",
        provider.model,
        sid,
        log_file,
    )
    # SANDBOX: --sandbox workspace-write with cwd=_OUT_DIR makes that dir the sole
    # writable root, so codex can drop its batch JSON there but CANNOT modify the source
    # tree / tests / .git (they are outside the workspace, read-only) -- it can read any
    # source but never `git restore` the working tree out from under a live edit. The
    # agent scores nothing itself (the optimizer scores its batch), so it needs no GPU
    # and no full-access sandbox. --search (top-level, before `exec`) enables the native
    # web_search tool so the agent can research external technique. Stream stdout+stderr
    # to log_file so the session is observable. start_new_session=True puts codex in its
    # OWN session/process group: without it codex shares the optimizer's group +
    # controlling terminal, and a SIGINT/SIGTERM it (or a tool it spawns, or its tty
    # cleanup) generates propagates to the optimizer, tripping its "team cancelled"
    # shutdown. Isolation severs it.
    with log_file.open("wb") as logf:
        proc = await asyncio.create_subprocess_exec(
            "codex",
            "--search",
            "exec",
            "--sandbox",
            "workspace-write",
            "--skip-git-repo-check",
            "-c",
            f'model="{provider.model}"',
            "-",
            stdin=asyncio.subprocess.PIPE,
            stdout=logf,
            stderr=asyncio.subprocess.STDOUT,
            cwd=str(_OUT_DIR),
            env=worker_env,
            start_new_session=True,
        )
        assert (
            proc.stdin is not None
        )  # stdin=PIPE above guarantees a writer  # noqa: S101
        proc.stdin.write(mission.encode("utf-8"))
        await proc.stdin.drain()
        proc.stdin.close()
        try:
            await asyncio.wait_for(
                proc.wait(), timeout=max(idle_timeout_s, AGENTIC_TIMEOUT_S)
            )
        except (TimeoutError, asyncio.TimeoutError):
            # codex leads its own process group (start_new_session), so kill the whole
            # tree -- proc.kill() alone would orphan any child tools it spawned.
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            _log.warning(
                "agentic proposer (%s): session %s timed out", provider.model, sid
            )
            return [], [], ""
    if not out_file.exists():
        tail = (
            log_file.read_text("utf-8", "replace")[-2000:] if log_file.exists() else ""
        )
        _log.info(
            "agentic proposer (%s): session %s wrote no batch file",
            provider.model,
            sid,
        )
        return [], [], tail
    try:
        raw = out_file.read_text(encoding="utf-8")
        batch = SubmissionBatch.model_validate_json(raw)
    except (pydantic.ValidationError, json.JSONDecodeError) as exc:
        _log.info("agentic proposer (%s): batch invalid: %s", provider.model, exc)
        return [], [], ""
    _log.info(
        "agentic proposer (%s): %d proposed submissions",
        provider.model,
        len(batch.submissions),
    )
    return batch.submissions, batch.diagnoses, ""
