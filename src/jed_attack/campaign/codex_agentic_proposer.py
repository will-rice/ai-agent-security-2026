"""Agentic proposer lane: codex `exec` with tool access -- the SOLE proposer.

This lane runs the codex CLI agentically (``codex exec``) with the repo and the real
scorer (``src/jed_attack/scripts/score_probe.sh``) as instruments: it READS the SDK
source to understand the victims, GRADES constructions through the real scorer (the same
objective the optimizer minimizes), keeps only firing/lean ones, and only then emits the
batch. :func:`optimize_prompts._propose_batch_oneshot` calls it directly -- the
FunSearch islands, scoring, and archive downstream are unchanged.

Output handoff: the agent writes the final ``SubmissionBatch`` JSON to a unique file
(the agent's stdout carries its reasoning/tool-calls, not the answer); we validate that
file with ``SubmissionBatch.model_validate_json`` so an invalid batch drops WHOLE.
"""

import asyncio
import itertools
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

# Hard wall-clock cap for one agent session (investigate + probe + author). Agentic
# sessions are minutes, not the seconds a streaming call takes, so this lane ignores the
# streaming idle_timeout and uses its own generous budget.
AGENTIC_TIMEOUT_S = float(os.getenv("JED_AGENTIC_TIMEOUT_S", "1200"))
# Pin each concurrent worker's scorer probes to a DEDICATED GPU (round-robin), so N
# workers don't pile their model loads onto one device. Each session masks to its GPU
# via CUDA_VISIBLE_DEVICES; score_probe loads ONE victim (index-0) at a time there.
_NUM_GPUS = int(os.getenv("JED_NUM_GPUS", "2"))
_GPU_CYCLE = itertools.cycle(range(_NUM_GPUS))
_REPO = Path(__file__).resolve().parents[3]  # src/jed_attack/campaign/ -> repo root
_OUT_DIR = _REPO / "run" / "agentic_batches"
# The committed, versioned mission prompts (domain facts + instruments the agent gets on
# top of the optimizer prompt). Which one loads is selected by JED_MISSION_PATH so the
# same code runs both the exfil optimizer (default) and the deputy hedge -- no
# scratchpad file-swap. Default is the packaged exfil mission next to this module.
_MISSIONS_DIR = Path(__file__).resolve().parent / "missions"
_MISSION_PATH = Path(os.getenv("JED_MISSION_PATH", str(_MISSIONS_DIR / "exfil.md")))


def _mission(prompt: str, out_file: Path) -> str:
    """Glue only: the (hot-reloaded) mission .md carries the agentic job + instrument.

    Everything static -- the agentic framing, the score_probe instrument, the process
    -- lives in the mission file (``_MISSION_PATH``) so it is versioned and hot-reloads.
    The only thing that MUST be code is the per-session output handoff (``out_file`` is
    a fresh path each run) and appending the optimizer's own prompt (schema + history).
    """
    extra = _MISSION_PATH.read_text(encoding="utf-8") if _MISSION_PATH.exists() else ""
    return (
        f"{extra}\n\n"
        f"OUTPUT: write ONLY the final valid SubmissionBatch JSON to `{out_file}` "
        '(an object {"diagnoses":[...],"submissions":[{"gpt_oss":[...],'
        '"gemma_4":[...]},...]}) that matches the JSON SCHEMA below EXACTLY. Nothing '
        "else you do matters except that this file holds a schema-valid batch of "
        "DISTINCT, scorer-graded firing shapes.\n\n"
        "=== OPTIMIZER PROMPT (context + schema) ===\n"
        f"{prompt}"
    )


async def propose_batch_codex_agentic(
    prompt: str, provider: providers.Provider, idle_timeout_s: float
) -> tuple[list[Submission], list[str], str]:
    """Author a batch by running codex agentically against the scorer + source.

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
    # Pin this session's scorer probes to a dedicated GPU: mask to it with
    # CUDA_VISIBLE_DEVICES (so it is the sole, index-0 device); score_probe loads one
    # victim at a time there (JED_GPU_*=0). Round-robin so concurrent workers land on
    # different GPUs. The optimizer's own resident scoring models sit on the other GPU
    # (run_optimizer.sh pins them), so the two don't fight for VRAM.
    gpu = next(_GPU_CYCLE)
    worker_env = os.environ.copy()
    worker_env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    worker_env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    worker_env["JED_GPU_GPT"] = "0"
    worker_env["JED_GPU_GEMMA"] = "0"
    _log.info(
        "agentic proposer (%s): session %s on GPU %d -> watch %s",
        provider.model,
        sid,
        gpu,
        log_file,
    )
    # YOLO: --dangerously-bypass-approvals-and-sandbox so codex's scorer probes reach
    # the GPU (the workspace-write bubblewrap sandbox blocks /dev/nvidia*). We are the
    # external sandbox (our own box, our own repo). --search (top-level, before `exec`)
    # enables the native web_search tool so the agent can research external technique,
    # not just the repo. Stream stdout+stderr to log_file so the session is observable.
    # start_new_session=True puts codex in its OWN session/process group: without it
    # codex shares the optimizer's group + controlling terminal, and a SIGINT/SIGTERM
    # that codex (or a tool it spawns, or its tty cleanup) generates propagates to the
    # optimizer, tripping its handler -> "team cancelled" shutdown. Isolation severs it.
    with log_file.open("wb") as logf:
        proc = await asyncio.create_subprocess_exec(
            "codex",
            "--search",
            "exec",
            "--dangerously-bypass-approvals-and-sandbox",
            "--skip-git-repo-check",
            "-c",
            f'model="{provider.model}"',
            "-",
            stdin=asyncio.subprocess.PIPE,
            stdout=logf,
            stderr=asyncio.subprocess.STDOUT,
            cwd=str(_REPO),
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
            # tree -- proc.kill() alone would orphan its score_probe model-load children
            # and leak GPU memory.
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
        "agentic proposer (%s): %d scorer-graded submissions",
        provider.model,
        len(batch.submissions),
    )
    return batch.submissions, batch.diagnoses, ""
