"""Proposer lane backed by the codex ChatGPT-account Responses API.

Distinct from the OpenAI-compatible ``api`` lanes. The codex backend
(``chatgpt.com/backend-api/codex``) differs from a normal OpenAI endpoint in four
ways this module has to accommodate:

1. It speaks the **Responses API** (``text.format`` structured output), not
   chat-completions -- the chat-completions ``response_format`` parameter is rejected
   (``Unsupported parameter``).
2. Auth is the **ChatGPT OAuth token** that ``codex login`` stores in
   ``~/.codex/auth.json`` (Bearer ``access_token`` + ``chatgpt-account-id`` header),
   not an API key. We re-read it every call so a token refreshed by the codex CLI is
   picked up without restarting the optimizer.
3. It **requires** ``stream=true``.
4. Its terminal ``response.completed`` event carries an EMPTY ``output`` -- so the SDK's
   ``get_final_response()``/``output_parsed`` helper recovers nothing. We instead
   accumulate the ``response.output_text.delta`` chunks and validate the whole reply
   once, exactly like :func:`optimize_prompts.propose_batch_async`.

Only the ``gpt-5.5``/``gpt-5.4`` models are usable: they comply with the red-team
proposer prompt, whereas the ``gpt-5.6`` family (sol/daybreak) is input-moderation
blocked, and the daybreak "cyber" model is not entitled on a ChatGPT plan (it silently
substitutes to sol). The strict JSON schema is derived from :class:`SubmissionBatch`
with the SDK's ``to_strict_json_schema`` -- the single source for constrained decoding
and validation, same as the chat-completions lane.
"""

import asyncio
import contextlib
import json
import logging
from pathlib import Path

import pydantic
from openai import AsyncOpenAI
from openai.lib._pydantic import to_strict_json_schema

from jed_attack.campaign import providers
from jed_attack.campaign.submission import Submission, SubmissionBatch

_log = logging.getLogger("codex_proposer")

CODEX_BASE_URL = "https://chatgpt.com/backend-api/codex"
CODEX_AUTH_PATH = Path.home() / ".codex" / "auth.json"
_STRICT_SCHEMA = to_strict_json_schema(SubmissionBatch)


def _client() -> AsyncOpenAI:
    """Build a client for the codex Responses backend from the on-disk ChatGPT token.

    Reads ``~/.codex/auth.json`` on every call so a token the codex CLI has refreshed is
    used without an optimizer restart. Raises if the file or its token is missing (the
    worker loop backs off and rotates -- a broken auth is not silently swallowed).
    """
    auth = json.loads(CODEX_AUTH_PATH.read_text(encoding="utf-8"))
    tokens = auth["tokens"]
    return AsyncOpenAI(
        base_url=CODEX_BASE_URL,
        api_key=tokens["access_token"],
        default_headers={
            "chatgpt-account-id": tokens["account_id"],
            "OpenAI-Beta": "responses=experimental",
            "originator": "codex_cli_rs",
        },
    )


async def propose_batch_codex(
    prompt: str, provider: providers.Provider, idle_timeout_s: float
) -> tuple[list[Submission], str]:
    """Author a batch on the codex Responses backend (ChatGPT-account ``gpt-5.x``).

    Mirrors :func:`optimize_prompts.propose_batch_async`: streams a single structured
    reply, accumulates raw chunks, and parses the whole accumulated content once with
    ``SubmissionBatch.model_validate_json`` -- so the model's ``@model_validator`` ship
    invariants run in that parse and any invalid message drops the batch WHOLE.
    Streaming
    uses an IDLE timeout (rescheduled per event), so a slow reply finishes and only a
    stall aborts. A backend failure (e.g. moderation ``Request blocked``) is raised so
    the lane's breakage is visible, not disguised as an empty drop.

    Args:
        prompt: The batch-proposer prompt text (the user turn).
        provider: The ``codex_responses`` lane to call (its ``model`` is the served id).
        idle_timeout_s: Max seconds to wait for the next streamed event before aborting.

    Returns:
        The parsed submissions (empty if the reply carried no valid batch) and the
        backend's reasoning summary text (empty if none).
    """
    # Lazy import breaks the optimize_prompts <-> codex_proposer cycle.
    from jed_attack.campaign.optimize_prompts import _load_prompts

    client = _client()
    stream = await client.responses.create(
        model=provider.model,
        instructions=_load_prompts()["system"],
        input=[
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": prompt}],
            }
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "submission_batch",
                "strict": True,
                "schema": _STRICT_SCHEMA,
            }
        },
        # The codex backend rejects max_output_tokens ("Unsupported parameter"); it caps
        # generation with its own truncation_policy, and the model stops when the batch
        # JSON is complete. The idle timeout is the only client-side cutoff.
        store=False,
        stream=True,
    )
    content: list[str] = []
    reasoning: list[str] = []
    loop = asyncio.get_running_loop()
    try:
        async with asyncio.timeout(idle_timeout_s) as timer:
            async for event in stream:
                timer.reschedule(loop.time() + idle_timeout_s)  # event -> reset idle
                if event.type == "response.output_text.delta":
                    content.append(event.delta)
                elif event.type == "response.reasoning_summary_text.delta":
                    reasoning.append(event.delta)
                elif event.type == "response.failed":
                    error = getattr(event.response, "error", None)
                    raise RuntimeError(f"codex response failed: {error}")
    finally:
        with contextlib.suppress(Exception):
            await stream.close()
    try:
        batch = SubmissionBatch.model_validate_json("".join(content))
    except pydantic.ValidationError as exc:
        not_json = any(e.get("type") == "json_invalid" for e in exc.errors())
        reason = "reply was not JSON (refusal/prose)" if not_json else "ship-invariants"
        _log.info("proposed batch dropped (%s): %s", provider.model, reason)
        return [], "".join(reasoning)
    _log.info(
        "proposed batch (%s): %d submissions", provider.model, len(batch.submissions)
    )
    return batch.submissions, "".join(reasoning)
