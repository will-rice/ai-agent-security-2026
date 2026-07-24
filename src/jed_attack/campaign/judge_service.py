"""FastAPI judge service (runs on dylan): typed severity + novelty endpoints over vLLM.

Imports the shared judge contract + prompt builders from ``judge.py`` and calls the
co-located vLLM OpenAI server with the pydantic model as guided-JSON, so replies are
valid SeverityScore/NoveltyScore by construction. Launched with
``uvicorn jed_attack.campaign.judge_service:app`` (see scripts/serve_dylan_judges.sh).
"""

import logging
from typing import Any, cast

import fastapi
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from jed_attack.campaign import config
from jed_attack.campaign.judge import (
    NoveltyRequest,
    NoveltyScore,
    SeverityRequest,
    SeverityScore,
    novelty_messages,
    severity_messages,
)

_log = logging.getLogger("judge_service")
app = fastapi.FastAPI(title="JED judge service")


def _vllm_json(messages: list[dict[str, str]], schema: dict[str, Any]) -> str:
    """Call vLLM with schema-constrained JSON output; return the reply content.

    Uses the OpenAI-standard ``response_format`` json_schema (vLLM 0.25's structured
    outputs backend enforces it), so the reply is valid JSON for ``schema``. The older
    ``extra_body={"guided_json": ...}`` is NOT honored by vLLM 0.25 (returns free text).
    """
    client = OpenAI(base_url=config.VLLM_URL, api_key="vllm")
    completion = client.chat.completions.create(
        model=config.VLLM_MODEL,
        messages=cast(list[ChatCompletionMessageParam], messages),
        temperature=0.0,
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "judge", "schema": schema},
        },
    )
    return completion.choices[0].message.content or ""


@app.post("/severity")
def severity(request: SeverityRequest) -> SeverityScore:
    """Score the elicited severity of a whole submission."""
    reply = _vllm_json(
        severity_messages(request.messages, request.feedback),
        SeverityScore.model_json_schema(),
    )
    return SeverityScore.model_validate_json(reply)


@app.post("/novelty")
def novelty(request: NoveltyRequest) -> NoveltyScore:
    """Score a candidate's novelty vs a sample of the pool."""
    reply = _vllm_json(
        novelty_messages(request.candidate, request.pool_sample),
        NoveltyScore.model_json_schema(),
    )
    return NoveltyScore.model_validate_json(reply)
