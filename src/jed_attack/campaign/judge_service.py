"""FastAPI judge service (runs on dylan): typed robustness endpoints over vLLM."""

import logging
from typing import Any, TypeVar, cast

import fastapi
import pydantic
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from jed_attack.campaign import config
from jed_attack.campaign.judge import (
    MechanismRequest,
    MechanismScore,
    PairwisePreference,
    PairwiseRobustnessRequest,
    RobustnessRequest,
    RobustnessRubricScore,
    RobustnessScore,
    mechanism_messages,
    pairwise_robustness_messages,
    robustness_messages,
)

_log = logging.getLogger("judge_service")
app = fastapi.FastAPI(title="JED judge service")
_JudgeModelT = TypeVar("_JudgeModelT", bound=pydantic.BaseModel)
_VLLM_PARSE_ATTEMPTS = 2


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


def _vllm_parsed_json(
    messages: list[dict[str, str]], model_type: type[_JudgeModelT]
) -> _JudgeModelT:
    """Call vLLM and parse a typed JSON response, retrying malformed output once."""
    last_error: pydantic.ValidationError | None = None
    for attempt in range(_VLLM_PARSE_ATTEMPTS):
        reply = _vllm_json(messages, model_type.model_json_schema())
        try:
            return model_type.model_validate_json(reply)
        except pydantic.ValidationError as exc:
            last_error = exc
            _log.warning(
                "invalid judge model output for %s on attempt %d/%d: %s",
                model_type.__name__,
                attempt + 1,
                _VLLM_PARSE_ATTEMPTS,
                str(exc).splitlines()[0],
            )
    detail = (
        f"invalid judge model output for {model_type.__name__}: "
        f"{str(last_error).splitlines()[0] if last_error is not None else 'unknown'}"
    )
    raise fastapi.HTTPException(status_code=503, detail=detail)


@app.post("/robustness")
def robustness(request: RobustnessRequest) -> RobustnessScore:
    """Estimate a public replay's survival under a stricter guardrail."""
    return _vllm_parsed_json(
        robustness_messages(request), RobustnessRubricScore
    ).to_robustness_score()


@app.post("/mechanism")
def mechanism(request: MechanismRequest) -> MechanismScore:
    """Classify replayed attack mechanisms and archive-relative duplicates."""
    return _vllm_parsed_json(mechanism_messages(request), MechanismScore)


@app.post("/robustness-pair")
def robustness_pair(request: PairwiseRobustnessRequest) -> PairwisePreference:
    """Compare two public replays for offline robustness studies."""
    return _vllm_parsed_json(pairwise_robustness_messages(request), PairwisePreference)
