"""Grader-faithful tool-call parser for gemma (verbatim from the competition harness).

The Kaggle grading env builds its gemma agent with a ``KaggleGemma4ToolCallParser``
(``kaggle_evaluation/jed_attack_134815/gemma_model_server.py``) -- a subclass of the
SDK's ``Gemma4ToolCallParser`` that routes a ``{``-prefixed argument blob (JSON, incl.
the double-brace ``{{...}}`` gemma sometimes emits -- the Aug-5 "robust parsing" fix)
through ``normalize_tool_arguments`` before falling back to the plain bare-key parser.
Our local scorer must use the SAME parser so local fire matches the grader's for EVERY
shape, not only the clean single-brace tool call we currently ship (where both parsers
agree). The gpt_oss server uses the plain ``GPTOSSAgent`` with no custom parser, so only
gemma needs this.

Copied verbatim from the grader so it cannot drift; the ``_``-prefixed symbols are the
exact ones the grader imports from ``aicomp_sdk`` (private, but mirroring the grader is
the whole point -- faithfulness beats the usual "prefer public" rule here).
"""

from aicomp_sdk.agents.gemma4_agent import (
    _GEMMA4_TOOL_CALL_PATTERN,
    Gemma4ToolCallParser,
    _clean_fallback_assistant_text,
    _parse_gemma4_arguments,
)
from aicomp_sdk.agents.hf_chat_template.response_parsing import (
    normalize_parsed_response,
    normalize_tool_arguments,
)
from aicomp_sdk.agents.hf_chat_template.types import HFGenerationResponse
from aicomp_sdk.agents.types import AgentDecision, JsonObject


def _parse_gemma4_arguments_kaggle(args_blob: str) -> JsonObject:
    if args_blob.lstrip().startswith("{"):
        return normalize_tool_arguments(args_blob)
    return _parse_gemma4_arguments(args_blob)


class KaggleGemma4ToolCallParser(Gemma4ToolCallParser):
    """Verbatim copy of the grader's gemma tool-call parser (see module docstring)."""

    def parse(
        self, response: HFGenerationResponse, *, fallback_call_id: str
    ) -> AgentDecision:
        """Parse gemma output, routing ``{``-prefixed args through the normalizer."""
        tool_calls = [
            {
                "type": "function",
                "function": {
                    "name": match.group("name"),
                    "arguments": _parse_gemma4_arguments_kaggle(
                        match.group("arguments")
                    ),
                },
            }
            for match in _GEMMA4_TOOL_CALL_PATTERN.finditer(response.raw_text)
        ]

        if not tool_calls:
            return super().parse(response, fallback_call_id=fallback_call_id)

        return normalize_parsed_response(
            {
                "role": "assistant",
                "content": _clean_fallback_assistant_text(response.text),
                "tool_calls": tool_calls,
            },
            fallback_call_id=fallback_call_id,
        )
