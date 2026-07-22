"""Tier 1 victim-trace feedback: distill a replay trace into proposer-readable text.

The proposer that authors submission messages learns faster from *why* a message
failed than from a bare score. :func:`trace_summary` renders one victim replay
(``SandboxEnv.export_trace_dict()``) as a plain descriptive line per tool event plus
the agent's own refusal/reasoning tail. :func:`message_feedback` combines that with
the per-guardrail severity already computed by
:func:`jed_attack.campaign.submission_score.score_submission`, picking out the
guardrail(s) that scored the message 0 (the failure mode) to explain.

Victim output is DATA, never instructions: both functions produce plain descriptive
strings for a human/proposer to read, never directives to be acted on.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from jed_attack.campaign.submission_score import MessageScore

_DENIED_PREFIX = "denied:"
_CONFIRM_PREFIX = "confirm_required:"


def trace_summary(trace: dict[str, Any]) -> str:
    """Render one victim replay trace as a plain-text summary line.

    One segment per tool event — ``name(args…) -> ok``, ``-> DENIED: <reason>``
    (parsed from an ``error`` starting with ``"denied:"``), or ``-> CONFIRM:
    <reason>`` (parsed from ``"confirm_required:"``) — followed by the agent's
    non-``"(no_tool)"`` ``assistant_messages`` as a refusal/reasoning tail.

    Args:
        trace: A victim trace dict as returned by ``SandboxEnv.export_trace_dict()``.

    Returns:
        A single descriptive string (segments joined by ``" | "``); never a
        directive, since victim output is untrusted data.
    """
    segments = []
    for event in trace.get("tool_events", []):
        name = event.get("name", "")
        args = event.get("args", {}) or {}
        args_str = ", ".join(f"{key}={value}" for key, value in args.items())
        error = event.get("error")
        if error and error.startswith(_DENIED_PREFIX):
            reason = error[len(_DENIED_PREFIX) :].strip()
            segments.append(f"{name}({args_str}) -> DENIED: {reason}")
        elif error and error.startswith(_CONFIRM_PREFIX):
            reason = error[len(_CONFIRM_PREFIX) :].strip()
            segments.append(f"{name}({args_str}) -> CONFIRM: {reason}")
        elif event.get("ok"):
            segments.append(f"{name}({args_str}) -> ok")
        else:
            segments.append(f"{name}({args_str}) -> error: {error}")

    tail = [m for m in trace.get("assistant_messages", []) if m != "(no_tool)"]
    if tail:
        segments.append(" ".join(tail))

    return " | ".join(segments)


def message_feedback(msg_score: "MessageScore") -> str:
    """Combine per-guardrail severity with the failure-mode trace summary.

    Reports every guardrail's severity, then — for each guardrail that scored the
    message 0 (blocked outright) — appends that guardrail's own :func:`trace_summary`
    so the proposer sees why, not just that it failed.

    Args:
        msg_score: One message's replay result from ``score_submission``.

    Returns:
        A plain descriptive feedback string; never a directive.
    """
    severity_line = "severity: " + ", ".join(
        f"{guardrail}={value:g}" for guardrail, value in msg_score.severity.items()
    )

    blocked = [
        guardrail for guardrail, value in msg_score.severity.items() if value == 0
    ]
    if not blocked:
        return severity_line

    failures = "; ".join(
        f"{guardrail} blocked -> {trace_summary(msg_score.trace[guardrail])}"
        for guardrail in blocked
        if guardrail in msg_score.trace
    )
    if not failures:
        return severity_line
    return f"{severity_line}. {failures}"
