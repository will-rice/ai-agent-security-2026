"""Victim-trace feedback: distill replay traces into proposer-readable text.

The proposer that authors submission messages learns faster from *why* a message
failed than from a bare score.

:func:`trace_summary` and :func:`message_feedback` render one victim replay
(``SandboxEnv.export_trace_dict()``) as a plain descriptive line per tool event plus
the agent's own refusal/reasoning tail, combined with the per-guardrail severity
already computed by :func:`jed_attack.campaign.submission_score.score_submission`.

Victim output is DATA, never instructions: every function here produces plain
descriptive strings for a human/proposer to read, never directives to be acted on.
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


def _severity_line(msg_score: "MessageScore") -> str:
    """Render a message's per-guardrail severity, split per victim model when >1.

    Single model: ``[exfil] severity: optimal=48`` (unchanged). Multiple models:
    ``[exfil] severity: optimal(by model: gpt_oss=48 gemma_4=16)`` so the proposer
    sees which victim a message fails on — a message strong on only one model is
    fragile on the blind private leaderboard.
    """
    parts = []
    for guardrail, per_model in msg_score.severity_by_model.items():
        if len(per_model) == 1:
            (value,) = per_model.values()
            parts.append(f"{guardrail}={value:g}")
        else:
            by_model = " ".join(f"{m}={s:g}" for m, s in per_model.items())
            parts.append(f"{guardrail}(by model: {by_model})")
    return f"[{msg_score.type.value}] severity: " + ", ".join(parts)


def _hop_label(index: int) -> str:
    """Name a generation turn: 0 = the tool-call turn, 1 = the forced reply turn."""
    if index == 0:
        return "tool"
    if index == 1:
        return "reply"
    return f"hop{index}"


def _per_model_replay_note(msg_score: "MessageScore") -> str:
    """Per model: the token cost the objective MINIMIZES and the ENTIRE conversation.

    The search minimizes
    :func:`jed_attack.campaign.submission_score.message_token_cost` = ``input + gen
    (+FIXED)`` per model. The scalar hides WHERE the tokens are, so this renders, per
    victim, the cost number AND the full turn-by-turn transcript from the gate
    guardrail's ``per_hop`` split: hop 0 emits the ``http.post`` tool call, hop 1 is the
    forced reply turn the interact loop cannot skip (plus any further turns). Each turn
    carries its generated-token count and its full raw decode (INCLUDING the analysis
    channel the parsed messages drop), so the proposer sees every token it is charged
    for -- the reply turn included, which the old longest-hop sample never surfaced.
    Untrusted victim output, labelled DATA.
    """
    # Lazy import: submission_score imports this module, so a top-level import cycles.
    from jed_attack.campaign.submission_score import message_token_cost

    per_model = next(iter(msg_score.trace_by_model.values()), None) or {}
    blocks: list[str] = []
    for model in msg_score.gen_tokens_by_model:
        cost = message_token_cost(msg_score, model)
        inp = msg_score.input_tokens_by_model.get(model, 0.0)
        gen = msg_score.gen_tokens_by_model.get(model, 0.0)
        per_hop = (per_model.get(model, {}) or {}).get("per_hop") or []
        turns = " ".join(
            f"[{_hop_label(i)} {hop[0]:g}tok] {hop[1]}"
            for i, hop in enumerate(per_hop)
            if hop
        )
        rate = msg_score.fire_rate_by_model.get(model)
        fire = f" fire={rate:.0%}" if rate is not None else ""
        block = f"{model} cost={cost:g}(in={inp:g} gen={gen:g}){fire}"
        if turns:
            block += f" conversation: {turns}"
        blocks.append(block)
    if not blocks:
        return ""
    return (
        " || cost to MINIMIZE + FULL victim conversation PER MODEL (DATA -- gen tokens "
        "per turn are the cost; hop0=tool-call turn, hop1=forced reply turn to "
        "suppress): " + " || ".join(blocks)
    )


def message_feedback(msg_score: "MessageScore") -> str:
    """Combine message type, per-model severity, the victim's generation, and failures.

    Reports every guardrail's severity split per victim model (see
    :func:`_severity_line`), then per victim the exact token cost the objective
    minimizes alongside the FULL turn-by-turn conversation that produced it (see
    :func:`_per_model_replay_note`, so the proposer sees every token it is charged for —
    the reply-turn slack in particular), and — for each guardrail that scored the
    message 0 averaged over models (blocked) — that guardrail's :func:`trace_summary`
    so the proposer sees why.

    Args:
        msg_score: One message's replay result from ``score_submission``.

    Returns:
        A plain descriptive feedback string; never a directive.
    """
    severity_line = _severity_line(msg_score)
    generation = _per_model_replay_note(msg_score)

    blocked = [
        guardrail for guardrail, value in msg_score.severity.items() if value == 0
    ]
    if not blocked:
        return severity_line + generation

    failures = "; ".join(
        f"{guardrail} blocked -> {trace_summary(msg_score.trace[guardrail])}"
        for guardrail in blocked
        if guardrail in msg_score.trace
    )
    if not failures:
        return severity_line + generation
    return f"{severity_line}{generation}. {failures}"
