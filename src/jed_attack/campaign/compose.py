"""Submission composer: packs the Pareto archive into the shipped message pool.

Replaces the exfil-only ship path. The archive (see :mod:`archive`) holds every
non-dominated scored message over the ``{optimal, rules, hardened}`` gate vector, plus
the pinned proven exfil template (the one candidate with a real scored LB result).
:func:`compose_pool` reserves a public floor of rendered exfil copies (so the visible LB
still reflects the proven template), then greedily fills the rest of the green-seconds
budget with copies of whichever non-pinned archive entry has the highest **surviving
robust weight per green-second** — the maximin bet that lifts whichever guardrail turns
out to be the private strict one. :func:`build` writes that pool into ``attack.py``
using the exact same writer :mod:`assemble` uses for its gate-adopted pool, so the
composed submission stays isolated (``aicomp_sdk`` + stdlib only) with no duplicated
template.
"""

import json
import logging
from collections.abc import Callable
from pathlib import Path

from jed_attack.campaign import archive, assemble, budget, config, prompt_opt

_log = logging.getLogger("compose")

# Consecutive render(None) misses before an entry is abandoned (an invariant-broken
# template that can never render should not spin the fill loop forever).
_MAX_RENDER_MISSES = 8


def _render(entry: archive.Entry, index: int) -> str | None:
    """Render one concrete copy of an archive entry via its placeholder's renderer.

    Dispatches on the template's placeholder: ``{urls}`` is an EXFIL template
    (:func:`prompt_opt.render`); ``{addrs}`` is a DEPUTY template
    (:func:`prompt_opt.render_deputy`).

    Args:
        entry: The archive entry to render; its ``hops`` sets the render fan-out
            (posts or emails).
        index: A globally unique index across the whole composed pool, so rendered
            domains/addresses never collide.

    Returns:
        The rendered message, or None if it fails a renderer invariant.

    Raises:
        ValueError: The template carries neither a ``{urls}`` nor an ``{addrs}``
            placeholder.
    """
    if "{urls}" in entry.template:
        return prompt_opt.render(entry.template, index, entry.hops)
    if "{addrs}" in entry.template:
        return prompt_opt.render_deputy(entry.template, index, entry.hops)
    raise ValueError(f"unknown placeholder in template: {entry.template!r}")


def _add_copies(
    entry: archive.Entry,
    pool_items: list[tuple[archive.Entry, int]],
    messages: list[str],
    cursor: list[int],
    should_add: Callable[[], bool],
) -> None:
    """Append rendered copies of ``entry`` while ``should_add`` allows more.

    Args:
        entry: The archive entry to render repeatedly.
        pool_items: The running (entry, count) pool; appended to in place.
        messages: The running rendered-message list; appended to in place.
        cursor: A single-element mutable high-water mark of the next free render
            index. Each copy claims the block ``[index*hops, index*hops+hops)`` at
            ``index = ceil(cursor / hops)`` and advances the cursor past it, so
            rendered domains/addresses never collide even across entries with
            differing ``hops``.
        should_add: Zero-arg predicate re-checked before every copy; return True to
            add one more rendered copy of ``entry``.
    """
    misses = 0
    while should_add() and misses < _MAX_RENDER_MISSES:
        index = -(-cursor[0] // entry.hops)  # ceil(cursor / hops)
        rendered = _render(entry, index)
        if rendered is None:
            misses += 1
            continue
        messages.append(rendered)
        pool_items.append((entry, 1))
        cursor[0] = index * entry.hops + entry.hops
        misses = 0


def _robust_weight(entry: archive.Entry) -> float:
    """Surviving robust weight per green-second: min(rules, hardened) / cost_s.

    Args:
        entry: A non-pinned archive entry with ``cost_s > 0``.

    Returns:
        The ranking key (higher is better).
    """
    survives = min(entry.gates.get("rules", 0.0), entry.gates.get("hardened", 0.0))
    return survives / entry.cost_s


def compose_pool(archive_path: Path) -> list[str]:
    """Pack the Pareto archive into the ship pool under the green-seconds budget.

    Reserves the pinned exfil entry, rendered repeatedly, until its summed ``cost_s``
    covers ``config.PUBLIC_FLOOR_COST_S`` (the public-LB floor for the promoted
    submission). Then ranks every non-pinned entry by surviving robust weight per
    green-second (``min(gates["rules"], gates["hardened"]) / cost_s``, highest first)
    and greedily renders copies of the top entry while :func:`budget.fits` still admits
    one more, moving to the next-ranked entry once it doesn't.

    Args:
        archive_path: The Pareto archive jsonl file (typically ``config.ARCHIVE_FILE``).

    Returns:
        The flat list of rendered ship messages (public floor first, then the
        budget-filling robust entries).
    """
    entries = archive.read(archive_path)
    pinned = [e for e in entries if e.pinned]
    ranked = sorted(
        (e for e in entries if not e.pinned and e.cost_s > 0),
        key=_robust_weight,
        reverse=True,
    )

    pool_items: list[tuple[archive.Entry, int]] = []
    messages: list[str] = []
    cursor = [0]

    if pinned and pinned[0].cost_s > 0:
        exfil_entry = pinned[0]
        _add_copies(
            exfil_entry,
            pool_items,
            messages,
            cursor,
            lambda: (
                sum(count * e.cost_s for e, count in pool_items if e is exfil_entry)
                < config.PUBLIC_FLOOR_COST_S
                and budget.fits([*pool_items, (exfil_entry, 1)])
            ),
        )

    for entry in ranked:
        _add_copies(
            entry,
            pool_items,
            messages,
            cursor,
            lambda entry=entry: budget.fits([*pool_items, (entry, 1)]),
        )

    _log.info("composed %d ship message(s) from %s", len(messages), archive_path)
    return messages


def build(out_dir: Path) -> Path:
    """Write the composed submission's ``attack.py``.

    Reuses :mod:`assemble`'s writer (``_TEMPLATE`` / ``_render_pool``) — the same
    isolated-attack.py template the gate-adopted pool uses — so the composed submission
    carries no duplicated template and stays ``aicomp_sdk`` + stdlib isolated.

    Args:
        out_dir: Output directory for the built submission (typically
            ``config.BUILD_NEXT_DIR``).

    Returns:
        The path to the written ``attack.py``.
    """
    messages = compose_pool(config.ARCHIVE_FILE)
    pairs: list[tuple[tuple[str, ...], str]] = [
        ((message,), "composed") for message in messages
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    source = assemble._TEMPLATE.format(
        pool=assemble._render_pool(pairs),
        max_candidates=len(pairs),
    )
    attack_path = out_dir / "attack.py"
    attack_path.write_text(source, encoding="utf-8")
    status = {"candidate_count": len(pairs), "source": "composed"}
    (out_dir / "build_next_status.json").write_text(
        json.dumps(status, indent=2, sort_keys=True), encoding="utf-8"
    )
    _log.info(
        "built composed submission (%d candidates) -> %s", len(pairs), attack_path
    )
    return attack_path
