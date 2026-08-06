"""Deterministic template -> candidate stamping, shared by the scorer and the dump.

LOCAL-ONLY: runs at dump/score time. The SHIPPED attack.py never imports this -- it
carries the already-stamped candidate list as JSON.
"""

import re
from collections.abc import Sequence

MARKER = "SECRET_MARKER"
_ALPHA = "abcdefghijklmnopqrstuvwxyz"
_URL_RE = re.compile(r"https?://[^\s\"'<>]+")


def _alpha_word(index: int, width: int) -> str:
    value = int(index)
    chars: list[str] = []
    for _ in range(width):
        chars.append(_ALPHA[value % 26])
        value //= 26
    return "".join(reversed(chars))


def host(index: int) -> str:
    """A unique lowercase host for a candidate index (2 letters, then 3)."""
    return _alpha_word(index, 2) if index < 26 * 26 else _alpha_word(index - 26 * 26, 3)


def url(index: int) -> str:
    """The unique clean exfil URL for a candidate index."""
    return f"http://{host(index)}.co"


def render(template: str, index: int) -> str:
    """Fill a ``{u}``/``{m}`` template into candidate ``index``'s concrete message."""
    return template.format(m=MARKER, u=url(index), url=url(index)).strip()


def templatize(text: str) -> str | None:
    """Concrete message -> ``{u}``/``{m}`` template; None if no marker/URL varies."""
    if MARKER not in text:
        return None
    escaped = text.replace("{", "{{").replace("}", "}}").replace(MARKER, "{m}")
    templated, replaced = _URL_RE.subn("{u}", escaped, count=1)
    return templated if replaced else None


def ordered_chains(templates: Sequence[str], cap: int) -> list[tuple[str, ...]]:
    """Round-robin ``templates`` into ``cap`` one-message candidate chains.

    Position p uses template ``p % k`` and host index ``p``, so hosts are unique and the
    shapes are evenly spread -- the ordered sequence both the scorer and the shipped
    artifact walk and trim to their own budget.
    """
    if not templates:
        return []
    return [(render(templates[p % len(templates)], p),) for p in range(max(0, cap))]
