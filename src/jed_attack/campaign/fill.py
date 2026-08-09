"""Deterministic template -> candidate stamping, shared by the scorer and the dump.

LOCAL-ONLY: runs at dump/score time. The SHIPPED attack.py never imports this -- it
carries the already-stamped candidate list as JSON.

The target regexes (``_URL_RE``, ``_EMAIL_RE``, ``TARGET_RE``) live in
:mod:`jed_attack.campaign.targets` -- the single home shared with ``submission`` so
typing, validation, target counting, and templatizing can never disagree on what a
target is.
"""

from collections.abc import Sequence

from jed_attack.campaign.targets import _EMAIL_RE, _URL_RE

MARKER = "SECRET_MARKER"
_ALPHA = "abcdefghijklmnopqrstuvwxyz"


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


def addr(index: int) -> str:
    """The unique clean deputy recipient address for a candidate index."""
    return f"user@{host(index)}.invalid"


def render(template: str, index: int) -> str:
    """Fill a ``{u}``/``{m}``/``{a}`` template into candidate ``index``'s message."""
    return template.format(
        m=MARKER, u=url(index), url=url(index), a=addr(index)
    ).strip()


def templatize(text: str) -> str | None:
    """Concrete message -> fill template; None if nothing varies.

    Three shapes, distinguished by content:
    - marker-exfil (has SECRET_MARKER): ``{m}`` for the marker and ``{u}`` for the URL.
    - read-exfil (no marker, has an http URL): ``{u}`` for the URL (reads secret.txt and
      posts its contents; the secret comes from the file, not the text).
    - deputy (no marker, no URL, has a recipient address): ``{a}`` for the address.

    Braces are escaped first, so ``render``'s ``str.format`` treats any literal braces
    in the text as data, not fields -- a body containing ``{...}`` cannot crash the
    fill.
    """
    escaped = text.replace("{", "{{").replace("}", "}}")
    if MARKER in escaped:
        escaped = escaped.replace(MARKER, "{m}")
        templated, replaced = _URL_RE.subn("{u}", escaped, count=1)
        return templated if replaced else None
    # No marker: read-exfil names an http URL; deputy names an email address.
    templated, replaced = _URL_RE.subn("{u}", escaped, count=1)
    if replaced:
        return templated
    templated, replaced = _EMAIL_RE.subn("{a}", escaped, count=1)
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
