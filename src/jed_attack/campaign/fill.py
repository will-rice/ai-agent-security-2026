"""Deterministic template -> candidate stamping, shared by the scorer and the dump.

LOCAL-ONLY: runs at dump/score time. The SHIPPED attack.py never imports this -- it
carries the already-stamped candidate list as JSON.

The target regexes (``_URL_RE``, ``_EMAIL_RE``, ``TARGET_RE``) live in
:mod:`jed_attack.campaign.targets` -- the single home shared with ``submission`` so
typing, validation, target counting, and templatizing can never disagree on what a
target is.
"""

import json
from collections.abc import Sequence
from pathlib import Path

from jed_attack.campaign.targets import _EMAIL_RE, _URL_RE

MARKER = "SECRET_MARKER"
# Default URL scheme -- reproduces the historical ``http://<host>.co`` when a message,
# elite, or persisted record does not author its own scheme, so every existing ``{u}``
# caller renders byte-identically.
DEFAULT_URL_SCHEME = "http://{h}.co"

# The single-token host pool (2-char then 3-char lowercase words that tokenize to
# exactly one token under BOTH victim GGUF tokenizers). Regenerate with
# jed_attack.scripts.gen_single_token_hosts whenever a victim model changes.
SINGLE_TOKEN_HOSTS: list[str] = json.loads(
    (Path(__file__).parent / "single_token_hosts.json").read_text(encoding="utf-8")
)


def host(index: int) -> str:
    """The unique single-token host for a candidate index.

    Draws from :data:`SINGLE_TOKEN_HOSTS` (2-char then 3-char lowercase words that are
    one token under both victim tokenizers), so every candidate's divergent host is a
    single token -- minimal decode + maximal prefix-cache reuse. Raises past the set (a
    4-char extension is the future hook; unreachable under ``MAX_REPLAY_FINDINGS``).
    """
    return SINGLE_TOKEN_HOSTS[int(index)]


def render_url(url_scheme: str, index: int) -> str:
    """Render a ``{h}``-slotted URL scheme into candidate ``index``'s concrete URL."""
    return url_scheme.format(h=host(index))


def url(index: int) -> str:
    """The unique clean exfil URL for a candidate index (default http scheme)."""
    return render_url(DEFAULT_URL_SCHEME, index)


def addr(index: int) -> str:
    """The unique clean deputy recipient address for a candidate index."""
    return f"user@{host(index)}.invalid"


def exfil_template(text: str) -> str:
    """Protect a raw exfil body for ``str.format``: escape literal braces, keep ``{u}``.

    Exfil text is authored with a live ``{u}`` placeholder (the URL slot) but is not run
    through :func:`templatize` (there is no concrete URL to extract). This escapes any
    stray literal brace so a model-authored body cannot crash ``render``'s
    ``str.format``, while leaving ``{u}`` as the one live field.
    """
    return text.replace("{", "{{").replace("}", "}}").replace("{{u}}", "{u}")


def render(template: str, index: int, url_scheme: str = DEFAULT_URL_SCHEME) -> str:
    """Fill a protected ``{u}``/``{m}``/``{a}``/``{h}`` template for candidate index.

    ``{u}`` fills from ``url_scheme`` (default reproduces ``http://<host>.co``), so
    every caller that does not author a scheme renders exactly as before.
    """
    u = render_url(url_scheme, index)
    return template.format(m=MARKER, u=u, url=u, a=addr(index), h=host(index)).strip()


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


def message_pair(text: str, mtype: str, url_scheme: str) -> tuple[str, str]:
    """Protected fill template + scheme for one message, type/backward-compat aware.

    Exfil text carrying a live ``{u}`` is new-style: the scheme provides the URL. Exfil
    text with a concrete URL (persisted pre-schema records) and every deputy message go
    through :func:`templatize` at the default scheme, preserving today's behavior.
    """
    if mtype == "exfil" and "{u}" in text:
        return exfil_template(text), url_scheme
    return (templatize(text) or text), DEFAULT_URL_SCHEME


def render_message(text: str, mtype: str, url_scheme: str, index: int) -> str:
    """Render one message to candidate ``index``'s concrete string."""
    tmpl, scheme = message_pair(text, mtype, url_scheme)
    return render(tmpl, index, scheme)


def ordered_chains(
    specs: Sequence[tuple[str, str, str]], cap: int
) -> list[tuple[str, ...]]:
    """Round-robin ``(text, mtype, url_scheme)`` specs into ``cap`` one-message chains.

    Position p uses spec ``p % k`` at host index ``p``, so hosts are unique and the
    shapes are evenly spread -- the ordered sequence both the scorer and the shipped
    artifact walk and trim to their own budget.
    """
    if not specs:
        return []
    k = len(specs)
    return [(render_message(*specs[p % k], index=p),) for p in range(max(0, cap))]
