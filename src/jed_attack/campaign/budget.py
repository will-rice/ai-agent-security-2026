"""Green-seconds budget check for the submission composer.

The composer (a later task) packs archive messages into a submission under a
green-seconds budget. :func:`fits` is the single source of truth for whether a
candidate pool fits that budget, so the composer cannot silently drift from the
calibrated ceiling.
"""

from jed_attack.campaign import archive, config


def fits(items: list[tuple[archive.Entry, int]]) -> bool:
    """Check whether a pool of (entry, count) pairs fits the green-seconds budget.

    Args:
        items: Pairs of an archived entry and how many times it is repeated in the
            candidate pool.

    Returns:
        True if the summed cost (count times each entry's ``cost_s``) is within
        ``config.GREEN_SECONDS_CEILING * config.BUDGET_FILL_FRACTION``.
    """
    total = sum(count * entry.cost_s for entry, count in items)
    return total <= config.GREEN_SECONDS_CEILING * config.BUDGET_FILL_FRACTION
