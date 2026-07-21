"""Tool-hop budget check for the submission composer.

The binding T4 resource is replay work, which is ``∝`` total tool hops — a DETERMINISTIC
quantity, identical on green and T4 (unlike measured wall-time, which fleet contention
inflates). So the composer packs against a hop budget: :func:`fits` is the single source
of truth for whether a candidate pool fits, calibrated to the empirical pass/fail
boundary (``80×K5 = 400`` hops passed; ``105×K5 = 525`` / ``230×K2 = 460`` blanked).
"""

from jed_attack.campaign import archive, config


def fits(items: list[tuple[archive.Entry, int]]) -> bool:
    """Check whether a pool of (entry, count) pairs fits the tool-hop budget.

    Args:
        items: Pairs of an archived entry and how many times it is repeated in the
            candidate pool.

    Returns:
        True if the summed tool hops (count times each entry's ``hops``) is within
        ``config.HOP_CEILING * config.BUDGET_FILL_FRACTION``.
    """
    total = sum(count * entry.hops for entry, count in items)
    return total <= config.HOP_CEILING * config.BUDGET_FILL_FRACTION
