#!/usr/bin/env python3
"""Early stopping criterion."""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """Determine whether to stop early and return updated count."""
    if (opt_cost - cost) > threshold:
        count = 0
    else:
        count += 1

    return count >= patience, count
