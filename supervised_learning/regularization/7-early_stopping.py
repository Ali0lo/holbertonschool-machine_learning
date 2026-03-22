#!/usr/bin/env python3
"""Module for early stopping."""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """
    Determine if you should stop gradient descent early.

    Early stopping should occur when the validation cost of the network has
    not decreased relative to the optimal validation cost by more than the
    threshold over a specific patience count.

    Args:
        cost: The current validation cost of the neural network
        opt_cost: The lowest recorded validation cost of the neural network
        threshold: The threshold used for early stopping
        patience: The patience count used for early stopping
        count: The count of how long the threshold has not been met

    Returns:
        A boolean of whether the network should be stopped early, followed
        by the updated count
    """
    # Calculate improvement (how much better the optimal cost is vs current)
    improvement = opt_cost - cost

    # Check if improvement exceeds threshold
    if improvement > threshold:
        # Significant improvement found, reset count
        count = 0
    else:
        # No significant improvement, increment count
        count += 1

    # Check if we should stop early
    if count >= patience:
        return (True, count)
    else:
        return (False, count)
