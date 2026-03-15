#!/usr/bin/env python3
"""Module to update learning rate using inverse time decay in NumPy."""
import numpy as np


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    Updates the learning rate using inverse time decay in a stepwise fashion.

    Parameters:
    - alpha: original learning rate
    - decay_rate: weight used to determine decay rate of alpha
    - global_step: number of gradient descent passes elapsed
    - decay_step: number of passes before decaying alpha further

    Returns:
    - updated_alpha: the updated learning rate
    """
    # Number of decay steps completed
    step_count = global_step // decay_step

    # Inverse time decay formula
    updated_alpha = alpha / (1 + decay_rate * step_count)

    return updated_alpha
