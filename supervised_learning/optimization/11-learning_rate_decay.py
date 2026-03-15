#!/usr/bin/env python3
"""Calculates learning rate decay using inverse time decay (stepwise)."""

import numpy as np


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """Returns the updated learning rate."""
    return alpha / (1 + decay_rate * np.floor(global_step / decay_step))
