#!/usr/bin/env python3
"""Calculates the weighted moving average of a dataset."""


def moving_average(data, beta):
    """Returns the bias-corrected moving averages of data."""
    avg = []
    v = 0

    for i, x in enumerate(data, 1):
        v = beta * v + (1 - beta) * x
        v_corr = v / (1 - beta ** i)
        avg.append(v_corr)

    return avg
