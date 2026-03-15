#!/usr/bin/env python3
"""Module to calculate bias-corrected moving averages."""


def moving_average(data, beta):
    """
    Calculates the bias-corrected moving average of a data set.

    Parameters:
    - data: list of numerical values to calculate the moving average of
    - beta: weight used for the moving average (0 < beta < 1)

    Returns:
    - moving_averages: list containing the bias-corrected moving averages
    """
    moving_averages = []
    v = 0  # running weighted average
    for t, x in enumerate(data, 1):  # t starts from 1
        v = beta * v + (1 - beta) * x
        v_corrected = v / (1 - beta**t)  # bias correction
        moving_averages.append(v_corrected)
    return moving_averages
