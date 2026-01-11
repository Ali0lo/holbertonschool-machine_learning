#!/usr/bin/env python3
"""Plot y as a solid red line with an x-axis range from 0 to 10."""

import numpy as np
import matplotlib.pyplot as plt


def line():
    """Plot y = x^3 for x in [0, 10] as a solid red line."""
    x = np.arange(0, 11)
    y = x ** 3

    plt.figure(figsize=(6.4, 4.8))
    plt.plot(x, y, "r-")
    plt.xlim(0, 10)
    plt.show()
