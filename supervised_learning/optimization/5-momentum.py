#!/usr/bin/env python3
"""Updates a variable using gradient descent with momentum."""

import numpy as np


def update_variables_momentum(alpha, beta1, var, grad, v):
    """Returns the updated variable and new first moment."""
    v_new = beta1 * v + (1 - beta1) * grad
    var_new = var - alpha * v_new
    return var_new, v_new
