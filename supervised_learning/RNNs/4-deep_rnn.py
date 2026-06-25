#!/usr/bin/env python3
"""Module for deep RNN forward propagation."""
import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """Perform forward propagation for a deep RNN.

    Args:
        rnn_cells: list of RNNCell instances, one per layer
        X: input data, shape (t, m, i)
        h_0: initial hidden states, shape (l, m, h)

    Returns:
        H: all hidden states, shape (t+1, l, m, h)
        Y: outputs from final layer, shape (t, m, o)
    """
    t, m, _ = X.shape
    l = len(rnn_cells)
    _, _, h = h_0.shape
    H = np.zeros((t + 1, l, m, h))
    H[0] = h_0
    Y = []
    for step in range(t):
        layer_input = X[step]
        for layer in range(l):
            h_next, y = rnn_cells[layer].forward(H[step, layer], layer_input)
            H[step + 1, layer] = h_next
            layer_input = h_next
        Y.append(y)
    return H, np.array(Y)
