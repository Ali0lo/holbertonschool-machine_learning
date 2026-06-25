#!/usr/bin/env python3
"""Module for RNN forward propagation."""
import numpy as np


def rnn(rnn_cell, X, h_0):
    """Perform forward propagation for a simple RNN.

    Args:
        rnn_cell: instance of RNNCell
        X: input data, shape (t, m, i)
        h_0: initial hidden state, shape (m, h)

    Returns:
        H: all hidden states, shape (t+1, m, h)
        Y: all outputs, shape (t, m, o)
    """
    t, m, _ = X.shape
    _, h = h_0.shape
    H = np.zeros((t + 1, m, h))
    H[0] = h_0
    Y = []
    for step in range(t):
        H[step + 1], y = rnn_cell.forward(H[step], X[step])
        Y.append(y)
    return H, np.array(Y)
