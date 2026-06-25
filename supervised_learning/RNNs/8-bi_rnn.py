#!/usr/bin/env python3
"""Module for bidirectional RNN forward propagation."""
import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """Perform forward propagation for a bidirectional RNN.

    Args:
        bi_cell: instance of BidirectionalCell
        X: input data, shape (t, m, i)
        h_0: initial forward hidden state, shape (m, h)
        h_t: initial backward hidden state, shape (m, h)

    Returns:
        H: concatenated hidden states, shape (t, m, 2 * h)
        Y: outputs, shape (t, m, o)
    """
    t, m, _ = X.shape
    _, h = h_0.shape
    Hf = np.zeros((t, m, h))
    Hb = np.zeros((t, m, h))
    h_forward = h_0
    for step in range(t):
        h_forward = bi_cell.forward(h_forward, X[step])
        Hf[step] = h_forward
    h_backward = h_t
    for step in range(t - 1, -1, -1):
        h_backward = bi_cell.backward(h_backward, X[step])
        Hb[step] = h_backward
    H = np.concatenate((Hf, Hb), axis=2)
    Y = bi_cell.output(H)
    return H, Y
