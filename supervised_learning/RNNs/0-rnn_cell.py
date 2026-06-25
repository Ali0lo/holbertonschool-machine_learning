#!/usr/bin/env python3
"""Module for RNNCell implementation."""
import numpy as np


class RNNCell:
    """Represents a single vanilla RNN cell."""

    def __init__(self, i, h, o):
        """Initialize RNNCell.

        Args:
            i: input dimensionality
            h: hidden state dimensionality
            o: output dimensionality
        """
        self.Wh = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """Perform forward pass for one time step.

        Args:
            h_prev: previous hidden state, shape (m, h)
            x_t: input at current time step, shape (m, i)

        Returns:
            h_next: next hidden state, shape (m, h)
            y: output, shape (m, o)
        """
        concat = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(concat @ self.Wh + self.bh)
        logits = h_next @ self.Wy + self.by
        logits -= np.max(logits, axis=1, keepdims=True)
        exp = np.exp(logits)
        y = exp / np.sum(exp, axis=1, keepdims=True)
        return h_next, y
