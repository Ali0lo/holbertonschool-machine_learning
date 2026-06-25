#!/usr/bin/env python3
"""Module for LSTMCell implementation."""
import numpy as np


class LSTMCell:
    """Represents a Long Short-Term Memory cell."""

    def __init__(self, i, h, o):
        """Initialize LSTMCell.

        Args:
            i: input dimensionality
            h: hidden state dimensionality
            o: output dimensionality
        """
        self.Wf = np.random.randn(i + h, h)
        self.Wu = np.random.randn(i + h, h)
        self.Wc = np.random.randn(i + h, h)
        self.Wo = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)
        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, c_prev, x_t):
        """Perform forward pass for one time step.

        Args:
            h_prev: previous hidden state, shape (m, h)
            c_prev: previous cell state, shape (m, h)
            x_t: input at current time step, shape (m, i)

        Returns:
            h_next: next hidden state, shape (m, h)
            c_next: next cell state, shape (m, h)
            y: output, shape (m, o)
        """
        concat = np.concatenate((h_prev, x_t), axis=1)
        f = 1 / (1 + np.exp(-(concat @ self.Wf + self.bf)))
        u = 1 / (1 + np.exp(-(concat @ self.Wu + self.bu)))
        c_bar = np.tanh(concat @ self.Wc + self.bc)
        c_next = f * c_prev + u * c_bar
        o = 1 / (1 + np.exp(-(concat @ self.Wo + self.bo)))
        h_next = o * np.tanh(c_next)
        logits = h_next @ self.Wy + self.by
        logits -= np.max(logits, axis=1, keepdims=True)
        exp = np.exp(logits)
        y = exp / np.sum(exp, axis=1, keepdims=True)
        return h_next, c_next, y
