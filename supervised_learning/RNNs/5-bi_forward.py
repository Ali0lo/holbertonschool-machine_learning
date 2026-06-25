#!/usr/bin/env python3
"""Module for BidirectionalCell forward direction implementation."""
import numpy as np


class BidirectionalCell:
    """Represents a bidirectional cell of an RNN."""

    def __init__(self, i, h, o):
        """Initialize BidirectionalCell.

        Args:
            i: dimensionality of the data
            h: dimensionality of the hidden states
            o: dimensionality of the outputs
        """
        self.Whf = np.random.randn(i + h, h)
        self.Whb = np.random.randn(i + h, h)
        self.Wy = np.random.randn(2 * h, o)
        self.bhf = np.zeros((1, h))
        self.bhb = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """Calculate hidden state in forward direction for one time step.

        Args:
            h_prev: previous hidden state, shape (m, h)
            x_t: data input for the cell, shape (m, i)

        Returns:
            h_next: next hidden state, shape (m, h)
        """
        concat = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(concat @ self.Whf + self.bhf)
        return h_next
