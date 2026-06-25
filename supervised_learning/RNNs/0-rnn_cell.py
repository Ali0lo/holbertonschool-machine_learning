#!/usr/bin/env python3
import numpy as np


class RNNCell:
    def __init__(self, i, h, o):
        self.Wh = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        concat = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(concat @ self.Wh + self.bh)
        logits = h_next @ self.Wy + self.by
        logits -= np.max(logits, axis=1, keepdims=True)
        exp = np.exp(logits)
        y = exp / np.sum(exp, axis=1, keepdims=True)
        return h_next, y
