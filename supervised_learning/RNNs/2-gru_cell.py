#!/usr/bin/env python3
import numpy as np


class GRUCell:
    def __init__(self, i, h, o):
        self.Wz = np.random.randn(i + h, h)
        self.Wr = np.random.randn(i + h, h)
        self.Wh = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)
        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        concat = np.concatenate((h_prev, x_t), axis=1)
        z = 1 / (1 + np.exp(-(concat @ self.Wz + self.bz)))
        r = 1 / (1 + np.exp(-(concat @ self.Wr + self.br)))
        concat_r = np.concatenate((r * h_prev, x_t), axis=1)
        h_hat = np.tanh(concat_r @ self.Wh + self.bh)
        h_next = (1 - z) * h_prev + z * h_hat
        logits = h_next @ self.Wy + self.by
        logits -= np.max(logits, axis=1, keepdims=True)
        exp = np.exp(logits)
        y = exp / np.sum(exp, axis=1, keepdims=True)
        return h_next, y
