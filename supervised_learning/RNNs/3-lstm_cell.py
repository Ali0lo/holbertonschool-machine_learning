#!/usr/bin/env python3
import numpy as np


class LSTMCell:
    def __init__(self, i, h, o):
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
