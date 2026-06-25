#!/usr/bin/env python3
import numpy as np


def rnn(rnn_cell, X, h_0):
    t, m, _ = X.shape
    _, h = h_0.shape
    H = np.zeros((t + 1, m, h))
    H[0] = h_0
    Y = []
    for step in range(t):
        H[step + 1], y = rnn_cell.forward(H[step], X[step])
        Y.append(y)
    return H, np.array(Y)
