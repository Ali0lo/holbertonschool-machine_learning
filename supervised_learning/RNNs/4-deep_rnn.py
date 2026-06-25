#!/usr/bin/env python3
import numpy as np


def deep_rnn(rnn_cells, X, h_0):
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
