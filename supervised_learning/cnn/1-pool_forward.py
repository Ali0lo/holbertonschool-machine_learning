#!/usr/bin/env python3
"""Pooling forward propagation."""

import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Perform forward propagation over a pooling layer.

    Args:
        A_prev (np.ndarray): shape (m, h_prev, w_prev, c_prev)
        kernel_shape (tuple): (kh, kw)
        stride (tuple): (sh, sw)
        mode (str): 'max' or 'avg'

    Returns:
        np.ndarray: output of the pooling layer
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    h_new = int((h_prev - kh) / sh) + 1
    w_new = int((w_prev - kw) / sw) + 1

    output = np.zeros((m, h_new, w_new, c_prev))

    for i in range(m):
        for h in range(h_new):
            for w in range(w_new):
                vert_start = h * sh
                vert_end = vert_start + kh
                horiz_start = w * sw
                horiz_end = horiz_start + kw

                for c in range(c_prev):
                    a_slice = A_prev[
                        i,
                        vert_start:vert_end,
                        horiz_start:horiz_end,
                        c
                    ]

                    if mode == 'max':
                        output[i, h, w, c] = np.max(a_slice)
                    elif mode == 'avg':
                        output[i, h, w, c] = np.mean(a_slice)
                    else:
                        raise ValueError("mode must be 'max' or 'avg'")

    return output
