#!/usr/bin/env python3
"""Pooling backward propagation."""

import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Perform backward propagation over a pooling layer.

    Args:
        dA (np.ndarray): shape (m, h_new, w_new, c)
        A_prev (np.ndarray): shape (m, h_prev, w_prev, c)
        kernel_shape (tuple): (kh, kw)
        stride (tuple): (sh, sw)
        mode (str): 'max' or 'avg'

    Returns:
        np.ndarray: dA_prev, same shape as A_prev
    """
    m, h_new, w_new, c_new = dA.shape
    m_prev, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    if m != m_prev or c_new != c_prev:
        raise ValueError("dA and A_prev must match on batch size and channels")

    dA_prev = np.zeros_like(A_prev)

    for i in range(m):
        for h in range(h_new):
            for w in range(w_new):
                vert_start = h * sh
                vert_end = vert_start + kh
                horiz_start = w * sw
                horiz_end = horiz_start + kw

                for c in range(c_new):
                    grad = dA[i, h, w, c]

                    if mode == 'max':
                        a_slice = A_prev[
                            i,
                            vert_start:vert_end,
                            horiz_start:horiz_end,
                            c
                        ]
                        mask = (a_slice == np.max(a_slice))
                        dA_prev[
                            i,
                            vert_start:vert_end,
                            horiz_start:horiz_end,
                            c
                        ] += mask * grad
                    elif mode == 'avg':
                        average_grad = grad / (kh * kw)
                        dA_prev[
                            i,
                            vert_start:vert_end,
                            horiz_start:horiz_end,
                            c
                        ] += np.ones((kh, kw)) * average_grad
                    else:
                        raise ValueError("mode must be 'max' or 'avg'")

    return dA_prev
