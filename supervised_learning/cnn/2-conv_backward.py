#!/usr/bin/env python3
"""Convolution backward propagation."""

import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    Perform backward propagation over a convolutional layer.

    Args:
        dZ (np.ndarray): shape (m, h_new, w_new, c_new)
        A_prev (np.ndarray): shape (m, h_prev, w_prev, c_prev)
        W (np.ndarray): shape (kh, kw, c_prev, c_new)
        b (np.ndarray): shape (1, 1, 1, c_new)
        padding (str): "same" or "valid"
        stride (tuple): (sh, sw)

    Returns:
        tuple: (dA_prev, dW, db)
    """
    m, h_new, w_new, c_new = dZ.shape
    m_prev, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev_w, c_new_w = W.shape
    sh, sw = stride

    if m != m_prev:
        raise ValueError("dZ and A_prev must have the same batch size")
    if c_prev != c_prev_w or c_new != c_new_w:
        raise ValueError("Shapes of dZ and W are incompatible")

    if padding == "same":
        ph = int(np.ceil(((h_prev - 1) * sh + kh - h_prev) / 2))
        pw = int(np.ceil(((w_prev - 1) * sw + kw - w_prev) / 2))
    elif padding == "valid":
        ph = 0
        pw = 0
    else:
        raise ValueError('padding must be either "same" or "valid"')

    A_prev_pad = np.pad(
        A_prev,
        ((0, 0), (ph, ph), (pw, pw), (0, 0)),
        mode="constant"
    )
    dA_prev_pad = np.zeros_like(A_prev_pad)
    dW = np.zeros_like(W)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    for i in range(m):
        a_prev = A_prev_pad[i]
        da_prev = dA_prev_pad[i]

        for h in range(h_new):
            for w in range(w_new):
                vert_start = h * sh
                vert_end = vert_start + kh
                horiz_start = w * sw
                horiz_end = horiz_start + kw

                a_slice = a_prev[
                    vert_start:vert_end,
                    horiz_start:horiz_end,
                    :
                ]

                for c in range(c_new):
                    da_prev[
                        vert_start:vert_end,
                        horiz_start:horiz_end,
                        :
                    ] += W[:, :, :, c] * dZ[i, h, w, c]

                    dW[:, :, :, c] += a_slice * dZ[i, h, w, c]

        dA_prev_pad[i] = da_prev

    if ph == 0 and pw == 0:
        dA_prev = dA_prev_pad
    else:
        dA_prev = dA_prev_pad[:, ph:ph + h_prev, pw:pw + w_prev, :]

    return dA_prev, dW, db
