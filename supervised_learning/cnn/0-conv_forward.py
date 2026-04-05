#!/usr/bin/env python3
"""Convolution forward propagation."""

import numpy as np


def conv_forward(
    A_prev,
    W,
    b,
    activation,
    padding="same",
    stride=(1, 1)
):
    """
    Perform forward propagation over a convolutional layer.

    Args:
        A_prev (np.ndarray): shape (m, h_prev, w_prev, c_prev)
        W (np.ndarray): shape (kh, kw, c_prev, c_new)
        b (np.ndarray): shape (1, 1, 1, c_new)
        activation (callable): activation function applied to convolution
        padding (str): "same" or "valid"
        stride (tuple): (sh, sw)

    Returns:
        np.ndarray: output of the convolutional layer
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev_w, c_new = W.shape
    sh, sw = stride

    if c_prev != c_prev_w:
        raise ValueError(
            "A_prev and W must have the same number of input channels"
        )

    if padding == "same":
        ph = int(np.ceil(((h_prev - 1) * sh + kh - h_prev) / 2))
        pw = int(np.ceil(((w_prev - 1) * sw + kw - w_prev) / 2))
    elif padding == "valid":
        ph = 0
        pw = 0
    else:
        raise ValueError('padding must be either "same" or "valid"')

    A_pad = np.pad(
        A_prev,
        pad_width=((0, 0), (ph, ph), (pw, pw), (0, 0)),
        mode="constant",
        constant_values=0
    )

    h_new = int((h_prev + 2 * ph - kh) / sh) + 1
    w_new = int((w_prev + 2 * pw - kw) / sw) + 1

    Z = np.zeros((m, h_new, w_new, c_new))

    for i in range(m):
        for h in range(h_new):
            for w in range(w_new):
                vert_start = h * sh
                vert_end = vert_start + kh
                horiz_start = w * sw
                horiz_end = horiz_start + kw

                a_slice = A_pad[
                    i,
                    vert_start:vert_end,
                    horiz_start:horiz_end,
                    :
                ]

                for c in range(c_new):
                    Z[i, h, w, c] = (
                        np.sum(a_slice * W[:, :, :, c]) +
                        b[0, 0, 0, c]
                    )

    return activation(Z)
