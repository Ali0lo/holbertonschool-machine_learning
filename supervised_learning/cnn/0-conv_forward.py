#!/usr/bin/env python3
""" Write a function that performs forward prop over a conv layer """
import numpy as np

def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """ Function for performing forward prop """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride

    if padding == 'same':
        h_new = h_prev
        w_new = w_prev

        ph = max((h_prev - 1) * sh + kh - h_prev, 0)
        pw = max((w_prev - 1) * sw + kw - w_prev, 0)

        pad_top = ph // 2
        pad_bottom = ph - pad_top

        pad_left = pw // 2
        pad_right = pw - pad_left

        A_prev_padded = np.pad(
            A_prev,
            ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode='constant'
        )
    else:
        A_prev_padded = A_prev
        h_new = (h_prev - kh) // sh + 1
        w_new = (w_prev - kw) // sw + 1

    conv_output = np.zeros((m, h_new, w_new, c_new))
    for example in range(m):
        for i in range(h_new):
            for j in range(w_new):
                for kernel in range(c_new):
                    A_slice = A_prev_padded[example, i*sh:i*sh+kh, j*sw:j*sw+kw, :]
                    Z = np.sum(A_slice * W[:, :, :, kernel]) + b[0, 0, 0, kernel]
                    conv_output[example, i, j, kernel] = Z

    return activation(conv_output)
