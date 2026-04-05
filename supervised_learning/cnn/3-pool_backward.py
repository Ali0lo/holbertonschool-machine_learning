#!/usr/bin/env python3
"""
Module containing the function pool_backward
"""
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs back propagation over a pooling layer of a neural network.

    Parameters:
    - dA: numpy.ndarray of shape (m, h_new, w_new, c_new) containing the
      partial derivatives with respect to the output of the pooling layer.
    - A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c) containing the
      output of the previous layer.
    - kernel_shape: tuple of (kh, kw) containing the size of the kernel.
    - stride: tuple of (sh, sw) containing the strides.
    - mode: string 'max' or 'avg'.

    Returns:
    dA_prev: the partial derivatives with respect to the previous layer.
    """
    m, h_new, w_new, c_new = dA.shape
    m, h_prev, w_prev, c = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    # Initialize the output gradient with zeros
    dA_prev = np.zeros_like(A_prev)

    for i in range(m):
        for h in range(h_new):
            for w in range(w_new):
                for f in range(c):
                    # Define the corners of the current window
                    v_start = h * sh
                    v_end = v_start + kh
                    h_start = w * sw
                    h_end = h_start + kw

                    if mode == 'max':
                        # In max pooling, only the maximum value gets the gradient
                        a_prev_slice = A_prev[i, v_start:v_end, h_start:h_end, f]
                        mask = (a_prev_slice == np.max(a_prev_slice))
                        dA_prev[i, v_start:v_end, h_start:h_end, f] += (
                            mask * dA[i, h, w, f]
                        )

                    elif mode == 'avg':
                        # In average pooling, the gradient is split equally
                        average_gradient = dA[i, h, w, f] / (kh * kw)
                        dA_prev[i, v_start:v_end, h_start:h_end, f] += (
                            np.ones((kh, kw)) * average_gradient
                        )

    return dA_prev
