#!/usr/bin/env python3
"""
Module that contains the function conv_backward
"""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    Performs back propagation over a convolutional layer of a neural network.

    Parameters:
    - dZ: numpy.ndarray of shape (m, h_new, w_new, c_new) containing the
      partial derivatives with respect to the unactivated output of the
      convolutional layer.
    - A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c_prev) containing
      the output of the previous layer.
    - W: numpy.ndarray of shape (kh, kw, c_prev, c_new) containing the kernels.
    - b: numpy.ndarray of shape (1, 1, 1, c_new) containing the biases.
    - padding: string that is either 'same' or 'valid'.
    - stride: tuple of (sh, sw) containing the strides for the convolution.

    Returns:
    The partial derivatives with respect to the previous layer (dA_prev),
    the kernels (dW), and the biases (db), respectively.
    """
    m, h_new, w_new, c_new = dZ.shape
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride

    # Determine padding values
    if padding == "same":
        ph = int(np.ceil(((h_prev - 1) * sh + kh - h_prev) / 2))
        pw = int(np.ceil(((w_prev - 1) * sw + kw - w_prev) / 2))
    else:
        ph, pw = 0, 0

    # Initialize variables with zeros
    dW = np.zeros(W.shape)
    dA_prev_pad = np.zeros((m, h_prev + 2 * ph, w_prev + 2 * pw, c_prev))
    
    # Calculate db directly by summing across examples, height, and width
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    # Pad the previous layer activation
    A_prev_pad = np.pad(A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                        mode='constant', constant_values=0)

    # Compute dW and dA_prev_pad
    for i in range(m):
        a_pad = A_prev_pad[i]
        da_pad = dA_prev_pad[i]
        
        for h in range(h_new):
            for w in range(w_new):
                for c in range(c_new):
                    # Define the corners of the current slice
                    v_start = h * sh
                    v_end = v_start + kh
                    h_start = w * sw
                    h_end = h_start + kw

                    # Extract the slice from A_prev_pad
                    a_slice = a_pad[v_start:v_end, h_start:h_end, :]

                    # Update gradients for the window and the weights
                    da_pad[v_start:v_end, h_start:h_end, :] += W[:, :, :, c] * dZ[i, h, w, c]
                    dW[:, :, :, c] += a_slice * dZ[i, h, w, c]

    # Extract the unpadded dA_prev from the padded version
    dA_prev = dA_prev_pad[:, ph:ph+h_prev, pw:pw+w_prev, :]

    return dA_prev, dW, db
