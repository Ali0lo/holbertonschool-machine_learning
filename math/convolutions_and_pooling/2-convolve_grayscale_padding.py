#!/usr/bin/env python3
""" Convolution with Padding """
import numpy as np


import numpy as np

def convolve_grayscale_padding(images, kernel, padding):
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph, pw = padding

    # Pad images with zeros
    padded = np.pad(
        images,
        ((0, 0), (ph, ph), (pw, pw)),
        mode='constant'
    )

    # Compute output dimensions
    H_out = h + 2*ph - kh + 1
    W_out = w + 2*pw - kw + 1

    # Initialize output
    output = np.zeros((m, H_out, W_out))

    # Convolution (2 loops only)
    for i in range(H_out):
        for j in range(W_out):
            patch = padded[:, i:i+kh, j:j+kw]
            output[:, i, j] = np.sum(patch * kernel, axis=(1, 2))

    return output
