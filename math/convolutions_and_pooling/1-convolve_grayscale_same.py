#!/usr/bin/env python3
""" Same Convolution """
import numpy as np


def convolve_grayscale_same(images, kernel):
    """ Function that performs a valid conv on grayscale image """
    m, h, w = np.shape(images)
    kh, kw = np.shape(kernel)

    # Compute padding
    p_h = kh // 2
    p_w = kw // 2

    # Pad images
    padded = np.pad(
        images,
        ((0, 0), (p_h, p_h), (p_w, p_w)),
        mode='constant'
    )

    # Output has same size as input
    output = np.zeros((m, h, w))

    # Convolution (same as before)
    for i in range(h):
        for j in range(w):
            patch = padded[:, i:i+kh, j:j+kw]
            output[:, i, j] = np.sum(patch * kernel, axis=(1, 2))

    return output
