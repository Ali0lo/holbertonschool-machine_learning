#!/usr/bin/env python3
""" Valid Convolution """
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """ Function that performs a valid conv on grayscale image """
    m, h, w = np.shape(images)
    kh, kw = np.shape(kernel)
    output = np.zeros((m, h - kh + 1, w - kw + 1))
    for i in range(h - kh + 1):
        for j in range(w - kw + 1):
            patch = images[:, i:i+kh, j:j+kw]
            output[:, i, j] = np.sum(patch * kernel, axis=(1, 2))
    return output
