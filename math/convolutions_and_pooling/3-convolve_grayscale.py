#!/usr/bin/env python3
"""Module for performing strided convolution on grayscale images."""
import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """Perform a convolution on grayscale images with stride and padding.

    Args:
        images: numpy.ndarray with shape (m, h, w) containing grayscale images.
            m is the number of images, h is the height, w is the width.
        kernel: numpy.ndarray with shape (kh, kw) containing the convolution
            kernel. kh is the kernel height, kw is the kernel width.
        padding: either 'same', 'valid', or a tuple (ph, pw) specifying
            the padding for height and width respectively.
        stride: a tuple (sh, sw) where sh is the stride for height and
            sw is the stride for width.

    Returns:
        numpy.ndarray containing the convolved images.
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride

    if padding == 'same':
        # Calculate padding to ensure output size is ceil(input / stride)
        ph = int(np.ceil(((h - 1) * sh + kh - h) / 2))
        pw = int(np.ceil(((w - 1) * sw + kw - w) / 2))
    elif padding == 'valid':
        ph, pw = 0, 0
    elif isinstance(padding, tuple):
        ph, pw = padding

    # Apply padding
    padded = np.pad(images, ((0, 0), (ph, ph), (pw, pw)), mode='constant')

    out_h = (h + 2 * ph - kh) // sh + 1
    out_w = (w + 2 * pw - kw) // sw + 1

    output = np.zeros((m, out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            region = padded[:, i * sh:i * sh + kh, j * sw:j * sw + kw]
            output[:, i, j] = np.sum(region * kernel, axis=(1, 2))

    return output
