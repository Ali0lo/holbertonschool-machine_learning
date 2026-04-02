#!/usr/bin/env python3
"""Module for performing convolution on images using multiple kernels."""
import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """Perform a convolution on images using multiple kernels.

    Args:
        images: numpy.ndarray with shape (m, h, w, c) containing images.
            m is the number of images, h is the height, w is the width,
            c is the number of channels.
        kernels: numpy.ndarray with shape (kh, kw, c, nc) containing kernels.
            kh is the kernel height, kw is the kernel width, c is channels,
            nc is the number of kernels.
        padding: either 'same', 'valid', or a tuple (ph, pw) specifying
            the padding for height and width respectively.
        stride: a tuple (sh, sw) where sh is the stride for height and
            sw is the stride for width.

    Returns:
        numpy.ndarray of shape (m, out_h, out_w, nc) containing the convolved
        images.
    """
    m, h, w, c = images.shape
    kh, kw, _, nc = kernels.shape
    sh, sw = stride

    if padding == 'same':
        ph = int(np.ceil(((h - 1) * sh + kh - h) / 2))
        pw = int(np.ceil(((w - 1) * sw + kw - w) / 2))
    elif padding == 'valid':
        ph, pw = 0, 0
    else:
        ph, pw = padding

    padded = np.pad(images, ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                    mode='constant')

    out_h = (h + 2 * ph - kh) // sh + 1
    out_w = (w + 2 * pw - kw) // sw + 1

    output = np.zeros((m, out_h, out_w, nc))

    for k in range(nc):
        for i in range(out_h):
            for j in range(out_w):
                region = padded[:,
                                i * sh:i * sh + kh,
                                j * sw:j * sw + kw,
                                :]
                output[:, i, j, k] = np.sum(
                    region * kernels[:, :, :, k], axis=(1, 2, 3)
                )

    return output
