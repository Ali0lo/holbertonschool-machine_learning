#!/usr/bin/env python3
"""Module for performing pooling on images."""
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """Perform pooling on images.

    Args:
        images: numpy.ndarray with shape (m, h, w, c) containing images.
            m is the number of images, h is the height, w is the width,
            c is the number of channels.
        kernel_shape: tuple (kh, kw) containing the kernel shape for pooling.
            kh is the kernel height, kw is the kernel width.
        stride: tuple (sh, sw) where sh is the stride for height and
            sw is the stride for width.
        mode: type of pooling - 'max' for max pooling, 'avg' for average
            pooling.

    Returns:
        numpy.ndarray of shape (m, out_h, out_w, c) containing pooled images.
    """
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride

    out_h = (h - kh) // sh + 1
    out_w = (w - kw) // sw + 1

    output = np.zeros((m, out_h, out_w, c))

    for i in range(out_h):
        for j in range(out_w):
            region = images[:, i * sh:i * sh + kh, j * sw:j * sw + kw, :]
            if mode == 'max':
                output[:, i, j, :] = np.max(region, axis=(1, 2))
            else:
                output[:, i, j, :] = np.mean(region, axis=(1, 2))

    return output
