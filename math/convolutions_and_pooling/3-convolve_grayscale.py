#!/usr/bin/env python3
""" Convolution with Padding and Strides """
import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """ Padding and Strides """
    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride

    # -------- Determine padding --------
    if padding == 'valid':
        ph, pw = 0, 0
        H_out = (h - kh) // sh + 1
        W_out = (w - kw) // sw + 1

    elif padding == 'same':
        H_out = int(np.ceil(h / sh))
        W_out = int(np.ceil(w / sw))
        ph = int(np.ceil(((H_out - 1) * sh + kh - h) / 2))
        pw = int(np.ceil(((W_out - 1) * sw + kw - w) / 2))

    else:  # custom padding
        ph, pw = padding
        H_out = (h + 2*ph - kh) // sh + 1
        W_out = (w + 2*pw - kw) // sw + 1

    # -------- Convolution --------
    for i in range(H_out):
        for j in range(W_out):
            patch = padded[:, i*sh:i*sh+kh, j*sw:j*sw+kw]
            output[:, i, j] = np.sum(patch * kernel, axis=(1, 2))

    return output
