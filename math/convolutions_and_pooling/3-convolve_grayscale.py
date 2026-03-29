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

    elif padding == 'same':
        H_out = int(np.ceil(h / sh))
        W_out = int(np.ceil(w / sw))

        ph = int(np.ceil(((H_out - 1) * sh + kh - h) / 2))
        pw = int(np.ceil(((W_out - 1) * sw + kw - w) / 2))

    else:
        ph, pw = padding

    # -------- Pad images --------
    padded = np.pad(images, ((0,0), (ph,ph), (pw,pw)), mode='constant')

    # -------- Output size --------
    H_out = (h + 2*ph - kh) // sh + 1
    W_out = (w + 2*pw - kw) // sw + 1

    output = np.zeros((m, H_out, W_out))

    # -------- Convolution --------
    for i in range(H_out):
        for j in range(W_out):
            patch = padded[:, i*sh:i*sh+kh, j*sw:j*sw+kw]
            output[:, i, j] = np.sum(patch * kernel, axis=(1, 2))

    return output
