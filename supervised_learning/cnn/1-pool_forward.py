#!/usr/bin/env python3
""" Implement forward prop for pooling """
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """ Perform forward prop for pooling layer """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape.shape
    sh, sw = stride
    h_new = (h_prev - kh) // sh + 1
    w_new = (w_prev - kw) // sw + 1

    pooling_output = np.zeros((m, h_new, w_new, c_new))
    for e in range(m):
        for i in range(h_new):
            for j in range(w_new):
                for kernel in range(c_prev):
                    A_s = A_prev_padded[e, i*sh:i*sh+kh, j*sw:j*sw+kw, :]
                    if mode == 'max':
                        Z = np.sum(np.max(A_s))
                    else:
                        Z = np.sum(np.mean(A_s))
                    pooling_output[e, i, j, kernel] = Z
    return pooling_output
