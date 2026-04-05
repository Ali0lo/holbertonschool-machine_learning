#!/usr/bin/env python3
""" Convolutional backward propagation """
import numpy as np

def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, c_new = W.shape
    _, h_new, w_new, _ = dZ.shape
    sh, sw = stride

    # ---- Padding calculation ----
    if padding == "same":
        ph = max((h_new - 1) * sh + kh - h_prev, 0)
        pw = max((w_new - 1) * sw + kw - w_prev, 0)
        pad_top = ph // 2
        pad_bottom = ph - pad_top
        pad_left = pw // 2
        pad_right = pw - pad_left
    else:
        pad_top = pad_bottom = pad_left = pad_right = 0

    # ---- Pad inputs ----
    A_prev_pad = np.pad(
        A_prev,
        ((0,0), (pad_top, pad_bottom), (pad_left, pad_right), (0,0)),
        mode='constant'
    )
    dA_prev_pad = np.zeros_like(A_prev_pad)
    dW = np.zeros_like(W)
    db = np.sum(dZ, axis=(0,1,2), keepdims=True)

    # ---- Loop over examples, height, width, channels ----
    for e in range(m):
        for i in range(h_new):
            for j in range(w_new):
                for c in range(c_new):
                    vert_start = i*sh
                    vert_end = vert_start + kh
                    horiz_start = j*sw
                    horiz_end = horiz_start + kw

                    A_slice = A_prev_pad[e, vert_start:vert_end, horiz_start:horiz_end, :]

                    dW[:,:,:,c] += A_slice * dZ[e,i,j,c]
                    dA_prev_pad[e, vert_start:vert_end, horiz_start:horiz_end, :] += W[:,:,:,c] * dZ[e,i,j,c]

    # ---- Unpad dA_prev ----
    if padding == "same":
        dA_prev = dA_prev_pad[:, pad_top:dA_prev_pad.shape[1]-pad_bottom,
                                     pad_left:dA_prev_pad.shape[2]-pad_right, :]
    else:
        dA_prev = dA_prev_pad

    return dA_prev, dW, db
