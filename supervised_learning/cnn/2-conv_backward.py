#!/usr/bin/env python3
""" Convolutional backward propagation module. """
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """ Performs BP over a convolutional layer of a neural network.

    Args:
        dZ: numpy.ndarray of shape (m, h_new, w_new, c_new) containing the
            partial derivatives with respect to the unactivated output.
        A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
            containing the output of the previous layer.
        W: numpy.ndarray of shape (kh, kw, c_prev, c_new)
            containing the kernels for the convolution.
        b: numpy.ndarray of shape (1, 1, 1, c_new)
            containing the biases applied to the convolution.
        padding: string, either 'same' or 'valid', indicating padding type.
        stride: tuple (sh, sw) containing strides for height and width.

    Returns:
        dA_prev: partial derivatives with respect to the previous layer.
        dW: partial derivatives with respect to the kernels.
        db: partial derivatives with respect to the biases.
    """
    m, h_new, w_new, c_new = dZ.shape
    _, h_prev, w_prev, _ = A_prev.shape
    kh, kw, _, _ = W.shape
    sh, sw = stride

    if padding == 'same':
        total_pad_h = max(0, (h_new - 1) * sh + kh - h_prev)
        total_pad_w = max(0, (w_new - 1) * sw + kw - w_prev)
        pad_top = total_pad_h - total_pad_h // 2
        pad_bottom = total_pad_h // 2
        pad_left = total_pad_w - total_pad_w // 2
        pad_right = total_pad_w // 2
    elif padding == 'valid':
        pad_top = pad_bottom = pad_left = pad_right = 0
    else:
        raise ValueError("padding must be 'valid' or 'same'")

    A_prev_padded = np.pad(
        A_prev,
        ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
        mode='constant',
        constant_values=0
    )
    dA_prev_padded = np.zeros_like(A_prev_padded)
    dW = np.zeros_like(W)

    for i in range(m):
        for h in range(h_new):
            h_start = h * sh
            h_end = h_start + kh
            for w in range(w_new):
                w_start = w * sw
                w_end = w_start + kw
                A_slice = A_prev_padded[i, h_start:h_end, w_start:w_end, :]
                for k in range(c_new):
                    dZ_val = dZ[i, h, w, k]
                    # Gradient w.r.t. kernel
                    dW[:, :, :, k] += A_slice * dZ_val
                    # Gradient w.r.t. previous layer (accumulate into padded)
                    dA_prev_padded[i, h_start:h_end, w_start:w_end, :] += \
                        W[:, :, :, k] * dZ_val

    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    dA_prev = dA_prev_padded[
        :, pad_top:pad_top + h_prev, pad_left:pad_left + w_prev, :
    ]

    return dA_prev, dW, db
