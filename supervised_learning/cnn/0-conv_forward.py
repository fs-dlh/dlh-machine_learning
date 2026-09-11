#!/usr/bin/env python3
""" Convolutional forward propagation module. """
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """ Performs forward propagation over a convolutional layer.

    Args:
        A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
            containing the output of the previous layer.
        W: numpy.ndarray of shape (kh, kw, c_prev, c_new)
            containing the kernels for the convolution.
        b: numpy.ndarray of shape (1, 1, 1, c_new)
            containing the biases applied to the convolution.
        activation: activation function applied to the convolution.
        padding: string, either 'same' or 'valid', indicating padding type.
        stride: tuple (sh, sw) containing strides for height and width.

    Returns:
        The output of the convolutional layer (after activation).
    """
    m, h_prev, w_prev, _ = A_prev.shape
    kh, kw, _, c_new = W.shape
    sh, sw = stride

    if padding == 'same':
        pad_h = int(((h_prev - 1) * sh + kh - h_prev) / 2)
        pad_w = int(((w_prev - 1) * sw + kw - w_prev) / 2)
    elif padding == 'valid':
        pad_h = pad_w = 0
    else:
        raise ValueError("padding must be 'valid' or 'same'")

    out_h = int((h_prev + 2 * pad_h - kh) / sh) + 1
    out_w = int((w_prev + 2 * pad_w - kw) / sw) + 1

    A_pad = np.pad(
        A_prev,
        ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)),
        mode='constant',
        constant_values=0
    )
    Z = np.zeros((m, out_h, out_w, c_new))
    for i in range(out_h):
        h_start = i * sh
        h_end = h_start + kh
        for j in range(out_w):
            w_start = j * sw
            w_end = w_start + kw
            A_slice = A_pad[:, h_start:h_end, w_start:w_end, :]
            for k in range(c_new):
                W_k = W[:, :, :, k]
                conv = np.sum(A_slice * W_k, axis=(1, 2, 3))
                Z[:, i, j, k] = conv + b[0, 0, 0, k]

    A = activation(Z)
    return A
