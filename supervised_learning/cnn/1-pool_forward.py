#!/usr/bin/env python3
"""Pooling forward propagation module."""

import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs forward propagation over a pooling layer of a neural network.

    Args:
        A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
            containing the output of the previous layer.
        kernel_shape: tuple (kh, kw) containing the size of the pooling kernel.
        stride: tuple (sh, sw) containing the strides for pooling.
        mode: string, either 'max' or 'avg', indicating pooling type.

    Returns:
        The output of the pooling layer.
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    out_h = (h_prev - kh) // sh + 1
    out_w = (w_prev - kw) // sw + 1

    A = np.zeros((m, out_h, out_w, c_prev))

    for i in range(out_h):
        h_start = i * sh
        h_end = h_start + kh
        for j in range(out_w):
            w_start = j * sw
            w_end = w_start + kw

            A_slice = A_prev[:, h_start:h_end, w_start:w_end, :]

            if mode == 'max':
                A[:, i, j, :] = np.max(A_slice, axis=(1, 2))
            elif mode == 'avg':
                A[:, i, j, :] = np.mean(A_slice, axis=(1, 2))
            else:
                raise ValueError("mode must be 'max' or 'avg'")

    return A
