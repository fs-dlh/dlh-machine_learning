#!/usr/bin/env python3
""" Scaled dot product attention module. """
import tensorflow as tf


def sdp_attention(Q, K, V, mask=None):
    """ Calculate the scaled dot product attention.

    Args:
        Q : Query matrix of shape (..., seq_len_q, dk).
        K : Key matrix of shape (..., seq_len_v, dk).
        V : Value matrix of shape (..., seq_len_v, dv).
        mask : Mask broadcastable to
            (..., seq_len_q, seq_len_v). Defaults to None.

    Returns:
        output : Scaled dot product attention of shape
            (..., seq_len_q, dv).
        weights : Attention weights of shape
            (..., seq_len_q, seq_len_v).
    """
    dk = tf.cast(tf.shape(Q)[-1], tf.float32)
    matmul_qk = tf.matmul(Q, K, transpose_b=True)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    output = tf.matmul(weights, V)

    return output, weights
