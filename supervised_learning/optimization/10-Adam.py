#!/usr/bin/env python3
""" 10-Adam module """

import tensorflow as tf


def create_Adam_op(alpha, beta1, beta2, epsilon):
    """ Sets up the Adam optimization algorithm in TensorFlow.

    Args:
        alpha : learning rate.
        beta1 : weight for first moment (momentum).
        beta2 : weight for second moment (RMSProp).
        epsilon : small number to avoid division by zero.

    Returns:
        tf.keras.optimizers.Optimizer: configured Adam optimizer.
    """
    return tf.keras.optimizers.Adam(
        learning_rate=alpha,
        beta_1=beta1,
        beta_2=beta2,
        epsilon=epsilon
    )
