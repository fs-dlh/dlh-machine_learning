#!/usr/bin/env python3
""" 5-momentum module """
import tensorflow as tf


def create_momentum_op(alpha, beta1):
    """ Sets up the gd with momentum optimization algorithm in TensorFlow.

    Args:
        alpha : learning rate.
        beta1 : momentum weight.

    Returns:
        tf.keras.optimizers.Optimizer: configured SGD optimizer with momentum.
    """
    return tf.keras.optimizers.SGD(learning_rate=alpha, momentum=beta1)
