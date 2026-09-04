#!/usr/bin/env python3
""" 8-RMSProp module """
import tensorflow as tf


def create_RMSProp_op(alpha, beta2, epsilon):
    """ Sets up the RMSProp optimization algorithm in TensorFlow.

    Args:
        alpha : learning rate.
        beta2 : RMSProp weight (decay rate for second moment).
        epsilon : small number to avoid division by zero.

    Returns:
        tf.keras.optimizers.Optimizer: configured RMSProp optimizer.
    """
    return tf.keras.optimizers.RMSprop(
        learning_rate=alpha, rho=beta2, epsilon=epsilon)
