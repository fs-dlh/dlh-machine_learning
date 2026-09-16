#!/usr/bin/env python3
""" 14-batch_norm module """
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """ Creates a batch normalization layer for a neural network in TensorFlow.

    Args:
        prev : activated output of the previous layer.
        n : number of nodes in the layer to be created.
        activation : activation function to apply to the output.

    Returns:
        tf.Tensor: activated output of the new layer.
    """

    initializer = tf.keras.initializers.VarianceScaling(mode='fan_avg')

    dense = tf.keras.layers.Dense(
        units=n,
        kernel_initializer=initializer
    )
    Z = dense(prev)

    mean, var = tf.nn.moments(Z, axes=[0])
    gamma = tf.Variable(tf.ones((1, n)))
    beta = tf.Variable(tf.zeros((1, n)))

    Z_norm = tf.nn.batch_normalization(
        Z, mean, var, beta, gamma, variance_epsilon=1e-7)

    if activation is not None:
        return activation(Z_norm)
    return Z_norm
