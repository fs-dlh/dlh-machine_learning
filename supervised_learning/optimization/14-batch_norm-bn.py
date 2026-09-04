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

    bn = tf.keras.layers.BatchNormalization(
        epsilon=1e-7,
        center=True,
        scale=True,
        gamma_initializer='ones',
        beta_initializer='zeros'
    )
    Z_norm = bn(Z)

    if activation is not None:
        return activation(Z_norm)
    return Z_norm
