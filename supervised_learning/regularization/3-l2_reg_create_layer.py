#!/usr/bin/env python3
""" Layer creation with L2 regularization for TensorFlow/Keras. """
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """ Create a fully connected layer with L2 regularization on its kernel.

    Args:
        prev : input tensor to the layer.
        n : number of neurons (units) in the layer.
        activation : activation function to apply.
        lambtha : L2 regularization parameter.

    Returns:
        tf.Tensor: the output of the new layer.
    """

    layer = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=tf.keras.initializers.VarianceScaling(
            scale=2.0,
            mode='fan_avg'
        ),
        kernel_regularizer=tf.keras.regularizers.l2(lambtha)
    )

    return layer(prev)
