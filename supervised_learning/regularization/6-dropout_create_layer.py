#!/usr/bin/env python3
""" Layer creation with Dropout regularization for TensorFlow/Keras. """
import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob, training=True):
    """ Create a fully connected layer with Dropout.

    Args:
        prev : input tensor to the layer.
        n : number of neurons (units) in the layer.
        activation : activation function to apply.
        keep_prob : probability of keeping a node (0 < keep_prob <= 1).
        training : whether the model is in training mode; dropout is
                         applied only when this is True.

    Returns:
        tf.Tensor: the output of the new layer after dropout.
    """

    initializer = tf.keras.initializers.VarianceScaling(
        scale=2.0,
        mode='fan_avg'
    )

    dense = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=initializer
    )
    output = dense(prev)

    dropout = tf.keras.layers.Dropout(rate=1 - keep_prob)

    return dropout(output, training=training)
