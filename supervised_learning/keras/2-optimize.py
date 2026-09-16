#!/usr/bin/env python3
""" 2-optimize module """
import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """ Sets up Adam optimization for a Keras model
    with categorical crossentropy loss and accuracy metrics.

    Args:
        network (keras.Model): the model to compile
        alpha (float): learning rate for Adam
        beta1 (float): first Adam optimization parameter
        beta2 (float): second Adam optimization parameter

    Returns:
        None (compiles the model in place)
    """
    optimizer = K.optimizers.Adam(
        learning_rate=alpha,
        beta_1=beta1,
        beta_2=beta2
    )

    network.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
