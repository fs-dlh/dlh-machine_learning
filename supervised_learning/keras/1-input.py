#!/usr/bin/env python3
""" 1-input module """
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """ Builds a neural network with the Keras using the Functional API.

    Args:
        nx : number of input features
        layers : list containing the number of nodes in each layer
        activations : list of activation functions for each layer
        lambtha : L2 regularization parameter
        keep_prob : probability that a node will be kept for dropout;
                    if None, dropout is not applied.

    Returns:
        keras.Model: the constructed Keras model
    """

    inputs = K.layers.Input(shape=(nx,))
    x = inputs

    for i, units in enumerate(layers):

        x = K.layers.Dense(
            units=units,
            activation=activations[i],
            kernel_regularizer=K.regularizers.l2(lambtha)
        )(x)

        if i != len(layers) - 1 and keep_prob is not None:
            dropout_rate = 1 - keep_prob
            x = K.layers.Dropout(rate=dropout_rate)(x)

    model = K.Model(inputs=inputs, outputs=x)
    return model
