#!/usr/bin/env python3
"""  0-sequential module  """
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """ Builds a sequential neural network with L2 regularization and Dropout.

    Args:
        nx : Dimensionality of the input data.
        layers : List representing the number of neurons in each layer.
        activations : List representing activation functions for each layer.
        lambtha : L2 regularization coefficient.
        keep_prob : Probability of keeping a neuron during dropout
                    (1 - dropout rate).

    Returns:
        keras.Model: The constructed Keras model.
    """

    model = K.Sequential()
    for i, units in enumerate(layers):
        # Create a Dense layer with L2 regularization on kernel
        dense = K.layers.Dense(
            units=units,
            activation=activations[i],
            kernel_regularizer=K.regularizers.l2(lambtha),
            input_shape=(nx,) if i == 0 else None
        )
        model.add(dense)

        if i != len(layers) - 1:
            dropout_rate = 1 - keep_prob
            model.add(K.layers.Dropout(dropout_rate))

    return model
