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
    for i in range(len(layers)):

        dense = K.layers.Dense(
            layers[i],
            input_dim=nx,
            activation=activations[i],
            kernel_regularizer=K.regularizers.l2(lambtha)
        )
        model.add(dense)

        if (i != len(layers) - 1) and (keep_prob is not None):
            model.add(K.layers.Dropout(rate=1 - keep_prob))

    return model
