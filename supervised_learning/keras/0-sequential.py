#!/usr/bin/env python
"""  0-sequential module  """
from tensorflow import keras as K


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
    for i, layer_size in enumerate(layers):
        if i == 0:
            input_shape = (nx,)
        else:
            input_shape = None
        model.add(K.layers.Dense(
            units=layer_size,
            activation=activations[i],
            input_shape=input_shape,
            kernel_regularizer=K.regularizers.l2(lambtha)
        ))
        model.add(K.layers.Dropout(1 - keep_prob))

    return model
