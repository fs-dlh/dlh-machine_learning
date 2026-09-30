#!/usr/bin/env python3
""" Builds an identity block for a ResNet architecture. """
from tensorflow import keras as K


def identity_block(A_prev, filters):
    """ Build an identity block as described in ResNet (2015).

    Args:
        A_prev: output from the previous layer
        filters: tuple/list of (F11, F3, F12)
            F11: filters in first 1x1 convolution
            F3: filters in 3x3 convolution
            F12: filters in second 1x1 convolution

    Returns:
        The activated output of the identity block.
    """
    F11, F3, F12 = filters

    X = K.layers.Conv2D(
        F11, (1, 1), padding='same',
        kernel_initializer=K.initializers.HeNormal(seed=0)
    )(A_prev)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)

    X = K.layers.Conv2D(
        F3, (3, 3), padding='same',
        kernel_initializer=K.initializers.HeNormal(seed=0)
    )(X)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)

    X = K.layers.Conv2D(
        F12, (1, 1), padding='same',
        kernel_initializer=K.initializers.HeNormal(seed=0)
    )(X)
    X = K.layers.BatchNormalization(axis=3)(X)

    X = K.layers.Add()([X, A_prev])
    X = K.layers.Activation('relu')(X)

    return X
