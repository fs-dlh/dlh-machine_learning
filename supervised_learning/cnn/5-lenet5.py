#!/usr/bin/env python3
""" LeNet-5 model using Keras. """
from tensorflow import keras as K


def lenet5(X):
    """ Builds a modified LeNet-5 model using Keras.

    Args:
        X: K.Input of shape (m, 28, 28, 1).

    Returns:
        A compiled K.Model using Adam and accuracy metrics.
    """

    conv1 = K.layers.Conv2D(
        filters=6,
        kernel_size=(5, 5),
        padding='same',
        kernel_initializer=K.initializers.HeNormal(seed=0),
        activation='relu'
    )(X)

    pool1 = K.layers.MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2)
    )(conv1)

    conv2 = K.layers.Conv2D(
        filters=16,
        kernel_size=(5, 5),
        padding='valid',
        kernel_initializer=K.initializers.HeNormal(seed=0),
        activation='relu'
    )(pool1)

    pool2 = K.layers.MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2)
    )(conv2)

    flatten = K.layers.Flatten()(pool2)

    fc1 = K.layers.Dense(
        units=120,
        kernel_initializer=K.initializers.HeNormal(seed=0),
        activation='relu'
    )(flatten)

    fc2 = K.layers.Dense(
        units=84,
        kernel_initializer=K.initializers.HeNormal(seed=0),
        activation='relu'
    )(fc1)

    output = K.layers.Dense(
        units=10,
        kernel_initializer=K.initializers.HeNormal(seed=0),
        activation='softmax'
    )(fc2)

    model = K.Model(inputs=X, outputs=output)
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
