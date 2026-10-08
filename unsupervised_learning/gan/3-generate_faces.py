#!/usr/bin/env python3
""" Convolutional generator and discriminator for 16x16 grayscale faces. """
import tensorflow as tf
from tensorflow import keras


def convolutional_GenDiscr():
    """ Build a convolutional generator and discriminator.

    The generator maps a latent vector of shape (16,) to a
    16x16x1 image. The discriminator maps a 16x16x1 image to a
    single scalar. All convolutions use ``padding="same"`` and
    ``"tanh"`` activations.

    Returns:
        tuple: the generator model and the discriminator model.
    """

    def get_generator():
        """ Return the generator model. """
        inputs = keras.Input(shape=(16,))
        hidden = keras.layers.Dense(2048, activation='tanh')(inputs)
        hidden = keras.layers.Reshape((2, 2, 512))(hidden)

        hidden = keras.layers.UpSampling2D()(hidden)
        hidden = keras.layers.Conv2D(64, (3, 3), padding='same')(hidden)
        hidden = keras.layers.BatchNormalization()(hidden)
        hidden = keras.layers.Activation('tanh')(hidden)

        hidden = keras.layers.UpSampling2D()(hidden)
        hidden = keras.layers.Conv2D(16, (3, 3), padding='same')(hidden)
        hidden = keras.layers.BatchNormalization()(hidden)
        hidden = keras.layers.Activation('tanh')(hidden)

        hidden = keras.layers.UpSampling2D()(hidden)
        hidden = keras.layers.Conv2D(1, (3, 3), padding='same')(hidden)
        hidden = keras.layers.BatchNormalization()(hidden)
        outputs = keras.layers.Activation('tanh')(hidden)

        return keras.Model(inputs, outputs, name="generator")

    def get_discriminator():
        """ Return the discriminator model. """
        inputs = keras.Input(shape=(16, 16, 1))

        hidden = keras.layers.Conv2D(32, (3, 3), padding='same')(inputs)
        hidden = keras.layers.MaxPooling2D()(hidden)
        hidden = keras.layers.Activation('tanh')(hidden)

        hidden = keras.layers.Conv2D(64, (3, 3), padding='same')(hidden)
        hidden = keras.layers.MaxPooling2D()(hidden)
        hidden = keras.layers.Activation('tanh')(hidden)

        hidden = keras.layers.Conv2D(128, (3, 3), padding='same')(hidden)
        hidden = keras.layers.MaxPooling2D()(hidden)
        hidden = keras.layers.Activation('tanh')(hidden)

        hidden = keras.layers.Conv2D(256, (3, 3), padding='same')(hidden)
        hidden = keras.layers.MaxPooling2D()(hidden)
        hidden = keras.layers.Activation('tanh')(hidden)

        hidden = keras.layers.Flatten()(hidden)
        outputs = keras.layers.Dense(1, activation='tanh')(hidden)

        return keras.Model(inputs, outputs, name="discriminator")

    return get_generator(), get_discriminator()
