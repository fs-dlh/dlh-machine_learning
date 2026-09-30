#!/usr/bin/env python3
""" Convolutional autoencoder module."""
import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """ Create a convolutional autoencoder.

    Args:
        input_dims: Tuple of ints, dimensions of the model input.
        filters: List of ints, filters for each encoder conv layer.
        latent_dims: Tuple of ints, dimensions of latent representation.

    Returns:
        encoder: The encoder model.
        decoder: The decoder model.
        auto: The full autoencoder model.
    """
    encoder_input = keras.Input(shape=input_dims)
    x = encoder_input

    for f in filters:
        x = keras.layers.Conv2D(
            f, (3, 3), padding='same', activation='relu'
        )(x)
        x = keras.layers.MaxPooling2D(
            (2, 2), padding='same'
        )(x)

    encoder = keras.Model(encoder_input, x)

    decoder_input = keras.Input(shape=latent_dims)
    x = decoder_input

    for f in reversed(filters[:-1]):
        x = keras.layers.Conv2D(
            f, (3, 3), padding='same', activation='relu'
        )(x)
        x = keras.layers.UpSampling2D((2, 2))(x)

    x = keras.layers.Conv2D(
        filters[0], (3, 3), padding='valid', activation='relu'
    )(x)
    x = keras.layers.UpSampling2D((2, 2))(x)

    x = keras.layers.Conv2D(
        input_dims[-1], (3, 3), padding='same', activation='sigmoid'
    )(x)

    decoder = keras.Model(decoder_input, x)

    auto_input = keras.Input(shape=input_dims)
    auto_encoded = encoder(auto_input)
    auto_decoded = decoder(auto_encoded)
    auto = keras.Model(auto_input, auto_decoded)

    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
