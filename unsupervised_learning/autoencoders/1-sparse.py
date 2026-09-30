#!/usr/bin/env python3
"""Sparse autoencoder module."""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims, lambtha):
    """ Create a sparse autoencoder.

    Args:
        input_dims: Integer, dimensions of the model input.
        hidden_layers: List of integers, nodes per encoder hidden layer.
        latent_dims: Integer, dimensions of the latent representation.
        lambtha: Float, L1 regularization parameter for the encoded output.

    Returns:
        encoder: The encoder model.
        decoder: The decoder model.
        auto: The sparse autoencoder model.
    """
    encoder_input = keras.Input(shape=(input_dims,))
    x = encoder_input

    for nodes in hidden_layers:
        x = keras.layers.Dense(nodes, activation='relu')(x)

    latent = keras.layers.Dense(
        latent_dims,
        activation='relu',
        activity_regularizer=keras.regularizers.l1(lambtha)
    )(x)
    encoder = keras.Model(encoder_input, latent)

    decoder_input = keras.Input(shape=(latent_dims,))
    x = decoder_input

    for nodes in reversed(hidden_layers):
        x = keras.layers.Dense(nodes, activation='relu')(x)

    decoder_output = keras.layers.Dense(
        input_dims, activation='sigmoid'
    )(x)
    decoder = keras.Model(decoder_input, decoder_output)

    auto_input = keras.Input(shape=(input_dims,))
    auto_encoded = encoder(auto_input)
    auto_decoded = decoder(auto_encoded)
    auto = keras.Model(auto_input, auto_decoded)

    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
