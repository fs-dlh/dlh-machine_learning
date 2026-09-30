#!/usr/bin/env python3
""" Variational autoencoder module. """
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """ Create a variational autoencoder.

    Args:
        input_dims: Integer, dimensions of the model input.
        hidden_layers: List of ints, nodes per encoder hidden layer.
        latent_dims: Integer, dimensions of the latent representation.

    Returns:
        encoder: Encoder model outputting (z, mean, log_var).
        decoder: Decoder model.
        auto: Full variational autoencoder model.
    """
    # Encoder
    encoder_input = keras.Input(shape=(input_dims,))
    x = encoder_input
    for nodes in hidden_layers:
        x = keras.layers.Dense(nodes, activation='relu')(x)

    mean = keras.layers.Dense(latent_dims, activation=None)(x)
    log_var = keras.layers.Dense(latent_dims, activation=None)(x)

    def sampling(args):
        mean, log_var = args
        epsilon = keras.backend.random_normal(
            shape=keras.backend.shape(mean),
            mean=0.0,
            stddev=1.0
        )
        return mean + keras.backend.exp(log_var / 2) * epsilon

    z = keras.layers.Lambda(
        sampling, output_shape=(latent_dims,)
    )([mean, log_var])

    encoder = keras.Model(encoder_input, [z, mean, log_var])

    # Decoder
    decoder_input = keras.Input(shape=(latent_dims,))
    x = decoder_input
    for nodes in reversed(hidden_layers):
        x = keras.layers.Dense(nodes, activation='relu')(x)
    decoder_output = keras.layers.Dense(
        input_dims, activation='sigmoid'
    )(x)

    decoder = keras.Model(decoder_input, decoder_output)

    # Full autoencoder
    auto_input = keras.Input(shape=(input_dims,))
    z_encoded = encoder(auto_input)[0]
    auto_decoded = decoder(z_encoded)
    auto = keras.Model(auto_input, auto_decoded)

    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
