#!/usr/bin/env python3
""" RNN Encoder module for machine translation. """
import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    """ Encode input sequences using an embedding layer and a GRU layer. """

    def __init__(self, vocab, embedding, units, batch):
        """ Initialize the RNN encoder.

        Args:
            vocab : Size of the input vocabulary.
            embedding : Dimensionality of the embedding vector.
            units : Number of hidden units in the GRU cell.
            batch : Batch size.
        """
        super(RNNEncoder, self).__init__()
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(
            input_dim=vocab,
            output_dim=embedding
        )
        self.gru = tf.keras.layers.GRU(
            units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer='glorot_uniform'
        )

    def initialize_hidden_state(self):
        """ Initialize the hidden state to zeros.

        Returns:
            tf.Tensor: Tensor of shape (batch, units) containing zeros.
        """
        return tf.zeros((self.batch, self.units))

    def call(self, x, initial):
        """ Encode the input sequence.

        Args:
            x (tf.Tensor): Input tensor of shape (batch, input_seq_len).
            initial (tf.Tensor): Initial hidden state of shape (batch, units).

        Returns:
            outputs (tf.Tensor): Encoder outputs of shape
                (batch, input_seq_len, units).
            hidden (tf.Tensor): Last hidden state of shape (batch, units).
        """
        x = self.embedding(x)
        outputs, hidden = self.gru(x, initial_state=initial)
        return outputs, hidden
