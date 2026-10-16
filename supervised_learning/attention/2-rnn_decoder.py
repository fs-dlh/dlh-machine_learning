#!/usr/bin/env python3
""" RNN Decoder module for machine translation. """

import tensorflow as tf

SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """ Decode sequences using an embedding, attention, and a GRU layer. """

    def __init__(self, vocab, embedding, units, batch):
        """ Initialize the RNN decoder.

        Args:
            vocab : Size of the output vocabulary.
            embedding : Dimensionality of the embedding vector.
            units : Number of hidden units in the GRU cell.
            batch : Batch size.
        """
        super(RNNDecoder, self).__init__()
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(
            units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer='glorot_uniform'
        )
        self.F = tf.keras.layers.Dense(vocab)
        self.attention = SelfAttention(units)

    def call(self, x, s_prev, hidden_states):
        """ Decode a single step.

        Args:
            x : Tensor of shape (batch, 1) with previous target
                word indices.
            s_prev : Tensor of shape (batch, units) with the
                previous decoder hidden state.
            hidden_states : Tensor of shape
                (batch, input_seq_len, units) with encoder outputs.

        Returns:
            y : Tensor of shape (batch, vocab) with output logits.
            s : Tensor of shape (batch, units) with the new
                decoder hidden state.
        """
        x = self.embedding(x)
        context, _ = self.attention(s_prev, hidden_states)
        context = tf.expand_dims(context, 1)
        x = tf.concat([context, x], axis=-1)
        outputs, s = self.gru(x, initial_state=s_prev)
        y = self.F(outputs)
        y = tf.squeeze(y, axis=1)
        return y, s
