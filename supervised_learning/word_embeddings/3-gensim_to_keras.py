#!/usr/bin/env python3
""" Converts a gensim Word2Vec model to a Keras Embedding layer. """
import tensorflow as tf


def gensim_to_keras(model):
    """ Convert a trained gensim Word2Vec model to a Keras Embedding layer.

    Args:
        model : Trained gensim Word2Vec model.

    Returns:
        tf.keras.layers.Embedding: Trainable Keras Embedding layer whose
            weights are initialized from the model's word vectors.
    """
    weights = model.wv.vectors
    vocab_size, vector_size = weights.shape

    embedding = tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=vector_size,
        weights=[weights],
        trainable=True
    )

    return embedding
