#!/usr/bin/env python3
""" Converts a gensim Word2Vec model to a Keras Embedding layer. """
from tensorflow.keras.layers import Embedding


def gensim_to_keras(model):
    """ Convert a trained gensim Word2Vec model to a Keras Embedding layer.

    Args:
        model : Trained gensim Word2Vec model.

    Returns:
        keras.layers.Embedding: layer initialized with Word2Vec weights
    """
    embedding_matrix = model.wv.vectors
    vocab_size, embedding_dim = embedding_matrix.shape

    return Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        weights=[embedding_matrix],
        trainable=True,
    )
