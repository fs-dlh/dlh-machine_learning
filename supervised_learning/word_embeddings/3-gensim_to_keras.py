#!/usr/bin/env python3
""" Converts a gensim Word2Vec model to a Keras Embedding layer. """
import tensorflow as tf


def gensim_to_keras(model):
    """ Convert a trained gensim Word2Vec model to a Keras Embedding layer.

    Rows follow descending word frequency.

    Args:
        model : Trained gensim Word2Vec model.

    Returns:
        keras.layers.Embedding: layer initialized with Word2Vec weights

    model.wv.sort_by_descending_frequency() """"


    return tf.keras.layers.Embedding(
        input_dim=model.wv.vectors.shape[0],
        output_dim=model.wv.vectors.shape[1],
        weights=[model.wv.vectors],
        trainable=True,
    )
