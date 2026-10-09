#!/usr/bin/env python3
""" Trains a Gensim Word2Vec model. """
import gensim


def word2vec_model(sentences,
                   vector_size=100,
                   min_count=5,
                   window=5,
                   negative=5,
                   cbow=True,
                   epochs=5,
                   seed=1,
                   workers=1):
    """ Create, build, and train a Gensim Word2Vec model.

    Args:
        sentences : List of sentences to be trained on.
        vector_size : Dimensionality of the embedding layer.
        min_count : Minimum word frequency for use in training.
        window : Maximum distance between current and predicted word.
        negative : Size of negative sampling.
        cbow : If True, use CBOW; if False, use Skip-gram.
        epochs : Number of training iterations.
        seed : Random number generator seed.
        workers : Number of worker threads.

    Returns:
        Word2Vec: The trained Word2Vec model.
    """
    sg = 0 if cbow else 1

    model = gensim.models.Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        min_count=min_count,
        window=window,
        negative=negative,
        sg=sg,
        epochs=epochs,
        seed=seed,
        workers=workers
    )

    return model
