#!/usr/bin/env python3
"""  Creates, builds and trains a genism fastText model. """
from gensim.models import FastText


def fasttext_model(sentences, vector_size=100, min_count=5, negative=5,
                   window=5, cbow=True, epochs=5, seed=0, workers=1):
    """  Creates, builds and trains a genism fastText model. """

    model = FastText(
        vector_size=vector_size,
        min_count=min_count,
        negative=negative,
        window=window,
        sg=0 if cbow else 1,
        epochs=epochs,
        seed=seed,
        workers=workers,
    )

    model.build_vocab(sentences)
    model.train(
        sentences,
        total_examples=len(sentences),
        epochs=epochs,
    )

    return model
