#!/usr/bin/env python3
""" Creates a bag-of-words embedding matrix. """
import re
import numpy as np


def bag_of_words(sentences, vocab=None):
    """ Create a bag-of-words embedding matrix.

    Args:
        sentences : List of sentences to analyze.
        vocab : List of vocabulary words to use. If None, all words
            from the sentences are used.

    Returns:
        tuple: (embeddings, features)
            embeddings (numpy.ndarray): Matrix of shape (s, f).
            features (list): List of features used for the embeddings.
    """
    processed = []

    for sentence in sentences:
        sentence = sentence.lower()
        sentence = re.sub(r"'s\b", "", sentence)
        words = re.findall(r"[a-z]+", sentence)
        processed.append(words)

    if vocab is None:
        features = sorted({word for words in processed for word in words})
    else:
        features = list(vocab)

    word_to_index = {word: idx for idx, word in enumerate(features)}

    embeddings = np.zeros((len(sentences), len(features)), dtype=int)

    for i, words in enumerate(processed):
        for word in words:
            if word in word_to_index:
                embeddings[i, word_to_index[word]] += 1

    return embeddings, np.array(features)
