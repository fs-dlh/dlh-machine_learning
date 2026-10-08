#!/usr/bin/env python3
""" Creates a TF-IDF embedding matrix. """
import re
import numpy as np


def tf_idf(sentences, vocab=None):
    """ Create a TF-IDF embedding matrix.

    Args:
        sentences : List of sentences to analyze.
        vocab : List of vocabulary words to use. If None, all words
            from the sentences are used.

    Returns:
        tuple: (embeddings, features)
            embeddings (numpy.ndarray): Matrix of shape (s, f).
            features (numpy.ndarray): Features used for the embeddings.
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

    n_docs = len(sentences)
    df = {word: 0 for word in features}

    for words in processed:
        seen = set(words)
        for word in features:
            if word in seen:
                df[word] += 1

    idf = {}
    for word in features:
        idf[word] = np.log((1 + n_docs) / (1 + df[word])) + 1

    embeddings = np.zeros((n_docs, len(features)), dtype=float)

    word_to_index = {word: idx for idx, word in enumerate(features)}

    for i, words in enumerate(processed):
        tf = {}
        for word in words:
            if word in word_to_index:
                tf[word] = tf.get(word, 0) + 1

        for word, count in tf.items():
            idx = word_to_index[word]
            embeddings[i, idx] = count * idf[word]

        norm = np.sqrt(np.sum(embeddings[i] ** 2))
        if norm > 0:
            embeddings[i] /= norm

    return embeddings, np.array(features)
