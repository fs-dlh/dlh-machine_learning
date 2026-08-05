#!/usr/bin/env python3
""" that performs a t-SNE transformation """
import numpy as np
pca = __import__('1-pca').pca
P_affinities = __import__('4-P_affinities').P_affinities
grads = __import__('6-grads').grads
cost = __import__('7-cost').cost


def tsne(X, ndims=2, idims=50, perplexity=30.0, iterations=1000, lr=500):
    """ Perform t‑SNE on the dataset X.

    Parameters:
    X :            High‑dimensional data.
    ndims :        Dimension of the low‑dimensional embedding.
    idims :        Intermediate dimension after PCA.
    perplexity :   Perplexity used in P‑affinity computation.
    iterations :   Number of gradient descent iterations.
    lr :           Learning rate for the gradient descent.

    Returns:
    Y :            Optimised low‑dimensional embedding.
    """

    X_pca = pca(X, idims)
    P = P_affinities(X_pca, 1e-5, perplexity)

    n, _ = X_pca.shape
    Y = np.random.randn(n, ndims) * 1e-8
    Y_prev = np.copy(Y)

    for i in range(1, iterations + 1):
        if i <= 100:
            P_eff = P * 4.0
        else:
            P_eff = P

        dY, Q = grads(Y, P_eff)

        gradient = 4.0 * dY
        alpha = 0.5 if i <= 20 else 0.8
        Y_new = Y - lr * gradient + alpha * (Y - Y_prev)
        Y_new = Y_new - np.mean(Y_new, axis=0)
        Y_prev = Y
        Y = Y_new
        if i % 100 == 0:
            _, Q = grads(Y, P_eff)
            C = cost(P_eff, Q)
            print(f"Cost at iteration {i}: {C}")

    return Y
