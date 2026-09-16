#!/usr/bin/env python3
""" 4-train module """
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs, verbose=True,
                shuffle=False):
    """ Trains a model using mini-batch gradient descent.

    Args:
        network : the compiled model to train.
        data : shape (m, nx) input data.
        labels : one-hot shape (m, classes) labels.
        batch_size : size of the batch for mini-batch gradient descent.
        epochs : number of passes through the data.
        verbose : whether to print progress during training.
        shuffle : whether to shuffle the data before each epoch.

    Returns:
        keras.callbacks.History: the training history object.
    """
    history = network.fit(
        x=data,
        y=labels,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        shuffle=shuffle
    )
    return history
