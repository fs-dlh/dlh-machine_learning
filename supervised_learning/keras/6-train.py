#!/usr/bin/env python3
""" 6-train module """
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                verbose=True, shuffle=False):
    """ Trains a model using mini-batch gradient descent
    with optional early stopping.

    Args:
        network : compiled model to train.
        data : shape (m, nx) input data.
        labels : one-hot shape (m, classes) labels.
        batch_size : batch size.
        epochs : number of epochs.
        validation_data : (x_val, y_val) for validation.
        early_stopping : whether to use early stopping based on val_loss.
        patience : patience for early stopping.
        verbose : whether to print progress.
        shuffle : whether to shuffle data before each epoch.

    Returns:
        keras.callbacks.History: training history object.
    """
    callbacks = []
    if validation_data is not None and early_stopping:
        callbacks.append(
            K.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=patience
            )
        )

    history = network.fit(
        x=data,
        y=labels,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        shuffle=shuffle,
        validation_data=validation_data,
        callbacks=callbacks
    )
    return history
