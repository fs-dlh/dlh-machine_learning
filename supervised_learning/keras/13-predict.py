#!/usr/bin/env python3
""" 13-predict module """
import tensorflow.keras as K


def predict(network, data, verbose=False):
    """ Makes a prediction using a neural network. """

    verbose_flag = 1 if verbose else 0
    return network.predict(data, verbose=verbose_flag)
