#!/usr/bin/env python3
""" 12-test module """
import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """ Tests a neural network by evaluating it on provided data and labels."""
    verbose_flag = 1 if verbose else 0
    loss, accuracy = network.evaluate(data, labels, verbose=verbose_flag)
    return [loss, accuracy]
