#!/usr/bin/env python3
""" 9-model module """
import tensorflow.keras as K


def save_model(network, filename):
    """ Saves an entire Keras model to a file.   """
    network.save(filename)


def load_model(filename):
    """ Loads an entire Keras model from a file. """
    return K.models.load_model(filename)
