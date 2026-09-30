#!/usr/bin/env python3
""" Module for randomly adjusting the contrast of an image using TF. """
import tensorflow as tf


def change_contrast(image, lower, upper):
    """Randomly adjust the contrast of an image.

    Args:
        image: A 3D tf.Tensor representing the input image.
        lower: A float representing the lower bound of ...
        upper: A float representing the upper bound of ...
                                            the random contrast factor range.

    Returns:
        The contrast-adjusted image as a tf.Tensor.
    """

    return tf.image.random_contrast(image, lower, upper, seed=None)
