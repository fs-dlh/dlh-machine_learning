#!/usr/bin/env python3
""" Module for randomly changing the brightness of an image using TF. """
import tensorflow as tf


def change_brightness(image, max_delta):
    """Randomly change the brightness of an image.

    Args:
        image: A 3D tf.Tensor containing the image to change.
        max_delta: The maximum amount the image should be brightened.

    Returns:
        The brightness-altered image as a tf.Tensor.
    """
    return tf.image.random_brightness(image, max_delta, seed=None)
