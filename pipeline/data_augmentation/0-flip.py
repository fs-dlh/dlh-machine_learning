#!/usr/bin/env python3
""" Module for flipping an image horizontally using TensorFlow. """
import tensorflow as tf


def flip_image(image):
    """Flip an image horizontally.

    Args:
        image: A 3D tf.Tensor containing the image to flip.

    Returns:
        The horizontally flipped image as a tf.Tensor.
    """
    return tf.image.flip_left_right(image)
