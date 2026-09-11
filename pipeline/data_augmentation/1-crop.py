#!/usr/bin/env python3
""" Module for performing a random crop on an image using TensorFlow. """
import tensorflow as tf


def crop_image(image, size):
    """ Perform a random crop of an image.

    Args:
        image: A 3D tf.Tensor containing the image to crop.
        size: A tuple containing the size of the crop (height, width, channels)

    Returns:
        The randomly cropped image as a tf.Tensor.
    """
    return tf.image.random_crop(image, size)
