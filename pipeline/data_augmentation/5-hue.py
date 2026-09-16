#!/usr/bin/env python3
""" Module for changing the hue of an image using TensorFlow. """
import tensorflow as tf


def change_hue(image, delta):
    """Change the hue of an image.

    Args:
        image: A 3D tf.Tensor containing the image to change.
        delta: The amount the hue should change.

    Returns:
        The hue-altered image as a tf.Tensor.
    """
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.adjust_hue(image, delta)
    return tf.image.convert_image_dtype(image, tf.uint8)
