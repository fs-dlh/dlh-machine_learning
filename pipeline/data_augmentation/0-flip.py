#!/usr/bin/env python3
""" Module for flipping an image horizontally using TensorFlow. """
import tensorflow as tf


def flip_image(image):
    """
    Flips an image left to right
    """
    return tf.image.flip_left_right(image)
