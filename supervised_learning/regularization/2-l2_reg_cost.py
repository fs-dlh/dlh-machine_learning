#!/usr/bin/env python3
""" L2 regularization cost for a Keras model. """
import tensorflow as tf


def l2_reg_cost(cost, model):
    """ Extract the L2 regularization cost for each layer of a Keras model.

    Args:
        cost : tensor containing the unregularized cost
                          (e.g., cross-entropy loss). This parameter is
                          included for interface consistency but is not
                          used in the computation.
        model : Keras model that includes layers with L2
                                regularization.

    Returns:
        tf.Tensor: a 1D tensor containing the L2 regularization loss for
                   each regularized layer in the model. The order matches
                   the order in which the layers were added.
    """

    reg_losses = model.losses
    if reg_losses:

        return tf.stack(reg_losses)
    else:

        return tf.constant([], dtype=tf.float32)
