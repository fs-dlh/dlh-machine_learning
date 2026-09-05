#!/usr/bin/env python3
""" L2 regularization cost for a Keras model. """
import tensorflow as tf


def l2_reg_cost(cost, model):
    """ Extract the L2 regularization cost for each layer of a Keras model.

    Args:
        cost : tensor containing the unregularized cost
        model : Keras model that includes layers with L2 regularization.

    Returns:
        tf.Tensor: a 1D tensor containing the L2 regularization loss for
                   each regularized layer in the model. The order matches
                   the order in which the layers were added.
    """

    reg_losses = model.losses
    if not reg_losses:
        return tf.constant([], dtype=tf.float32)

    total_costs = cost + tf.reduce_sum(model.losses)
    return total_costs
