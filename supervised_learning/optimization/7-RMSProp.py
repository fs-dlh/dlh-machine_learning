#!/usr/bin/env python3
""" 7-RMSProp module """
import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """ Updates a variable using the RMSProp optimization algorithm.

    Args:
        alpha : learning rate.
        beta2 : RMSProp weight (decay rate for second moment).
        epsilon : small number to avoid division by zero.
        var : variable to be updated.
        grad : gradient of the variable.
        s : previous second moment (running average of squared gradients).

    Returns:
        updated_var : updated variable.
        new_s : new second moment.
    """
    s_new = beta2 * s + (1 - beta2) * grad ** 2
    var_new = var - alpha * grad / (np.sqrt(s_new) + epsilon)
    return var_new, s_new
