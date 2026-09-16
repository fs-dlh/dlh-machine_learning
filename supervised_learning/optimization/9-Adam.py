#!/usr/bin/env python3
""" 9-Adam module """
import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """ Updates a variable using the Adam optimization algorithm.

    Args:
        alpha : learning rate.
        beta1 : weight for first moment (momentum).
        beta2 : weight for second moment (RMSProp).
        epsilon : small number to avoid division by zero.
        var : variable to update.
        grad : gradient of the variable.
        v : previous first moment.
        s : previous second moment.
        t : time step (iteration number, starting from 1).

    Returns:
        updated_var : variable after update.
        new_v  : updated first moment.
        new_s : updated second moment.
    """

    v_new = beta1 * v + (1 - beta1) * grad
    s_new = beta2 * s + (1 - beta2) * (grad ** 2)

    v_corrected = v_new / (1 - beta1 ** t)
    s_corrected = s_new / (1 - beta2 ** t)

    var_new = var - alpha * v_corrected / (np.sqrt(s_corrected) + epsilon)

    return var_new, v_new, s_new
