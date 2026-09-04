#!/usr/bin/env python3
""" 5-momentum module """


def update_variables_momentum(alpha, beta1, var, grad, v):
    """ Updates a variable using gradient descent with momentum.

    Args:
        alpha : learning rate.
        beta1 : momentum weight.
        var : variable to update.
        grad : gradient of the variable.
        v : previous first moment (velocity).

    Returns:
        updated_var : updated variable.
        new_v : new velocity (momentum).
    """
    v_new = beta1 * v + (1 - beta1) * grad
    var_new = var - alpha * v_new
    return var_new, v_new
