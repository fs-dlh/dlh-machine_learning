#!/usr/bin/env python3
""" 11-learning_rate_decay module """


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """ Updates the learning rate using itd in a stepwise fashion.

    Args:
        alpha : original learning rate.
        decay_rate : weight determining how fast alpha decays.
        global_step : number of gradient descent passes elapsed.
        decay_step : number of passes before alpha is decayed further.

    Returns:
        alphaU : updated learning rate after applying stepwise decay.
    """
    alphaU = alpha / (1 + decay_rate * (global_step // decay_step))

    return alphaU
