#!/usr/bin/env python3
""" 4-moving_average module """


def moving_average(data, beta):
    """ Calculates the weighted moving average with bias correction.

    Args:
        data : list of numerical data points.
        beta : weight for the moving average (0 < beta < 1).

    Returns:
        list: bias-corrected moving averages for each data point.
    """
    v = 0
    moving_avgs = []
    for t, x in enumerate(data, 1):
        v = beta * v + (1 - beta) * x
        v_corrected = v / (1 - beta ** t)
        moving_avgs.append(v_corrected)
    return moving_avgs
