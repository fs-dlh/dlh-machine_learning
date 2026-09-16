#!/usr/bin/env python3
""" Early stopping for gradient descent based on validation cost. """


def early_stopping(cost, opt_cost, threshold, patience, count):
    """ Determine if gradient descent should stop early.

    Args:
        cost : current validation cost.
        opt_cost : lowest recorded validation cost so far.
        threshold : minimum improvement required.
        patience : number of epochs to tolerate without sufficient
                        improvement.
        count : current count of epochs without sufficient improvement.

    Returns:
        tuple: (bool, int) – whether to stop early, and the updated count.
    """

    if cost < opt_cost - threshold:
        return False, 0
    else:
        count += 1

        if count >= patience:
            return True, count
        else:
            return False, count
