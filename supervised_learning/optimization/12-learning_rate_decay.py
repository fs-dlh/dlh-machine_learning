#!/usr/bin/env python3
""" 12-learning_rate_decay module """
import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, decay_step):
    """ Creates a learning rate decay operation using itd in a stepwise fashion

    Args:
        alpha : original learning rate.
        decay_rate  : weight determining how fast alpha decays.
        decay_step : number of gradient descent passes
                        before alpha is decayed further.

    Returns:
        the learning rate decay operation.
    """

    return tf.keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate=alpha,
        decay_steps=decay_step,
        decay_rate=decay_rate,
        staircase=True
    )
