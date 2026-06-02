#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

def line():
    """Plots y = x^3 as a solid red line with x from 0 to 10."""
    y = np.arange(0, 11) ** 3
    x = np.arange(0, 11)

    plt.figure(figsize=(6.4, 4.8))

    # Plot y as a solid red line
    plt.plot(x, y, 'r-')

    # Set x-axis limits from 0 to 10
    plt.xlim(0, 10)

    # Display the plot
    plt.show()
    