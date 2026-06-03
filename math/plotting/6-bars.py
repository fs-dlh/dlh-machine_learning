#!/usr/bin/env python3
""" Stacking Bars Graph. """
import numpy as np
import matplotlib.pyplot as plt


def bars():
    """Plots a stacked bar graph of fruit counts."""

    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4, 3))
    plt.figure(figsize=(6.4, 4.8))

    # your code here

    fruit_names = ['apples', 'bananas', 'oranges', 'peaches']
    colors = ['red', 'yellow', '#ff8000', '#ffe5b4']
    persons = ['Farrah', 'Fred', 'Felicia']
    x = np.arange(len(persons))
    width = 0.5
    bottom = np.zeros(len(persons))
    for i in range(len(fruit_names)):
        plt.bar(x, fruit[i], width=width, bottom=bottom,
                color=colors[i], label=fruit_names[i])
        bottom += fruit[i]
    plt.ylabel('Quantity of Fruit')
    plt.title('Number of Fruit per Person')
    plt.ylim(0, 80)
    plt.yticks(np.arange(0, 81, 10))
    plt.xticks(x, persons)
    plt.legend()
    plt.show()
