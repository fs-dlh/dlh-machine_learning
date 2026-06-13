#!/usr/bin/env python3
"""
The subplot thickens
"""
import numpy as np
import matplotlib.pyplot as plt


def all_in_one():
    """5 graphs in one"""
    # 0-line
    y0 = np.arange(0, 11) ** 3
    # 1-scatter
    mean = [69, 0]
    cov = [[15, 8], [8, 15]]
    np.random.seed(5)
    x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
    y1 += 180

    # change scale
    x2 = np.arange(0, 28651, 5730)
    r2 = np.log(0.5)
    t2 = 5730
    y2 = np.exp((r2 / t2) * x2)

    # 3 two
    x3 = np.arange(0, 21000, 1000)
    r3 = np.log(0.5)
    t31 = 5730
    t32 = 1600
    y31 = np.exp((r3 / t31) * x3)
    y32 = np.exp((r3 / t32) * x3)

    # 4 frequency
    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)

    # your code here
    plt.figure(figsize=(6.4, 4.8))
    plt.title("All in One")

    plt.clf()  # clears anything that might have leaked

    # 0-line
    plt.subplot(3, 2, 1)  # let's see how important it is to have sth in the ()
    plt.plot(np.arange(0, 11), y0, color='red')
    plt.xlim(0, 10)

    # 1 -scatter
    plt.subplot(3, 2, 2)
    plt.scatter(x1, y1, color='magenta')
    plt.title("Men's Height vs Weight", fontsize='x-small')
    plt.xlabel('Height (in)', fontsize='x-small')
    plt.ylabel('Weight (lbs)', fontsize='x-small')

    # 2- change_scale
    plt.subplot(3, 2, 3)  # (?)
    plt.plot(x2, y2)
    plt.title("Exponential Decay of C-14", fontsize='x-small')
    plt.xlabel("Time (years)", fontsize='x-small')
    plt.ylabel("Fraction Remaining", fontsize='x-small')
    plt.gca().autoscale(enable=True, axis='x', tight=True)
    plt.yscale('log')  # log scaling of the y achsis

    # 3-two
    plt.subplot(3, 2, 4)  # (?)
    plt.plot(x3, y31, 'r--', label='C-14')
    plt.plot(x3, y32, 'g-', label='Ra-226')
    plt.title('Exponential Decay of Radioactive Elements',
              fontsize='x-small')
    plt.xlabel('Time (years)', fontsize='x-small')
    plt.ylabel('Fraction Remaining', fontsize='x-small')
    plt.legend(loc='upper right')
    plt.xlim(0, 20000)
    plt.ylim(0, 1)

    # 4 - frequency
    plt.subplot(3, 2, (5, 6))
    bin = np.arange(0, 101, 10)
    n, bins, patches = plt.hist(
        student_grades, bins=bin, edgecolor='black'
        )
    plt.xlabel('Grades', fontsize='x-small')
    plt.ylabel('Number of Students', fontsize='x-small')
    plt.title('Project A', fontsize='x-small')
    plt.xlim(0, 100)
    plt.ylim(0, 30)
    plt.xticks(bin)

    plt.tight_layout()  # tightens the subplots
    # show them all
    plt.show()