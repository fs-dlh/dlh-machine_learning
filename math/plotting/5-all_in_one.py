#!/usr/bin/env python3
""" Plot All Graphs."""
import numpy as np
import matplotlib.pyplot as plt


def all_in_one():
    """Plots all graphs in one figure."""
    y0 = np.arange(0, 11) ** 3

    mean = [69, 0]
    cov = [[15, 8], [8, 15]]
    np.random.seed(5)
    x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
    y1 += 180

    x2 = np.arange(0, 28651, 5730)
    r2 = np.log(0.5)
    t2 = 5730
    y2 = np.exp((r2 / t2) * x2)

    x3 = np.arange(0, 21000, 1000)
    r3 = np.log(0.5)
    t31 = 5730
    t32 = 1600
    y31 = np.exp((r3 / t31) * x3)
    y32 = np.exp((r3 / t32) * x3)

    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)

    # your code here

    fig = plt.figure(figsize=(10, 8))
    grid = fig.add_gridspec(3, 2)

    """Plots y = x^3 as a solid red line with x from 0 to 10."""
    ax00 = fig.add_subplot(grid[0, 0])
    ax00.plot(np.arange(0, 11), y0, 'r-')
    ax00.set_xlabel('x', fontsize='x-small')
    ax00.tick_params(labelsize='x-small')   
    ax00.set_xticks(np.arange(0, 11, 2))
    ax00.set_yticks(np.arange(0, 1001, 500))

    """Plots a scatter plot of height vs weight for men."""
    ax01 = fig.add_subplot(grid[0, 1])
    ax01.scatter(x1, y1, c='m', s=10)
    ax01.set_title("Men's Height vs Weight", fontsize='x-small')
    ax01.set_xlabel('Height (in)', fontsize='x-small')
    ax01.set_ylabel('Weight (lbs)', fontsize='x-small')
    ax01.tick_params(labelsize='x-small')
    ax01.set_xticks(np.arange(60, 81, 10))
    ax01.set_yticks(np.arange(170, 191, 10))   

    """Plots a line graph of x vs y."""
    ax10 = fig.add_subplot(grid[1, 0])
    ax10.plot(x2, y2)
    ax10.set_title('Exponential Decay of C-14', fontsize='x-small')
    ax10.set_xlabel('Time (years)', fontsize='x-small')
    ax10.set_ylabel('Fraction Remaining', fontsize='x-small')
    ax10.set_yscale('log')
    ax10.tick_params(labelsize='x-small')
    ax10.set_xticks(np.arange(10000, 20001, 10000))    

    """ Plot x ↦ y1 and x ↦ y2 Line Graphs."""
    ax11 = fig.add_subplot(grid[1, 1])
    ax11.plot(x3, y31, 'r--', label='C-14')
    ax11.plot(x3, y32, 'g-', label='Ra-226')
    ax11.set_title('Exponential Decay of Radioactive Elements', fontsize='x-small')
    ax11.set_xlabel('Time (years)', fontsize='x-small')
    ax11.set_ylabel('Fraction Remaining', fontsize='x-small')
    ax11.legend(fontsize='x-small')
    ax11.tick_params(labelsize='x-small')
    ax11.set_xticks(np.arange(0, 20001, 5000))
    ax11.set_yticks(np.arange(0, 1.0001, 0.5))     

    """ Plot Histogram Graph."""
    ax2 = fig.add_subplot(grid[2, :])
    ax2.hist(student_grades, bins=10, edgecolor='black')
    ax2.set_title('Project A', fontsize='x-small')
    ax2.set_xlabel('Grades', fontsize='x-small')
    ax2.set_ylabel('Number of Students', fontsize='x-small')
    ax2.set_xlim(0, 100)
    ax2.set_ylim(0, 30)
    ax2.tick_params(labelsize='x-small')
    ax2.set_xticks(np.arange(0, 101, 10))
    ax2.set_yticks(np.arange(0, 31, 10))    
    
    fig.suptitle('All in One')
    plt.tight_layout()
    plt.show()