#!/usr/bin/env python3

poly_integral = __import__('17-integrate').poly_integral


print(poly_integral([]))               # [0]
print(poly_integral(['A']))               # [0]
print(poly_integral([0]))             # [0]
print(poly_integral([0, 0, 0]))       # [0]
print(poly_integral([1]))             # [0, 1]
print(poly_integral([2, 4, 6]))       # [0, 2, 2, 2]
print(poly_integral([5, 3, 0, 1]))    # [0, 5, 1.5, 0, 0.25]
