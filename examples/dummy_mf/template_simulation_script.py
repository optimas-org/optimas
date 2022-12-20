"""
Dummy script for fast test; this does not actually run a simulation,
but instead calculate the result of a test function and store
the results in a file result.txt
"""
import numpy as np

# 2D function with multiple minima
result =  -( {{x0}} + 10*np.cos({{x0}} + 0.1*{{resolution}}) )*( {{x1}} + 5*np.cos({{x1}} - 0.2*{{resolution}}) )

with open('result.txt', 'w') as f:
    f.write("%f" %result)
