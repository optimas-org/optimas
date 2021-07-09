"""
Dummy script for fast test ; this does not actually run fbpic,
but instead calculate the result of a test function and store
the results in a file result.txt
"""
import numpy as np
import time

# 2D function with multiple minima
result =  ( {{x0}} + 10*np.cos({{x0}}) + 4*np.sin({{x0}}) )

time.sleep( 4 )

with open('result.txt', 'w') as f:
    f.write("%f" %result)
