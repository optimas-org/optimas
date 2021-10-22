"""
Dummy script for fast test ; this does not actually run fbpic,
but instead calculate the result of a test function and store
the results in a file result.txt
"""
import numpy as np
import time

task = {{task}}

if task == 'expensive_model':
    resolution = 60
elif task == 'cheap_model':
    resolution = 1

# 2D function with multiple minima
result =  -( {{x0}} + 10*np.cos({{x0}} + 0.1*resolution) )*( {{x1}} + 5*np.cos({{x1}} - 0.2*resolution) )
time.sleep(resolution)

with open('result.txt', 'w') as f:
    f.write("%f" %result)
