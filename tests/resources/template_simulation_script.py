"""
Dummy simulation template used for testing. It takes x0 and x1 as input
parameters and stores the result in `result.txt`.
"""
import numpy as np

# 2D function with multiple minima
result =  -( {{x0}} + 10*np.cos({{x0}}) )*( {{x1}} + 5*np.cos({{x1}}) )

with open('result.txt', 'w') as f:
    f.write("%f" %result)
