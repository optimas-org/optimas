"""
Simple template script that evaluates an analytical expression and stores
the results in a `result.txt` file.
"""
import numpy as np

# 2D function with multiple minima
result = -(
    ({{x0}} + 10 * np.cos({{x0}} + 0.1 * {{resolution}}))
    * ({{x1}} + 5 * np.cos({{x1}} - 0.2 * {{resolution}}))
)

with open("result.txt", "w") as f:
    f.write("%f" % result)
