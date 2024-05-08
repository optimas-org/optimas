"""Simple template script used for demonstration.

The script evaluates an analytical expression and stores the results in a
`result.txt` file that is later read by the analysis function.
"""

import numpy as np

# 2D function with multiple minima
result = -({{x0}} + 10 * np.cos({{x0}})) * ({{x1}} + 5 * np.cos({{x1}}))

with open("result.txt", "w") as f:
    f.write("%f" % result)
