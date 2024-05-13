"""Simple template script used for demonstration.

The script evaluates an analytical expression and stores the results in a
`result.txt` file that is later read by the analysis function.
"""

import numpy as np

# 2D function with multiple minima
# result = -({{x0}} + 10 * np.cos({{x0}})) * ({{x1}} + 5 * np.cos({{x1}}))

x1 = {{x0}}
x2 = {{x1}}

term1 = (4 - 2.1 * x1**2 + (x1**4) / 3) * x1**2
term2 = x1 * x2
term3 = (-4 + 4 * x2**2) * x2**2

result = term1 + term2 + term3

with open("result.txt", "w") as f:
    f.write("%f" % result)
