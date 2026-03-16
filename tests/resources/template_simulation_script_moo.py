"""Dummy simulation template used for MOO testing.

The template takes two parameters x0 and x1 as input
and stores the result in `f1.txt` and `f2.txt`.
"""

import numpy as np

# 2D function with multiple minima
f1 = -({{x0}} + 10 * np.cos({{x0}})) * ({{x1}} + 5 * np.cos({{x1}}))
f2 = -({{x0}} + 10 * np.cos({{x0}} + 10)) * ({{x1}} + 5 * np.cos({{x1}} - 5))

with open("f1.txt", "w") as f:
    f.write("%f" % f1)

with open("f2.txt", "w") as f:
    f.write("%f" % f2)
