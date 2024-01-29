"""Dummy simulation template used for testing.

The template takes two parameters x0 and x1 as input
and stores the result in `result.txt`.
"""

import os
import numpy as np

test_env_var = os.getenv("LIBE_TEST_SUB_ENV_VAR")

# 2D function with multiple minima
result = -({{x0}} + 10 * np.cos({{x0}})) * ({{x1}} + 5 * np.cos({{x1}}))

with open("result.txt", "w") as f:
    output = [str(result) + "\n"]
    if test_env_var is not None:
        output.append(test_env_var)
    f.writelines(output)
