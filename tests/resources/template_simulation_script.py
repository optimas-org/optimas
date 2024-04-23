"""Dummy simulation template used for testing.

The template takes two parameters x0 and x1 as input
and stores the result in `result.txt`.
"""

import os
import time
import numpy as np

test_env_var = os.getenv("LIBE_TEST_SUB_ENV_VAR")
sleep = os.getenv("OPTIMAS_TEST_SLEEP")

if sleep is not None:
    time.sleep(float(sleep))

# 2D function with multiple minima
result = -({{x0}} + 10 * np.cos({{x0}})) * ({{x1}} + 5 * np.cos({{x1}}))

with open("result.txt", "w") as f:
    output = [str(result) + "\n"]
    if test_env_var is not None:
        output.append(test_env_var)
    f.writelines(output)

with open("cuda_visible_devices.txt", "w") as f:
    f.write(os.getenv("CUDA_VISIBLE_DEVICES"))
