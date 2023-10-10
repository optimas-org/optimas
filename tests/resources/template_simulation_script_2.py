"""Dummy simulation template used for testing a ChainedEvaluator.

The template takes the result from a previous evaluation and uses
it to perform a followup evaluation.
"""

with open("result.txt", "r") as f:
    result_1 = float(f.read())

result_2 = 2 * result_1

with open("result_2.txt", "w") as f:
    f.write("%f" % result_2)
