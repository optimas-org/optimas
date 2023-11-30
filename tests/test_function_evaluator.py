import os

import numpy as np
import matplotlib.pyplot as plt

from optimas.explorations import Exploration
from optimas.generators import RandomSamplingGenerator
from optimas.evaluators import FunctionEvaluator
from optimas.core import VaryingParameter, Objective, Parameter


def eval_func(input_params, output_params):
    """Evaluation function used for testing"""
    x0 = input_params["x0"]
    x1 = input_params["x1"]
    result = -(x0 + 10 * np.cos(x0)) * (x1 + 5 * np.cos(x1))
    output_params["f"] = result
    output_params["p0"] = np.array([[1, 2, 3, 4], [2, 6, 7, 4]])
    output_params["p1"] = np.array([[1, 2, 3, 4], [2, 6, 7, 4]])
    plt.figure()
    plt.plot(output_params["p1"][0], output_params["p1"][1])
    output_params["fig"] = plt.gcf()


def test_function_evaluator():
    """Test that an exploration runs successfully with a function evaluator."""
    # Define variables and objectives.
    var1 = VaryingParameter("x0", -50.0, 5.0)
    var2 = VaryingParameter("x1", -5.0, 15.0)
    obj = Objective("f", minimize=False)
    # Test also more complex analyzed parameters.
    p0 = Parameter("p0", dtype=(float, (2, 4)))
    p1 = Parameter("p1", dtype="O")
    p2 = Parameter("fig", dtype="O")

    # Create generator.
    gen = RandomSamplingGenerator(
        varying_parameters=[var1, var2],
        objectives=[obj],
        analyzed_parameters=[p0, p1, p2],
    )

    # Create function evaluator.
    ev = FunctionEvaluator(function=eval_func)

    # Create exploration.
    exploration = Exploration(
        generator=gen,
        evaluator=ev,
        max_evals=10,
        sim_workers=2,
        exploration_dir_path="./tests_output/test_function_evaluator",
    )

    # Run exploration.
    exploration.run()

    # Check that the multidimensional analyzed parameters worked as expected.
    for p0_data in exploration.history["p0"]:
        np.testing.assert_array_equal(
            np.array(p0_data), np.array([[1, 2, 3, 4], [2, 6, 7, 4]])
        )
    for p1_data in exploration.history["p1"]:
        np.testing.assert_array_equal(
            np.array(p1_data), np.array([[1, 2, 3, 4], [2, 6, 7, 4]])
        )
    for i, fig in enumerate(exploration.history["fig"]):
        fig.savefig(
            os.path.join(exploration.exploration_dir_path, f"test_fig_{i}.png")
        )


if __name__ == "__main__":
    test_function_evaluator()
