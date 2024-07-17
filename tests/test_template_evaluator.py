import os

import numpy as np
import matplotlib.pyplot as plt

from optimas.explorations import Exploration
from optimas.generators import RandomSamplingGenerator
from optimas.evaluators import TemplateEvaluator
from optimas.core import VaryingParameter, Objective, Parameter


def analysis_func(sim_dir, output_params):
    """Analysis function used by the template evaluator."""
    # Read back result from file
    with open("result.txt") as f:
        result = float(f.read())
    output_params["f"] = result
    output_params["p0"] = np.array([[1, 2, 3, 4], [2, 6, 7, 4]])
    output_params["p1"] = np.array([[1, 2, 3, 4], [2, 6, 7, 4]])
    plt.figure()
    plt.plot(output_params["p1"][0], output_params["p1"][1])
    output_params["fig"] = plt.gcf()


def test_template_evaluator():
    # Define variables and objectives.
    var1 = VaryingParameter("x0", -50.0, 5.0)
    var2 = VaryingParameter("x1", -5.0, 15.0)
    obj = Objective("f", minimize=False)
    # Test also more complex analyzed parameters.
    p0 = Parameter("p0", dtype=(float, (2, 4)))
    p1 = Parameter("p1", dtype="O")
    p2 = Parameter("fig", dtype="O")

    # Define variables and objectives.
    gen = RandomSamplingGenerator(
        varying_parameters=[var1, var2],
        objectives=[obj],
        analyzed_parameters=[p0, p1, p2],
    )

    # Create template evaluator.
    ev = TemplateEvaluator(
        sim_template=os.path.join(
            os.path.abspath(os.path.dirname(__file__)),
            "resources",
            "template_simulation_script.py",
        ),
        analysis_func=analysis_func,
    )

    # Create exploration.
    exploration = Exploration(
        generator=gen,
        evaluator=ev,
        max_evals=10,
        sim_workers=2,
        exploration_dir_path="./tests_output/test_template_evaluator",
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


def test_template_evaluator_timeout():
    """Test the evaluation timeout.

    All evaluations will sleep for 20 seconds. This should trigger the 1
    second timeout set in the `TemplateEvaluator`, causing all evaluations
    to fail.
    """
    # Make evaluations sleep for 20 seconds.
    os.environ["OPTIMAS_TEST_SLEEP"] = "20"

    # Define variables and objectives.
    var1 = VaryingParameter("x0", -50.0, 5.0)
    var2 = VaryingParameter("x1", -5.0, 15.0)
    obj = Objective("f", minimize=False)

    # Define variables and objectives.
    gen = RandomSamplingGenerator(
        varying_parameters=[var1, var2],
        objectives=[obj],
    )

    # Create template evaluator with 1s timeout.
    ev = TemplateEvaluator(
        sim_template=os.path.join(
            os.path.abspath(os.path.dirname(__file__)),
            "resources",
            "template_simulation_script.py",
        ),
        timeout=1,
    )

    # Create exploration.
    exploration = Exploration(
        generator=gen,
        evaluator=ev,
        max_evals=10,
        sim_workers=2,
        exploration_dir_path="./tests_output/test_template_evaluator_timeout",
    )

    # Run exploration.
    exploration.run()

    # Check that no evaluations were successful.
    np.testing.assert_array_equal(exploration.history.trial_status, "FAILED")
    np.testing.assert_array_equal(exploration.history.f, np.nan)


if __name__ == "__main__":
    test_template_evaluator()
    test_template_evaluator_timeout()
