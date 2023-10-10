import re

import numpy as np
import pytest

from optimas.explorations import Exploration
from optimas.generators import LineSamplingGenerator
from optimas.evaluators import FunctionEvaluator
from optimas.core import VaryingParameter, Objective


def eval_func(input_params, output_params):
    """Evaluation function for single-fidelity test"""
    x0 = input_params["x0"]
    x1 = input_params["x1"]
    result = -(x0 + 10 * np.cos(x0)) * (x1 + 5 * np.cos(x1))
    output_params["f"] = result


def test_line_sampling():
    """Test that line sampling generates the expected configurations."""

    # Create varying parameters.
    names = ["x0", "x1"]
    lower_bounds = [-3.0, 2.0]
    upper_bounds = [1.0, 5.0]
    defaults = [0, 0]
    n_steps = [7, 15]
    vars = []
    for name, lb, ub, dv in zip(names, lower_bounds, upper_bounds, defaults):
        vars.append(VaryingParameter(name, lb, ub, default_value=dv))

    # Set number of evaluations.
    n_evals = np.sum(n_steps)

    # Define objective.
    obj = Objective("f", minimize=False)

    # Create generator and run exploration.
    gen = LineSamplingGenerator(
        varying_parameters=vars, objectives=[obj], n_steps=n_steps
    )
    ev = FunctionEvaluator(function=eval_func)
    exploration = Exploration(
        generator=gen,
        evaluator=ev,
        max_evals=n_evals,
        sim_workers=2,
        exploration_dir_path="./tests_output/test_line_sampling",
    )
    exploration.run()

    # Get generated points.
    h = exploration.history
    h = h[h["sim_ended"]]
    x0_gen = h["x0"]
    x1_gen = h["x1"]

    # Check that the amount of evaluations with default values is as expected.
    assert len(x0_gen[x0_gen == defaults[0]]) == n_steps[1]
    assert len(x1_gen[x1_gen == defaults[1]]) == n_steps[0]

    # Check that the line scans along each direction are as expected.
    np.testing.assert_array_equal(
        x0_gen[: n_steps[0]],
        np.linspace(lower_bounds[0], upper_bounds[0], n_steps[0]),
    )
    np.testing.assert_array_equal(
        x1_gen[n_steps[0] :],
        np.linspace(lower_bounds[1], upper_bounds[1], n_steps[1]),
    )


def test_line_sampling_errors():
    """Test that the line sampling raises the correct exceptions."""

    # Create varying parameters with missing default value.
    var1 = VaryingParameter("x0", -3, 1)
    var2 = VaryingParameter("x0", -3, 1)

    # Define objective.
    obj = Objective("f", minimize=False)

    # Check that an exception is raised when default values are missing.
    with pytest.raises(
        AssertionError, match="Parameter x0 does not have a default value."
    ):
        gen = LineSamplingGenerator(
            varying_parameters=[var1, var2], objectives=[obj], n_steps=[3, 5]
        )

    # Create varying parameters.
    var1 = VaryingParameter("x0", -3, 1, default_value=0.0)
    var2 = VaryingParameter("x0", -3, 1, default_value=0.0)

    # Define objective.
    obj = Objective("f", minimize=False)

    # Check that an exception is raised when n_steps is not correct.
    with pytest.raises(
        AssertionError,
        match=re.escape(
            "Length of `n_steps` (1) and"
            " `varying_parameters` (2) do not match."
        ),
    ):
        gen = LineSamplingGenerator(
            varying_parameters=[var1, var2], objectives=[obj], n_steps=[3]
        )


if __name__ == "__main__":
    test_line_sampling()
    test_line_sampling_errors()
