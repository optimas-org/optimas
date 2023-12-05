import numpy as np

from optimas.explorations import Exploration
from optimas.generators import GridSamplingGenerator
from optimas.evaluators import FunctionEvaluator
from optimas.core import VaryingParameter, Objective


def eval_func(input_params, output_params):
    """Evaluation function for single-fidelity test"""
    x0 = input_params["x0"]
    x1 = input_params["x1"]
    result = -(x0 + 10 * np.cos(x0)) * (x1 + 5 * np.cos(x1))
    output_params["f"] = result


def test_grid_sampling():
    """Test that grid sampling generates the expected configurations."""

    # Create varying parameters.
    names = ["x0", "x1"]
    lower_bounds = [-3.0, 2.0]
    upper_bounds = [1.0, 5.0]
    vars = []
    n_steps = [7, 15]
    for name, lb, ub in zip(names, lower_bounds, upper_bounds):
        vars.append(VaryingParameter(name, lb, ub))

    # Set number of evaluations.
    n_evals = np.prod(n_steps)

    # Define objective.
    obj = Objective("f", minimize=False)

    # Create generator and run exploration.
    gen = GridSamplingGenerator(
        varying_parameters=vars, objectives=[obj], n_steps=n_steps
    )
    ev = FunctionEvaluator(function=eval_func)
    exploration = Exploration(
        generator=gen,
        evaluator=ev,
        max_evals=n_evals,
        sim_workers=2,
        exploration_dir_path="./tests_output/test_grid_sampling",
    )
    exploration.run()

    # Get generated points.
    h = exploration.history
    h = h[h["sim_ended"]]
    x0_gen = h["x0"]
    x1_gen = h["x1"]

    # Get expected 1D steps along each variable.
    x0_steps = np.linspace(lower_bounds[0], upper_bounds[0], n_steps[0])
    x1_steps = np.linspace(lower_bounds[1], upper_bounds[1], n_steps[1])

    # Check that the scan along each variable is as expected.
    np.testing.assert_array_equal(np.unique(x0_gen), x0_steps)
    np.testing.assert_array_equal(np.unique(x1_gen), x1_steps)

    # Check that for every x0 step, the expected x1 steps are performed.
    for x0_step in x0_steps:
        x1_in_x0_step = x1_gen[x0_gen == x0_step]
        np.testing.assert_array_equal(x1_in_x0_step, x1_steps)


if __name__ == "__main__":
    test_grid_sampling()
