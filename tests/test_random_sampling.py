import numpy as np
from gest_api.vocs import VOCS

from optimas.explorations import Exploration
from optimas.generators import RandomSamplingGenerator
from optimas.evaluators import FunctionEvaluator


def eval_func(input_params, output_params):
    """Evaluation function for single-fidelity test"""
    x0 = input_params["x0"]
    x1 = input_params["x1"]
    result = -(x0 + 10 * np.cos(x0)) * (x1 + 5 * np.cos(x1))
    output_params["f"] = result


def test_uniform_sampling():
    """Test that random sampling with a uniform distribution generates the
    expected configurations."""

    # Set random seed for reproducible results.
    seed = 1

    # Create varying parameters.
    lower_bounds = [-3.0, 2.0]
    upper_bounds = [1.0, 5.0]

    # Set number of evaluations.
    n_evals = 10

    vocs = VOCS(
        variables={
            "x0": [lower_bounds[0], upper_bounds[0]],
            "x1": [lower_bounds[1], upper_bounds[1]],
        },
        objectives={"f": "MAXIMIZE"},
    )

    # Create generator and run exploration.
    gen = RandomSamplingGenerator(
        vocs=vocs,
        distribution="uniform",
        seed=1,
    )
    ev = FunctionEvaluator(function=eval_func)
    exploration = Exploration(
        generator=gen,
        evaluator=ev,
        max_evals=n_evals,
        sim_workers=2,
        exploration_dir_path="./tests_output/test_uniform_sampling",
    )
    exploration.run()

    # Get generated points.
    h = exploration.history
    h = h[h["sim_ended"]]
    x0_gen = h["x0"]
    x1_gen = h["x1"]

    # Generate expected points.
    rng = np.random.default_rng(seed=seed)
    configs = rng.uniform(
        lower_bounds, upper_bounds, (n_evals, len(lower_bounds))
    )
    x0_test = configs[:, 0]
    x1_test = configs[:, 1]

    # Check that the generated points are as expected.
    np.testing.assert_array_equal(x0_gen, x0_test)
    np.testing.assert_array_equal(x1_gen, x1_test)


def test_normal_sampling():
    """Test that random sampling with a normal distribution generates the
    expected configurations."""

    # Set random seed for reproducible results.
    seed = 1

    # Create varying parameters.
    center = [0.0, 0.0]
    sigma = [1.0, 5.0]

    # Set number of evaluations.
    n_evals = 10

    vocs = VOCS(
        variables={
            "x0": [center[0] - sigma[0], center[0] + sigma[0]],
            "x1": [center[1] - sigma[1], center[1] + sigma[1]],
        },
        objectives={"f": "MAXIMIZE"},
    )

    # Create generator and run exploration.
    gen = RandomSamplingGenerator(vocs=vocs, distribution="normal", seed=1)
    ev = FunctionEvaluator(function=eval_func)
    exploration = Exploration(
        generator=gen,
        evaluator=ev,
        max_evals=n_evals,
        sim_workers=2,
        exploration_dir_path="./tests_output/test_normal_sampling",
    )
    exploration.run()

    # Get generated points.
    h = exploration.history
    h = h[h["sim_ended"]]
    x0_gen = h["x0"]
    x1_gen = h["x1"]

    # Generate expected points.
    rng = np.random.default_rng(seed=seed)
    configs = rng.normal(center, sigma, (n_evals, len(center)))
    x0_test = configs[:, 0]
    x1_test = configs[:, 1]

    # Check that the generated points are as expected.
    np.testing.assert_array_equal(x0_gen, x0_test)
    np.testing.assert_array_equal(x1_gen, x1_test)


if __name__ == "__main__":
    test_uniform_sampling()
    test_normal_sampling()
