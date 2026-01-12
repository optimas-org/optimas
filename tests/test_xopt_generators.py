"""Tests Optimas with Xopt ExpectedImprovementGenerator generator."""

import numpy as np

from optimas.generators import ExternalGenerator
from xopt.generators.bayesian.expected_improvement import (
    ExpectedImprovementGenerator,
)

from gest_api.vocs import VOCS
from optimas.evaluators import FunctionEvaluator
from optimas.explorations import Exploration


def xtest_callable(input_params, output_params):
    """Single-objective callable test function"""
    x1 = input_params["x1"]
    x2 = input_params["x2"]
    y1 = x2
    output_params["y1"] = y1


def test_xopt_EI():
    """Test xopt_EI with a 2D function."""
    initial_sample_size = 8

    # Create vocs
    vocs = VOCS(
        variables={"x1": [0, 1.0], "x2": [0, 10.0]},
        objectives={"y1": "MINIMIZE"},
    )

    gen = ExpectedImprovementGenerator(vocs=vocs)

    # Create 4 initial points and ingest them
    initial_points = [
        {"x1": 0.2, "x2": 2.0, "constant1": 1.0, "y1": 2.0, "c1": 0.2},
        {"x1": 0.5, "x2": 5.0, "constant1": 1.0, "y1": 5.0, "c1": 0.5},
        {"x1": 0.7, "x2": 7.0, "constant1": 1.0, "y1": 7.0, "c1": 0.7},
        {"x1": 0.9, "x2": 9.0, "constant1": 1.0, "y1": 9.0, "c1": 0.9},
    ]
    gen.ingest(initial_points)

    # Create generator.
    gen = ExternalGenerator(
        ext_gen=gen,
        vocs=vocs,
        save_model=True,
    )

    # Create evaluator.
    ev = FunctionEvaluator(function=xtest_callable)

    # Create exploration.
    exp = Exploration(
        generator=gen,
        evaluator=ev,
        max_evals=20,
        sim_workers=4,
        run_async=True,
        exploration_dir_path="./tests_output/test_xopt_EI",
    )

    # Run exploration
    exp.run()

    if exp.is_manager:
        H = exp.history
        print(f"Completed {len(H)} simulations")
        assert np.array_equal(H["y1"], H["x2"])


if __name__ == "__main__":
    test_xopt_EI()
