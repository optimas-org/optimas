"""Tests Optimas with Xopt ExpectedImprovementGenerator generator."""

import numpy as np

from optimas.generators import ExternalGenerator
from xopt.generators.bayesian.expected_improvement import (
    ExpectedImprovementGenerator,
)
from xopt.generators.sequential.neldermead import NelderMeadGenerator

from gest_api.vocs import VOCS
from optimas.evaluators import FunctionEvaluator
from optimas.explorations import Exploration


def xtest(input_params, output_params):
    """Single-objective callable test function"""
    x1 = input_params["x1"]
    x2 = input_params["x2"]
    y1 = x2
    output_params["y1"] = y1


def rosenbrock(input_params, output_params):
    """2D Rosenbrock function"""
    x1 = input_params["x1"]
    x2 = input_params["x2"]
    y1 = 100 * (x2 - x1**2) ** 2 + (1 - x1) ** 2
    output_params["y1"] = y1


def test_xopt_EI():
    """Test xopt ExpectedImprovementGenerator with a 2D function."""
    initial_sample_size = 8

    # Create vocs
    vocs = VOCS(
        variables={"x1": [0, 1.0], "x2": [0, 10.0]},
        objectives={"y1": "MINIMIZE"},
    )

    gen = ExpectedImprovementGenerator(vocs=vocs)

    # Create 4 initial points and ingest them
    initial_points = [
        {"x1": 0.2, "x2": 2.0, "y1": 2.0},
        {"x1": 0.5, "x2": 5.0, "y1": 5.0},
        {"x1": 0.7, "x2": 7.0, "y1": 7.0},
        {"x1": 0.9, "x2": 9.0, "y1": 9.0},
    ]
    gen.ingest(initial_points)

    # Create generator.
    gen = ExternalGenerator(
        ext_gen=gen,
        vocs=vocs,
        save_model=True,
    )

    # Create evaluator.
    ev = FunctionEvaluator(function=xtest)

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


def test_xopt_neldermead():
    """Test xopt NelderMeadGenerator with a 2D function."""
    initial_sample_size = 8

    # Create vocs
    vocs = VOCS(
        variables={"x1": [0, 1.0], "x2": [0, 10.0]},
        objectives={"y1": "MINIMIZE"},
    )

    gen = NelderMeadGenerator(vocs=vocs)

    # Create 4 initial points and ingest them
    initial_points = [
        {"x1": -1.2, "x2": 1.0, "y1": 24.2},
        {"x1": -1.0, "x2": 1.0, "y1": 4.0},
        {"x1": -0.8, "x2": 0.8, "y1": 5.8},
    ]
    gen.ingest(initial_points)

    # Create generator.
    gen = ExternalGenerator(
        ext_gen=gen,
        vocs=vocs,
        save_model=True,
    )

    # Create evaluator.
    ev = FunctionEvaluator(function=rosenbrock)

    # Create exploration.
    exp = Exploration(
        generator=gen,
        evaluator=ev,
        max_evals=30,
        sim_workers=1,
        run_async=True,
        exploration_dir_path="./tests_output/test_xopt_neldermead",
    )

    # Run exploration
    exp.run()

    if exp.is_manager:
        H = exp.history
        print(f"Completed {len(H)} simulations")
        initial_value = H["y1"][0]
        best_value = H["y1"][np.argmin(H["y1"])]
        assert best_value <= initial_value


if __name__ == "__main__":
    test_xopt_EI()
    test_xopt_neldermead()
