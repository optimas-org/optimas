import os

from gest_api.vocs import VOCS
from optimas.explorations import Exploration
from optimas.generators import RandomSamplingGenerator
from optimas.evaluators import FunctionEvaluator


def eval_func(input_params, output_params):
    raise ValueError("Exception to break exploration")


def test_exception_during_exploration_run():
    """Test that the Exploration handles exceptions during the run correctly.

    When using `create_evaluation_dirs=True`, the current working directory
    will change during exploration and should be restored when `.run` finishes,
    even if an exception occurs.
    """
    # Define variables and objectives.
    vocs = VOCS(
        variables={"x0": [-50.0, 5.0], "x1": [-5.0, 15.0]},
        objectives={"f": "MAXIMIZE"},
    )

    # Create generator.
    gen = RandomSamplingGenerator(vocs=vocs)

    # Create function evaluator.
    ev = FunctionEvaluator(function=eval_func, create_evaluation_dirs=True)

    # Create exploration.
    exploration = Exploration(
        generator=gen,
        evaluator=ev,
        max_evals=10,
        sim_workers=2,
        exploration_dir_path="./tests_output/test_exception_during_run",
    )

    cwd = os.getcwd()

    # Run exploration without raising an exception.
    exploration.run()

    # Check that the cwd remains unchanged after a failed run.
    assert os.getcwd() == cwd


if __name__ == "__main__":
    test_exception_during_exploration_run()
