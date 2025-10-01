import numpy as np
from gest_api.vocs import VOCS

from optimas.explorations import Exploration
from optimas.generators import RandomSamplingGenerator
from optimas.evaluators import FunctionEvaluator


def eval_func(input_params, output_params):
    """Evaluation function used for testing"""
    x0 = input_params["x0"]
    x1 = input_params["x1"]
    result = -(x0 + 10 * np.cos(x0)) * (x1 + 5 * np.cos(x1))
    output_params["f"] = result


def test_libe_comms():
    """Test local and local_threading communications."""
    # Define variables and objectives.
    vocs = VOCS(
        variables={"x0": [-50.0, 5.0], "x1": [-5.0, 15.0]},
        objectives={"f": "MAXIMIZE"},
    )

    max_evals = 10

    for comm in ["local", "local_threading"]:
        # Create generator.
        gen = RandomSamplingGenerator(vocs=vocs)

        # Create function evaluator.
        ev = FunctionEvaluator(function=eval_func)

        # Create exploration.
        exploration = Exploration(
            generator=gen,
            evaluator=ev,
            max_evals=max_evals,
            sim_workers=2,
            exploration_dir_path=f"./tests_output/test_comms_{comm}",
            libe_comms=comm,
        )

        # Run exploration.
        exploration.run()

        # Check that all trials were evaluated.
        assert np.all(exploration.history["f"] != 0.0)
        assert len(exploration.history) == max_evals


if __name__ == "__main__":
    test_libe_comms()
