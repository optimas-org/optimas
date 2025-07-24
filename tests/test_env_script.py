import os

import numpy as np
from generator_standard.vocs import VOCS

from optimas.explorations import Exploration
from optimas.generators import RandomSamplingGenerator
from optimas.evaluators import TemplateEvaluator
from optimas.core import VaryingParameter, Objective, Parameter


def analysis_func(sim_dir, output_params):
    """Analysis function used by the template evaluator."""
    # Read back result from file
    with open("result.txt") as f:
        result = f.readlines()
        f = float(result[0])
        test_var = result[1]
    output_params["f"] = f
    output_params["test_var"] = test_var


def test_env_script():
    # Define variables and objectives.
    vocs = VOCS(
        variables={
            "x0": [-50.0, 5.0],
            "x1": [-5.0, 15.0]
        },
        objectives={"f": "MAXIMIZE"},
        observables={"test_var": "U10"}
    )

    # Define variables and objectives.
    gen = RandomSamplingGenerator(vocs=vocs)

    # Create template evaluator.
    ev = TemplateEvaluator(
        sim_template=os.path.join(
            os.path.abspath(os.path.dirname(__file__)),
            "resources",
            "template_simulation_script.py",
        ),
        analysis_func=analysis_func,
        env_script=os.path.join(
            os.path.abspath(os.path.dirname(__file__)),
            "resources",
            "env_script.sh",
        ),
    )

    # Create exploration.
    exploration = Exploration(
        generator=gen,
        evaluator=ev,
        max_evals=10,
        sim_workers=2,
        exploration_dir_path="./tests_output/test_env_script",
    )

    # Run exploration.
    exploration.run()

    assert np.all(exploration.history["test_var"] == "testvalue")


if __name__ == "__main__":
    test_env_script()
