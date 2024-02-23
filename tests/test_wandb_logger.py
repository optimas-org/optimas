import os
# os.environ['WANDB_START_METHOD'] = 'thread'
# os.environ['WANDB_DISABLE_SERVICE'] = "1"
# API_KEY = "4b810c5f384e91c204b67371249db1a6b1e1aca4"

import wandb
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

from optimas.explorations import Exploration
from optimas.generators import RandomSamplingGenerator
from optimas.evaluators import FunctionEvaluator
from optimas.core import VaryingParameter, Objective, Parameter
from optimas.loggers import WandBLogger


def eval_func(input_params, output_params):
    """Evaluation function used for testing"""
    x0 = input_params["x0"]
    x1 = input_params["x1"]
    result = -(x0 + 10 * np.cos(x0)) * (x1 + 5 * np.cos(x1))
    output_params["f"] = result
    output_params["p0"] = np.array([[1, 2, 3, 4], [2, 6, 7, 4]])
    
    plt.figure()
    plt.plot(output_params["p0"][0], output_params["p0"][1])
    output_params["fig"] = deepcopy(plt.gcf())
    plt.figure()
    plt.imshow(output_params["p0"])
    output_params["p1"] = deepcopy(plt.gcf())


def user_function(last_trial, generator: RandomSamplingGenerator):    
    all_trials = generator.completed_trials
    n_trials = len(all_trials)
    shape_1 = np.array(all_trials[0].data["p0"]).shape[1]
    history = np.zeros((n_trials, shape_1))
    for i, trial in enumerate(all_trials):
        history[i] = np.array(trial.data["p0"]).sum(axis=0)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.imshow(history.T, aspect="auto")
    return {"history": wandb.Image(fig)}


def test_wandb_logger():
    """Test that an exploration with a Weights and Biases logger."""
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
        sim_workers=1,
        exploration_dir_path="./tests_output/test_wandb_logger",
        # libe_comms="threads",
        logger=WandBLogger(
            api_key="API_KEY",
            project="project_name",
            run_name="run_01",
            data_types={
                "p0": {"type": wandb.Histogram, "type_kwargs": {}},
            },
            user_function=user_function,
        ),
    )

    exploration.attach_evaluations(
        {
            "x0": [1.0],
            "x1": [2.0],
            "f": [0.],
            "p0": [np.array([[1, 2, 3, 4], [2, 6, 7, 4]])],
            "p1": [plt.figure()],
            "fig": [plt.figure()],
        }
    )

    # Run exploration.
    exploration.run(3)
    exploration.run()

    # Check that the multidimensional analyzed parameters worked as expected.
    # for p0_data in exploration.history["p0"]:
    #     np.testing.assert_array_equal(
    #         np.array(p0_data), np.array([[1, 2, 3, 4], [2, 6, 7, 4]])
    #     )
    # for p1_data in exploration.history["p1"]:
    #     np.testing.assert_array_equal(
    #         np.array(p1_data), np.array([[1, 2, 3, 4], [2, 6, 7, 4]])
    #     )
    # for i, fig in enumerate(exploration.history["fig"]):
    #     print(threading.current_thread().name)
    #     fig = deepcopy(fig)
    #     fig.savefig(
    #         os.path.join(exploration.exploration_dir_path, f"test_fig_{i}.png")
    #     )


if __name__ == "__main__":
    test_wandb_logger()