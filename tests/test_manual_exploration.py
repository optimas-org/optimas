import numpy as np
import pandas as pd

from optimas.explorations import Exploration
from optimas.generators import RandomSamplingGenerator
from optimas.evaluators import FunctionEvaluator
from optimas.core import VaryingParameter, Objective, Parameter


def eval_func(input_params, output_params):
    """Evaluation function for single-fidelity test"""
    x0 = input_params["x0"]
    x1 = input_params["x1"]
    result = -(x0 + 10 * np.cos(x0)) * (x1 + 5 * np.cos(x1))
    output_params["f"] = result
    output_params["par1"] = 1.0


def test_manual_exploration():
    """Tests methods for manually attaching trials and evaluations."""

    # Create varying parameters.
    names = ["x0", "x1"]
    lower_bounds = [-3.0, 2.0]
    upper_bounds = [1.0, 5.0]
    vars = []
    for name, lb, ub in zip(names, lower_bounds, upper_bounds):
        vars.append(VaryingParameter(name, lb, ub))
    par1 = Parameter("par1")

    # Set number of evaluations.
    n_evals = 10
    n_evals_substep = 3

    # Define objective.
    obj = Objective("f", minimize=False)

    # Trials to attach and evaluate manually.
    trials_to_attach = {"x0": [-2.3, 1.0, 0.5], "x1": [2.4, 1.0, 3.0]}

    # Dummy evaluation data to attach manually.
    evals_to_attach = {
        "x0": [-2.0, 1.3, 0.5],
        "x1": [2.7, 1.0, 3.5],
        "f": [2.0, 4.0, 6.0],
        "par1": [1.0, 1.0, 1.0],
    }
    n_attached_trials = len(trials_to_attach["x0"])
    n_attached_evals = len(evals_to_attach["x0"])

    test_trials = [trials_to_attach, *convert_dict(trials_to_attach)]
    test_evals = [evals_to_attach, *convert_dict(evals_to_attach)]

    # Test attaching trials with all supported formats.
    for i, (trials, evals) in enumerate(zip(test_trials, test_evals)):
        # Create generator and run exploration.
        gen = RandomSamplingGenerator(
            varying_parameters=vars,
            objectives=[obj],
            analyzed_parameters=[par1],
            distribution="uniform",
            seed=1,
        )
        ev = FunctionEvaluator(function=eval_func)
        exploration = Exploration(
            generator=gen,
            evaluator=ev,
            max_evals=n_evals + n_attached_trials,
            sim_workers=2,
            exploration_dir_path=f"./tests_output/test_manual_exploration_{i}",
            run_async=False,
        )

        # Attach evaluations
        exploration.attach_evaluations(evals)
        assert len(exploration.history) == n_attached_evals
        assert gen.n_completed_trials == n_attached_evals
        for param, vals in evals_to_attach.items():
            np.testing.assert_array_equal(exploration.history[param], vals)

        # Evaluate manually the trials and check that the history is correct.
        exploration.evaluate_trials(trials)
        assert len(exploration.history) == n_attached_trials + n_attached_evals
        i_start = n_attached_evals
        i_end = i_start + n_attached_trials
        history_attached_trials = exploration.history[i_start:i_end]
        for param, vals in trials_to_attach.items():
            np.testing.assert_array_equal(history_attached_trials[param], vals)

        # Run an intermediate step using the generator.
        exploration.run(n_evals_substep)

        # Attach trials and run the exploration until completion.
        exploration.attach_trials(trials)
        exploration.run()

        # Check that the final history has the expected size.
        assert (
            len(exploration.history)
            == n_evals + n_attached_trials + n_attached_evals
        )

        # Check that the trials attached after the intermediate step were
        # evaluates in the correct order.
        i_start = n_attached_evals + n_attached_trials + n_evals_substep
        i_end = i_start + n_attached_trials
        history_attached_trials = exploration.history[i_start:i_end]
        assert len(history_attached_trials) == n_attached_trials
        for param, vals in trials_to_attach.items():
            np.testing.assert_array_equal(history_attached_trials[param], vals)


def convert_dict(input_dict):
    """Convert dictionary to list, numpy array and pandas dataframe."""
    output_list = [dict(zip(input_dict, t)) for t in zip(*input_dict.values())]

    dtype = [(key, type(vals[0])) for key, vals in input_dict.items()]
    n_attached_trials = len(input_dict["x0"])
    output_array = np.zeros(n_attached_trials, dtype=dtype)
    for key, vals in input_dict.items():
        output_array[key] = vals

    output_df = pd.DataFrame(input_dict)
    return output_list, output_array, output_df


if __name__ == "__main__":
    test_manual_exploration()
