import os

import numpy as np
import matplotlib.pyplot as plt

from optimas.explorations import Exploration
from optimas.generators import RandomSamplingGenerator
from optimas.evaluators import FunctionEvaluator
from optimas.core import VaryingParameter, Objective
from optimas.diagnostics import ExplorationDiagnostics


def eval_func(input_params, output_params):
    """Evaluation function used for testing"""
    x0 = input_params["x0"]
    x1 = input_params["x1"]
    result = -(x0 + 10 * np.cos(x0)) * (x1 + 5 * np.cos(x1))
    result2 = -(x0 + 10 * np.cos(x0 + 10)) * (x1 + 5 * np.cos(x1 - 5))
    output_params["f1"] = result
    output_params["f2"] = result2


def test_exploration_diagnostics():
    """Test the `ExplorationDiagnostics` class."""

    exploration_dir_path = "./tests_output/test_exploration_diagnostics"

    # Define variables and objectives.
    var1 = VaryingParameter("x0", -50.0, 5.0)
    var2 = VaryingParameter("x1", -5.0, 15.0)
    obj = Objective("f1", minimize=False)
    obj2 = Objective("f2", minimize=True)

    # Create generator.
    gen = RandomSamplingGenerator(
        varying_parameters=[var1, var2], objectives=[obj, obj2]
    )

    # Create function evaluator.
    ev = FunctionEvaluator(function=eval_func)

    # Create exploration.
    exploration = Exploration(
        generator=gen,
        evaluator=ev,
        max_evals=50,
        sim_workers=2,
        exploration_dir_path=exploration_dir_path,
    )

    # Run exploration.
    exploration.run()

    # Test diagnostics with both possible initializations.
    diags_from_file = ExplorationDiagnostics(exploration_dir_path)
    diags_from_instance = ExplorationDiagnostics(exploration)
    for diags in [diags_from_file, diags_from_instance]:
        for name in exploration.history:
            np.testing.assert_array_equal(
                diags.history[name], exploration.history[name]
            )

        for p_i, p_o in zip(gen.varying_parameters, diags.varying_parameters):
            assert p_i.model_dump_json() == p_o.model_dump_json()
        for p_i, p_o in zip(gen.objectives, diags.objectives):
            assert p_i.model_dump_json() == p_o.model_dump_json()
        for p_i, p_o in zip(gen.analyzed_parameters, diags.analyzed_parameters):
            assert p_i.model_dump_json() == p_o.model_dump_json()

        diags.plot_objective(show_trace=True)
        plt.savefig(os.path.join(exploration_dir_path, "optimization.png"))

        diags.plot_pareto_frontier(show_best_evaluation_indices=True)
        plt.savefig(os.path.join(exploration_dir_path, "pareto_frontier.png"))

        diags.plot_worker_timeline()
        plt.savefig(os.path.join(exploration_dir_path, "timeline.png"))

        # Check that all 3 possible objective inputs give the same result.
        _, trace1 = diags.get_objective_trace()
        _, trace2 = diags.get_objective_trace("f1")
        _, trace3 = diags.get_objective_trace(obj)
        np.testing.assert_array_equal(trace1, trace2)
        np.testing.assert_array_equal(trace1, trace3)

        # Test making plot using the diagnostics API.
        fig, ax = plt.subplots()
        vps = diags.varying_parameters
        df = diags.history
        f1 = diags.objectives[0]
        ax.axvline(vps[0].lower_bound)
        ax.axvline(vps[0].upper_bound)
        ax.set_xlabel(vps[0].name)
        ax.axhline(vps[1].lower_bound)
        ax.axhline(vps[1].upper_bound)
        ax.set_ylabel(vps[1].name)
        ax.scatter(df[vps[0].name], df[vps[1].name], c=df[f1.name])
        fig.savefig(os.path.join(exploration_dir_path, "search_space.png"))


if __name__ == "__main__":
    test_exploration_diagnostics()
