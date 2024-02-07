import os

import numpy as np
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import pytest

from optimas.explorations import Exploration
from optimas.generators import RandomSamplingGenerator
from optimas.evaluators import TemplateEvaluator
from optimas.core import VaryingParameter, Objective
from optimas.diagnostics import ExplorationDiagnostics


def analysis_func(sim_dir, output_params):
    """Analysis function used by the template evaluator."""
    # Read back result from file
    with open("f1.txt") as f:
        f1 = float(f.read())
    with open("f2.txt") as f:
        f2 = float(f.read())
    output_params["f1"] = f1
    output_params["f2"] = f2


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
        varying_parameters=[var1, var2], objectives=[obj, obj2], seed=0
    )

    # Create template evaluator.
    ev = TemplateEvaluator(
        sim_template=os.path.join(
            os.path.abspath(os.path.dirname(__file__)),
            "resources",
            "template_simulation_script_moo.py",
        ),
        analysis_func=analysis_func,
    )

    # Create exploration.
    exploration = Exploration(
        generator=gen,
        evaluator=ev,
        max_evals=50,
        sim_workers=2,
        exploration_dir_path=exploration_dir_path,
    )

    # Run exploration.
    exploration.run(15)
    # Attach an external evaluation to test later the evaluation folders.
    exploration.attach_evaluations(
        [{"x0": -10.0, "x1": 0.0, "f1": 10.0, "f2": 7.0}]
    )
    exploration.run()

    # Test diagnostics with both possible initializations.
    for i, diags_source in enumerate([exploration_dir_path, exploration]):
        diags = ExplorationDiagnostics(diags_source)
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

        diags.plot_pareto_front(show_best_evaluation_indices=True)
        plt.savefig(os.path.join(exploration_dir_path, "pareto_front.png"))

        diags.plot_worker_timeline()
        plt.savefig(os.path.join(exploration_dir_path, "timeline.png"))

        diags.plot_history(top=5, show_legend=True)
        plt.savefig(os.path.join(exploration_dir_path, "history.png"))

        fig = plt.figure(figsize=(8, 5))
        gs = GridSpec(2, 2, wspace=0.4, hspace=0.3, top=0.95, right=0.95)
        diags.plot_history(top=10, show_legend=True, subplot_spec=gs[:, 0])
        diags.plot_pareto_front(
            show_best_evaluation_indices=True,
            show_legend=True,
            subplot_spec=gs[0, 1],
        )
        diags.plot_worker_timeline(subplot_spec=gs[1, 1])
        plt.savefig(os.path.join(exploration_dir_path, "combined_plots.png"))

        # Check the simulation paths.
        delete_index = 10
        if i == 0:
            diags.delete_evaluation_dir(delete_index)
        else:
            with pytest.raises(ValueError):
                diags.delete_evaluation_dir(delete_index)
        assert 15 not in diags._sim_dir_paths
        for trial_index in diags.history["trial_index"]:
            if trial_index in [15, delete_index]:
                with pytest.raises(ValueError):
                    diags.get_evaluation_dir_path(trial_index)
            else:
                ev_path = diags.get_evaluation_dir_path(trial_index)
                assert int(ev_path[-4:]) == trial_index

        # Check best evaluations.
        best_ev_f1 = diags.get_best_evaluation("f1")
        best_ev_f2 = diags.get_best_evaluation("f2")
        assert best_ev_f1.index == np.argmax(diags.history["f1"])
        assert best_ev_f2.index == np.argmin(diags.history["f2"])
        pareto_evs = diags.get_pareto_front_evaluations()
        assert best_ev_f1.index.to_numpy() in pareto_evs.index.to_numpy()
        assert best_ev_f2.index.to_numpy() in pareto_evs.index.to_numpy()
        best_ev_f1_path = diags.get_best_evaluation_dir_path()
        assert best_ev_f1_path == diags.get_evaluation_dir_path(
            best_ev_f1["trial_index"].item()
        )

        # Check printing methods.
        diags.print_best_evaluations(top=3, objective="f1")
        diags.print_evaluation(best_ev_f1["trial_index"].item())

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
