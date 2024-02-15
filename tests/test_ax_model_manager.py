import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pytest

from optimas.explorations import Exploration
from optimas.core import VaryingParameter, Objective
from optimas.generators import (
    AxSingleFidelityGenerator,
)
from optimas.evaluators import FunctionEvaluator
from optimas.diagnostics import ExplorationDiagnostics, AxModelManager


def eval_func_sf_moo(input_params, output_params):
    """Evaluation function for multi-objective single-fidelity test"""
    x0 = input_params["x0"]
    x1 = input_params["x1"]
    result = -(x0 + 10 * np.cos(x0)) * (x1 + 5 * np.cos(x1))
    output_params["f"] = result
    output_params["f2"] = result * 2


def test_ax_model_manager():
    """
    Test that an exploration with a multi-objective single-fidelity generator
    runs and that the generator and Ax client are updated after running.
    """

    var1 = VaryingParameter("x0", -50.0, 5.0)
    var2 = VaryingParameter("x1", -5.0, 15.0)
    var3 = VaryingParameter("x2", -5.0, 15.0)
    obj = Objective("f", minimize=True)
    obj2 = Objective("f2", minimize=False)

    gen = AxSingleFidelityGenerator(
        varying_parameters=[var1, var2, var3], objectives=[obj, obj2]
    )
    ev = FunctionEvaluator(function=eval_func_sf_moo)
    exploration = Exploration(
        generator=gen,
        evaluator=ev,
        max_evals=10,
        sim_workers=2,
        exploration_dir_path="./tests_output/test_ax_model_manager",
    )

    # Get reference to original AxClient.
    ax_client = gen._ax_client

    # Run exploration.
    exploration.run()

    # Open diagnostics and extract parameter sample `df`
    diags = ExplorationDiagnostics(exploration)
    varpar_names = [var.name for var in diags.varying_parameters]
    df = diags.history[varpar_names]

    # Get model manager directly from the existing `AxClient` instance.
    mm_axcl = AxModelManager(source=ax_client)
    mean_axcl, sem_axcl = mm_axcl.evaluate_model(sample=df, metric_name="f")

    # Get model manager from the `AxClient` dumped json file.
    max_evals = exploration.max_evals
    exploration_dir_path = exploration.exploration_dir_path
    path = os.path.join(
        exploration_dir_path,
        f"model_history/ax_client_at_eval_{max_evals}.json",
    )
    mm_json = AxModelManager(source=path)
    mean_json, sem_json = mm_json.evaluate_model(sample=df, metric_name="f")

    # Get model manager from diagnostics data.
    mm_diag = diags.build_model(objname="f")
    mean_diag, sem_diag = mm_diag.evaluate_model(sample=df, metric_name="f")

    # Add model evaluations to sample and print results.
    df["f_mean_axcl"] = mean_axcl
    df["f_mean_json"] = mean_json
    df["f_mean_diag"] = mean_diag
    print(df)

    # Check that different model initializations match within a 1% tolerance.
    assert np.allclose(mean_axcl, mean_json, rtol=1e-2)
    assert np.allclose(mean_axcl, mean_diag, rtol=1e-2)

    # Make example figure with two models in 2D.
    fig = plt.figure(figsize=(10, 4.8))
    gs = GridSpec(1, 2, wspace=0.2, hspace=0.3)

    # plot model for `f`
    ax1 = mm_axcl.plot_model(
        mname="f", pcolormesh_kw={"cmap": "GnBu"}, subplot_spec=gs[0, 0]
    )

    # Get and draw top 3 evaluations for `f`
    top_f = diags.get_best_evaluations_index(top=3, objective="f")
    df_top = diags.history.loc[top_f][varpar_names]
    ax1.scatter(df_top["x0"], df_top["x1"], c="red", marker="x")

    # plot model for `f2`
    ax2 = mm_axcl.plot_model(
        mname="f2", pcolormesh_kw={"cmap": "OrRd"}, subplot_spec=gs[0, 1]
    )

    # Get and draw top 3 evaluations for `f`
    top_f2 = diags.get_best_evaluations_index(top=3, objective="f2")
    df2_top = diags.history.loc[top_f2][varpar_names]
    ax2.scatter(df2_top["x0"], df2_top["x1"], c="blue", marker="x")
    plt.savefig(os.path.join(exploration_dir_path, "models.png"))

    # Make example figure of the models in 1D with errors.
    x1 = np.ones(100) * 0.5 * (var2.lower_bound + var2.upper_bound)
    x2 = np.ones(100) * 0.5 * (var3.lower_bound + var3.upper_bound)
    x0 = np.linspace(var1.lower_bound, var1.upper_bound, 100)
    metric_names = mm_axcl.ax_client.objective_names
    fig, axs = plt.subplots(len(metric_names), 1, sharex=True)
    for i, (ax, metric_name) in enumerate(zip(axs, metric_names)):
        mean, sed = mm_axcl.evaluate_model(
            sample={"x0": x0, "x1": x1, "x2": x2}, metric_name=metric_name
        )
        ax.plot(x0, mean, color=f"C{i}", label=f"x1 = {x1[0]}")
        ax.fill_between(
            x0, mean - sed, mean + sed, color="lightgray", alpha=0.5
        )
        ax.set_ylabel(metric_name)
        ax.legend(frameon=False)
    plt.xlabel("x0")
    plt.savefig(os.path.join(exploration_dir_path, "models_1d.png"))


if __name__ == "__main__":
    test_ax_model_manager()
