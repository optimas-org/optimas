import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from optimas.explorations import Exploration
from optimas.core import VaryingParameter, Objective
from optimas.generators import AxSingleFidelityGenerator
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
    df = diags.history[varpar_names + ["f"]]

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
    mm_diag = diags.build_gp_model(parameter="f")
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
    fig = plt.figure(figsize=(8, 8))
    gs = GridSpec(2, 2, wspace=0.2, hspace=0.3)

    # center coordinates
    x1_c = 0.5 * (var2.lower_bound + var2.upper_bound)
    x2_c = 0.5 * (var3.lower_bound + var3.upper_bound)

    # plot model for `f` with custom slice value
    fig, ax1 = mm_axcl.plot_contour(
        metric_name="f",
        slice_values={"x2": x2_c},
        pcolormesh_kw={"cmap": "GnBu"},
        subplot_spec=gs[0, 0],
    )

    # Get and draw top 3 evaluations for `f`
    df_top = diags.get_best_evaluations(top=3, objective="f")
    ax1.scatter(df_top["x0"], df_top["x1"], c="red", marker="x")

    # plot model for `f` with default settings (mid point)
    fig, ax1 = mm_axcl.plot_contour(
        metric_name="f",
        subplot_spec=gs[0, 1],
    )

    # plot model for `f2` with custom slice value
    fig, ax2 = mm_axcl.plot_contour(
        metric_name="f2",
        slice_values={"x2": x2_c},
        pcolormesh_kw={"cmap": "OrRd"},
        subplot_spec=gs[1, 0],
    )

    # plot model for `f2` along best slice
    fig, ax2 = mm_axcl.plot_contour(
        metric_name="f2",
        slice_values="best",
        pcolormesh_kw={"cmap": "OrRd"},
        subplot_spec=gs[1, 1],
    )

    # Get and draw top 3 evaluations for `f2`
    df2_top = diags.get_best_evaluations(top=3, objective="f2")
    ax2.scatter(df2_top["x0"], df2_top["x1"], c="blue", marker="x")
    plt.savefig(os.path.join(exploration_dir_path, "models_2d.png"))

    fig, axs = mm_axcl.plot_contour(mode="both", figsize=(8, 4))
    fig.savefig("models_2d_both.png")

    # Make figure of the models in 1D with errors.
    fig = plt.figure()
    gs = GridSpec(2, 1, hspace=0.3)
    fig, ax = mm_axcl.plot_slice(
        "x0",
        metric_name="f",
        slice_values={"x1": x1_c, "x2": x2_c},
        subplot_spec=gs[0],
        plot_kw={"color": "C0"},
    )
    fig, ax = mm_axcl.plot_slice(
        "x0",
        metric_name="f2",
        slice_values={"x1": x1_c, "x2": x2_c},
        subplot_spec=gs[1],
        plot_kw={"color": "C1"},
    )
    fig.savefig(os.path.join(exploration_dir_path, "models_1d.png"))


if __name__ == "__main__":
    test_ax_model_manager()
