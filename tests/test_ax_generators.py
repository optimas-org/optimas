import threading

import numpy as np
from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.utils.measurement.synthetic_functions import hartmann6

from optimas.explorations import Exploration
from optimas.generators import (
    AxSingleFidelityGenerator,
    AxMultiFidelityGenerator,
    AxMultitaskGenerator,
    AxClientGenerator,
)
from optimas.evaluators import FunctionEvaluator, MultitaskEvaluator
from optimas.core import VaryingParameter, Objective, Task, Parameter


# Some tests will use threading (instead of multiprocessing) to be able to
# trigger failed trials based on a global counter.
threadLock = threading.Lock()
trial_count = 0
trials_to_fail = []


def eval_func_sf(input_params, output_params):
    """Evaluation function for single-fidelity test.

    This function can trigger a failed evaluation by not filling in the
    output parameters.
    """
    global trial_count
    global trials_to_fail
    with threadLock:
        trial_count += 1
        x0 = input_params["x0"]
        x1 = input_params["x1"]
        result = -(x0 + 10 * np.cos(x0)) * (x1 + 5 * np.cos(x1))
        if trial_count - 1 not in trials_to_fail:
            output_params["f"] = result
            if "p1" in output_params.dtype.names:
                output_params["p1"] = x0**2


def eval_func_sf_moo(input_params, output_params):
    """Evaluation function for multi-objective single-fidelity test.

    This function can trigger a failed evaluation by not filling in the
    output parameters.
    """
    global trial_count
    global trials_to_fail
    with threadLock:
        trial_count += 1
        x0 = input_params["x0"]
        x1 = input_params["x1"]
        result = -(x0 + 10 * np.cos(x0)) * (x1 + 5 * np.cos(x1))
        if trial_count - 1 not in trials_to_fail:
            output_params["f"] = result
            output_params["f2"] = result * 2


def eval_func_mf(input_params, output_params):
    """Evaluation function for multifidelity test.

    This function can trigger a failed evaluation by not filling in the
    output parameters.
    """
    global trial_count
    global trials_to_fail
    with threadLock:
        trial_count += 1
        x0 = input_params["x0"]
        x1 = input_params["x1"]
        resolution = input_params["res"]
        result = -(
            (x0 + 10 * np.cos(x0 + 0.1 * resolution))
            * (x1 + 5 * np.cos(x1 - 0.2 * resolution))
        )
        if trial_count - 1 not in trials_to_fail:
            output_params["f"] = result
            if "p1" in output_params.dtype.names:
                output_params["p1"] = x0**2


def eval_func_ax_client(input_params, output_params):
    """Evaluation function for the AxClient test.

    This function can trigger a failed evaluation by not filling in the
    output parameters.
    """
    global trial_count
    global trials_to_fail
    with threadLock:
        trial_count += 1
        x = np.array([input_params.get(f"x{i+1}") for i in range(6)])
        if trial_count - 1 not in trials_to_fail:
            output_params["hartmann6"] = hartmann6(x)
            output_params["l2norm"] = np.sqrt((x**2).sum())


def eval_func_task_1(input_params, output_params):
    """Evaluation function for task1 in multitask test"""
    x0 = input_params["x0"]
    x1 = input_params["x1"]
    result = -(x0 + 10 * np.cos(x0)) * (x1 + 5 * np.cos(x1))
    output_params["f"] = result


def eval_func_task_2(input_params, output_params):
    """Evaluation function for task1 in multitask test"""
    x0 = input_params["x0"]
    x1 = input_params["x1"]
    result = -0.5 * (x0 + 10 * np.cos(x0)) * (x1 + 5 * np.cos(x1))
    output_params["f"] = result


def check_run_ax_service(ax_client, gen, exploration, n_failed_expected):
    # Check that the generator has been updated and that the failed trials are
    # accounted.
    assert gen.n_evaluated_trials == exploration.history.shape[0]
    assert gen.n_failed_trials == n_failed_expected
    assert gen.n_completed_trials == gen.n_evaluated_trials - n_failed_expected
    ax_trials = ax_client.get_trials_data_frame()
    assert (ax_trials.trial_status == "ABANDONED").sum() == n_failed_expected
    np.testing.assert_array_equal(
        exploration.history.trial_status == "FAILED",
        ax_trials.trial_status == "ABANDONED",
    )

    # Check that the original ax client has been updated.
    n_ax_trials = ax_trials.shape[0]
    assert n_ax_trials == exploration.history.shape[0]


def test_ax_single_fidelity():
    """
    Test that an exploration with a single-fidelity generator runs
    and that the generator and Ax client are updated after running.
    """
    global trial_count
    global trials_to_fail
    trial_count = 0
    trials_to_fail = [2, 6]

    var1 = VaryingParameter("x0", -50.0, 5.0)
    var2 = VaryingParameter("x1", -5.0, 15.0)
    obj = Objective("f", minimize=False)
    p1 = Parameter("p1")

    gen = AxSingleFidelityGenerator(
        varying_parameters=[var1, var2],
        objectives=[obj],
        analyzed_parameters=[p1],
        parameter_constraints=["x0 + x1 <= 10"],
        outcome_constraints=["p1 <= 30"],
    )
    ev = FunctionEvaluator(function=eval_func_sf)
    exploration = Exploration(
        generator=gen,
        evaluator=ev,
        max_evals=10,
        sim_workers=2,
        exploration_dir_path="./tests_output/test_ax_single_fidelity",
        libe_comms="local_threading",
    )

    # Get reference to original AxClient.
    ax_client = gen._ax_client

    # Run exploration.
    exploration.run()

    # Mark a trials as failed by hand
    exploration.mark_evaluation_as_failed(8)

    # Perform checks.
    check_run_ax_service(ax_client, gen, exploration, len(trials_to_fail) + 1)

    # Check constraints.
    history = exploration.history
    assert all(history["x0"] + history["x1"] <= 10.0 + 1e-3)
    ocs = gen._ax_client.experiment.optimization_config.outcome_constraints
    assert len(ocs) == 1
    assert ocs[0].metric.name == p1.name

    # Save history for later restart test
    np.save("./tests_output/ax_sf_history", exploration._libe_history.H)


def test_ax_single_fidelity_int():
    """
    Test that an exploration with a single-fidelity generator runs
    correctly with an integer parameter.
    """
    global trial_count
    global trials_to_fail
    trial_count = 0
    trials_to_fail = [2, 6]

    var1 = VaryingParameter("x0", -50.0, 5.0, dtype=int)
    var2 = VaryingParameter("x1", -5.0, 15.0)
    obj = Objective("f", minimize=False)

    gen = AxSingleFidelityGenerator(
        varying_parameters=[var1, var2], objectives=[obj]
    )
    ev = FunctionEvaluator(function=eval_func_sf)
    exploration = Exploration(
        generator=gen,
        evaluator=ev,
        max_evals=10,
        sim_workers=2,
        exploration_dir_path="./tests_output/test_ax_single_fidelity_int",
        libe_comms="local_threading",
    )

    # Get reference to original AxClient.
    ax_client = gen._ax_client
    assert ax_client.experiment.search_space.parameters["x0"].python_type == int

    # Run exploration.
    exploration.run()

    # Perform checks.
    check_run_ax_service(ax_client, gen, exploration, len(trials_to_fail))

    # Check correct variable type.
    assert exploration.history["x0"].to_numpy().dtype == int


def test_ax_single_fidelity_moo():
    """
    Test that an exploration with a multi-objective single-fidelity generator
    runs and that the generator and Ax client are updated after running.
    """
    global trial_count
    global trials_to_fail
    trial_count = 0
    trials_to_fail = [2, 6]

    var1 = VaryingParameter("x0", -50.0, 5.0)
    var2 = VaryingParameter("x1", -5.0, 15.0)
    obj = Objective("f", minimize=False)
    obj2 = Objective("f2", minimize=False)

    gen = AxSingleFidelityGenerator(
        varying_parameters=[var1, var2], objectives=[obj, obj2]
    )
    ev = FunctionEvaluator(function=eval_func_sf_moo)
    exploration = Exploration(
        generator=gen,
        evaluator=ev,
        max_evals=10,
        sim_workers=2,
        exploration_dir_path="./tests_output/test_ax_single_fidelity_moo",
        libe_comms="local_threading",
    )

    # Get reference to original AxClient.
    ax_client = gen._ax_client

    # Run exploration.
    exploration.run()

    # Perform checks.
    check_run_ax_service(ax_client, gen, exploration, len(trials_to_fail))


def test_ax_single_fidelity_fb():
    """
    Test that an exploration with a fully Bayesian single-fidelity generator
    runs and that the generator and Ax client are updated after running.
    """
    global trial_count
    global trials_to_fail
    trial_count = 0
    trials_to_fail = [2, 6]

    var1 = VaryingParameter("x0", -50.0, 5.0)
    var2 = VaryingParameter("x1", -5.0, 15.0)
    obj = Objective("f", minimize=False)

    gen = AxSingleFidelityGenerator(
        varying_parameters=[var1, var2], objectives=[obj], fully_bayesian=True
    )
    ev = FunctionEvaluator(function=eval_func_sf)
    exploration = Exploration(
        generator=gen,
        evaluator=ev,
        max_evals=10,
        sim_workers=2,
        exploration_dir_path="./tests_output/test_ax_single_fidelity_fb",
        libe_comms="local_threading",
    )

    # Get reference to original AxClient.
    ax_client = gen._ax_client

    # Run exploration.
    exploration.run()

    # Perform checks.
    check_run_ax_service(ax_client, gen, exploration, len(trials_to_fail))


def test_ax_single_fidelity_moo_fb():
    """
    Test that an exploration with a fully Bayesian multi-objective
    single-fidelity generator runs and that the generator and Ax client
    are updated after running.
    """
    global trial_count
    global trials_to_fail
    trial_count = 0
    trials_to_fail = [2, 6]

    var1 = VaryingParameter("x0", -50.0, 5.0)
    var2 = VaryingParameter("x1", -5.0, 15.0)
    obj = Objective("f", minimize=False)
    obj2 = Objective("f2", minimize=False)

    gen = AxSingleFidelityGenerator(
        varying_parameters=[var1, var2],
        objectives=[obj, obj2],
        fully_bayesian=True,
    )
    ev = FunctionEvaluator(function=eval_func_sf_moo)
    exploration = Exploration(
        generator=gen,
        evaluator=ev,
        max_evals=10,
        sim_workers=2,
        exploration_dir_path="./tests_output/test_ax_single_fidelity_moo_fb",
        libe_comms="local_threading",
    )

    # Get reference to original AxClient.
    ax_client = gen._ax_client

    # Run exploration.
    exploration.run()

    # Perform checks.
    check_run_ax_service(ax_client, gen, exploration, len(trials_to_fail))


def test_ax_single_fidelity_updated_params():
    """
    Test that an exploration with a single-fidelity generator runs
    as expected when the varing parameters are updated.
    """
    # Prevent trials from failing in this test.
    global trial_count
    global trials_to_fail
    trial_count = 0
    trials_to_fail = []

    var1 = VaryingParameter("x0", -50.0, 5.0)
    var2 = VaryingParameter("x1", -5.0, 15.0)
    obj = Objective("f", minimize=False)

    # Start with a fixed value of x0.
    var1.fix_value(-10.0)

    gen = AxSingleFidelityGenerator(
        varying_parameters=[var1, var2],
        objectives=[obj],
        fit_out_of_design=True,
    )
    ev = FunctionEvaluator(function=eval_func_sf)
    exploration = Exploration(
        generator=gen,
        evaluator=ev,
        sim_workers=2,
        exploration_dir_path="./tests_output/test_ax_single_fidelity_up",
        libe_comms="local_threading",
    )

    # Run exploration.
    exploration.run(n_evals=5)
    assert all(exploration.history["x0"] == -10)

    # Free value of x0 and run 5 evals.
    var1.free_value()
    gen.update_parameter(var1)
    exploration.run(n_evals=5)
    assert not all(exploration.history["x0"][-5:] == -10)

    # Update range of x0 and run 10 evals.
    var1.update_range(-20.0, 0.0)
    gen.update_parameter(var1)
    exploration.run(n_evals=10)
    assert all(exploration.history["x0"][-10:] >= -20)
    assert all(exploration.history["x0"][-10:] <= 0.0)

    # Fix of x0 and run 5 evals.
    var1.fix_value(-9)
    gen.update_parameter(var1)
    exploration.run(n_evals=5)
    assert all(exploration.history["x0"][-5:] == -9)

    # Evaluate a custom trial.
    exploration.evaluate_trials([{"x0": -7.0, "x1": 10.0}])
    assert exploration.history["x0"].to_numpy()[-1] == -7

    # Free value and run 3 evals.
    var1.free_value()
    gen.update_parameter(var1)
    exploration.run(n_evals=3)
    assert all(exploration.history["x0"][-3:] != -9)


def test_ax_multi_fidelity():
    """Test that an exploration with a multifidelity generator runs"""

    global trial_count
    global trials_to_fail
    trial_count = 0
    trials_to_fail = [2, 5]

    var1 = VaryingParameter("x0", -50.0, 5.0)
    var2 = VaryingParameter("x1", -5.0, 15.0)
    var3 = VaryingParameter(
        "res", 1.0, 8.0, is_fidelity=True, fidelity_target_value=8.0
    )
    obj = Objective("f", minimize=False)
    p1 = Parameter("p1")

    gen = AxMultiFidelityGenerator(
        varying_parameters=[var1, var2, var3],
        objectives=[obj],
        analyzed_parameters=[p1],
        outcome_constraints=["p1 <= 30"],
    )
    ev = FunctionEvaluator(function=eval_func_mf)
    exploration = Exploration(
        generator=gen,
        evaluator=ev,
        max_evals=6,
        sim_workers=2,
        run_async=False,
        exploration_dir_path="./tests_output/test_ax_multi_fidelity",
        libe_comms="local_threading",
    )

    # Get reference to original AxClient.
    ax_client = gen._ax_client

    # Run exploration.
    exploration.run()

    # Check constraints.
    ocs = gen._ax_client.experiment.optimization_config.outcome_constraints
    assert len(ocs) == 1
    assert ocs[0].metric.name == p1.name

    # Perform checks.
    check_run_ax_service(ax_client, gen, exploration, len(trials_to_fail))

    # Save history for later restart test
    np.save("./tests_output/ax_mf_history", exploration._libe_history.H)


def test_ax_multitask():
    """Test that an exploration with a multitask generator runs"""

    var1 = VaryingParameter("x0", -50.0, 5.0)
    var2 = VaryingParameter("x1", -5.0, 15.0)
    obj = Objective("f", minimize=False)

    task1 = Task("task_1", n_init=2, n_opt=1)
    task2 = Task("task_2", n_init=5, n_opt=3)

    gen = AxMultitaskGenerator(
        varying_parameters=[var1, var2],
        objectives=[obj],
        hifi_task=task1,
        lofi_task=task2,
    )
    ev1 = FunctionEvaluator(function=eval_func_task_1)
    ev2 = FunctionEvaluator(function=eval_func_task_2)
    ev = MultitaskEvaluator(tasks=[task1, task2], task_evaluators=[ev1, ev2])
    exploration = Exploration(
        generator=gen,
        evaluator=ev,
        max_evals=15,
        sim_workers=2,
        exploration_dir_path="./tests_output/test_ax_multitask",
    )

    exploration.run()

    # Save history for later restart test
    np.save("./tests_output/ax_mt_history", exploration._libe_history.H)


def test_ax_client():
    """Test that an exploration with a user-given AxClient runs"""

    global trial_count
    global trials_to_fail
    trial_count = 0
    trials_to_fail = [2, 5]

    # Create the AxClient from https://ax.dev/tutorials/gpei_hartmann_service.html.
    ax_client = AxClient()
    ax_client.create_experiment(
        name="hartmann_test_experiment",
        parameters=[
            {
                "name": "x1",
                "type": "range",
                "bounds": [0.0, 1.0],
            },
            {
                "name": "x2",
                "type": "range",
                "bounds": [0.0, 1.0],
            },
            {
                "name": "x3",
                "type": "range",
                "bounds": [0.0, 1.0],
            },
            {
                "name": "x4",
                "type": "range",
                "bounds": [0.0, 1.0],
            },
            {
                "name": "x5",
                "type": "range",
                "bounds": [0.0, 1.0],
            },
            {
                "name": "x6",
                "type": "range",
                "bounds": [0.0, 1.0],
            },
        ],
        objectives={
            "hartmann6": ObjectiveProperties(minimize=True),
        },
        parameter_constraints=["x1 + x2 <= 2.0"],  # Optional.
        outcome_constraints=["l2norm <= 1.25"],  # Optional.
    )

    gen = AxClientGenerator(ax_client=ax_client)
    ev = FunctionEvaluator(function=eval_func_ax_client)
    exploration = Exploration(
        generator=gen,
        evaluator=ev,
        max_evals=6,
        sim_workers=2,
        run_async=False,
        exploration_dir_path="./tests_output/test_ax_client",
        libe_comms="local_threading",
    )

    exploration.run()

    # Perform checks.
    check_run_ax_service(ax_client, gen, exploration, len(trials_to_fail))


def test_ax_single_fidelity_with_history():
    """
    Test that an exploration with a single-fidelity generator runs when
    restarted from a history file
    """
    global trial_count
    global trials_to_fail
    trial_count = 0
    trials_to_fail = []

    var1 = VaryingParameter("x0", -50.0, 5.0)
    var2 = VaryingParameter("x1", -5.0, 15.0)
    obj = Objective("f", minimize=False)
    p1 = Parameter("p1")

    gen = AxSingleFidelityGenerator(
        varying_parameters=[var1, var2],
        objectives=[obj],
        analyzed_parameters=[p1],
    )
    ev = FunctionEvaluator(function=eval_func_sf)
    exploration = Exploration(
        generator=gen,
        evaluator=ev,
        max_evals=10,
        sim_workers=2,
        history="./tests_output/ax_sf_history.npy",
        exploration_dir_path="./tests_output/test_ax_single_fidelity_with_history",
    )

    # Get reference to original AxClient.
    ax_client = gen._ax_client

    # Run exploration.
    exploration.run()

    # Perform checks.
    check_run_ax_service(ax_client, gen, exploration, n_failed_expected=3)


def test_ax_multi_fidelity_with_history():
    """
    Test that an exploration with a multifidelity generator runs when
    restarted from a history file
    """
    global trial_count
    global trials_to_fail
    trial_count = 0
    trials_to_fail = []

    var1 = VaryingParameter("x0", -50.0, 5.0)
    var2 = VaryingParameter("x1", -5.0, 15.0)
    var3 = VaryingParameter(
        "res", 1.0, 8.0, is_fidelity=True, fidelity_target_value=8.0
    )
    obj = Objective("f", minimize=False)
    p1 = Parameter("p1")

    gen = AxMultiFidelityGenerator(
        varying_parameters=[var1, var2, var3],
        objectives=[obj],
        analyzed_parameters=[p1],
    )
    ev = FunctionEvaluator(function=eval_func_mf)
    exploration = Exploration(
        generator=gen,
        evaluator=ev,
        max_evals=4,
        sim_workers=2,
        run_async=False,
        history="./tests_output/ax_mf_history.npy",
        exploration_dir_path="./tests_output/test_ax_multi_fidelity_with_history",
    )

    # Get reference to original AxClient.
    ax_client = gen._ax_client

    # Run exploration.
    exploration.run()

    # Perform checks.
    check_run_ax_service(ax_client, gen, exploration, n_failed_expected=2)


def test_ax_multitask_with_history():
    """
    Test that an exploration with a multitask generator runs when
    restarted from a history file
    """

    var1 = VaryingParameter("x0", -50.0, 5.0)
    var2 = VaryingParameter("x1", -5.0, 15.0)
    obj = Objective("f", minimize=False)

    task1 = Task("task_1", n_init=2, n_opt=1)
    task2 = Task("task_2", n_init=5, n_opt=3)

    gen = AxMultitaskGenerator(
        varying_parameters=[var1, var2],
        objectives=[obj],
        hifi_task=task1,
        lofi_task=task2,
    )
    ev1 = FunctionEvaluator(function=eval_func_task_1)
    ev2 = FunctionEvaluator(function=eval_func_task_2)
    ev = MultitaskEvaluator(tasks=[task1, task2], task_evaluators=[ev1, ev2])
    exploration = Exploration(
        generator=gen,
        evaluator=ev,
        max_evals=10,
        sim_workers=2,
        history="./tests_output/ax_mt_history.npy",
        exploration_dir_path="./tests_output/test_ax_multitask_with_history",
    )
    exploration.run()


def test_ax_service_init():
    """
    Test that an exploration with using an AxServiceGenerator correctly
    reduces the number of `n_init` SOBOL evaluations if external trials
    or evaluations are given.
    """

    var1 = VaryingParameter("x0", -50.0, 5.0)
    var2 = VaryingParameter("x1", -5.0, 15.0)
    obj = Objective("f", minimize=False)

    n_init = 4
    n_external = 6

    for i in range(n_external):
        gen = AxSingleFidelityGenerator(
            varying_parameters=[var1, var2], objectives=[obj], n_init=n_init
        )
        ev = FunctionEvaluator(function=eval_func_sf)
        exploration = Exploration(
            generator=gen,
            evaluator=ev,
            max_evals=6,
            sim_workers=2,
            exploration_dir_path=f"./tests_output/test_ax_service_init_{i}",
        )

        # Get reference to AxClient.
        ax_client = gen._ax_client

        for _ in range(i):
            exploration.evaluate_trials(
                {
                    "x0": [-2.0 + np.random.rand(1)[0]],
                    "x1": [2.7 + np.random.rand(1)[0]],
                }
            )
        # Run exploration.
        exploration.run()

        # Check that the number of SOBOL trials is reduced and that they
        # are replaced by Manual trials.
        df = ax_client.get_trials_data_frame()
        for j in range(i):
            assert df["generation_method"][j] == "Manual"
        for k in range(i, n_init - 1):
            assert df["generation_method"][k] == "Sobol"
        df["generation_method"][min(i, n_init)] == "GPEI"

    # Test single case with `enforce_n_init=True`
    gen = AxSingleFidelityGenerator(
        varying_parameters=[var1, var2],
        objectives=[obj],
        n_init=n_init,
        enforce_n_init=True,
    )
    ev = FunctionEvaluator(function=eval_func_sf)
    exploration = Exploration(
        generator=gen,
        evaluator=ev,
        max_evals=15,
        sim_workers=2,
        exploration_dir_path="./tests_output/test_ax_service_init_enforce",
    )

    # Get reference to AxClient.
    ax_client = gen._ax_client

    for _ in range(n_external):
        exploration.evaluate_trials(
            {
                "x0": [-2.0 + np.random.rand(1)[0]],
                "x1": [2.7 + np.random.rand(1)[0]],
            }
        )
    # Run exploration.
    exploration.run()

    # Check that the number of SOBOL trials is still `n_init` after adding
    # `n_external` Manual trials.
    df = ax_client.get_trials_data_frame()
    for j in range(n_external):
        assert df["generation_method"][j] == "Manual"
    for k in range(n_external, n_external + n_init):
        assert df["generation_method"][k] == "Sobol"
    df["generation_method"][n_external + n_init] == "GPEI"


if __name__ == "__main__":
    test_ax_single_fidelity()
    test_ax_single_fidelity_int()
    test_ax_single_fidelity_moo()
    test_ax_single_fidelity_fb()
    test_ax_single_fidelity_moo_fb()
    test_ax_single_fidelity_updated_params()
    test_ax_multi_fidelity()
    test_ax_multitask()
    test_ax_client()
    test_ax_single_fidelity_with_history()
    test_ax_multi_fidelity_with_history()
    test_ax_multitask_with_history()
    test_ax_service_init()
