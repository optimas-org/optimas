import numpy as np

from optimas.explorations import Exploration
from optimas.generators import (
    AxSingleFidelityGenerator, AxMultiFidelityGenerator, AxMultitaskGenerator)
from optimas.evaluators import FunctionEvaluator, MultitaskEvaluator
from optimas.core import VaryingParameter, Objective, Task


def eval_func_sf(input_params, output_params):
    """Evaluation function for single-fidelity test"""
    x0 = input_params['x0']
    x1 = input_params['x1']
    result = -(x0 + 10 * np.cos(x0)) * (x1 + 5 * np.cos(x1))
    output_params['f'] = result


def eval_func_mf(input_params, output_params):
    """Evaluation function for multifidelity test"""
    x0 = input_params['x0']
    x1 = input_params['x1']
    resolution = input_params['res']
    result = -((x0 + 10 * np.cos(x0 + 0.1 * resolution)) *
               (x1 + 5 * np.cos(x1 - 0.2 * resolution)))
    output_params['f'] = result


def eval_func_task_1(input_params, output_params):
    """Evaluation function for task1 in multitask test"""
    x0 = input_params['x0']
    x1 = input_params['x1']
    result = -(x0 + 10 * np.cos(x0)) * (x1 + 5 * np.cos(x1))
    output_params['f'] = result


def eval_func_task_2(input_params, output_params):
    """Evaluation function for task1 in multitask test"""
    x0 = input_params['x0']
    x1 = input_params['x1']
    result = - 0.5 * (x0 + 10 * np.cos(x0)) * (x1 + 5 * np.cos(x1))
    output_params['f'] = result


def test_ax_single_fidelity():
    """Test that an exploration with a single-fidelity generator runs"""

    var1 = VaryingParameter('x0', -50., 5.)
    var2 = VaryingParameter('x1', -5., 15.)
    obj = Objective('f', minimize=False)

    gen = AxSingleFidelityGenerator(varying_parameters=[var1, var2], objectives=[obj])
    ev = FunctionEvaluator(function=eval_func_sf)
    exploration = Exploration(
        generator=gen,
        evaluator=ev,
        max_evals=10,
        sim_workers=2,
        exploration_dir_path='./tests_output/test_ax_single_fidelity'
    )

    exploration.run()

    # Save history for later restart test
    np.save('./tests_output/ax_sf_history' , exploration.history)


def test_ax_multi_fidelity():
    """Test that an exploration with a multifidelity generator runs"""

    var1 = VaryingParameter('x0', -50., 5.)
    var2 = VaryingParameter('x1', -5., 15.)
    var3 = VaryingParameter('res', 1., 8., is_fidelity=True,
                            fidelity_target_value=8.)
    obj = Objective('f', minimize=False)

    gen = AxMultiFidelityGenerator(
        varying_parameters=[var1, var2, var3],
        objectives=[obj]
    )
    ev = FunctionEvaluator(function=eval_func_mf)
    exploration = Exploration(
        generator=gen,
        evaluator=ev,
        max_evals=6,
        sim_workers=2,
        run_async=False,
        exploration_dir_path='./tests_output/test_ax_multi_fidelity'
    )

    exploration.run()

    # Save history for later restart test
    np.save('./tests_output/ax_mf_history' , exploration.history)


def test_ax_multitask():
    """Test that an exploration with a multitask generator runs"""

    var1 = VaryingParameter('x0', -50., 5.)
    var2 = VaryingParameter('x1', -5., 15.)
    obj = Objective('f', minimize=False)

    task1 = Task('task_1', n_init=2, n_opt=1)
    task2 = Task('task_2', n_init=5, n_opt=3)

    gen = AxMultitaskGenerator(
        varying_parameters=[var1, var2],
        objectives=[obj],
        hifi_task=task1,
        lofi_task=task2
    )
    ev1 = FunctionEvaluator(function=eval_func_task_1)
    ev2 = FunctionEvaluator(function=eval_func_task_2)
    ev = MultitaskEvaluator(tasks=[task1, task2], task_evaluators=[ev1, ev2])
    exploration = Exploration(
        generator=gen,
        evaluator=ev,
        max_evals=15,
        sim_workers=2,
        exploration_dir_path='./tests_output/test_ax_multitask'
    )

    exploration.run()

    # Save history for later restart test
    np.save('./tests_output/ax_mt_history' , exploration.history)


def test_ax_single_fidelity_with_history():
    """
    Test that an exploration with a single-fidelity generator runs when
    restarted from a history file
    """

    var1 = VaryingParameter('x0', -50., 5.)
    var2 = VaryingParameter('x1', -5., 15.)
    obj = Objective('f', minimize=False)

    gen = AxSingleFidelityGenerator(
        varying_parameters=[var1, var2],
        objectives=[obj]
    )
    ev = FunctionEvaluator(function=eval_func_sf)
    exploration = Exploration(
        generator=gen,
        evaluator=ev,
        max_evals=10,
        sim_workers=2,
        history='./tests_output/ax_sf_history.npy',
        exploration_dir_path='./tests_output/test_ax_single_fidelity_with_history'
    )

    exploration.run()


def test_ax_multi_fidelity_with_history():
    """
    Test that an exploration with a multifidelity generator runs when
    restarted from a history file
    """

    var1 = VaryingParameter('x0', -50., 5.)
    var2 = VaryingParameter('x1', -5., 15.)
    var3 = VaryingParameter('res', 1., 8., is_fidelity=True,
                            fidelity_target_value=8.)
    obj = Objective('f', minimize=False)

    gen = AxMultiFidelityGenerator(
        varying_parameters=[var1, var2, var3],
        objectives=[obj]
    )
    ev = FunctionEvaluator(function=eval_func_mf)
    exploration = Exploration(
        generator=gen,
        evaluator=ev,
        max_evals=4,
        sim_workers=2,
        run_async=False,
        history='./tests_output/ax_mf_history.npy',
        exploration_dir_path='./tests_output/test_ax_multi_fidelity_with_history'
    )

    exploration.run()


def test_ax_multitask_with_history():
    """
    Test that an exploration with a multitask generator runs when
    restarted from a history file
    """

    var1 = VaryingParameter('x0', -50., 5.)
    var2 = VaryingParameter('x1', -5., 15.)
    obj = Objective('f', minimize=False)

    task1 = Task('task_1', n_init=2, n_opt=1)
    task2 = Task('task_2', n_init=5, n_opt=3)

    gen = AxMultitaskGenerator(
        varying_parameters=[var1, var2],
        objectives=[obj],
        hifi_task=task1,
        lofi_task=task2
    )
    ev1 = FunctionEvaluator(function=eval_func_task_1)
    ev2 = FunctionEvaluator(function=eval_func_task_2)
    ev = MultitaskEvaluator(tasks=[task1, task2], task_evaluators=[ev1, ev2])
    exploration = Exploration(
        generator=gen,
        evaluator=ev,
        max_evals=10,
        sim_workers=2,
        history='./tests_output/ax_mt_history.npy',
        exploration_dir_path='./tests_output/test_ax_multitask_with_history'
    )
    exploration.run()


if __name__ == '__main__':
    test_ax_single_fidelity()
    test_ax_multi_fidelity()
    test_ax_multitask()
    test_ax_single_fidelity_with_history()
    test_ax_multi_fidelity_with_history()
    test_ax_multitask_with_history()
