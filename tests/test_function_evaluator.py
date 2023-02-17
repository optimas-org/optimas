import numpy as np

from optimas.explorations import Exploration
from optimas.generators import RandomSamplingGenerator
from optimas.evaluators import FunctionEvaluator
from optimas.core import VaryingParameter, Objective


def eval_func(input_params, output_params):
    """Evaluation function used for testing"""
    x0 = input_params['x0']
    x1 = input_params['x1']
    result = -(x0 + 10 * np.cos(x0)) * (x1 + 5 * np.cos(x1))
    output_params['f'] = result


def test_function_evaluator():
    """Test that an exploration runs successfully with a function evaluator."""
    # Define variables and objectives.
    var1 = VaryingParameter('x0', -50., 5.)
    var2 = VaryingParameter('x1', -5., 15.)
    obj = Objective('f', minimize=False)

    # Create generator.
    gen = RandomSamplingGenerator(
        varying_parameters=[var1, var2],
        objectives=[obj]
    )

    # Create function evaluator.
    ev = FunctionEvaluator(function=eval_func)

    # Create exploration.
    exploration = Exploration(
        generator=gen,
        evaluator=ev,
        max_evals=10,
        sim_workers=2,
        exploration_dir_path='./tests_output/test_function_evaluator'
    )

    # Run exploration.
    exploration.run()


if __name__ == '__main__':
    test_function_evaluator()
