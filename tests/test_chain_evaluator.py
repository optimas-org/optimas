import os

from optimas.explorations import Exploration
from optimas.generators import RandomSamplingGenerator
from optimas.evaluators import TemplateEvaluator, ChainEvaluator
from optimas.core import VaryingParameter, Objective, Parameter


def analysis_func_1(sim_dir, output_params):
    """Analysis function used by the template evaluator."""
    # Read back result from file
    with open('result.txt') as f:
        result = float(f.read())
    output_params['result_1'] = result


def analysis_func_2(sim_dir, output_params):
    """Analysis function used by the template evaluator."""
    # Read back result from file
    with open('result_2.txt') as f:
        result = float(f.read())
    output_params['f'] = result


def test_chain_evaluator():
    # Define variables and objectives.
    var1 = VaryingParameter('x0', -50., 5.)
    var2 = VaryingParameter('x1', -5., 15.)
    par1 = Parameter('result_1')
    obj = Objective('f', minimize=False)

    # Define variables and objectives.
    gen = RandomSamplingGenerator(
        varying_parameters=[var1, var2],
        objectives=[obj],
        analyzed_parameters=[par1]
    )

    # Create template evaluator.
    ev1 = TemplateEvaluator(
        sim_template=os.path.join(
            os.path.abspath(os.path.dirname(__file__)),
            'resources',
            'template_simulation_script.py'
        ),
        analysis_func=analysis_func_1
    )
    ev2 = TemplateEvaluator(
        sim_template=os.path.join(
            os.path.abspath(os.path.dirname(__file__)),
            'resources',
            'template_simulation_script_2.py'
        ),
        analysis_func=analysis_func_2
    )
    ev = ChainEvaluator([ev1, ev2])

    # Create exploration.
    exploration = Exploration(
        generator=gen,
        evaluator=ev,
        max_evals=10,
        sim_workers=2,
        exploration_dir_path='./tests_output/test_chain_evaluator'
    )

    # Run exploration.
    exploration.run()


if __name__ == '__main__':
    test_chain_evaluator()
