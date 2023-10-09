import os

from optimas.explorations import Exploration
from optimas.generators import RandomSamplingGenerator
from optimas.evaluators import TemplateEvaluator
from optimas.core import VaryingParameter, Objective


def analysis_func(sim_dir, output_params):
    """Analysis function used by the template evaluator."""
    # Read back result from file
    with open('result.txt') as f:
        result = float(f.read())
    output_params['f'] = result


def test_exploration_in_steps():
    """Test that an exploration runs correctly when doing so in several steps.
    """
    # Define variables and objectives.
    var1 = VaryingParameter('x0', -50., 5.)
    var2 = VaryingParameter('x1', -5., 15.)
    obj = Objective('f', minimize=False)

    # Define variables and objectives.
    gen = RandomSamplingGenerator(
        varying_parameters=[var1, var2],
        objectives=[obj]
    )

    # Create template evaluator.
    ev = TemplateEvaluator(
        sim_template=os.path.join(
            os.path.abspath(os.path.dirname(__file__)),
            'resources',
            'template_simulation_script.py'
        ),
        analysis_func=analysis_func
    )

    # Create exploration.
    exploration = Exploration(
        generator=gen,
        evaluator=ev,
        max_evals=30,
        sim_workers=2,
        exploration_dir_path='./tests_output/test_template_evaluator'
    )

    # Run exploration in several steps.
    exploration.run(3)
    exploration.run(4)
    exploration.run(10)
    exploration.run(5)
    exploration.run()

    # Check final state.
    assert exploration._n_evals == len(exploration.history)
    assert exploration._n_evals == gen.n_trials
    assert exploration._n_evals == exploration.max_evals
    assert exploration.history['gen_informed'][-1]


def test_exploration_resume():
    """Test that an exploration correctly resumes from a previous run.
    """
    # Define variables and objectives.
    var1 = VaryingParameter('x0', -50., 5.)
    var2 = VaryingParameter('x1', -5., 15.)
    obj = Objective('f', minimize=False)

    # Define variables and objectives.
    gen = RandomSamplingGenerator(
        varying_parameters=[var1, var2],
        objectives=[obj]
    )

    # Create template evaluator.
    ev = TemplateEvaluator(
        sim_template=os.path.join(
            os.path.abspath(os.path.dirname(__file__)),
            'resources',
            'template_simulation_script.py'
        ),
        analysis_func=analysis_func
    )

    # Create exploration.
    exploration = Exploration(
        generator=gen,
        evaluator=ev,
        max_evals=40,
        sim_workers=2,
        exploration_dir_path='./tests_output/test_template_evaluator',
        resume=True
    )

    # Run exploration.
    exploration.run()

    # Check final state.
    assert exploration._n_evals == len(exploration.history)
    assert exploration._n_evals == gen.n_trials
    assert exploration._n_evals == exploration.max_evals
    assert exploration.history['gen_informed'][-1]


if __name__ == '__main__':
    test_exploration_in_steps()
    test_exploration_resume()
