import os

from optimas.explorations import Exploration
from optimas.generators import RandomSamplingGenerator
from optimas.evaluators import TemplateEvaluator
from optimas.core import VaryingParameter, Objective


def analysis_func(sim_dir, output_params):
    """Analysis function used by the template evaluator."""
    # Read back result from file
    with open("result.txt") as f:
        result = float(f.read())
    output_params["f"] = result


def test_exploration_in_steps():
    """Test that an exploration runs correctly when doing so in several steps."""
    # Define variables and objectives.
    var1 = VaryingParameter("x0", -50.0, 5.0)
    var2 = VaryingParameter("x1", -5.0, 15.0)
    obj = Objective("f", minimize=False)

    # Define variables and objectives.
    gen = RandomSamplingGenerator(
        varying_parameters=[var1, var2], objectives=[obj]
    )

    # Create template evaluator.
    ev = TemplateEvaluator(
        sim_template=os.path.join(
            os.path.abspath(os.path.dirname(__file__)),
            "resources",
            "template_simulation_script.py",
        ),
        analysis_func=analysis_func,
    )

    # Create exploration.
    exploration = Exploration(
        generator=gen,
        evaluator=ev,
        max_evals=30,
        sim_workers=2,
        exploration_dir_path="./tests_output/test_exploration_steps",
    )

    # Run exploration in several steps.
    exploration.run(3)
    exploration.run(4)
    exploration.run(10)
    exploration.run(5)
    exploration.run()

    # Check final state.
    assert exploration._n_evals == len(exploration.history)
    assert exploration._n_evals == gen.n_completed_trials
    assert exploration._n_evals == exploration.max_evals
    assert exploration.history["gen_informed"].to_numpy()[-1]
    assert count_history_files(exploration.exploration_dir_path) == 1


def test_exploration_in_steps_without_limit():
    """
    Test that an exploration runs correctly when doing so in several steps
    without a limit on the maximum number of evaluations.
    """
    # Define variables and objectives.
    var1 = VaryingParameter("x0", -50.0, 5.0)
    var2 = VaryingParameter("x1", -5.0, 15.0)
    obj = Objective("f", minimize=False)

    # Define evaluator.
    gen = RandomSamplingGenerator(
        varying_parameters=[var1, var2], objectives=[obj]
    )

    # Create template evaluator.
    ev = TemplateEvaluator(
        sim_template=os.path.join(
            os.path.abspath(os.path.dirname(__file__)),
            "resources",
            "template_simulation_script.py",
        ),
        analysis_func=analysis_func,
    )

    # Create exploration.
    exploration = Exploration(
        generator=gen,
        evaluator=ev,
        sim_workers=2,
        exploration_dir_path="./tests_output/test_exploration_steps_no_limit",
    )

    # Run exploration in several steps.
    steps = [3, 4, 10, 5]
    for step in steps:
        exploration.run(step)

    # Check final state.
    assert exploration._n_evals == len(exploration.history)
    assert exploration._n_evals == gen.n_completed_trials
    assert exploration._n_evals == sum(steps)
    assert exploration.history["gen_informed"].to_numpy()[-1]


def test_exploration_resume():
    """Test that an exploration correctly resumes from a previous run."""
    # Define variables and objectives.
    var1 = VaryingParameter("x0", -50.0, 5.0)
    var2 = VaryingParameter("x1", -5.0, 15.0)
    obj = Objective("f", minimize=False)

    # Define variables and objectives.
    gen = RandomSamplingGenerator(
        varying_parameters=[var1, var2], objectives=[obj]
    )

    # Create template evaluator.
    ev = TemplateEvaluator(
        sim_template=os.path.join(
            os.path.abspath(os.path.dirname(__file__)),
            "resources",
            "template_simulation_script.py",
        ),
        analysis_func=analysis_func,
    )

    # Create exploration.
    exploration = Exploration(
        generator=gen,
        evaluator=ev,
        max_evals=40,
        sim_workers=2,
        exploration_dir_path="./tests_output/test_exploration_steps",
        resume=True,
    )

    # Run exploration.
    exploration.run()

    # Check final state.
    assert exploration._n_evals == len(exploration.history)
    assert exploration._n_evals == gen.n_completed_trials
    assert exploration._n_evals == exploration.max_evals
    assert exploration.history["gen_informed"].to_numpy()[-1]
    assert count_history_files(exploration.exploration_dir_path) == 1


def count_history_files(exploration_dir):
    """ "Count the number of history files in a directory."""
    files = os.listdir(exploration_dir)
    count = 0
    for file in files:
        file = str(file)
        if file.endswith(".npy") and "_history_" in file:
            count += 1
    return count


if __name__ == "__main__":
    test_exploration_in_steps()
    test_exploration_in_steps_without_limit()
    test_exploration_resume()
