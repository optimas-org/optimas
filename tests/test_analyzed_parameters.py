import numpy as np

from optimas.explorations import Exploration
from optimas.generators import RandomSamplingGenerator
from optimas.evaluators import FunctionEvaluator
from optimas.core import VaryingParameter, Objective, Parameter


def eval_func(input_params, output_params):
    """Evaluation function used for testing"""
    x0 = input_params["x0"]
    x1 = input_params["x1"]
    result = -(x0 + 10 * np.cos(x0)) * (x1 + 5 * np.cos(x1))
    output_params["f"] = result
    output_params["analyzed_parameter_1"] = result * 3
    output_params["analyzed_parameter_2"] = result * np.pi


def test_analyzed_parameters():
    """
    Test that an exploration runs successfully when including not only an
    objective, but also a set of additional analyzed parameters.
    """
    # Define varying parameters.
    var1 = VaryingParameter("x0", -50.0, 5.0)
    var2 = VaryingParameter("x1", -5.0, 15.0)

    # Define objective and other parameters to analyze.
    obj = Objective("f", minimize=False)
    par1 = Parameter("analyzed_parameter_1")
    par2 = Parameter("analyzed_parameter_2")

    # Create generator.
    gen = RandomSamplingGenerator(
        varying_parameters=[var1, var2],
        objectives=[obj],
        analyzed_parameters=[par1, par2],
    )

    # Create function evaluator.
    ev = FunctionEvaluator(function=eval_func)

    # Create exploration.
    exploration = Exploration(
        generator=gen,
        evaluator=ev,
        max_evals=10,
        sim_workers=2,
        exploration_dir_path="./tests_output/test_analyzed_parameters",
    )

    # Run exploration.
    exploration.run()

    # Get f and analyzed parameters from history array.
    h = exploration.history
    h = h[h["sim_ended"]]
    f_out = h["f"]
    par1_out = h["analyzed_parameter_1"]
    par2_out = h["analyzed_parameter_2"]

    # Check that the values of the analyzed parameters are as expected.
    np.testing.assert_array_equal(par1_out, f_out * 3)
    np.testing.assert_array_equal(par2_out, f_out * np.pi)

    # Save history for later restart test
    np.save(
        "./tests_output/ax_sf_history_with_analyzed_parameters",
        exploration._libe_history.H,
    )


def test_analyzed_parameters_from_history():
    """
    Test that an exploration with analyzed parameters can be successfully
    initialized from a history file. This includes checking that the past
    values of the analyzed parameters in the history file are correctly
    loaded back into the exploration.
    """
    # Define varying parameters.
    var1 = VaryingParameter("x0", -50.0, 5.0)
    var2 = VaryingParameter("x1", -5.0, 15.0)

    # Define objective and other parameters to analyze.
    obj = Objective("f", minimize=False)
    par1 = Parameter("analyzed_parameter_1")
    par2 = Parameter("analyzed_parameter_2")

    # Create generator.
    gen = RandomSamplingGenerator(
        varying_parameters=[var1, var2],
        objectives=[obj],
        analyzed_parameters=[par1, par2],
    )

    # Create function evaluator.
    ev = FunctionEvaluator(function=eval_func)

    # Create exploration.
    exploration = Exploration(
        generator=gen,
        evaluator=ev,
        max_evals=10,
        sim_workers=2,
        history="./tests_output/ax_sf_history_with_analyzed_parameters.npy",
        exploration_dir_path="./tests_output/test_analyzed_parameters_with_history",
    )

    # Run exploration.
    exploration.run()

    # Get f and analyzed parameters from history array.
    h = exploration.history
    h = h[h["sim_ended"]]
    f_out = h["f"]
    par1_out = h["analyzed_parameter_1"]
    par2_out = h["analyzed_parameter_2"]

    # Check that the values of the analyzed parameters are as expected.
    np.testing.assert_array_equal(par1_out, f_out * 3)
    np.testing.assert_array_equal(par2_out, f_out * np.pi)


if __name__ == "__main__":
    test_analyzed_parameters()
    test_analyzed_parameters_from_history()
