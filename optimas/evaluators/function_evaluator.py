"""Contains the definition of the FunctionEvaluator class."""

from typing import Callable, Dict, List

from optimas.sim_functions import run_function
from optimas.core import VaryingParameter, Objective, Parameter
from .base import Evaluator


class FunctionEvaluator(Evaluator):
    """Evaluator class for executing an arbitrary function.

    Parameters
    ----------
    function : callable
        The function to be evaluated.

    """

    def __init__(self, function: Callable) -> None:
        super().__init__(sim_function=run_function)
        self.function = function

    def get_sim_specs(
        self,
        varying_parameters: List[VaryingParameter],
        objectives: List[Objective],
        analyzed_parameters: List[Parameter],
    ) -> Dict:
        """Get the `sim_specs` for `libEnsemble`."""
        # Get base sim_specs.
        sim_specs = super().get_sim_specs(
            varying_parameters, objectives, analyzed_parameters
        )
        # Add evaluation function to sim_specs.
        sim_specs["user"]["evaluation_func"] = self.function
        return sim_specs
