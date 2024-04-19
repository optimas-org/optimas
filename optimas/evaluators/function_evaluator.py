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
    create_evaluation_dirs : bool
        Whether to create a directory for each evaluation. The directories will
        be located in `./evaluations` and be named `sim{trial_index}`. When
        using this option, the current working directory inside the ``function``
        will be changed to the corresponding evaluation directory.
        By default, ``False``.

    """

    def __init__(
        self, function: Callable, create_evaluation_dirs: bool = False
    ) -> None:
        super().__init__(sim_function=run_function)
        self.function = function
        self._create_evaluation_dirs = create_evaluation_dirs

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

    def get_libe_specs(self) -> Dict:
        """Get the `libE_specs` for `libEnsemble`."""
        libE_specs = super().get_libe_specs()
        libE_specs["sim_dirs_make"] = self._create_evaluation_dirs
        return libE_specs
