"""Contains the definition of the FunctionEvaluator class."""

from typing import Callable, Optional, Dict, List

from optimas.sim_functions import run_function
from optimas.core import VaryingParameter, Objective, Parameter
from .base import Evaluator


class FunctionEvaluator(Evaluator):
    """Evaluator class to use when the evaluations are carried out by calling
    a function.

    Parameters
    ----------
    function : callable
        The function to be evaluated.
    n_procs : int, optional
        The number of processes that will be used for each evaluation. By
        default, ``n_procs=1`` if ``n_gpus`` is not given. Otherwise, the
        default behavior is to match the number of processes to the number
        of GPUs, i.e., ``n_procs=n_gpus``.
    n_gpus : int, optional
        The number of GPUs that will be made available for each evaluation. By
        default, 0.
    """
    def __init__(
        self,
        function: Callable,
        n_procs: Optional[int] = None,
        n_gpus: Optional[int] = None
    ) -> None:
        super().__init__(
            sim_function=run_function,
            n_procs=n_procs,
            n_gpus=n_gpus
        )
        self.function = function

    def get_sim_specs(
        self,
        varying_parameters: List[VaryingParameter],
        objectives: List[Objective],
        analyzed_parameters: List[Parameter],
    ) -> Dict:
        """Get a dictionary with the ``sim_specs`` as expected
        by ``libEnsemble``
        """
        # Get base sim_specs.
        sim_specs = super().get_sim_specs(varying_parameters, objectives,
                                          analyzed_parameters)
        # Add evaluation function to sim_specs.
        sim_specs['user']['evaluation_func'] = self.function
        return sim_specs
