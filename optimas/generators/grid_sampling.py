"""Contains the definition of the grid sampling generator."""

from typing import List, Optional

import numpy as np

from optimas.core import Objective, Trial, VaryingParameter, Parameter
from .base import Generator


class GridSamplingGenerator(Generator):
    r"""Sample an n-dimensional space with a uniform grid.

    This generator samples the given objectives in a uniform grid of ``n``
    dimensions, where ``n`` is the number of ``varying_parameters``. Along each
    direction :math:`i` (i.e., along each varying parameter), the space is
    divided in :math:`n_\mathrm{steps,i}` evenly spaced steps, resulting in a
    total number of evaluations :math:`\prod_i n_\mathrm{steps,i}`.

    Parameters
    ----------
    varying_parameters : list of VaryingParameter
        List of input parameters to vary.
    objectives : list of Objective
        List of optimization objectives.
    n_steps : list of int
        Number of grid steps along each direction.
    analyzed_parameters : list of Parameter, optional
        List of parameters to analyze at each trial, but which are not
        optimization objectives. By default ``None``.

    """

    def __init__(
        self,
        varying_parameters: List[VaryingParameter],
        objectives: List[Objective],
        n_steps: List[int],
        analyzed_parameters: Optional[List[Parameter]] = None,
    ) -> None:
        super().__init__(
            varying_parameters=varying_parameters,
            objectives=objectives,
            analyzed_parameters=analyzed_parameters,
        )
        self._n_steps = n_steps if n_steps is np.ndarray else np.array(n_steps)
        self._create_configurations()

    def _create_configurations(self) -> None:
        """Create a list will all configurations to be evaluated."""
        # Create a flattened meshgrid for each varying parameter.
        var_linspaces = []
        for var, n_steps_var in zip(self._varying_parameters, self._n_steps):
            var_linspaces.append(
                np.linspace(var.lower_bound, var.upper_bound, n_steps_var)
            )
        var_mgrids = np.meshgrid(*var_linspaces, indexing="ij")
        var_mgrids_flat = [np.ravel(var_mgrid) for var_mgrid in var_mgrids]

        # Calculate total amount of trials and create all configurations.
        all_configs = []
        n_trials = np.prod(self._n_steps)
        for i in range(n_trials):
            config = {}
            for var, mgrid in zip(self._varying_parameters, var_mgrids_flat):
                config[var.name] = mgrid[i]
            all_configs.append(config)
        self._all_configs = all_configs

    def _ask(self, trials: List[Trial]) -> List[Trial]:
        """Fill in the parameter values of the requested trials."""
        for trial in trials:
            if self._all_configs:
                config = self._all_configs.pop(0)
                trial.parameter_values = [
                    config[var.name] for var in trial.varying_parameters
                ]
        return trials

    def _mark_trial_as_failed(self, trial: Trial):
        """No need to do anything, since there is no surrogate model."""
        pass
