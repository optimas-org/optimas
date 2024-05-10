"""Contains the definition of the line sampling generator."""

from typing import List, Optional, Union

import numpy as np

from optimas.core import Objective, Trial, VaryingParameter, Parameter
from .base import Generator


class LineSamplingGenerator(Generator):
    r"""Sample an n-dimensional space one parameter at a time.

    This generator samples the given objectives along ``n`` dimensions, where
    ``n`` is the number of ``varying_parameters``, by varying only one
    parameter at a time. Along each direction :math:`i` (i.e., along each
    varying parameter), the space is divided in :math:`n_\mathrm{steps,i}`
    evenly spaced steps, resulting in a total number of evaluations
    :math:`\sum_i n_\mathrm{steps,i}`.

    Since only one parameter is varied at a time, a default value that will be
    used when a parameter is not being varied needs to be provided for all
    varying parameters.

    Parameters
    ----------
    varying_parameters : list of VaryingParameter
        List of input parameters to vary.
    objectives : list of Objective
        List of optimization objectives.
    n_steps : ndarray or list of int
        A 1D array or list with the number of steps along each direction.
    analyzed_parameters : list of Parameter, optional
        List of parameters to analyze at each trial, but which are not
        optimization objectives. By default ``None``.

    """

    def __init__(
        self,
        varying_parameters: List[VaryingParameter],
        objectives: List[Objective],
        n_steps: Union[np.ndarray, List[int]],
        analyzed_parameters: Optional[List[Parameter]] = None,
    ) -> None:
        super().__init__(
            varying_parameters=varying_parameters,
            objectives=objectives,
            analyzed_parameters=analyzed_parameters,
        )
        self._check_inputs(varying_parameters, objectives, n_steps)
        self._n_steps = n_steps if n_steps is np.ndarray else np.array(n_steps)
        self._create_configurations()

    def _check_inputs(
        self,
        varying_parameters: List[VaryingParameter],
        objectives: List[Objective],
        n_steps: int,
    ) -> None:
        """Check that the generator inputs are valid."""
        # Check as many n_steps as varying_parameters are provided.
        assert len(n_steps) == len(
            varying_parameters
        ), "Length of `n_steps` ({}) and ".format(
            len(n_steps)
        ) + "`varying_parameters` ({}) do not match.".format(
            len(varying_parameters)
        )
        # Check that all varying parameters have a default value.
        for var in varying_parameters:
            assert (
                var.default_value is not None
            ), "Parameter {} does not have a default value.".format(var.name)

    def _create_configurations(self) -> None:
        """Create a list will all configurations to be evaluated."""
        # Determine all required quantities.
        lb = [var.lower_bound for var in self._varying_parameters]
        ub = [var.upper_bound for var in self._varying_parameters]
        n_vars = len(self._varying_parameters)
        n_trials = np.sum(self._n_steps)
        default_values = np.array(
            [var.default_value for var in self._varying_parameters]
        )

        # Generate configurations.
        all_configs_array = np.ones((n_trials, n_vars)) * default_values
        for i in range(n_vars):
            i_var_min = np.sum(self._n_steps[:i])
            i_var_max = i_var_min + self._n_steps[i]
            var_vals = np.linspace(lb[i], ub[i], self._n_steps[i])
            all_configs_array[i_var_min:i_var_max, i] = var_vals

        # Turn them into a list of dictionaries.
        all_configs = []
        for config_array in all_configs_array:
            config = {}
            for var, value in zip(self._varying_parameters, config_array):
                config[var.name] = value
            all_configs.append(config)

        # Store configurations.
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
