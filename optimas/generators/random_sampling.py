"""Contains the definition of the random sampling generator."""

from typing import List, Optional

import numpy as np

from optimas.core import Objective, Trial, VaryingParameter, Parameter
from .base import Generator


class RandomSamplingGenerator(Generator):
    r"""Sample an n-dimensional space with random distributions.

    This generator uses a random distribution to generate a sample of
    configurations where to evaluate the given objectives.

    Parameters
    ----------
    varying_parameters : list of VaryingParameter
        List of input parameters to vary.
    objectives : list of Objective
        List of optimization objectives.
    distribution : {'uniform', 'normal'}, optional
        The random distribution to use. The ``'uniform'`` option draws samples
        from a uniform distribution within the lower :math:`l_b` and upper
        :math:`u_b` bounds of each parameter. The ``'normal'`` option draws
        samples from a normal distribution that, for each parameter, is
        centered at :math:`c = l_b - u_b` with standard deviation
        :math:`\sigma = u_b - c`. By default, ``'uniform'``.
    seed : int, optional
        Seed to initialize the random generator.
    analyzed_parameters : list of Parameter, optional
        List of parameters to analyze at each trial, but which are not
        optimization objectives. By default ``None``.

    """

    def __init__(
        self,
        varying_parameters: List[VaryingParameter],
        objectives: List[Objective],
        distribution: Optional[str] = "uniform",
        seed: Optional[int] = None,
        analyzed_parameters: Optional[List[Parameter]] = None,
    ) -> None:
        super().__init__(
            varying_parameters,
            objectives,
            analyzed_parameters=analyzed_parameters,
        )
        self._generate_sampling = {
            "uniform": self._generate_uniform_sampling,
            "normal": self._generate_normal_sampling,
        }
        self._check_inputs(varying_parameters, objectives, distribution)
        self._distribution = distribution
        self._rng = np.random.default_rng(seed)
        self._define_generator_parameters()

    def _ask(self, trials: List[Trial]) -> List[Trial]:
        """Fill in the parameter values of the requested trials."""
        n_trials = len(trials)
        configs = self._generate_sampling[self._distribution](n_trials)
        for trial, config in zip(trials, configs):
            trial.parameter_values = config
        return trials

    def _check_inputs(
        self,
        varying_parameters: List[VaryingParameter],
        objectives: List[Objective],
        distribution: str,
    ) -> None:
        """Check that the generator inputs are valid."""
        # Check that the specified distribution is supported.
        supported_distributions = list(self._generate_sampling.keys())
        assert (
            distribution in supported_distributions
        ), "Distribution '{}' not recognized. Possible values are {}".format(
            distribution, supported_distributions
        )

    def _define_generator_parameters(self) -> None:
        """Define parameters used by the random generator."""
        self._n_vars = len(self._varying_parameters)
        self._lb = np.array(
            [var.lower_bound for var in self._varying_parameters]
        )
        self._ub = np.array(
            [var.upper_bound for var in self._varying_parameters]
        )
        self._center = (self._lb + self._ub) / 2
        self._width = self._ub - self._center

    def _generate_uniform_sampling(self, n_trials: int) -> np.ndarray:
        """Generate trials using a uniform distribution."""
        return self._rng.uniform(self._lb, self._ub, (n_trials, self._n_vars))

    def _generate_normal_sampling(self, n_trials: int) -> np.ndarray:
        """Generate trials using a normal distribution."""
        return self._rng.normal(
            self._center, self._width, (n_trials, self._n_vars)
        )

    def _mark_trial_as_failed(self, trial: Trial):
        """No need to do anything, since there is no surrogate model."""
        pass
