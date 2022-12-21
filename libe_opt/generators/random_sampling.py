import numpy as np

from .base import Generator


class RandomSamplingGenerator(Generator):
    def __init__(self, varying_parameters, objectives, distribution='uniform',
                 seed=None):
        super().__init__(varying_parameters, objectives)
        self._generate_sampling = {
            'uniform': self._generate_uniform_sampling,
            'normal': self._generate_normal_sampling,
        }
        self._check_inputs(varying_parameters, objectives, distribution)
        self._distribution = distribution
        self._rng = np.random.default_rng(seed)
        self._define_generator_parameters()

    def _ask(self, trials):
        n_trials = len(trials)
        configs = self._generate_sampling[self._distribution](n_trials)
        for trial, config in zip(trials, configs):
            trial.parameter_values = config
        return trials

    def _check_inputs(self, varying_parameters, objectives, distribution):
        supported_distributions = list(self._generate_sampling.keys())
        assert distribution in supported_distributions, (
            "Distribution '{}' not recognized. Possible values are {}".format(
                distribution, supported_distributions)
        )

    def _define_generator_parameters(self):
        self._n_vars = len(self._varying_parameters)
        self._lb = np.array([var.lower_bound
                             for var in self._varying_parameters])
        self._ub = np.array([var.upper_bound
                             for var in self._varying_parameters])
        self._center = (self._lb + self._ub) / 2
        self._width = (self._ub - self._center)

    def _generate_uniform_sampling(self, n_trials):
        return self._rng.uniform(self._lb, self._ub, (n_trials, self._n_vars))

    def _generate_normal_sampling(self, n_trials):
        return self._rng.normal(self._center, self._width,
                                (n_trials, self._n_vars))
