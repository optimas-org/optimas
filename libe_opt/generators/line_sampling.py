import numpy as np

from .base import Generator


class LineSamplingGenerator(Generator):
    def __init__(self, varying_parameters, objectives, n_steps,
                 analyzed_parameters=None):
        super().__init__(varying_parameters, objectives,
                         analyzed_parameters=analyzed_parameters)
        self._check_inputs(varying_parameters, objectives, n_steps)
        self._n_steps = n_steps if n_steps is np.ndarray else np.array(n_steps)
        self._create_configurations()

    def _check_inputs(self, varying_parameters, objectives, n_steps):
        assert len(n_steps) == len(varying_parameters), (
            'Length of `n_steps` ({}) and '.format(len(n_steps)) +
            '`varying_parameters` ({}) do not match.'.format(
                len(varying_parameters))
        )
        for var in varying_parameters:
            assert var.default_value is not None, (
                'Parameter {} does not have a default value.'.format(var.name)
            )

    def _create_configurations(self):
        # Determine all required quantities.
        lb = [var.lower_bound for var in self._varying_parameters]
        ub = [var.upper_bound for var in self._varying_parameters]
        n_vars = len(self._varying_parameters)
        n_trials = np.sum(self._n_steps)
        default_values = np.array([var.default_value
                                   for var in self._varying_parameters])

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

    def _ask(self, trials):
        for trial in trials:
            if self._all_configs:
                config = self._all_configs.pop(0)
                trial.parameter_values = [
                    config[var.name] for var in trial.varying_parameters]
        return trials
