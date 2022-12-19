import numpy as np

from .base import Generator


class GridSamplingGenerator(Generator):
    def __init__(self, variables, objectives, n_steps):
        super().__init__(variables, objectives)
        self._n_steps = n_steps if n_steps is np.ndarray else np.array(n_steps)
        self._create_configurations()

    def _create_configurations(self):
        var_linspaces = []
        for var, n_steps_var in zip(self.variables, self._n_steps):
            var_linspaces.append(
                np.linspace(var.lower_bound, var.upper_bound, n_steps_var))
        var_mgrids = np.meshgrid(*var_linspaces, indexing='ij')
        var_mgrids_flat = [np.ravel(var_mgrid) for var_mgrid in var_mgrids]

        all_configs = []
        n_trials = np.prod(self._n_steps)  
        for i in range(n_trials):
            config = {}
            for var, mgrid in zip(self.variables, var_mgrids_flat):
                config[var.name] = mgrid[i]
            all_configs.append(config)
        
        self._all_configs = all_configs

    def _ask(self, trials):
        for trial in trials:
            if self._all_configs:
                config = self._all_configs.pop(0)
                trial.variable_values = [config[var.name] for var in trial.variables]
        return trials
