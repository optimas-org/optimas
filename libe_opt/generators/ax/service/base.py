from libe_opt.generators.ax.base import AxGenerator


class AxServiceGenerator(AxGenerator):
    def __init__(self, varying_parameters, objectives, n_init=4,
                 use_cuda=False):
        super().__init__(varying_parameters, objectives, use_cuda=use_cuda)
        self._n_init = n_init
        self._create_ax_client()

    def _ask(self, trials):
        for trial in trials:
            parameters, trial_id = self._ax_client.get_next_trial()
            trial.parameter_values = [
                parameters.get(var.name) for var in self._varying_parameters]
            trial.ax_trial_id = trial_id
        return trials

    def _tell(self, trials):
        for trial in trials:
            objective_eval = {}
            for oe in trial.objective_evaluations:
                objective_eval[oe.objective.name] = (oe.value, oe.sem)
            try:
                self._ax_client.complete_trial(
                            trial_index=trial.ax_trial_id,
                            raw_data=objective_eval
                        )
            except AttributeError:
                params = {}
                for var, value in zip(trial.varying_parameters,
                                      trial.parameter_values):
                    params[var.name] = value
                _, trial_id = self._ax_client.attach_trial(params)
                self._ax_client.complete_trial(trial_id, objective_eval)

    def _create_ax_client(self):
        raise NotImplementedError
