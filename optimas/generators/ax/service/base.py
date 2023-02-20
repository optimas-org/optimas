import os

from optimas.generators.ax.base import AxGenerator


class AxServiceGenerator(AxGenerator):
    def __init__(self, varying_parameters, objectives,
                 analyzed_parameters=None, n_init=4, use_cuda=False, gpu_id=0,
                 dedicated_resources=False, save_model=True,
                 model_save_period=5, model_history_dir='model_history'):
        super().__init__(varying_parameters,
                         objectives,
                         analyzed_parameters=analyzed_parameters,
                         use_cuda=use_cuda,
                         gpu_id=gpu_id,
                         dedicated_resources=dedicated_resources,
                         save_model=save_model,
                         model_save_period=model_save_period,
                         model_history_dir=model_history_dir)
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
            for ev in trial.objective_evaluations:
                objective_eval[ev.parameter.name] = (ev.value, ev.sem)
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

    def _save_model_to_file(self):
        file_path = os.path.join(
            self._model_history_dir,
            'ax_client_at_eval_{}.json'.format(
                self._n_completed_trials_last_saved)
        )
        self._ax_client.save_to_json_file(file_path)
