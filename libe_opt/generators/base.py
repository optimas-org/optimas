import os

from libe_opt.utils.logger import get_logger
from libe_opt.gen_functions import persistent_generator
from libe_opt.core import Objective, Trial, ObjectiveEvaluation

logger = get_logger(__name__)


class Generator():
    def __init__(self, varying_parameters, objectives=None, constraints=None,
                 use_cuda=False, save_model=False, model_save_period=5,
                 model_history_dir='model_history',
                 custom_trial_parameters=None):
        if objectives is None:
            objectives = [Objective()]
        self._varying_parameters = varying_parameters
        self._objectives = objectives
        self._constraints = constraints
        self._save_model = save_model
        self._model_save_period = model_save_period
        self._model_history_dir = model_history_dir
        self._n_completed_trials_last_saved = 0
        self._use_cuda = use_cuda
        self._custom_trial_parameters = (
            [] if custom_trial_parameters is None else custom_trial_parameters)
        self._gen_function = persistent_generator
        self._trials = []

    @property
    def varying_parameters(self):
        return self._varying_parameters

    @property
    def objectives(self):
        return self._objectives

    @property
    def constraints(self):
        return self._constraints

    @property
    def use_cuda(self):
        return self._use_cuda

    def ask(self, n_trials):
        trials = []
        # Initialize as many trials as requested.
        for i in range(n_trials):
            trials.append(
                Trial(
                    varying_parameters=self._varying_parameters,
                    objectives=self._objectives,
                    index=len(self._trials) + i,
                    custom_parameters=self._custom_trial_parameters
                )
            )
        # Ask the generator to fill them.
        trials = self._ask(trials)
        # Keep only trials that have been given data.
        trials = [trial for trial in trials if len(trial.parameter_values) > 0]
        for trial in trials:
            logger.info(
                'Generated trial {} with parameters {}'.format(
                    trial.index, trial.parameters_as_dict()))
        # Store trials.
        self._trials.extend(trials)
        return trials

    def tell(self, trials, allow_saving_model=True):
        for trial in trials:
            if trial not in self._trials:
                trial.index = len(self._trials)
                self._trials.append(trial)
        self._tell(trials)
        for trial in trials:
            logger.info(
                'Completed trial {} with data {}'.format(
                    trial.index, trial.objectives_as_dict()))
        if allow_saving_model and self._save_model:
            self.save_model_to_file()

    def incorporate_history(self, history):
        # Keep only evaluations where the simulation finished sucessfully.
        history = history[history['sim_ended']]
        n_sims = len(history)
        trials = []
        for i in range(n_sims):
            trial = Trial(
                varying_parameters=self.varying_parameters,
                parameter_values=[
                    history[var.name][i] for var in self.varying_parameters],
                objectives=self._objectives,
                objective_evaluations=[
                    ObjectiveEvaluation(
                        objective=obj,
                        value=history[obj.name][i]
                    ) for obj in self._objectives
                ],
                custom_parameters=self._custom_trial_parameters
            )
            for par in self._custom_trial_parameters:
                setattr(trial, par.name, history[par.save_name][i].item())
            trials.append(trial)
        self.tell(trials, allow_saving_model=False)

    def save_model_to_file(self):
        n_completed_trials = 0
        for trial in self._trials:
            if trial.completed():
                n_completed_trials += 1
        n_new = n_completed_trials - self._n_completed_trials_last_saved
        if n_new >= self._model_save_period:
            self._n_completed_trials_last_saved = n_completed_trials
            if not os.path.exists(self._model_history_dir):
                os.mkdir(self._model_history_dir)
            self._save_model_to_file()
            logger.info(
                'Saved model to file after {} completed trials.'.format(
                    n_completed_trials)
            )

    def get_gen_specs(self, sim_workers):
        gen_specs = {
            # Generator function.
            'gen_f': self._gen_function,
            # Generator input. This is a RNG, no need for inputs.
            'in': ['sim_id'],
            'persis_in': (
                ['sim_id', 'trial_index'] +
                [obj.name for obj in self._objectives]
            ),
            'out': (
                [(var.name, float) for var in self._varying_parameters] +
                [('resource_sets', int), ('trial_index', int)] +
                [(par.save_name, par.dtype)
                 for par in self._custom_trial_parameters]
            ),
            'user': {
                'generator': self,
                # Total max number of sims running concurrently.
                'gen_batch_size': sim_workers,
                # Allow generator to run on GPU.
                'use_cuda': self._use_cuda
            }
        }
        return gen_specs

    def get_libe_specs(self):
        libE_specs = {}
        # If not using CUDA, do not allocate resources for generator.
        if not self._use_cuda:
            libE_specs['zero_resource_workers'] = [1]
        return libE_specs

    def _ask(self, trials):
        pass

    def _tell(self, trials):
        pass

    def _save_model_to_file(self):
        pass
