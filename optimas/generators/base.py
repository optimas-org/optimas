"""Contains the definition of the base Generator class."""

import os
from typing import List, Dict, Optional

import numpy as np

from optimas.utils.logger import get_logger
from optimas.gen_functions import persistent_generator
from optimas.core import (Objective, Trial, Evaluation, VaryingParameter,
                          Parameter, TrialParameter)

logger = get_logger(__name__)


class Generator():
    """Base class for all generators.

    Parameters
    ----------
    varying_parameters : list of VaryingParameter
        List of input parameters to vary.
    objectives : list of Objective
        List of optimization objectives.
    constraints : list of Parameter, optional
        [Not yet implemented] List of optimization constraints. By default
        ``None``.
    analyzed_parameters : list of Parameter, optional
        List of parameters to analyze at each trial, but which are not
        optimization objectives. By default ``None``.
    use_cuda : bool, optional
        Whether to allow the generator to run on a CUDA GPU. By default
        ``False``.
    gpu_id : int, optional
        The ID of the GPU in which to run the generator. By default, ``0``.
    dedicated_resources : bool, optional
        Whether to allocated dedicated resources (e.g., the GPU) for the
        generator. These resources will not be available to the
        simulation workers. By default, ``False``.
    save_model : bool, optional
        Whether to save the optimization model (e.g., the surrogate model) to
        disk. By default ``False``.
    model_save_period : int, optional
        Periodicity, in number of evaluated Trials, with which to save the
        model to disk. By default, ``5``.
    model_history_dir : str, optional
        Name of the directory in which the model will be saved. By default,
        ``'model_history'``.
    custom_trial_parameters : list of TrialParameter
        For some generators, it might be necessary to attach additional
        parameters to the trials. If so, they can be given here as a list.
        By default, ``None``.
    """
    def __init__(
        self,
        varying_parameters: List[VaryingParameter],
        objectives: List[Objective],
        constraints: Optional[List[Parameter]] = None,
        analyzed_parameters: Optional[List[Parameter]] = None,
        use_cuda: Optional[bool] = False,
        gpu_id: Optional[int] = 0,
        dedicated_resources: Optional[bool] = False,
        save_model: Optional[bool] = False,
        model_save_period: Optional[int] = 5,
        model_history_dir: Optional[str] = 'model_history',
        custom_trial_parameters: Optional[TrialParameter] = None
    ) -> None:
        if objectives is None:
            objectives = [Objective()]
        self._varying_parameters = varying_parameters
        self._objectives = objectives
        self._constraints = constraints
        self._analyzed_parameters = (
            [] if analyzed_parameters is None else analyzed_parameters
        )
        self._save_model = save_model
        self._model_save_period = model_save_period
        self._model_history_dir = model_history_dir
        self._n_completed_trials_last_saved = 0
        self._use_cuda = use_cuda
        self._gpu_id = gpu_id
        self._dedicated_resources = dedicated_resources
        self._custom_trial_parameters = (
            [] if custom_trial_parameters is None else custom_trial_parameters
        )
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
    def analyzed_parameters(self):
        return self._analyzed_parameters

    @property
    def use_cuda(self):
        return self._use_cuda

    @property
    def gpu_id(self):
        return self._gpu_id

    @property
    def dedicated_resources(self):
        return self._dedicated_resources

    def ask(
        self,
        n_trials: int
    ) -> List[Trial]:
        """Ask the generator to suggest the next ``n_trials`` to evaluate.

        Parameters
        ----------
        n_trials : int
            The number of trials to generate.

        Returns
        -------
        list of Trial
            A list with all the generated trials.
        """
        trials = []
        # Initialize as many trials as requested.
        for i in range(n_trials):
            trials.append(
                Trial(
                    varying_parameters=self._varying_parameters,
                    objectives=self._objectives,
                    analyzed_parameters=self._analyzed_parameters,
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

    def tell(
        self,
        trials: List[Trial],
        allow_saving_model: Optional[bool] = True
    ) -> None:
        """Give trials back to generator once they have been evaluated.

        Parameters
        ----------
        trials : list of Trial
            The list of evaluated trials.
        allow_saving_model : bool, optional
            Whether to allow the generator to save the model to file after
            incorporating the evaluated trials. By default ``True``.
        """
        for trial in trials:
            if trial not in self._trials:
                trial.index = len(self._trials)
                self._trials.append(trial)
        self._tell(trials)
        for trial in trials:
            log_msg = 'Completed trial {} with objective(s) {}'.format(
                trial.index, trial.objectives_as_dict())
            if trial.analyzed_parameters:
                log_msg += ' and analyzed parameter(s) {}'.format(
                    trial.analyzed_parameters_as_dict())
            logger.info(log_msg)
        if allow_saving_model and self._save_model:
            self.save_model_to_file()

    def incorporate_history(
        self,
        history: np.ndarray
    ) -> None:
        """Incorporate past history into the generator.

        Parameters
        ----------
        history : np.ndarray
            The libEnsemble history array.
        """
        # Keep only evaluations where the simulation finished successfully.
        history = history[history['sim_ended']]
        n_sims = len(history)
        trials = []
        for i in range(n_sims):
            trial = Trial(
                varying_parameters=self.varying_parameters,
                parameter_values=[
                    history[var.name][i] for var in self.varying_parameters],
                objectives=self._objectives,
                analyzed_parameters=self._analyzed_parameters,
                evaluations=[
                    Evaluation(
                        parameter=par,
                        value=history[par.name][i]
                    ) for par in self._objectives + self._analyzed_parameters
                ],
                custom_parameters=self._custom_trial_parameters
            )
            for par in self._custom_trial_parameters:
                setattr(trial, par.name, history[par.save_name][i].item())
            trials.append(trial)
        self.tell(trials, allow_saving_model=False)

    def save_model_to_file(self) -> None:
        """Save model to file"""
        # Get total number of completed trials.
        n_completed_trials = 0
        for trial in self._trials:
            if trial.completed():
                n_completed_trials += 1
        # Get number of completed trials since last model was saved.
        n_new = n_completed_trials - self._n_completed_trials_last_saved
        # Save model only if save period is reached.
        if n_new >= self._model_save_period:
            self._n_completed_trials_last_saved = n_completed_trials
            if not os.path.exists(self._model_history_dir):
                os.mkdir(self._model_history_dir)
            self._save_model_to_file()
            logger.info(
                'Saved model to file after {} completed trials.'.format(
                    n_completed_trials)
            )

    def get_gen_specs(
        self,
        sim_workers: int
    ) -> Dict:
        """Get the libEnsemble gen_specs.

        Parameters
        ----------
        sim_workers : int
            Total number of parallel simulation workers.
        """
        gen_specs = {
            # Generator function.
            'gen_f': self._gen_function,
            # Generator input. This is a RNG, no need for inputs.
            'in': ['sim_id'],
            'persis_in': (
                ['sim_id', 'trial_index'] +
                [obj.name for obj in self._objectives] +
                [par.name for par in self._analyzed_parameters]
            ),
            'out': (
                [(var.name, var.dtype) for var in self._varying_parameters] +
                [('resource_sets', int), ('trial_index', int)] +
                [(par.save_name, par.dtype)
                 for par in self._custom_trial_parameters]
            ),
            'user': {
                # Store the generator itself in gen_specs.
                'generator': self,
                # Total max number of sims running concurrently.
                'gen_batch_size': sim_workers,
                # Allow generator to run on GPU.
                'use_cuda': self._use_cuda,
                # GPU in which to run generator.
                'gpu_id': self._gpu_id
            }
        }
        return gen_specs

    def get_libe_specs(self) -> Dict:
        """Get the libEnsemble libe_specs."""
        libE_specs = {}
        return libE_specs

    def _ask(
        self,
        trials: List[Trial]
    ) -> List[Trial]:
        """Ask method to be implemented by the Generator subclasses.

        Parameters
        ----------
        trials : list of Trial
            A list with as many trials as requested to the generator. The
            trials do not yet contain the values of the varying parameters.
            These values should instead be supplied in this method.
        """
        pass

    def _tell(
        self,
        trials: List[Trial]
    ) -> None:
        """Tell method to be implemented by the Generator subclasses.

        Parameters
        ----------
        trials : list of Trial
            A list with all evaluated trials. All evaluations included in the
            trials should be incorporated to the generator model in this
            method.
        """
        pass

    def _save_model_to_file(self):
        """Save model method to be implemented by the Generator subclasses."""
        pass
