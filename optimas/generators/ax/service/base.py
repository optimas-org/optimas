"""Contains the definition of the base Ax generator using the service API."""

from typing import List, Optional

import os

from ax.service.ax_client import AxClient

from optimas.utils.other import update_object
from optimas.core import Objective, Trial, VaryingParameter, Parameter
from optimas.generators.ax.base import AxGenerator
from optimas.generators.base import Generator


class AxServiceGenerator(AxGenerator):
    """Base class for all Ax generators using the service API.

    Parameters
    ----------
    varying_parameters : list of VaryingParameter
        List of input parameters to vary.
    objectives : list of Objective
        List of optimization objectives.
    analyzed_parameters : list of Parameter, optional
        List of parameters to analyze at each trial, but which are not
        optimization objectives. By default ``None``.
    n_init : int, optional
        Number of evaluations to perform during the initialization phase using
        Sobol sampling. By default, ``4``.
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
        Whether to save the optimization model (in this case, the Ax client) to
        disk. By default ``True``.
    model_save_period : int, optional
        Periodicity, in number of evaluated Trials, with which to save the
        model to disk. By default, ``5``.
    model_history_dir : str, optional
        Name of the directory in which the model will be saved. By default,
        ``'model_history'``.
    """
    def __init__(
        self,
        varying_parameters: List[VaryingParameter],
        objectives: List[Objective],
        analyzed_parameters: Optional[List[Parameter]] = None,
        n_init: Optional[int] = 4,
        use_cuda: Optional[bool] = False,
        gpu_id: Optional[int] = 0,
        dedicated_resources: Optional[bool] = False,
        save_model: Optional[bool] = True,
        model_save_period: Optional[int] = 5,
        model_history_dir: Optional[str] = 'model_history',
    ) -> None:
        super().__init__(
            varying_parameters=varying_parameters,
            objectives=objectives,
            analyzed_parameters=analyzed_parameters,
            use_cuda=use_cuda,
            gpu_id=gpu_id,
            dedicated_resources=dedicated_resources,
            save_model=save_model,
            model_save_period=model_save_period,
            model_history_dir=model_history_dir
        )
        self._n_init = n_init
        self._ax_client = self._create_ax_client()

    def _ask(
        self,
        trials: List[Trial]
    ) -> List[Trial]:
        """Fill in the parameter values of the requested trials."""
        for trial in trials:
            parameters, trial_id = self._ax_client.get_next_trial()
            trial.parameter_values = [
                parameters.get(var.name) for var in self._varying_parameters]
            trial.ax_trial_id = trial_id
        return trials

    def _tell(
        self,
        trials: List[Trial]
    ) -> None:
        """Incorporate evaluated trials into Ax client."""
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

    def _create_ax_client(self) -> AxClient:
        """Create Ax client (must be implemented by subclasses)"""
        raise NotImplementedError

    def _save_model_to_file(self) -> None:
        """Save Ax client to json file."""
        file_path = os.path.join(
            self._model_history_dir,
            'ax_client_at_eval_{}.json'.format(
                self._n_completed_trials_last_saved)
        )
        self._ax_client.save_to_json_file(file_path)

    def _prepare_to_send(self) -> None:
        """Prepare generator to send to another process.

        Delete the fitted model from the generation strategy. It can contain
        pytorch tensors that prevent serialization.
        """
        generation_strategy = self._ax_client.generation_strategy
        if generation_strategy._model is not None:
            del generation_strategy._curr.model_spec._fitted_model
            generation_strategy._curr.model_spec._fitted_model = None
            del generation_strategy._model
            generation_strategy._model = None

    def _update(
        self,
        new_generator: Generator
    ) -> None:
        """Update generator with the attributes of a newer one.

        This method is overrides the base one to make sure that the original
        AxClient is updated and not simply replaced.

        Parameters
        ----------
        new_generator : Generator
            The newer version of the generator returned in ``persis_info``.
        """
        original_ax_client = self._ax_client
        super()._update(new_generator)
        update_object(original_ax_client, new_generator._ax_client)
        self._ax_client = original_ax_client
