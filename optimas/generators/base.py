"""Contains the definition of the base Generator class."""

from __future__ import annotations
import os
from copy import deepcopy
from typing import List, Dict, Optional, Union

import numpy as np
import pandas as pd

from optimas.utils.logger import get_logger
from optimas.utils.other import update_object, convert_to_dataframe
from optimas.gen_functions import persistent_generator
from optimas.core import (
    Objective,
    Trial,
    Evaluation,
    VaryingParameter,
    Parameter,
    TrialParameter,
    TrialStatus,
)

logger = get_logger(__name__)


class Generator:
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
    allow_fixed_parameters : bool, optional
        Whether the generator supports ``VaryingParameter``s whose value
        has been fixed. By default, False.
    allow_updating_parameters : list of TrialParameter
        Whether the generator supports updating the ``VaryingParameter``s.
        If so, the `_update_parameter` method must be implemented.
        By default, False.

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
        model_history_dir: Optional[str] = "model_history",
        custom_trial_parameters: Optional[List[TrialParameter]] = None,
        allow_fixed_parameters: Optional[bool] = False,
        allow_updating_parameters: Optional[bool] = False,
    ) -> None:
        if objectives is None:
            objectives = [Objective()]
        # Store copies to prevent unexpected behavior if parameters are changed
        # externally.
        self._varying_parameters = deepcopy(varying_parameters)
        self._objectives = deepcopy(objectives)
        self._constraints = constraints
        self._analyzed_parameters = (
            [] if analyzed_parameters is None else analyzed_parameters
        )
        self._save_model = save_model
        self._model_save_period = model_save_period
        self._model_history_dir = model_history_dir
        self._n_evaluated_trials_last_saved = 0
        self._use_cuda = use_cuda
        self._gpu_id = gpu_id
        self._dedicated_resources = dedicated_resources
        self._custom_trial_parameters = (
            [] if custom_trial_parameters is None else custom_trial_parameters
        )
        self._allow_fixed_parameters = allow_fixed_parameters
        self._allow_updating_parameters = allow_updating_parameters
        self._gen_function = persistent_generator
        self._given_trials = []  # Trials given for evaluation.
        self._queued_trials = []  # Trials queued to be given for evaluation.
        self._trial_count = 0
        self._check_parameters(self._varying_parameters)

    @property
    def varying_parameters(self) -> List[VaryingParameter]:
        """Get the list of varying parameters."""
        return self._varying_parameters

    @property
    def objectives(self) -> List[Objective]:
        """Get the list of objectives."""
        return self._objectives

    @property
    def constraints(self) -> Optional[List[Parameter]]:
        """Get the list of constraints."""
        return self._constraints

    @property
    def analyzed_parameters(self) -> List[Parameter]:
        """Get the list of analyzed parameters."""
        return self._analyzed_parameters

    @property
    def use_cuda(self) -> bool:
        """Get whether the generator can use CUDA."""
        return self._use_cuda

    @property
    def gpu_id(self) -> int:
        """Get the ID of the GPU used by the generator."""
        return self._gpu_id

    @property
    def dedicated_resources(self) -> bool:
        """Get whether the generator has dedicated resources allocated."""
        return self._dedicated_resources

    @property
    def n_queued_trials(self) -> int:
        """Get the number of trials queued for evaluation."""
        return len(self._queued_trials)

    @property
    def n_given_trials(self) -> int:
        """Get the number of trials given for evaluation."""
        return len(self._given_trials)

    @property
    def n_completed_trials(self) -> int:
        """Get the number of successfully evaluated trials."""
        n_completed = 0
        for trial in self._given_trials:
            if trial.completed:
                n_completed += 1
        return n_completed

    @property
    def n_failed_trials(self) -> int:
        """Get the number of unsuccessfully evaluated trials."""
        n_failed = 0
        for trial in self._given_trials:
            if trial.failed:
                n_failed += 1
        return n_failed

    @property
    def n_evaluated_trials(self) -> int:
        """Get the number of evaluated trials."""
        n_evaluated = 0
        for trial in self._given_trials:
            if trial.evaluated:
                n_evaluated += 1
        return n_evaluated

    def ask(self, n_trials: int) -> List[Trial]:
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
        # Generate as many trials as needed and add them to the queue.
        if n_trials > self.n_queued_trials:
            n_gen = n_trials - self.n_queued_trials
            gen_trials = []
            for _ in range(n_gen):
                gen_trials.append(
                    Trial(
                        varying_parameters=self._varying_parameters,
                        objectives=self._objectives,
                        analyzed_parameters=self._analyzed_parameters,
                        custom_parameters=self._custom_trial_parameters,
                    )
                )
            # Ask the generator to fill them.
            gen_trials = self._ask(gen_trials)
            # Keep only trials that have been given data.
            for trial in gen_trials:
                if len(trial.parameter_values) > 0:
                    self._add_trial_to_queue(trial)
                    logger.info(
                        "Generated trial {} with parameters {}".format(
                            trial.index, trial.parameters_as_dict()
                        )
                    )

        # Get trials from the queue.
        # The loop below properly handles the case in which the generator
        # was not able to generate as many trials as requested.
        trials = []
        for _ in range(n_trials):
            trial = self._get_next_trial()
            if trial is not None:
                trials.append(trial)
        return trials

    def tell(
        self, trials: List[Trial], allow_saving_model: Optional[bool] = True
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
            if trial not in self._given_trials:
                self._add_external_evaluated_trial(trial)
        self._tell(trials)
        for trial in trials:
            if not trial.failed:
                log_msg = "Completed trial {} with objective(s) {}".format(
                    trial.index, trial.objectives_as_dict()
                )
                if trial.analyzed_parameters:
                    log_msg += " and analyzed parameter(s) {}".format(
                        trial.analyzed_parameters_as_dict()
                    )
            else:
                log_msg = f"Failed to evaluate trial {trial.index}."
            logger.info(log_msg)
        if allow_saving_model and self._save_model:
            self.save_model_to_file()

    def incorporate_history(self, history: np.ndarray) -> None:
        """Incorporate past history into the generator.

        Parameters
        ----------
        history : np.ndarray
            The libEnsemble history array.

        """
        # Keep only evaluations where the simulation finished successfully.
        history = history[history["sim_ended"]]
        trials = self._create_trials_from_external_data(
            history, ignore_unrecognized_parameters=True
        )
        self.tell(trials, allow_saving_model=False)

    def attach_trials(
        self,
        trial_data: Union[Dict, List[Dict], np.ndarray, pd.DataFrame],
        ignore_unrecognized_parameters: Optional[bool] = False,
    ) -> None:
        """Manually add a list of trials to the generator.

        The given trials are placed at the top of the queue of trials that
        will be proposed by the generator (that is, they will be the first
        ones to be proposed the next time that `ask` is called).

        Parameters
        ----------
        trial_data : dict, list, NDArray or DataFrame
            The data containing the trial parameters.
        ignore_unrecognized_parameters : bool, optional
            Whether to ignore unrecognized parameters in the given data.
        """
        trials = self._create_trials_from_external_data(
            trial_data,
            include_evaluations=False,
            ignore_unrecognized_parameters=ignore_unrecognized_parameters,
        )
        # Attach trials to the top of the queue.
        for i, trial in enumerate(trials):
            self._add_trial_to_queue(trial, queue_index=i)
            logger.info(
                "Attached trial {} with parameters {}".format(
                    trial.index, trial.parameters_as_dict()
                )
            )

    def get_trial(self, trial_index) -> Union[Trial, None]:
        """Get trial by index.

        Parameters
        ----------
        trial_index : int
            Index of the trial to retrieve.
        """
        for trial in self._given_trials + self._queued_trials:
            if trial.index == trial_index:
                return trial

    def mark_trial_as_failed(self, trial_index: int):
        """Mark an already evaluated trial as failed.

        Parameters
        ----------
        trial_index : int
            The index of the trial.
        """
        trial = self.get_trial(trial_index)
        if trial.failed:
            return
        elif trial.completed:
            self._mark_trial_as_failed(trial)
            trial.mark_as(TrialStatus.FAILED)
        else:
            raise ValueError(
                "Cannot mark trial as failed because it has not yet been "
                "evaluated."
            )

    def _mark_trial_as_failed(self, trial: Trial):
        raise NotImplementedError(
            f"The trials of a {self.__class__.__name__} cannot be "
            "marked as failed after completion."
        )

    def update_parameter(self, parameter: VaryingParameter):
        """Update a varying parameter of the generator.

        This method should be called, for example, after updating the range
        of the parameter, or fixing its value.

        Parameters
        ----------
        parameter : VaryingParameter
            The updated parameter. It must have the name of one of the
            existing parameters.
        """
        if not self._allow_updating_parameters:
            raise ValueError(
                f"The parameters of a {self.__class__.__name__} cannot be "
                "updated."
            )
        if not isinstance(parameter, VaryingParameter):
            raise ValueError(
                "Updated parameter must be a VaryingParameter, not a "
                f"{type(parameter)}."
            )
        gen_vps = [vp.name for vp in self._varying_parameters]
        if parameter.name not in gen_vps:
            raise ValueError(
                f"Cannot update parameter {parameter.name}. "
                f"Available parameters are {gen_vps}."
            )
        self._check_parameters([parameter])
        for i, vp in enumerate(self._varying_parameters):
            if vp.name == parameter.name:
                self._varying_parameters[i] = parameter
                break
        self._update_parameter(parameter)

    def _add_trial_to_queue(
        self, trial: Trial, queue_index: Optional[int] = None
    ) -> None:
        """Add trial to the queue.

        By default, the trial will be appended to the end of the queue, unless
        a `queue_index` is given. Trials at the top of the queue will be the
        first ones to be given for evaluation when `ask` is called.

        Parameters
        ----------
        trial : Trial
            The trial to add to the queue.
        queue_index : int, optional
            Queue index in which to insert the trial. If not given, the trial
            will be appended to the end of the queue.
        """
        trial.index = self._trial_count
        if queue_index is None:
            self._queued_trials.append(trial)
        else:
            self._queued_trials.insert(queue_index, trial)
        self._trial_count += 1

    def _get_next_trial(self) -> Union[None, Trial]:
        """Get the next trial to evaluate."""
        if self._queued_trials:
            trial = self._queued_trials.pop(0)
            self._given_trials.append(trial)
            return trial

    def _add_external_evaluated_trial(self, trial: Trial) -> None:
        """Add an evaluated trial that was not generated by the generator."""
        trial.index = self._trial_count
        self._given_trials.append(trial)
        self._trial_count += 1

    def _create_trials_from_external_data(
        self,
        trial_data: Union[Dict, List[Dict], np.ndarray, pd.DataFrame],
        include_evaluations: Optional[bool] = True,
        ignore_unrecognized_parameters: Optional[bool] = False,
    ) -> List[Trial]:
        """Create a list of Trials from the given data."""
        # Convert to dataframe.
        trial_data = convert_to_dataframe(trial_data)

        # Get fields in given data.
        given_fields = trial_data.columns.values.tolist()

        # Check for missing fields in the data.
        required_parameters = self.varying_parameters
        if include_evaluations:
            required_parameters = (
                required_parameters + self.objectives + self.analyzed_parameters
            )
        required_fields = [p.name for p in required_parameters]
        required_fields += [p.save_name for p in self._custom_trial_parameters]
        if include_evaluations:
            required_fields += ["trial_status"]
        missing_fields = [f for f in required_fields if f not in given_fields]
        if missing_fields:
            raise ValueError(
                "Could not create trials from given data because the "
                f"required fields {missing_fields} are missing."
            )

        # Check if the given data has more fields than required.
        extra_fields = [f for f in given_fields if f not in required_fields]
        if extra_fields and not ignore_unrecognized_parameters:
            raise ValueError(
                f"The given data contains the fields {extra_fields}, which "
                "are unknown to the generator. If this is expected, ignore "
                "them by setting `ignore_unrecognized_parameters=True`."
            )

        # Create trials.
        n_sims = len(trial_data)
        trials = []
        for i in range(n_sims):
            trial = Trial(
                varying_parameters=self.varying_parameters,
                parameter_values=[
                    trial_data[var.name][i] for var in self.varying_parameters
                ],
                objectives=self._objectives,
                analyzed_parameters=self._analyzed_parameters,
                custom_parameters=self._custom_trial_parameters,
            )
            for par in self._custom_trial_parameters:
                setattr(trial, par.name, trial_data[par.save_name][i])
            if include_evaluations:
                if trial_data["trial_status"][i] == TrialStatus.FAILED.name:
                    trial.mark_as(TrialStatus.FAILED)
                else:
                    for par in self._objectives + self._analyzed_parameters:
                        ev = Evaluation(
                            parameter=par, value=trial_data[par.name][i]
                        )
                        trial.complete_evaluation(ev)
            trials.append(trial)
        return trials

    def save_model_to_file(self) -> None:
        """Save model to file."""
        # Get number of completed trials since last model was saved.
        n_new = self.n_evaluated_trials - self._n_evaluated_trials_last_saved
        # Save model only if save period is reached.
        if n_new >= self._model_save_period:
            self._n_evaluated_trials_last_saved = self.n_evaluated_trials
            if not os.path.exists(self._model_history_dir):
                os.mkdir(self._model_history_dir)
            self._save_model_to_file()
            logger.info(
                "Saved model to file after {} evaluated trials.".format(
                    self.n_evaluated_trials
                )
            )

    def get_gen_specs(
        self, sim_workers: int, run_params: Dict, max_evals: int
    ) -> Dict:
        """Get the libEnsemble gen_specs.

        Parameters
        ----------
        sim_workers : int
            Total number of parallel simulation workers.
        run_params : dict
            Dictionary containing the number of processes and gpus
            required.
        max_evals : int
            Maximum number of evaluations to generate.

        """
        self._prepare_to_send()
        gen_specs = {
            # Generator function.
            "gen_f": self._gen_function,
            # Generator input. This is a RNG, no need for inputs.
            "in": ["sim_id"],
            "persis_in": (
                ["sim_id", "trial_index", "trial_status"]
                + [obj.name for obj in self._objectives]
                + [par.name for par in self._analyzed_parameters]
            ),
            "out": (
                [(var.name, var.dtype) for var in self._varying_parameters]
                + [("num_procs", int), ("num_gpus", int)]
                + [("trial_index", int)]
                + [
                    (par.save_name, par.dtype)
                    for par in self._custom_trial_parameters
                ]
            ),
            "user": {
                # Store the generator itself in gen_specs.
                "generator": self,
                # Total max number of sims running concurrently.
                "gen_batch_size": sim_workers,
                # Allow generator to run on GPU.
                "use_cuda": self._use_cuda,
                # GPU in which to run generator.
                "gpu_id": self._gpu_id,
                # num of procs and gpus required
                "run_params": run_params,
                # Maximum number of evaluations to generate.
                "max_evals": max_evals,
            },
        }
        return gen_specs

    def get_libe_specs(self) -> Dict:
        """Get the libEnsemble libe_specs."""
        libE_specs = {}
        return libE_specs

    def _prepare_to_send(self) -> None:
        """Prepare generator to send to another process.

        This method is necessary because the generator, when given to
        libEnsemble, is sent to another process (the process of the generator
        worker) and then sent back to optimas at the end of the run. In order
        for it to be sent, the generator must be serialized, and sometimes
        some of the contents of the generator cannot be serialized. The
        purpose of this method is to take care of the attributes that prevent
        serialization, and is always called before the generator is sent
        to/from libEnsemble.

        It must be implemented by the subclasses, if needed.
        """
        pass

    def _update(self, new_generator: Generator) -> None:
        """Update generator with the attributes of a newer one.

        This method is only intended to be used internally by an
        ``Exploration`` after a run is completed. It is needed because the
        ``Generator`` given to ``libEnsemble`` is passed as a copy to the
        generator worker and is therefore not updated during the run.

        Parameters
        ----------
        new_generator : Generator
            The newer version of the generator returned in ``persis_info``.

        """
        update_object(self, new_generator)

    def _ask(self, trials: List[Trial]) -> List[Trial]:
        """Ask method to be implemented by the Generator subclasses.

        Parameters
        ----------
        trials : list of Trial
            A list with as many trials as requested to the generator. The
            trials do not yet contain the values of the varying parameters.
            These values should instead be supplied in this method.

        """
        pass

    def _tell(self, trials: List[Trial]) -> None:
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

    def _update_parameter(self, parameter: VaryingParameter):
        """Perform the operations needed by to update the parameter.

        This method must be implemented by the subclasses if
        `allow_updating_parameters=True`.
        """
        raise NotImplementedError

    def _check_parameters(self, parameters: List[VaryingParameter]):
        """Check the validity of the varying parameters."""
        if not self._allow_fixed_parameters:
            for vp in parameters:
                if vp.is_fixed:
                    raise ValueError(
                        f"{self.__class__.__name__} does not support fixing "
                        "the value of a VaryingParameter."
                    )
