"""Contains the definition of the base Ax generator using the service API."""

from typing import List, Optional, Dict
import os

import torch
from ax.service.ax_client import AxClient
from ax.service.utils.instantiation import (
    InstantiationBase,
    ObjectiveProperties,
    FixedFeatures,
)
from ax.modelbridge.registry import Models
from ax.modelbridge.generation_strategy import (
    GenerationStep,
    GenerationStrategy,
)
from ax.modelbridge.transition_criterion import MaxTrials, MinTrials
from ax import Arm

from optimas.core import (
    Objective,
    Trial,
    VaryingParameter,
    Parameter,
    TrialStatus,
)
from optimas.generators.ax.base import AxGenerator
from optimas.utils.ax import AxModelManager
from optimas.utils.ax.other import (
    convert_optimas_to_ax_parameters,
    convert_optimas_to_ax_objectives,
)


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
    parameter_constraints : list of str, optional
        List of string representation of parameter
        constraints, such as ``"x3 >= x4"`` or ``"-x3 + 2*x4 - 3.5*x5 >= 2"``.
        For the latter constraints, any number of arguments is accepted, and
        acceptable operators are ``<=`` and ``>=``.
    outcome_constraints : list of str, optional
        List of string representation of outcome constraints (i.e., constraints
        on any of the ``analyzed_parameters``) of form
        ``"metric_name >= bound"``, like ``"m1 <= 3."``.
    n_init : int, optional
        Number of evaluations to perform during the initialization phase using
        Sobol sampling. If external data is attached to the exploration, the
        number of initialization evaluations will be reduced by the same
        amount, unless `enforce_n_init=True`. By default, ``4``.
    enforce_n_init : bool, optional
        Whether to enforce the generation of `n_init` Sobol trials, even if
        external data is supplied. By default, ``False``.
    abandon_failed_trials : bool, optional
        Whether failed trials should be abandoned (i.e., not suggested again).
        By default, ``True``.
    fit_out_of_design : bool, optional
        Whether to fit the surrogate model taking into account evaluations
        outside of the range of the varying parameters. This can be useful
        if the range of parameter has been reduced during the optimization.
        By default, False.
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
        parameter_constraints: Optional[List[str]] = None,
        outcome_constraints: Optional[List[str]] = None,
        n_init: Optional[int] = 4,
        enforce_n_init: Optional[bool] = False,
        abandon_failed_trials: Optional[bool] = True,
        fit_out_of_design: Optional[bool] = False,
        use_cuda: Optional[bool] = False,
        gpu_id: Optional[int] = 0,
        dedicated_resources: Optional[bool] = False,
        save_model: Optional[bool] = True,
        model_save_period: Optional[int] = 5,
        model_history_dir: Optional[str] = "model_history",
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
            model_history_dir=model_history_dir,
            allow_fixed_parameters=True,
            allow_updating_parameters=True,
        )
        self._n_init = n_init
        self._enforce_n_init = enforce_n_init
        self._abandon_failed_trials = abandon_failed_trials
        self._fit_out_of_design = fit_out_of_design
        self._fixed_features = None
        self._parameter_constraints = parameter_constraints
        self._outcome_constraints = outcome_constraints
        self._ax_client = self._create_ax_client()
        self._model = AxModelManager(self._ax_client)

    @property
    def ax_client(self) -> AxClient:
        """Get the underlying AxClient."""
        return self._ax_client

    @property
    def model(self) -> AxModelManager:
        """Get access to the underlying model using an `AxModelManager`."""
        return self._model

    def _ask(self, trials: List[Trial]) -> List[Trial]:
        """Fill in the parameter values of the requested trials."""
        for trial in trials:
            parameters, trial_id = self._ax_client.get_next_trial(
                fixed_features=self._fixed_features
            )
            trial.parameter_values = [
                parameters.get(var.name) for var in self._varying_parameters
            ]
            trial.ax_trial_id = trial_id
        return trials

    def _tell(self, trials: List[Trial]) -> None:
        """Incorporate evaluated trials into Ax client."""
        for trial in trials:
            try:
                trial_id = trial.ax_trial_id
                ax_trial = self._ax_client.get_trial(trial_id)
            except AttributeError:
                params = {}
                for var, value in zip(
                    trial.varying_parameters, trial.parameter_values
                ):
                    params[var.name] = value
                try:
                    _, trial_id = self._ax_client.attach_trial(params)
                except ValueError as error:
                    # Bypass checks from AxClient and manually add a trial
                    # outside of the search space.
                    # https://github.com/facebook/Ax/issues/768#issuecomment-1036515242
                    if "not a valid value" in str(error):
                        if self._fit_out_of_design:
                            ax_trial = self._ax_client.experiment.new_trial()
                            ax_trial.add_arm(Arm(parameters=params))
                            ax_trial.mark_running(no_runner_required=True)
                            trial_id = ax_trial.index
                        else:
                            ignore_reason = (
                                f"The parameters {params} are outside of the "
                                "range of the varying parameters. "
                                "Set `fit_out_of_design=True` if you want "
                                "the model to use these data."
                            )
                            trial.ignore(reason=ignore_reason)
                            continue
                    else:
                        raise error
                ax_trial = self._ax_client.get_trial(trial_id)

                # Since data was given externally, reduce number of
                # initialization trials, but only if they have not failed.
                if trial.completed and not self._enforce_n_init:
                    generation_strategy = self._ax_client.generation_strategy
                    current_step = generation_strategy.current_step
                    # Reduce only if there are still Sobol trials left.
                    if current_step.model == Models.SOBOL:
                        for tc in current_step.transition_criteria:
                            # Looping over all criterial makes sure we reduce
                            # the transition thresholds due to `_n_init`
                            # (i.e., max trials) and `min_trials_observed=1` (
                            # i.e., min trials).
                            if isinstance(tc, (MinTrials, MaxTrials)):
                                tc.threshold -= 1
                        generation_strategy._maybe_transition_to_next_node()
            finally:
                if trial.ignored:
                    continue
                elif trial.completed:
                    outcome_evals = {}
                    # Add objective evaluations.
                    for ev in trial.objective_evaluations:
                        outcome_evals[ev.parameter.name] = (ev.value, ev.sem)
                    # Add outcome constraints evaluations.
                    ax_config = self._ax_client.experiment.optimization_config
                    if ax_config.outcome_constraints:
                        ocs = [
                            oc.metric.name
                            for oc in ax_config.outcome_constraints
                        ]
                        for ev in trial.parameter_evaluations:
                            par_name = ev.parameter.name
                            if par_name in ocs:
                                outcome_evals[par_name] = (ev.value, ev.sem)
                    self._ax_client.complete_trial(
                        trial_index=trial_id, raw_data=outcome_evals
                    )
                elif trial.failed:
                    if self._abandon_failed_trials:
                        ax_trial.mark_abandoned()
                    else:
                        ax_trial.mark_failed()

    def _create_ax_client(self) -> AxClient:
        """Create Ax client."""
        bo_model_kwargs = {
            "torch_dtype": torch.double,
            "torch_device": torch.device(self.torch_device),
            "fit_out_of_design": self._fit_out_of_design,
        }
        ax_client = AxClient(
            generation_strategy=GenerationStrategy(
                self._create_generation_steps(bo_model_kwargs)
            ),
            verbose_logging=False,
        )
        ax_client.create_experiment(
            parameters=self._create_ax_parameters(),
            objectives=self._create_ax_objectives(),
            outcome_constraints=self._outcome_constraints,
            parameter_constraints=self._parameter_constraints,
        )
        return ax_client

    def _create_ax_parameters(self) -> List:
        """Create list of parameters to pass to an Ax."""
        parameters = convert_optimas_to_ax_parameters(self.varying_parameters)
        fixed_parameters = {}
        for var in self._varying_parameters:
            if var.is_fixed:
                fixed_parameters[var.name] = var.default_value
        # Store fixed parameters as fixed features.
        self._fixed_features = FixedFeatures(fixed_parameters)
        return parameters

    def _create_ax_objectives(self) -> Dict[str, ObjectiveProperties]:
        """Create list of objectives to pass to an Ax."""
        return convert_optimas_to_ax_objectives(self.objectives)

    def _create_sobol_step(self) -> GenerationStep:
        """Create a Sobol generation step with `n_init` trials."""
        # Ensure that at least 1 trial is completed before moving onto the BO
        # step, and keep generating Sobol trials until that happens, even if
        # the number of Sobol trials exceeds `n_init`.
        # Otherwise, if we move to the BO step before any trial is completed,
        # the next `ask` would fail with a `DataRequiredError`.
        # This also allows the generator to work well when
        # `sim_workers` > `n_init`.
        return GenerationStep(
            model=Models.SOBOL,
            num_trials=self._n_init,
            min_trials_observed=1,
            enforce_num_trials=False,
        )

    def _create_generation_steps(
        self, bo_model_kwargs: Dict
    ) -> List[GenerationStep]:
        """Create generation steps (must be implemented by subclasses)."""
        raise NotImplementedError

    def _save_model_to_file(self) -> None:
        """Save Ax client to json file."""
        file_path = os.path.join(
            self._model_history_dir,
            "ax_client_at_eval_{}.json".format(
                self._n_evaluated_trials_last_saved
            ),
        )
        self._ax_client.save_to_json_file(file_path)

    def _update_parameter(self, parameter):
        """Update a parameter from the search space."""
        # Delete the fitted model from the generation strategy, otherwise
        # the parameter won't be updated.
        generation_strategy = self._ax_client.generation_strategy
        if generation_strategy._model is not None:
            del generation_strategy._curr.model_spec._fitted_model
        parameters = self._create_ax_parameters()
        new_search_space = InstantiationBase.make_search_space(parameters, None)
        self._ax_client.experiment.search_space.update_parameter(
            new_search_space.parameters[parameter.name]
        )

    def _mark_trial_as_failed(self, trial: Trial):
        """Mark a trial as failed so that is not used for fitting the model."""
        ax_trial = self._ax_client.get_trial(trial.ax_trial_id)
        if self._abandon_failed_trials:
            ax_trial.mark_abandoned(unsafe=True)
        else:
            ax_trial.mark_failed(unsafe=True)
