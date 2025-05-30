"""Contains the definition of the multitask Ax generator."""

import os
from copy import deepcopy
from typing import List, Dict, Tuple, Optional, Union
from pyre_extensions import assert_is_instance

import numpy as np
import torch

from ax.core.arm import Arm
from ax.core.batch_trial import BatchTrial
from ax.core.multi_type_experiment import MultiTypeExperiment
from ax.core.parameter import RangeParameter, ParameterType
from ax.core.search_space import SearchSpace
from ax.core.optimization_config import OptimizationConfig
from ax.core.objective import Objective as AxObjective
from ax.runners import SyntheticRunner
from ax.modelbridge.factory import get_sobol
from ax.modelbridge.torch import TorchModelBridge
from ax.core.observation import ObservationFeatures
from ax.core.generator_run import GeneratorRun
from ax.storage.json_store.save import save_experiment
from ax.storage.metric_registry import register_metrics

from ax.modelbridge.registry import Models, ST_MTGP_trans

try:
    # For Ax >= 0.5.0
    from ax.modelbridge.transforms.derelativize import Derelativize
    from ax.modelbridge.transforms.convert_metric_names import (
        ConvertMetricNames,
    )
    from ax.modelbridge.transforms.trial_as_task import TrialAsTask
    from ax.modelbridge.transforms.stratified_standardize_y import (
        StratifiedStandardizeY,
    )
    from ax.modelbridge.transforms.task_encode import TaskChoiceToIntTaskChoice
    from ax.modelbridge.registry import MBM_X_trans

    MT_MTGP_trans = MBM_X_trans + [
        Derelativize,
        ConvertMetricNames,
        TrialAsTask,
        StratifiedStandardizeY,
        TaskChoiceToIntTaskChoice,
    ]

except ImportError:
    # For Ax < 0.5.0
    from ax.modelbridge.registry import MT_MTGP_trans

from ax.core.experiment import Experiment
from ax.core.data import Data
from ax.modelbridge.transforms.convert_metric_names import (
    tconfig_from_mt_experiment,
)

from optimas.generators.ax.base import AxGenerator
from optimas.core import (
    TrialParameter,
    VaryingParameter,
    Objective,
    Parameter,
    Task,
    Trial,
    TrialStatus,
)
from .ax_metric import AxMetric

# Define generator states.
NOT_STARTED = "not_started"
LOFI_RETURNED = "lofi_returned"
HIFI_RETURNED = "hifi_returned"


# get_MTGP is not part of the Ax codebase, as of Ax 0.4.1, due to this PR:
# https://github.com/facebook/Ax/pull/2508
# Here we use `get_MTGP` https://ax.dev/docs/tutorials/multi_task/
def get_MTGP(
    experiment: Experiment,
    data: Data,
    search_space: Optional[SearchSpace] = None,
    trial_index: Optional[int] = None,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.double,
) -> TorchModelBridge:
    """Instantiate a Multi-task Gaussian Process (MTGP) model.

    Points are generated with EI (Expected Improvement).
    If the input experiment is a MultiTypeExperiment then a
    Multi-type Multi-task GP model will be instantiated.
    Otherwise, the model will be a Single-type Multi-task GP.
    """
    if isinstance(experiment, MultiTypeExperiment):
        trial_index_to_type = {
            t.index: t.trial_type for t in experiment.trials.values()
        }
        transforms = MT_MTGP_trans
        transform_configs = {
            "TrialAsTask": {
                "trial_level_map": {"trial_type": trial_index_to_type}
            },
            "ConvertMetricNames": tconfig_from_mt_experiment(experiment),
        }
    else:
        # Set transforms for a Single-type MTGP model.
        transforms = ST_MTGP_trans
        transform_configs = None

    # Choose the status quo features for the experiment from the selected
    # trial. If trial_index is None, we will look for a status quo from the
    # last experiment trial to use as a status quo for the experiment.
    if trial_index is None:
        trial_index = len(experiment.trials) - 1
    elif trial_index >= len(experiment.trials):
        raise ValueError(
            "trial_index is bigger than the number of experiment trials"
        )

    status_quo = experiment.trials[trial_index].status_quo
    if status_quo is None:
        status_quo_features = None
    else:
        status_quo_features = ObservationFeatures(
            parameters=status_quo.parameters,
            trial_index=trial_index,  # pyre-ignore[6]
        )

    return assert_is_instance(
        Models.ST_MTGP(
            experiment=experiment,
            search_space=search_space or experiment.search_space,
            data=data,
            transforms=transforms,
            transform_configs=transform_configs,
            torch_dtype=dtype,
            torch_device=device,
            status_quo_features=status_quo_features,
        ),
        TorchModelBridge,
    )


class AxMultitaskGenerator(AxGenerator):
    """Multitask Bayesian optimization using the Ax developer API.

    Two tasks need to be provided: one for low-fidelity evaluations and
    another for high-fidelity evaluations. The objective will be optimized
    by maximizing (or minimizing) its high-fidelity value. Only one objective
    can be provided.

    Parameters
    ----------
    varying_parameters : list of VaryingParameter
        List of input parameters to vary. One them should be a fidelity.
    objectives : list of Objective
        List of optimization objectives. Only one objective is supported.
    lofi_task, hifi_task : Task
        The low- and high-fidelity tasks.
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
        Whether to save the optimization model (in this case, the Ax
        experiment) to disk. By default ``True``.
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
        lofi_task: Task,
        hifi_task: Task,
        analyzed_parameters: Optional[List[Parameter]] = None,
        use_cuda: Optional[bool] = False,
        gpu_id: Optional[int] = 0,
        dedicated_resources: Optional[bool] = False,
        save_model: Optional[bool] = True,
        model_save_period: Optional[int] = 5,
        model_history_dir: Optional[str] = "model_history",
    ) -> None:
        # As trial parameters these get written to history array
        # Ax trial_index and arm toegther locate a point
        # Multiple points (Optimas trials) can share the same Ax trial_index
        custom_trial_parameters = [
            TrialParameter("arm_name", "ax_arm_name", dtype="U32"),
            TrialParameter("trial_type", "ax_trial_type", dtype="U32"),
            TrialParameter("ax_trial_id", "ax_trial_index", dtype=int),
        ]
        self._check_inputs(varying_parameters, objectives, lofi_task, hifi_task)
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
            custom_trial_parameters=custom_trial_parameters,
        )
        self.lofi_task = lofi_task
        self.hifi_task = hifi_task
        self.model_iteration = 0
        self.hifi_trials = []
        self.n_gen_lofi = 0
        self.n_gen_hifi = 0
        self.gen_state = NOT_STARTED
        self.returned_lofi_trials = 0
        self.returned_hifi_trials = 0
        self.init_batch_limit = 1000
        self.current_trial = None
        self.gr_lofi = None
        self._experiment = self._create_experiment()

    def get_gen_specs(
        self, sim_workers: int, run_params: Dict, sim_max: int
    ) -> Dict:
        """Get the libEnsemble gen_specs."""
        # Get base specs.
        gen_specs = super().get_gen_specs(sim_workers, run_params, sim_max)
        # Add task to output parameters.
        max_length = max([len(self.lofi_task.name), len(self.hifi_task.name)])
        gen_specs["out"].append(("task", str, max_length))
        return gen_specs

    def _check_inputs(
        self,
        varying_parameters: List[VaryingParameter],
        objectives: List[Objective],
        lofi_task: Task,
        hifi_task: Task,
    ) -> None:
        """Check that the generator inputs are valid."""
        # Check that only one objective has been given.
        n_objectives = len(objectives)
        assert n_objectives == 1, (
            "Multitask generator supports only a single objective. "
            "Objectives given: {}.".format(n_objectives)
        )
        # Check that the number of low-fidelity trials per iteration is larger
        # than that of high-fidelity trials.
        assert lofi_task.n_opt >= hifi_task.n_opt, (
            "The number of low-fidelity trials must be larger than or equal "
            "to the number of high-fidelity trials"
        )

    def suggest(self, num_points: Optional[int]) -> List[dict]:
        """Request the next set of points to evaluate."""
        points = []
        for _ in range(num_points):
            next_trial = self._get_next_trial_arm()
            if next_trial is not None:
                arm, trial_type, trial_index = next_trial
                point = {
                    var.name: arm.parameters.get(var.name)
                    for var in self._varying_parameters
                }
                # SH for VOCS standard these will need to be 'variables'
                # For now much match the trial parameter names.
                point["ax_trial_id"] = trial_index
                point["arm_name"] = arm.name
                point["trial_type"] = trial_type
                points.append(point)
        return points

    def ingest(self, results: List[dict]) -> None:
        """Incorporate evaluated trials into experiment."""
        # reconstruct Optimas trials
        trials = []
        for result in results:
            trial = Trial.from_dict(
                trial_dict=result,
                varying_parameters=self._varying_parameters,
                objectives=self._objectives,
                analyzed_parameters=self._analyzed_parameters,
                custom_parameters=self._custom_trial_parameters,
            )
            trials.append(trial)
        if self.gen_state == NOT_STARTED:
            self._incorporate_external_data(trials)
        else:
            self._complete_evaluations(trials)

    def _incorporate_external_data(self, trials: List[Trial]) -> None:
        """Incorporate external data (e.g., from history) into experiment."""
        # Get trial indices.
        # SH should have handling if ax_trial_ids are None...
        trial_indices = []
        for trial in trials:
            trial_indices.append(trial.ax_trial_id)
        trial_indices = np.unique(np.array(trial_indices))

        # Group trials by index.
        grouped_trials = {}
        for index in trial_indices:
            grouped_trials[index] = []
        for trial in trials:
            grouped_trials[trial.ax_trial_id].append(trial)

        # Add trials to experiment.
        for index in trial_indices:
            # Get all trials with current index.
            trials_i = grouped_trials[index]
            trial_type = trials_i[0].trial_type
            # Create arms.
            arms = []
            for trial in trials_i:
                params = {}
                for var, val in zip(
                    trial.varying_parameters, trial.parameter_values
                ):
                    params[var.name] = val
                arms.append(Arm(parameters=params, name=trial.arm_name))
            # Create new batch trial.
            gr = GeneratorRun(arms=arms, weights=[1.0] * len(arms))
            ax_trial = self._experiment.new_batch_trial(
                generator_run=gr, trial_type=trial_type
            )
            ax_trial.run()
            # Incorporate observations.
            for trial in trials_i:
                if trial.status != TrialStatus.FAILED:
                    objective_eval = {}
                    oe = trial.objective_evaluations[0]
                    objective_eval["f"] = (oe.value, oe.sem)
                    ax_trial.run_metadata[trial.arm_name] = objective_eval
                else:
                    ax_trial.mark_arm_abandoned(trial.arm_name)
            # Mark batch trial as completed.
            ax_trial.mark_completed()
            # Keep track of high-fidelity trials.
            if trial_type == self.hifi_task.name:
                self.hifi_trials.append(index)

    def _complete_evaluations(self, trials: List[Trial]) -> None:
        """Complete evaluated trials."""
        for trial in trials:
            # Make sure trial is part of current batch trial.
            current_trial_arms = list(self.current_trial.arms_by_name.keys())
            assert trial.arm_name in current_trial_arms, (
                "Arm {} is not part of current trial. "
                "External data can only be loaded into generator before "
                "initialization."
            )
            if trial.status != TrialStatus.FAILED:
                objective_eval = {}
                oe = trial.objective_evaluations[0]
                objective_eval["f"] = (oe.value, oe.sem)
                self.current_trial.run_metadata[trial.arm_name] = objective_eval
            else:
                self.current_trial.mark_arm_abandoned(trial.arm_name)
            if trial.trial_type == self.lofi_task.name:
                self.returned_lofi_trials += 1
                if self.returned_lofi_trials == self.n_gen_lofi:
                    self.current_trial.mark_completed()
                    self.gen_state = LOFI_RETURNED
            if trial.trial_type == self.hifi_task.name:
                self.returned_hifi_trials += 1
                if self.returned_hifi_trials == self.n_gen_hifi:
                    self.current_trial.mark_completed()
                    self.gen_state = HIFI_RETURNED

    def _create_experiment(self) -> MultiTypeExperiment:
        """Create Ax experiment."""
        # Create search space.
        parameters = []
        for var in self._varying_parameters:
            parameters.append(
                RangeParameter(
                    name=var.name,
                    parameter_type=ParameterType.FLOAT,
                    lower=float(var.lower_bound),
                    upper=float(var.upper_bound),
                )
            )
        search_space = SearchSpace(parameters=parameters)

        # Determine whether tominimize objective.
        minimize = self.objectives[0].minimize

        # Create metrics.
        hifi_objective = AxMetric(name="hifi_metric", lower_is_better=minimize)
        lofi_objective = AxMetric(name="lofi_metric", lower_is_better=minimize)

        # Create optimization config.
        opt_config = OptimizationConfig(
            objective=AxObjective(hifi_objective, minimize=minimize)
        )

        # Create experiment.
        experiment = MultiTypeExperiment(
            name="mt_exp",
            search_space=search_space,
            default_trial_type=self.hifi_task.name,
            default_runner=SyntheticRunner(),
            optimization_config=opt_config,
        )

        # Add low fidelity information.
        experiment.add_trial_type(self.lofi_task.name, SyntheticRunner())
        experiment.add_tracking_metric(
            metric=lofi_objective,
            trial_type=self.lofi_task.name,
            canonical_name="hifi_metric",
        )

        # Register metric in order to be able to save experiment to json file.
        _, encoder_registry, decoder_registry = register_metrics(
            {AxMetric: None}
        )
        self._encoder_registry = encoder_registry
        self._decoder_registry = decoder_registry

        return experiment

    def _get_next_trial_arm(self) -> Union[Tuple[Arm, str, int], None]:
        """Get the next trial arm to evaluate."""
        if self.gen_state in [NOT_STARTED, HIFI_RETURNED]:
            trial = self._get_lofi_batch()
            self.trials_list = [
                (arm, trial.trial_type, trial.index) for arm in trial.arms
            ]
        elif self.gen_state in [LOFI_RETURNED]:
            trial = self._get_hifi_batch()
            self.trials_list = [
                (arm, trial.trial_type, trial.index) for arm in trial.arms
            ]
            self.model_iteration += 1
        if self.trials_list:
            return self.trials_list.pop(0)
        else:
            return None

    def _get_lofi_batch(self) -> BatchTrial:
        """Get the next batch of low-fidelity trials to evaluate."""
        if self.model_iteration == 0:
            # Generate first batch using a Sobol sequence.
            m = get_sobol(self._experiment.search_space, scramble=True)
            n_gen = self.lofi_task.n_init
            gr = m.gen(n_gen)
        else:
            m = get_MTGP(
                experiment=self._experiment,
                data=self._experiment.fetch_data(),
                search_space=self._experiment.search_space,
                dtype=torch.double,
                device=torch.device(self.torch_device),
            )
            n_gen = self.lofi_task.n_opt

            generator_success = True
            while True:
                model_gen_options = {
                    "optimizer_kwargs": {
                        "options": {"init_batch_limit": self.init_batch_limit}
                    }
                }
                try:
                    # Try to generate the new points.
                    gr = m.gen(
                        n=n_gen,
                        optimization_config=(
                            self._experiment.optimization_config
                        ),
                        fixed_features=ObservationFeatures(
                            parameters={}, trial_index=self.hifi_trials[-1]
                        ),
                        model_gen_options=model_gen_options,
                    )
                    # When successful, break loop.
                    break
                except RuntimeError as e:
                    # Print exception.
                    print("RuntimeError: {}".format(e), flush=True)
                    # Divide batch size by 2.
                    self.init_batch_limit //= 2
                    print(
                        "Retrying with `init_batch_limit={}`".format(
                            self.init_batch_limit
                        ),
                        flush=True,
                    )
                finally:
                    # If all attempts have failed (even for batch size of 1),
                    # mark generation as failed and break loop.
                    if self.init_batch_limit == 0:
                        generator_success = False
                        break
            # If generation failed, stop optimization.
            if not generator_success:
                raise RuntimeError(
                    "Failed to generate multitask trials. "
                    "Not sufficient memory is available."
                )
        self.gr_lofi = gr
        trial = self._experiment.new_batch_trial(
            trial_type=self.lofi_task.name, generator_run=gr
        )
        trial.run()
        self.gen_state = "lofi_given"
        self.n_gen_lofi = n_gen
        self.current_trial = trial
        self.returned_lofi_trials = 0
        return trial

    def _get_hifi_batch(self) -> BatchTrial:
        """Get the next batch of high-fidelity trials to evaluate."""
        if self.model_iteration == 0:
            m = get_sobol(self._experiment.search_space, scramble=True)
            n_gen = self.hifi_task.n_init
            gr = m.gen(n_gen)
        else:
            m = get_MTGP(
                experiment=self._experiment,
                data=self._experiment.fetch_data(),
                search_space=self._experiment.search_space,
                dtype=torch.double,
                device=torch.device(self.torch_device),
            )
            n_gen = self.hifi_task.n_opt

            # Select max-utility points from the low fidelity batch to
            # generate a high fidelity batch.
            gr = max_utility_from_GP(
                n=n_gen, m=m, gr=self.gr_lofi, hifi_task=self.hifi_task.name
            )
        trial = self._experiment.new_batch_trial(
            trial_type=self.hifi_task.name, generator_run=gr
        )
        trial.run()
        self.gen_state = "hifi_given"
        self.n_gen_hifi = n_gen
        self.current_trial = trial
        self.returned_hifi_trials = 0
        self.hifi_trials.append(trial.index)
        return trial

    def _save_model_to_file(self) -> None:
        """Save experiment to json file."""
        file_path = os.path.join(
            self._model_history_dir,
            "ax_experiment_at_eval_{}.json".format(
                self._n_evaluated_trials_last_saved
            ),
        )
        save_experiment(
            experiment=self._experiment,
            filepath=file_path,
            encoder_registry=self._encoder_registry,
        )


def max_utility_from_GP(
    n: int, m: TorchModelBridge, gr: GeneratorRun, hifi_task: str
) -> GeneratorRun:
    """Select the max utility points according to the MTGP predictions.

    High fidelity batches are constructed by selecting the maximum utility
    points from the low fidelity batch, after updating the model with the low
    fidelity results.
    """
    obsf = []
    for arm in gr.arms:
        params = deepcopy(arm.parameters)
        params["trial_type"] = hifi_task
        obsf.append(ObservationFeatures(parameters=params))
    # Make predictions
    f, cov = m.predict(obsf)
    # Compute expected utility
    u = -np.array(f["hifi_metric"])
    best_arm_indx = np.flip(np.argsort(u))[:n]
    gr_new = GeneratorRun(
        arms=[gr.arms[i] for i in best_arm_indx],
        weights=[1.0] * n,
    )
    return gr_new
