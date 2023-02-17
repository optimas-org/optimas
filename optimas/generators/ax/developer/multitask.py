import os
from copy import deepcopy

import numpy as np
import torch

from ax.core.arm import Arm
from ax.core.multi_type_experiment import MultiTypeExperiment
from ax.core.parameter import RangeParameter, ParameterType
from ax.core.search_space import SearchSpace
from ax.core.optimization_config import OptimizationConfig
from ax.core.objective import Objective
from ax.runners import SyntheticRunner
from ax.modelbridge.factory import get_sobol, get_MTGP
from ax.core.observation import ObservationFeatures
from ax.core.generator_run import GeneratorRun
from ax.storage.json_store.save import save_experiment
from ax.storage.metric_registry import register_metric

from optimas.generators.ax.base import AxGenerator
from optimas.core import TrialParameter
from .ax_metric import AxMetric


# Define generator states.
NOT_STARTED = 'not_started'
LOFI_RETURNED = 'lofi_returned'
HIFI_RETURNED = 'hifi_returned'


class AxMultitaskGenerator(AxGenerator):
    def __init__(
            self, varying_parameters, objectives, lofi_task, hifi_task,
            analyzed_parameters=None, use_cuda=False, save_model=True,
            model_save_period=5, model_history_dir='model_history'):
        custom_trial_parameters = [
            TrialParameter('arm_name', 'ax_arm_name', dtype='U32'),
            TrialParameter('trial_type', 'ax_trial_type', dtype='U32'),
            TrialParameter('trial_index', 'ax_trial_index', dtype=int)
        ]
        self._check_inputs(varying_parameters, objectives, lofi_task,
                           hifi_task)
        super().__init__(varying_parameters,
                         objectives,
                         analyzed_parameters=analyzed_parameters,
                         use_cuda=use_cuda,
                         save_model=save_model,
                         model_save_period=model_save_period,
                         model_history_dir=model_history_dir,
                         custom_trial_parameters=custom_trial_parameters)
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
        self._create_experiment()

    def get_gen_specs(self, sim_workers):
        gen_specs = super().get_gen_specs(sim_workers)
        max_length = max([len(self.lofi_task.name), len(self.hifi_task.name)])
        gen_specs['out'].append(('task', str, max_length))
        return gen_specs

    def _check_inputs(self, varying_parameters, objectives, lofi_task,
                      hifi_task):
        n_objectives = len(objectives)
        assert n_objectives == 1, (
            'Multitask generator supports only a single objective. '
            'Objectives given: {}.'.format(n_objectives)
        )
        assert lofi_task.n_opt >= hifi_task.n_opt, (
            'The number of low-fidelity trials must be larger than or equal '
            'to the number of high-fidelity trials'
        )

    def _ask(self, trials):
        for trial in trials:
            next_trial = self._get_next_trial_arm()
            if next_trial is not None:
                arm, trial_type, trial_index = next_trial
                trial.parameter_values = [arm.parameters.get(var.name)
                                          for var in self._varying_parameters]
                trial.trial_type = trial_type
                trial.arm_name = arm.name
                trial.trial_index = trial_index
        return trials

    def _tell(self, trials):
        if self.gen_state == NOT_STARTED:
            self._incorporate_external_data(trials)
        else:
            self._complete_evaluations(trials)

    def _incorporate_external_data(self, trials):
        # Get trial indices.
        trial_indices = []
        for trial in trials:
            trial_indices.append(trial.trial_index)
        trial_indices = np.unique(np.array(trial_indices))

        # Group trials by index.
        grouped_trials = {}
        for index in trial_indices:
            grouped_trials[index] = []
        for trial in trials:
            grouped_trials[trial.trial_index].append(trial)

        # Add trials to experiment.
        for index in trial_indices:
            # Get all trials with current index.
            trials_i = grouped_trials[index]
            trial_type = trials_i[0].trial_type
            # Create arms.
            arms = []
            for trial in trials_i:
                params = {}
                for var, val in zip(trial.varying_parameters,
                                    trial.parameter_values):
                    params[var.name] = val
                arms.append(Arm(parameters=params, name=trial.arm_name))
            # Create new batch trial.
            gr = GeneratorRun(arms=arms, weights=[1.] * len(arms))
            ax_trial = self._experiment.new_batch_trial(
                generator_run=gr,
                trial_type=trial_type)
            ax_trial.run()
            # Incorporate observations.
            for trial in trials_i:
                objective_eval = {}
                oe = trial.objective_evaluations[0]
                objective_eval['f'] = (oe.value, oe.sem)
                ax_trial.run_metadata[trial.arm_name] = objective_eval
            # Mark batch trial as completed.
            ax_trial.mark_completed()
            # Keep track of high-fidelity trials.
            if trial_type == self.hifi_task.name:
                self.hifi_trials.append(index)

    def _complete_evaluations(self, trials):
        for trial in trials:
            # Make sure trial is part of current batch trial.
            current_trial_arms = list(self.current_trial.arms_by_name.keys())
            assert trial.arm_name in current_trial_arms, (
                'Arm {} is not part of current trial. '
                'External data can only be loaded into generator before '
                'initialization.'
            )
            objective_eval = {}
            oe = trial.objective_evaluations[0]
            objective_eval['f'] = (oe.value, oe.sem)
            self.current_trial.run_metadata[trial.arm_name] = objective_eval
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

    def _create_experiment(self):
        # Create search space.
        parameters = []
        for var in self._varying_parameters:
            parameters.append(
                RangeParameter(
                    name=var.name,
                    parameter_type=ParameterType.FLOAT,
                    lower=float(var.lower_bound),
                    upper=float(var.upper_bound))
            )
        search_space = SearchSpace(parameters=parameters)

        # Determine whether tominimize objective.
        minimize = self.objectives[0].minimize

        # Create metrics.
        hifi_objective = AxMetric(
            name='hifi_metric',
            lower_is_better=minimize
        )
        lofi_objective = AxMetric(
            name='lofi_metric',
            lower_is_better=minimize
        )

        # Create optimization config.
        opt_config = OptimizationConfig(
            objective=Objective(hifi_objective, minimize=minimize))

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
            canonical_name='hifi_metric')

        # Store experiment.
        self._experiment = experiment

        # Register metric in order to be able to save experiment to json file.
        _, encoder_registry, decoder_registry = register_metric(AxMetric)
        self._encoder_registry = encoder_registry
        self._decoder_registry = decoder_registry

    def _get_next_trial_arm(self):
        if self.gen_state in [NOT_STARTED, HIFI_RETURNED]:
            trial = self._get_lofi_batch()
            self.trials_list = [(arm, trial.trial_type, trial.index)
                                for arm in trial.arms]
        elif self.gen_state in [LOFI_RETURNED]:
            trial = self._get_hifi_batch()
            self.trials_list = [(arm, trial.trial_type, trial.index)
                                for arm in trial.arms]
            self.model_iteration += 1
        if self.trials_list:
            return self.trials_list.pop(0)
        else:
            return None

    def _get_lofi_batch(self):
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
                device=torch.device(self.torch_device)
            )
            n_gen = self.lofi_task.n_opt

            generator_success = True
            while True:
                try:
                    # Try to generate the new points.
                    gr = m.gen(
                        n=n_gen,
                        optimization_config=(
                            self._experiment.optimization_config),
                        fixed_features=ObservationFeatures(
                            parameters={}, trial_index=self.hifi_trials[-1]),
                        model_gen_options={
                            'optimizer_kwargs': {
                                'init_batch_limit': self.init_batch_limit
                            }
                        }
                    )
                    # When successful, break loop.
                    break
                except RuntimeError as e:
                    # Print exception.
                    print('RuntimeError: {}'.format(e), flush=True)
                    # Divide batch size by 2.
                    self.init_batch_limit //= 2
                    print('Retrying with `init_batch_limit={}`'.format(
                        self.init_batch_limit), flush=True)
                finally:
                    # If all attempts have failed (even for batch size of 1),
                    # mark generation as failed and break loop.
                    if self.init_batch_limit == 0:
                        generator_success = False
                        break
            # If generation failed, stop optimization.
            if not generator_success:
                raise RuntimeError(
                    'Failed to generate multitask trials. '
                    'Not sufficient memory is available.')
        self.gr_lofi = gr
        trial = self._experiment.new_batch_trial(
            trial_type=self.lofi_task.name,
            generator_run=gr
        )
        trial.run()
        self.gen_state = 'lofi_given'
        self.n_gen_lofi = n_gen
        self.current_trial = trial
        self.returned_lofi_trials = 0
        return trial

    def _get_hifi_batch(self):
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
                device=torch.device(self.torch_device)
            )
            n_gen = self.hifi_task.n_opt

            # Select max-utility points from the low fidelity batch to
            # generate a high fidelity batch.
            gr = max_utility_from_GP(
                n=n_gen,
                m=m,
                gr=self.gr_lofi,
                hifi_task=self.hifi_task.name
            )
        trial = self._experiment.new_batch_trial(
            trial_type=self.hifi_task.name,
            generator_run=gr
        )
        trial.run()
        self.gen_state = 'hifi_given'
        self.n_gen_hifi = n_gen
        self.current_trial = trial
        self.returned_hifi_trials = 0
        self.hifi_trials.append(trial.index)
        return trial

    def _save_model_to_file(self):
        file_path = os.path.join(
            self._model_history_dir,
            'ax_experiment_at_eval_{}.json'.format(
                self._n_completed_trials_last_saved)
        )
        save_experiment(
            experiment=self._experiment,
            filepath=file_path,
            encoder_registry=self._encoder_registry
        )


def max_utility_from_GP(n, m, gr, hifi_task):
    """
    High fidelity batches are constructed by selecting the maximum utility
    points from the low fidelity batch, after updating the model with the low
    fidelity results.
    This function selects the max utility points according to the MTGP
    predictions.
    """
    obsf = []
    for arm in gr.arms:
        params = deepcopy(arm.parameters)
        params['trial_type'] = hifi_task
        obsf.append(ObservationFeatures(parameters=params))
    # Make predictions
    f, cov = m.predict(obsf)
    # Compute expected utility
    u = -np.array(f['hifi_metric'])
    best_arm_indx = np.flip(np.argsort(u))[:n]
    gr_new = GeneratorRun(
        arms=[gr.arms[i] for i in best_arm_indx],
        weights=[1.] * n,
    )
    return gr_new
