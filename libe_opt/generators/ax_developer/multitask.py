from copy import deepcopy

import numpy as np
import torch

from ax.core.multi_type_experiment import MultiTypeExperiment
from ax.core.parameter import RangeParameter, ParameterType
from ax.core.search_space import SearchSpace
from ax.core.optimization_config import OptimizationConfig
from ax.core.objective import Objective
from ax.runners import SyntheticRunner
from ax.modelbridge.factory import get_sobol, get_MTGP
from ax.core.observation import ObservationFeatures
from ax.core.generator_run import GeneratorRun

from libe_opt.generators.base import Generator
from .ax_metric import AxMetric


# Define generator states.
NOT_STARTED = 'not_started'
LOFI_RETURNED = 'lofi_returned'
HIFI_RETURNED = 'hifi_returned'


class AxMultitaskGenerator(Generator):
    def __init__(
            self, variables, objectives, lofi_task, hifi_task,
            use_cuda=False):
        self._check_inputs(variables, objectives, lofi_task, hifi_task)
        super().__init__(variables, objectives, use_cuda=use_cuda)
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
        self._determine_torch_device()
        self._create_experiment()

    def get_gen_specs(self, sim_workers):
        gen_specs = super().get_gen_specs(sim_workers)
        gen_specs['out'].append(
            ('task', str, max(
                [len(self.lofi_task.name), len(self.hifi_task.name)]
                )
            )
        )
        return gen_specs

    def _check_inputs(self, variables, objectives, lofi_task, hifi_task):
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
                arm, trial_type = next_trial
                trial.variable_values = [arm.parameters.get(var.name) for var in self.variables]
                trial.trial_type = trial_type
                trial.arm_name = arm.name
        return trials

    def _tell(self, trials):
        for trial in trials:
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
        for var in self.variables:
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
        self.experiment = experiment

    def _get_next_trial_arm(self):
        if self.gen_state in [NOT_STARTED, HIFI_RETURNED]:
            trial = self._get_lofi_batch()
            self.trials_list = [(arm, trial.trial_type) for arm in trial.arms]
        elif self.gen_state in [LOFI_RETURNED]:
            trial = self._get_hifi_batch()
            self.trials_list = [(arm, trial.trial_type) for arm in trial.arms]
            self.model_iteration += 1
        if self.trials_list:
            return self.trials_list.pop(0)
        else:
            return None

    def _get_lofi_batch(self):
        if self.model_iteration == 0:
            # Generate first batch using a Sobol sequence.
            m = get_sobol(self.experiment.search_space, scramble=True)            
            n_gen = self.lofi_task.n_init
            gr = m.gen(n_gen)
        else:
            m = get_MTGP(
                experiment=self.experiment,
                data=self.experiment.fetch_data(),
                search_space=self.experiment.search_space,
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
                        optimization_config=self.experiment.optimization_config,
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
        trial = self.experiment.new_batch_trial(
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
            m = get_sobol(self.experiment.search_space, scramble=True)
            n_gen = self.hifi_task.n_init
            gr = m.gen(n_gen)
        else:
            m = get_MTGP(
                experiment=self.experiment,
                data=self.experiment.fetch_data(),
                search_space=self.experiment.search_space,
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
        trial = self.experiment.new_batch_trial(
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

    def _determine_torch_device(self):
        # If CUDA is available, run BO loop on the GPU.
        if self.use_cuda and torch.cuda.is_available():
            self.torch_device = 'cuda'
        else:
            self.torch_device = 'cpu'


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
