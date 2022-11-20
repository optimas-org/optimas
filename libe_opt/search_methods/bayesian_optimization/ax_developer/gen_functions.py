import os

import numpy as np
import torch
from libensemble.message_numbers import STOP_TAG, PERSIS_STOP, FINISHED_PERSISTENT_GEN_TAG, EVAL_GEN_TAG
from libensemble.tools.persistent_support import PersistentSupport
from libensemble.resources.resources import Resources


from copy import deepcopy

from ax.runners import SyntheticRunner
from ax.storage.json_store.save import save_experiment
from ax.storage.metric_registry import register_metric
from ax.storage.runner_registry import register_runner
from ax.core.search_space import SearchSpace
from ax.modelbridge.factory import get_sobol
from ax.core.observation import ObservationFeatures
from ax.core.multi_type_experiment import MultiTypeExperiment
from ax.core.generator_run import GeneratorRun

from .ax_metric import AxMetric
from .ax_runner import AxRunner

def persistent_gp_mt_ax_gen_f(H, persis_info, gen_specs, libE_info):
    """
    Create a Gaussian Process model for multi-task optimization
    and update it as new simulation results are
    available, and generate inputs for the next simulations.

    This is a persistent `genf` i.e. this function is called by a dedicated
    worker and does not return until the end of the whole libEnsemble run.
    """
    # If CUDA is available, run BO loop on the GPU.
    if gen_specs['user']['use_cuda'] and torch.cuda.is_available():
        torch_device = 'cuda'
        resources = Resources.resources.worker_resources
        resources.set_env_to_slots('CUDA_VISIBLE_DEVICES')
    else:
        torch_device = 'cpu'

    # Get task names.
    hifi_task = gen_specs['user']['mt_params']['name_hifi']
    lofi_task = gen_specs['user']['mt_params']['name_lofi']

    # Number of points to generate initially and during optimization.
    n_init_hifi = gen_specs['user']['mt_params']['n_init_hifi']
    n_init_lofi = gen_specs['user']['mt_params']['n_init_lofi']
    n_opt_hifi = gen_specs['user']['mt_params']['n_opt_hifi']
    n_opt_lofi = gen_specs['user']['mt_params']['n_opt_lofi']    

    # Create runner.
    ax_runner = AxRunner(libE_info, gen_specs)

    # Create experiment.
    exp = gen_specs['user']['experiment']
    exp.update_runner(lofi_task, ax_runner)
    exp.update_runner(hifi_task, ax_runner)

    # TODO: Implement reading past history (by reading saved experiment or
    # libEnsemble hystory file).

    # Batch limit for the initialization of the aquisition function optimization.
    init_batch_limit = 1000

    # Receive information from the manager (or a STOP_TAG)
    tag = None
    model_iteration = 0
    hifi_trials = []
    while tag not in [STOP_TAG, PERSIS_STOP]:

        if model_iteration == 0:
            # Initialize with sobol sample.
            for model, n_gen in zip([hifi_task, lofi_task], [n_init_hifi, n_init_lofi]):
                s = get_sobol(exp.search_space, scramble=True)
                gr = s.gen(n_gen)
                trial = exp.new_batch_trial(trial_type=model, generator_run=gr)
                trial.run()
                trial.mark_completed()
                tag = trial.run_metadata['tag']
                if tag in [STOP_TAG, PERSIS_STOP]:
                    break
                if model == hifi_task:
                    hifi_trials.append(trial.index)

        else:
            # Run multi-task BO.

            # 1) Fit the MTGP.
            m = get_MTGP(
                experiment=exp,
                data=exp.fetch_data(),
                search_space=exp.search_space,
                dtype=torch.double,
                device=torch.device(torch_device)
            )

            # 2) Find the best points for the high fidelity task.

            # Too large initialization batches can lead to out-of-memory errors.
            # The loop below tries to generate the next points to evaluate
            # using `init_batch_limit`. If an RuntimeError is raised during
            # generation (namely an out-of-memory error), `init_batch_limit`
            # is divided by two and a new attempt is made. This is repeated
            # until the generation runs successfully.            
            generator_success = True
            while True:
                try:
                    # Try to generate the new points.
                    gr = m.gen(
                        n=n_opt_lofi,
                        optimization_config=exp.optimization_config,
                        fixed_features=ObservationFeatures(
                            parameters={}, trial_index=hifi_trials[-1]),
                        model_gen_options={
                            'optimizer_kwargs': {
                                'init_batch_limit': init_batch_limit
                            }
                        }
                    )
                    # When successful, break loop.
                    break
                except RuntimeError as e:
                    # Print exception.
                    print('RuntimeError: {}'.format(e), flush=True)
                    # Divide batch size by 2.
                    init_batch_limit //= 2
                    print('Retrying with `init_batch_limit={}`'.format(
                        init_batch_limit), flush=True)
                finally:
                    # If all attempts have failed (even for batch size of 1),
                    # mark generation as failed and break loop.
                    if init_batch_limit == 0:
                        generator_success = False
                        break
            # If generation failed, stop optimization.
            if not generator_success:
                tag = STOP_TAG
                break

            # 3) But launch them at low fidelity.
            tr = exp.new_batch_trial(trial_type=lofi_task, generator_run=gr)
            tr.run()
            tr.mark_completed()
            tag = tr.run_metadata['tag']
            if tag in [STOP_TAG, PERSIS_STOP]:
                break

            # 4) Update the model.
            m = get_MTGP(
                experiment=exp,
                data=exp.fetch_data(),
                search_space=exp.search_space,
                dtype=torch.double,
                device=torch.device(torch_device)
            )

            # 5) Select max-utility points from the low fidelity batch to generate a high fidelity batch.
            gr = max_utility_from_GP(
                n=n_opt_hifi,
                m=m,
                gr=gr,
                hifi_task=hifi_task
            )
            tr = exp.new_batch_trial(trial_type=hifi_task, generator_run=gr)
            tr.run()
            tr.mark_completed()
            tag = tr.run_metadata['tag']
            if tag in [STOP_TAG, PERSIS_STOP]:
                break
            hifi_trials.append(tr.index)

        if model_iteration == 0:
            # Initialize folder to log the model.
            if not os.path.exists('model_history'):
                os.mkdir('model_history')
            # Register metric and runner in order to be able to save to json.
            _, encoder_registry, decoder_registry = register_metric(AxMetric)
            _, encoder_registry, decoder_registry = register_runner(
                AxRunner,
                encoder_registry=encoder_registry,
                decoder_registry=decoder_registry
            )

        # Save current experiment.
        # Saving the experiment to a json file currently requires a bit of
        # trickery. The `AxRunner` cannot be serialized into a json file
        # due to the `libE_info` and `gen_specs` attributes. This also prevents
        # the experiment from being saved to file. In order to overcome this,
        # all instances of the `AxRunner` are replaced by a `SyntheticRunner`
        # before saving. Afterwards, the `AxRunner` is reasigned again to both
        # high- and low-fidelity tasks in order to allow the optimization to
        # continue.
        for i, trial in exp.trials.items():
            trial._runner = SyntheticRunner()
        exp.update_runner(lofi_task, SyntheticRunner())
        exp.update_runner(hifi_task, SyntheticRunner())
        save_experiment(
            exp,
            'model_history/experiment_%05d.json' % model_iteration,
            encoder_registry=encoder_registry
        )
        exp.update_runner(lofi_task, ax_runner)
        exp.update_runner(hifi_task, ax_runner)

        # Increase iteration counter.
        model_iteration += 1

    return [], persis_info, FINISHED_PERSISTENT_GEN_TAG


def max_utility_from_GP(n, m, gr, hifi_task):
    """
    High fidelity batches are constructed by selecting the maximum utility points
    from the low fidelity batch, after updating the model with the low fidelity results.
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
        arms = [
            gr.arms[i] for i in best_arm_indx
        ],
        weights = [1.] * n,
    )
    return gr_new


# Imports and definitions needed for custom `get_MTGP`
from ax.core.data import Data
from ax.core.experiment import Experiment
from typing import Optional
from ax.modelbridge.transforms.convert_metric_names import (
    tconfig_from_mt_experiment
)
from ax.modelbridge.registry import (
    MT_MTGP_trans,
    ST_MTGP_trans,
)
from ax.modelbridge.torch import TorchModelBridge
from ax.models.torch.botorch import BotorchModel
DEFAULT_TORCH_DEVICE = torch.device("cpu")

def get_MTGP(
    experiment: Experiment,
    data: Data,
    search_space: Optional[SearchSpace] = None,
    dtype: torch.dtype = torch.double,
    device: torch.device = DEFAULT_TORCH_DEVICE,
    trial_index: Optional[int] = None,
) -> TorchModelBridge:
    """Instantiates a Multi-task Gaussian Process (MTGP) model that generates
    points with EI.
    If the input experiment is a MultiTypeExperiment then a
    Multi-type Multi-task GP model will be instantiated.
    Otherwise, the model will be a Single-type Multi-task GP.

    This method is a custom version of `get_MTGP` in `ax.modelbridge.factory`
    that exposes a `device` parameter in order to be able to run on the GPU.
    See https://github.com/facebook/Ax/issues/928 for details.

    Implemented by S. Jalas.
    """

    if isinstance(experiment, MultiTypeExperiment):
        trial_index_to_type = {
            t.index: t.trial_type for t in experiment.trials.values()
        }
        transforms = MT_MTGP_trans
        transform_configs = {
            "TrialAsTask": {"trial_level_map": {"trial_type": trial_index_to_type}},
            "ConvertMetricNames": tconfig_from_mt_experiment(experiment),
        }
    else:
        # Set transforms for a Single-type MTGP model.
        transforms = ST_MTGP_trans
        transform_configs = None

    # Choose the status quo features for the experiment from the selected trial.
    # If trial_index is None, we will look for a status quo from the last
    # experiment trial to use as a status quo for the experiment.
    if trial_index is None:
        trial_index = len(experiment.trials) - 1
    elif trial_index >= len(experiment.trials):
        raise ValueError("trial_index is bigger than the number of experiment trials")

    status_quo = experiment.trials[trial_index].status_quo
    if status_quo is None:
        status_quo_features = None
    else:
        status_quo_features = ObservationFeatures(
            parameters=status_quo.parameters,
            trial_index=trial_index,
        )

    return TorchModelBridge(
        experiment=experiment,
        search_space=search_space or experiment.search_space,
        data=data,
        model=BotorchModel(),
        transforms=transforms,
        transform_configs=transform_configs,
        torch_dtype=torch.double,
        torch_device=device,
        status_quo_features=status_quo_features,
    )

