import os
from copy import deepcopy

import numpy as np
import torch
from ax.runners import SyntheticRunner
from ax.storage.json_store.save import save_experiment
from ax.storage.metric_registry import register_metric
from ax.storage.runner_registry import register_runner
from ax.modelbridge.factory import get_sobol, get_MTGP
from ax.core.observation import ObservationFeatures
from ax.core.generator_run import GeneratorRun
from libensemble.message_numbers import (
    STOP_TAG, PERSIS_STOP, FINISHED_PERSISTENT_GEN_TAG)
from libensemble.resources.resources import Resources

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

    # Batch limit for the initialization of the optimization of the
    # aquisition function.
    init_batch_limit = 1000

    # Receive information from the manager (or a STOP_TAG)
    tag = None
    model_iteration = 0
    hifi_trials = []
    while tag not in [STOP_TAG, PERSIS_STOP]:

        if model_iteration == 0:
            # Initialize with sobol sample.
            for model, n_gen in zip([hifi_task, lofi_task],
                                    [n_init_hifi, n_init_lofi]):
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

            # Too large initialization batches can lead to out-of-memory
            # errors.
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

            # 5) Select max-utility points from the low fidelity batch to
            # generate a high fidelity batch.
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
