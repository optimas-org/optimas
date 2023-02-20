import os

import numpy as np
from libensemble.message_numbers import (
    STOP_TAG, PERSIS_STOP, FINISHED_PERSISTENT_GEN_TAG, EVAL_GEN_TAG)
from libensemble.tools.persistent_support import PersistentSupport
from libensemble.resources.resources import Resources

from optimas.core import Evaluation


def persistent_generator(H, persis_info, gen_specs, libE_info):
    """
    Create a Gaussian Process model, update it as new simulation results
    are available, and generate inputs for the next simulations.
    This is a persistent `genf` i.e. this function is called by a dedicated
    worker and does not return until the end of the whole libEnsemble run.
    """
    # If CUDA is available, run BO loop on the GPU.
    if gen_specs['user']['use_cuda']:
        resources = Resources.resources.worker_resources
        # If there is no dedicated slot for the generator, use the GPU
        # specified by the user. This GPU will be shared with the simulation
        # workers.
        if resources.slot_count is None:
            gpu_id = str(gen_specs['user']['gpu_id'])
            os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
        # If there is a dedicated slot for the generator, use the corresponding
        # GPU. This GPU will only be used for the generator and will not be
        # available for the simulation workers.
        else:
            resources.set_env_to_slots('CUDA_VISIBLE_DEVICES')

    # Get generator, objectives, and parameters to analyze.
    generator = gen_specs['user']['generator']
    objectives = generator.objectives
    analyzed_parameters = generator.analyzed_parameters

    ps = PersistentSupport(libE_info, EVAL_GEN_TAG)

    # Number of points to generate initially.
    number_of_gen_points = gen_specs['user']['gen_batch_size']

    n_failed_gens = 0

    # Receive information from the manager (or a STOP_TAG)
    tag = None
    while tag not in [STOP_TAG, PERSIS_STOP]:

        # Ask the optimizer to generate `batch_size` new points
        # Store this information in the format expected by libE
        H_o = np.zeros(number_of_gen_points, dtype=gen_specs['out'])
        for i in range(number_of_gen_points):
            generated_trials = generator.ask(1)
            if generated_trials:
                trial = generated_trials[0]
                for var, val in zip(trial.varying_parameters,
                                    trial.parameter_values):
                    H_o[var.name][i] = val
                if 'task' in H_o.dtype.names:
                    H_o['task'][i] = trial.trial_type
                if trial.custom_parameters is not None:
                    for par in trial.custom_parameters:
                        H_o[par.save_name][i] = getattr(trial, par.name)
                H_o['trial_index'][i] = trial.index
                H_o['resource_sets'][i] = 1
        n_failed_gens = np.sum(H_o['resource_sets'] == 0)
        H_o = H_o[H_o['resource_sets'] > 0]

        # Send data and get results from finished simulation
        # Blocking call: waits for simulation results to be sent by the manager
        tag, Work, calc_in = ps.send_recv(H_o)
        if calc_in is not None:
            # Check how many simulations have returned
            n = len(calc_in['sim_id'])
            # Update the GP with latest simulation results
            for i in range(n):
                trial_index = int(calc_in['trial_index'][i])
                trial = generator._trials[trial_index]
                for par in objectives + analyzed_parameters:
                    y = calc_in[par.name][i]
                    ev = Evaluation(parameter=par, value=y)
                    trial.complete_evaluation(ev)
                # Register trial with unknown SEM
                generator.tell([trial])
            # Set the number of points to generate to that number:
            number_of_gen_points = n + n_failed_gens
            n_failed_gens = 0
        else:
            number_of_gen_points = 0

    return H_o, persis_info, FINISHED_PERSISTENT_GEN_TAG
