import os

import numpy as np
import torch
from libensemble.message_numbers import STOP_TAG, PERSIS_STOP, FINISHED_PERSISTENT_GEN_TAG, EVAL_GEN_TAG
from libensemble.tools.persistent_support import PersistentSupport
from libensemble.resources.resources import Resources


def persistent_ax_client(H, persis_info, gen_specs, libE_info):
    """
    Create a Gaussian Process model, update it as new simulation results
    are available, and generate inputs for the next simulations.
    This is a persistent `genf` i.e. this function is called by a dedicated
    worker and does not return until the end of the whole libEnsemble run.
    """
    # If CUDA is available, run BO loop on the GPU.
    if gen_specs['user']['use_cuda'] and torch.cuda.is_available():
        resources = Resources.resources.worker_resources
        resources.set_env_to_slots('CUDA_VISIBLE_DEVICES')

    # Create Ax client.
    ax_client = gen_specs['user']['client']
    
    # Detemine if optimization uses multiple fidelities.
    use_mf = 'mf_params' in gen_specs['user']

    # If so, get name of fidelity parameter.
    if use_mf:
        fidel_name = gen_specs['user']['mf_params']['name']
    
    # Metric name.
    metric_name = list(ax_client.experiment.metrics.keys())[0]

    ps = PersistentSupport(libE_info, EVAL_GEN_TAG)

    # Number of points to generate initially.
    number_of_gen_points = gen_specs['user']['gen_batch_size']

    # If there is any past history, feed it to the GP
    if len(H) > 0:
        names_list = gen_specs['user']['params']
        params = dict.fromkeys(names_list)

        for i in range(len(H)):
            for j, name in enumerate(names_list):
                params[name] = H['x'][i][j]

            if use_mf:
                params[fidel_name] = H['z'][i]

            _, trial_id = ax_client.attach_trial(params)
            ax_client.complete_trial(trial_id, {metric_name: (H['f'][i], np.nan)})

    # Receive information from the manager (or a STOP_TAG)
    tag = None
    model_iteration = 0
    while tag not in [STOP_TAG, PERSIS_STOP]:

        # Ask the optimizer to generate `batch_size` new points
        # Store this information in the format expected by libE
        H_o = np.zeros(number_of_gen_points, dtype=gen_specs['out'])
        for i in range(number_of_gen_points):
            parameters, _ = ax_client.get_next_trial()
            if use_mf:
                H_o['z'][i] = parameters.pop(fidel_name)
            H_o['x'][i] = list(parameters.values())
            H_o['resource_sets'][i] = 1

        # Send data and get results from finished simulation
        # Blocking call: waits for simulation results to be sent by the manager
        tag, Work, calc_in = ps.send_recv(H_o)
        if calc_in is not None:
            # Check how many simulations have returned
            n = len(calc_in['f'])
            # Update the GP with latest simulation results
            for i in range(n):
                trial_index = int(calc_in['sim_id'][i])
                y = calc_in['f'][i]
                # Register trial with unknown SEM
                ax_client.complete_trial(trial_index, {metric_name: (y, np.nan)})
            # Set the number of points to generate to that number:
            number_of_gen_points = n
        else:
            number_of_gen_points = 0

        # Save current model.
        if model_iteration == 0:
            # Initialize folder to log the model.
            if not os.path.exists('model_history'):
                os.mkdir('model_history')
        ax_client.save_to_json_file(
            'model_history/ax_client_%05d.json' % model_iteration)

        # Increase iteration counter.
        model_iteration += 1

    return H_o, persis_info, FINISHED_PERSISTENT_GEN_TAG
