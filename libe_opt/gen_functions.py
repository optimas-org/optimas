import numpy as np

from libensemble.message_numbers import (
    STOP_TAG, PERSIS_STOP, FINISHED_PERSISTENT_GEN_TAG, EVAL_GEN_TAG)
from libensemble.tools.persistent_support import PersistentSupport
from libensemble.resources.resources import Resources

from libe_opt.core import Evaluation, ObjectiveEvaluation


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
        resources.set_env_to_slots('CUDA_VISIBLE_DEVICES')

    # Get generator.
    generator = gen_specs['user']['generator']
    objectives = generator.objectives

    ps = PersistentSupport(libE_info, EVAL_GEN_TAG)

    # Number of points to generate initially.
    number_of_gen_points = gen_specs['user']['gen_batch_size']

    # If there is any past history, feed it to the GP
    # if len(H) > 0:
    #     names_list = gen_specs['user']['params']
    #     params = dict.fromkeys(names_list)

    #     for i in range(len(H)):
    #         for j, name in enumerate(names_list):
    #             params[name] = H['x'][i][j]

    #         generator.tell(params, (H['f'][i], np.nan))

    # Receive information from the manager (or a STOP_TAG)
    tag = None
    while tag not in [STOP_TAG, PERSIS_STOP]:

        # Ask the optimizer to generate `batch_size` new points
        # Store this information in the format expected by libE
        H_o = np.zeros(number_of_gen_points, dtype=gen_specs['out'])
        for i in range(number_of_gen_points):
            trial = generator.ask(1)[0]
            for variable, value in zip(trial.variables, trial.variable_values):
                H_o[variable.name][i] = value
            H_o['trial_index'][i] = trial.index
            H_o['resource_sets'][i] = 1

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
                for objective in objectives:
                    y = calc_in[objective.name][i]
                    ev = ObjectiveEvaluation(objective=objective, value=y)
                    trial.complete_evaluation(ev)
                # Register trial with unknown SEM
                generator.tell([trial])
            # Set the number of points to generate to that number:
            number_of_gen_points = n
        else:
            number_of_gen_points = 0

    return H_o, persis_info, FINISHED_PERSISTENT_GEN_TAG
