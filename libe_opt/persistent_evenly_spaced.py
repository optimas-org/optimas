"""
This file defines the `gen_f` for evenly-spaced evaluation.

The `gen_f` is called once by a dedicated worker and only returns at the end
of the whole libEnsemble run.

This `gen_f` is meant to be used with the `alloc_f` function
`only_persistent_gens`
"""
import numpy as np
from libensemble.message_numbers import STOP_TAG, PERSIS_STOP, FINISHED_PERSISTENT_GEN_TAG, EVAL_GEN_TAG
from libensemble.tools.persistent_support import PersistentSupport

def persistent_evenly_spaced(H, persis_info, gen_specs, libE_info):
    """
    Evaluate simulations on an evenly-spaced grid in parameter space

    This is a persistent `genf` i.e. this function is called by a dedicated
    worker and does not return until the end of the whole libEnsemble run.
    """
    # Extract bounds of the parameter space, and batch size
    ub_list = gen_specs['user']['ub']
    lb_list = gen_specs['user']['lb']
    ps = PersistentSupport(libE_info, EVAL_GEN_TAG)

    # TODO: Read this from user input
    # Number of points in each direction
    N = 5

    # Generate the array of data to be simulated
    scanned_params = np.meshgrid(
        *(np.linspace( lb, ub, N ) for lb, ub in zip(lb_list, ub_list)),
        indexing='ij' )
    scanned_params = np.array([ element.flatten() for element in scanned_params ]).T

    # If there is any past history, skip the corresponding elements
    i_sim = len(H)

    # Receive information from the manager (or a STOP_TAG)
    tag = None
    while tag not in [STOP_TAG, PERSIS_STOP]:

        # Ask for `batch_size` new points
        # Store this information in the format expected by libE
        H_o = np.zeros(number_of_gen_points, dtype=gen_specs['out'])
        for i in range(number_of_gen_points):
            H_o['x'][i] = scanned_params[i_sim]
            i_sim += 1
            H_o['resource_sets'][i] = 1

        # Send data and get results from finished simulation
        # Blocking call: waits for simulation results to be sent by the manager
        tag, Work, calc_in = ps.send_recv(H_o)
        if calc_in is not None:
            # Check how many simulations have returned
            n = len(calc_in['f'])
            # Set the number of points to generate to that number:
            number_of_gen_points = n
        else:
            number_of_gen_points = 0

    return H_o, persis_info, FINISHED_PERSISTENT_GEN_TAG
