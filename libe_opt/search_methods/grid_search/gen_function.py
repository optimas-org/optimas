import numpy as np

from libensemble.message_numbers import STOP_TAG, PERSIS_STOP, \
    FINISHED_PERSISTENT_GEN_TAG, EVAL_GEN_TAG
from libensemble.tools.persistent_support import PersistentSupport


def persistent_regular_grid_search(H, persis_info, gen_specs, libE_info):
    # underscore parameter for internal/testing arguments

    ps = PersistentSupport(libE_info, EVAL_GEN_TAG)

    # Pull out user parameters to perform calculations
    user_specs = gen_specs['user']

    # Get location and standard deviation bounds from gen_specs
    lb = user_specs['lb']
    ub = user_specs['ub']
    n_points = user_specs['n_steps']
    n_vars = len(lb)
    n_configs = np.prod(n_points)

    all_configs = np.zeros((n_configs, n_vars))

    var_linspaces = []
    for i in range(n_vars):
        var_linspaces.append(np.linspace(lb[i], ub[i], n_points[i]))
    var_mgrids = np.meshgrid(*var_linspaces, indexing='ij')
    var_mgrids_flat = [np.ravel(var_mgrid) for var_mgrid in var_mgrids]

    for i in range(n_configs):
        for j, flat_mgrid in enumerate(var_mgrids_flat):
            all_configs[i, j] = flat_mgrid[i]

    # Determine how many values to generate
    batch_size = user_specs['gen_batch_size']

    tag = None
    n_batch = 0
    while tag not in [STOP_TAG, PERSIS_STOP]:
        i_first = n_batch * batch_size
        i_last = (n_batch + 1) * batch_size

        if i_first > n_configs:
            tag = STOP_TAG
        else:
            if i_last > n_configs:
                i_last = n_configs
                batch_size = i_last - i_first

            # Create array of 'batch_size' zeros
            H_o = np.zeros(batch_size, dtype=gen_specs['out'])

            # Replace those zeros with the random numbers
            H_o['x'] = all_configs[i_first:i_last]
            H_o['resource_sets'] = np.ones(batch_size, dtype=int)
            tag, Work, calc_in = ps.send_recv(H_o)
            n_batch += 1

    # Send back our output and persis_info
    return H_o, persis_info, FINISHED_PERSISTENT_GEN_TAG
