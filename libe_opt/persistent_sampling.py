import numpy as np

from libensemble.message_numbers import STOP_TAG, PERSIS_STOP, \
    FINISHED_PERSISTENT_GEN_TAG, EVAL_GEN_TAG
from libensemble.tools.persistent_support import PersistentSupport


def persistent_normal(H, persis_info, gen_specs, libE_info):
    """
    This generation function returns ``gen_specs['user']['gen_batch_size']``
    normally sampled points with a given mean and standard deviation.
    """

    # Get location and standard deviation bounds from gen_specs
    loc = gen_specs['user']['lb']
    sigma = gen_specs['user']['ub']

    # Determine how many values to generate
    num = len(loc)
    batch_size = gen_specs['user']['gen_batch_size']

    ps = PersistentSupport(libE_info, EVAL_GEN_TAG)
    tag = None
    while tag not in [STOP_TAG, PERSIS_STOP]:
        # Create array of 'batch_size' zeros
        H_o = np.zeros(batch_size, dtype=gen_specs['out'])
        # Replace those zeros with the random numbers
        H_o['x'] = persis_info['rand_stream'].normal(loc, sigma, (batch_size, num))
        H_o['resource_sets'] = np.ones(batch_size, dtype=int)

        tag, Work, calc_in = ps.send_recv(H_o)
        if hasattr(calc_in, "__len__"):
            batch_size = len(calc_in)

    # Send back our output and persis_info
    return H_o, persis_info, FINISHED_PERSISTENT_GEN_TAG


def persistent_uniform(H, persis_info, gen_specs, libE_info):
    """
    This generation function returns ``gen_specs['user']['gen_batch_size']``
    uniformily sampled points within the specified bounds.
    """

    # Get lower and upper bounds from gen_specs
    lb = gen_specs['user']['lb']
    ub = gen_specs['user']['ub']

    # Determine how many values to generate
    num = len(lb)
    batch_size = gen_specs['user']['gen_batch_size']

    ps = PersistentSupport(libE_info, EVAL_GEN_TAG)
    tag = None
    while tag not in [STOP_TAG, PERSIS_STOP]:
        # Create array of 'batch_size' zeros
        H_o = np.zeros(batch_size, dtype=gen_specs['out'])
        # Replace those zeros with the random numbers
        H_o['x'] = persis_info['rand_stream'].uniform(lb, ub, (batch_size, num))
        H_o['resource_sets'] = np.ones(batch_size, dtype=int)

        tag, Work, calc_in = ps.send_recv(H_o)
        if hasattr(calc_in, "__len__"):
            batch_size = len(calc_in)

    # Send back our output and persis_info
    return H_o, persis_info, FINISHED_PERSISTENT_GEN_TAG
