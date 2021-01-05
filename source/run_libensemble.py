#!/usr/bin/env python
"""
This file is part of the suite of scripts to use LibEnsemble on top of PIC
simulations.
"""

# Either 'random' or 'bo', 'async_bo', 'async_bo_mf', 'async_bo_mf_disc' or 'aposmm'
generator_type = 'async_bo'
# Choose maximum number of simulations to be run within the optimization
sim_max = 10

import os
import numpy as np
from simf import run_fbpic

# Import libEnsemble modules
from libensemble.libE import libE
from libensemble.tools import check_inputs
is_mf = False
if generator_type == 'random':
    from libensemble.gen_funcs.persistent_uniform_sampling \
        import persistent_uniform as gen_f
    from libensemble.alloc_funcs.start_only_persistent \
        import only_persistent_gens as alloc_f
elif generator_type in ['bo', 'async_bo', 'async_bo_mf', 'async_bo_mf_disc']:
    from libensemble.alloc_funcs.start_only_persistent \
        import only_persistent_gens as alloc_f
    if generator_type == 'async_bo_mf':
        from libensemble.gen_funcs.persistent_gp \
            import persistent_gp_mf_gen_f as gen_f
        is_mf = True
    elif generator_type == 'async_bo_mf_disc':
        from libensemble.gen_funcs.persistent_gp \
            import persistent_gp_mf_disc_gen_f as gen_f
        is_mf = True
    else:
        from libensemble.gen_funcs.persistent_gp \
            import persistent_gp_gen_f as gen_f
elif generator_type == 'aposmm':
    import libensemble.gen_funcs
    libensemble.gen_funcs.rc.aposmm_optimizers = 'nlopt'
    from libensemble.gen_funcs.persistent_aposmm import aposmm as gen_f
    from libensemble.alloc_funcs.persistent_aposmm_alloc \
        import persistent_aposmm_alloc as alloc_f
else:
    raise RuntimeError('Unknown generator: %s' %generator_type)

from libensemble.tools import parse_args, save_libE_output, \
    add_unique_random_streams
from libensemble import libE_logger
from libensemble.executors.mpi_executor import MPIExecutor

# Import user-defined parameters
from sim_specific.varying_parameters import varying_parameters
from sim_specific.analysis_script import analyzed_quantities
if is_mf:
    from sim_specific.mf_parameters import mf_parameters


def determine_fidelity_type_and_length(mf_parameters):
    """
    Determine the type of the fidelity (i.e. float, int, str...) and, if it
    is a string, also its length.
    """
    # Check that all fidelities in 'range' are of the same type.
    fidel_types = [type(z) for z in mf_parameters['range']]
    if fidel_types.count(fidel_types[0]) != len(fidel_types):
        raise ValueError("The fidelities in 'range' are of different types.")
    fidel_type = fidel_types[0]
    fidel_len = None
    # If fidelities are strings, determine the lenght of the longest one
    # so that it can be fully stored in a numpy array.
    if fidel_type == str:
        str_lengths = [len(z) for z in mf_parameters['range']]
        fidel_len = max(str_lengths)
    return fidel_type, fidel_len


libE_logger.set_level('INFO')
nworkers, is_master, libE_specs, _ = parse_args()

# Problem dimension. This is the number of input parameters exposed,
# that LibEnsemble will vary in order to minimize a single output parameter.
n = len(varying_parameters)

exctr = MPIExecutor(central_mode=False, zero_resource_workers=[1])
exctr.register_calc(full_path='python', calc_type='sim')

# State the objective function, its arguments, output, and necessary parameters
# (and their sizes). Here, the 'user' field is for the user's (in this case,
# the simulation) convenience. Feel free to use it to pass number of nodes,
# number of ranks per note, time limit per simulation etc.
sim_specs = {
    # Function whose output is being minimized. The parallel WarpX run is
    # launched from run_WarpX.
    'sim_f': run_fbpic,
    # Name of input for sim_f, that LibEnsemble is allowed to modify.
    # May be a 1D array.
    'in': ['x'],
    'out': [ ('f', float) ] \
        # f is the single float output that LibEnsemble minimizes.
        + analyzed_quantities \
        # input parameters
        + [ (name, float, (1,)) for name in varying_parameters.keys() ],
}

if is_mf:
    sim_specs['in'].append('z')
    if mf_parameters['discrete']:
        sim_specs['out'].append((mf_parameters['name'], np.unicode_, 16))
    else:
        sim_specs['out'].append((mf_parameters['name'], float))

# Allocator function, decides what a worker should do.
# We use a LibEnsemble allocator.
alloc_specs = {'alloc_f': alloc_f, 'out': [('given_back', bool)]}

# Here, the 'user' field is for the user's (in this case,
# the RNG) convenience.
gen_specs = {
    # Generator function. Will randomly generate new sim inputs 'x'.
    'gen_f': gen_f,
    # Generator input. This is a RNG, no need for inputs.
    'in': ['sim_id', 'x', 'f'],
    'out': [
        # parameters to input into the simulation.
        ('x', float, (n,))
    ],
    'user': {
        # Total max number of sims running concurrently.
        'gen_batch_size': nworkers-1,
        # Lower bound for the n parameters.
        'lb': np.array([ v[0] for v in varying_parameters.values() ]),
        # Upper bound for the n parameters.
        'ub': np.array([ v[1] for v in varying_parameters.values() ])
    }
}
if is_mf:
    gen_specs['in'].append('z')

# State the generating function, its arguments, output,
# and necessary parameters.
if generator_type in ['random', 'bo', 'async_bo', 'async_bo_mf', 'async_bo_mf_disc']:
    # Here, the 'user' field is for the user's (in this case,
    # the RNG) convenience.
    gen_specs['user']['gen_batch_size'] = nworkers-1
    if generator_type in ['async_bo', 'async_bo_mf', 'async_bo_mf_disc']:
        gen_specs['user']['async'] = True

    if is_mf:
        if mf_parameters['discrete']:
            gen_specs['out'].append(('z', np.unicode_, 16))
        else:
            gen_specs['out'].append(('z', float))
        gen_specs['user'] = {**gen_specs['user'], **mf_parameters}

elif generator_type == 'aposmm':
    gen_specs['out'] = [
        # parameters to input into the simulation.
        ('x', float, (n,)),
        # x scaled to a unique cube.
        ('x_on_cube', float, (n,)),
        # unique ID of simulation.
        ('sim_id', int),
        # Whether this point is a local minimum.
        ('local_min', bool),
        # whether the point is from a local optimization run
        # or a random sample point.
        ('local_pt', bool)
    ]
    # Number of sims for initial random sampling.
    # Optimizer starts afterwards.
    gen_specs['user']['initial_sample_size'] =  max(nworkers-1, 1)
    # APOSMM/NLOPT optimization method
    gen_specs['user']['localopt_method'] =  'LN_BOBYQA'
    gen_specs['user']['num_pts_first_pass'] =  nworkers
    # Relative tolerance of inputs
    gen_specs['user']['xtol_rel'] =  1e-3
    # Absolute tolerance of output 'f'. Determines when
    # local optimization stops.
    gen_specs['user']['ftol_abs'] =  3e-8

# Save H to file every N simulation evaluations
libE_specs['save_every_k_sims'] = 5
libE_specs['sim_dir_copy_files'] = ['sim_specific/template_fbpic_script.py']

exit_criteria = {'sim_max': sim_max}  # Exit after running sim_max simulations

# Create a different random number stream for each worker and the manager
persis_info = add_unique_random_streams({}, nworkers + 1)

# Before starting libensemble, check whether there is past history file
if os.path.exists('past_history.npy'):
    H0 = np.load('past_history.npy')
    H0 = H0[ H0['returned']==True ] # Only include runs that completed
    check_inputs( H0=H0, sim_specs=sim_specs,
                  alloc_specs=alloc_specs, gen_specs=gen_specs)
else:
    H0 = None

# Run LibEnsemble, and store results in history array H
H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria,
                            persis_info, alloc_specs, libE_specs, H0=H0)

# Save results to numpy file
if is_master:
    save_libE_output(H, persis_info, __file__, nworkers)
