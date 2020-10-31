#!/usr/bin/env python

"""
This file is part of the suite of scripts to use LibEnsemble on top of WarpX
simulations. It is the entry point script that runs LibEnsemble. Libensemble
then launches WarpX simulations.

Execute locally via the following command:
    python run_libensemble_on_warpx.py --comms local --nworkers 3
On summit, use the submission script:
    bsub summit_submit_mproc.sh

The number of concurrent evaluations of the objective function will be
nworkers=1 as one worker is for the persistent gen_f.
"""

# Either 'random' or 'bo', 'async_bo', 'async_bo_mf', or 'aposmm'
generator_type = 'async_bo'
# Either 'local' or 'summit'
machine = 'local'

import os
import numpy as np
from simf import run_fbpic

# Import libEnsemble modules
from libensemble.libE import libE
from libensemble.tools import check_inputs
if generator_type == 'random':
    from libensemble.gen_funcs.persistent_uniform_sampling \
        import persistent_uniform as gen_f
    from libensemble.alloc_funcs.start_only_persistent \
        import only_persistent_gens as alloc_f
elif generator_type in ['bo', 'async_bo', 'async_bo_mf']:
    from libensemble.alloc_funcs.start_only_persistent \
        import only_persistent_gens as alloc_f
    if generator_type == 'async_bo_mf':
        from libensemble.gen_funcs.persistent_gp \
            import persistent_gp_mf_gen_f as gen_f
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
import all_machine_specs
from sim_specific.varying_parameters import varying_parameters
from sim_specific.analysis_script import analyzed_quantities

# Import machine-specific run parameters
if machine == 'local':
    machine_specs = all_machine_specs.local_specs
elif machine == 'summit':
    machine_specs = all_machine_specs.summit_specs

libE_logger.set_level('INFO')

nworkers, is_master, libE_specs, _ = parse_args()

# Set to full path of warp executable
sim_app = machine_specs['sim_app']

# Problem dimension. This is the number of input parameters exposed,
# that LibEnsemble will vary in order to minimize a single output parameter.
n = len(varying_parameters)

exctr = MPIExecutor(central_mode=True, zero_resource_workers=[1])
exctr.register_calc(full_path=sim_app, calc_type='sim')

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
    'user': {
        # machine-specific parameters
        'machine_specs': machine_specs
    }
}

# Allocator function, decides what a worker should do.
# We use a LibEnsemble allocator.
alloc_specs = {'alloc_f': alloc_f, 'out': [('given_back', bool)]}

# Here, the 'user' field is for the user's (in this case,
# the RNG) convenience.
gen_specs = {
    # Generator function. Will randomly generate new sim inputs 'x'.
    'gen_f': gen_f,
    # Generator input. This is a RNG, no need for inputs.
    'in': [],
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
        'ub': np.array([ v[1] for v in varying_parameters.values() ]),
    }
}

# State the generating function, its arguments, output,
# and necessary parameters.
if generator_type in ['random', 'bo', 'async_bo']:
    # Here, the 'user' field is for the user's (in this case,
    # the RNG) convenience.
    gen_specs['user']['gen_batch_size'] = nworkers-1
    if generator_type == 'async_bo':
        gen_specs['user']['async'] = True

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

sim_max = machine_specs['sim_max']  # Maximum number of simulations
exit_criteria = {'sim_max': sim_max}  # Exit after running sim_max simulations

# Create a different random number stream for each worker and the manager
persis_info = add_unique_random_streams({}, nworkers + 1)

# Before starting libensemble, check whether there is past history file
if os.path.exists('past_history.npy'):
    H0 = np.load('past_history.npy')
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
