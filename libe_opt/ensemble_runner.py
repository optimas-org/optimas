import os

import numpy as np
from libensemble.libE import libE
from libensemble.tools import check_inputs
from libensemble import logger
from libensemble.executors.mpi_executor import MPIExecutor
from libensemble.tools import save_libE_output, add_unique_random_streams

from libe_opt.utils import (
    create_alloc_specs, create_gen_specs, create_libe_specs, create_sim_specs)


def run_ensemble(
        nworkers, sim_max, is_master, gen_type, analyzed_params,
        var_params, analysis_func, sim_template=None,
        mf_params=None, mt_params=None,
        past_history=None, libE_specs={}, run_async=False, use_cuda=False,
        bo_backend='df', ax_client=None):

    if sim_template is None:
        # Automatically detect the template simulation script
        sim_template = [ filename for filename in os.listdir('./') \
                         if filename.startswith('template') ][0]

    # Create specs.
    sim_specs = create_sim_specs(
        analyzed_params, var_params, analysis_func, sim_template, mf_params,
        mt_params)
    alloc_specs = create_alloc_specs(gen_type, run_async)
    gen_specs = create_gen_specs(
        gen_type, nworkers, var_params, mf_params, mt_params,
        bo_backend, ax_client, use_cuda)
    libE_specs = create_libe_specs(sim_template, libE_specs)

    # Setup MPI executor
    exctr = MPIExecutor()
    if sim_template.endswith('.py'):
        sim_script = os.path.basename(sim_template)
        sim_script = sim_script[len('template_'):]  # Strip 'template_' from name
        exctr.register_app(full_path=sim_script, calc_type='sim')
    else:
        # By default, if the template is not a `.py` file, we run
        # it with an executable. The executable should start with `warpx`
        executables = [filename for filename in os.listdir() \
                     if filename.startswith('warpx')]
        if len(executables) == 0:
            raise ValueError('You need to copy the WarpX executable in this folder.')
        else:
            executable = './' + executables[0]
            exctr.register_app(full_path=executable, calc_type='sim')
        libE_specs['sim_dir_copy_files'].append(executable)

    # libE logger
    logger.set_level('INFO')

    # Exit criteria
    exit_criteria = {'sim_max': sim_max}  # Exit after running sim_max simulations

    # Create a different random number stream for each worker and the manager
    persis_info = add_unique_random_streams({}, nworkers + 1)

    # Indicate whether to allocate resources for the generator
    if use_cuda:
        persis_info['gen_resources'] = 1
    else:
        libE_specs['zero_resource_workers'] = [1]

    # Before starting libensemble, check whether there is past history file
    if past_history is not None and os.path.exists(past_history):
        H0 = np.load(past_history)
        H0 = H0[H0['sim_ended']]  # Only include runs that completed
        check_inputs(
            H0=H0, sim_specs=sim_specs, alloc_specs=alloc_specs,
            gen_specs=gen_specs)
    else:
        H0 = None

    # Run LibEnsemble, and store results in history array H
    H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria,
                                persis_info, alloc_specs, libE_specs, H0=H0)

    # Save results to numpy file
    if is_master:
        save_libE_output(H, persis_info, __file__, nworkers)
