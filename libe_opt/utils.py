import warnings

import numpy as np
import os

from libe_opt.gen_functions import get_generator_function
from libe_opt.alloc_functions import get_alloc_function_from_gen_type
from libe_opt.sim_functions import run_simulation


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


def create_sim_specs(analyzed_params, var_params, analysis_func, sim_template,
                     mf_params=None, mt_params=None):
    # State the objective function, its arguments, output, and necessary parameters
    # (and their sizes). Here, the 'user' field is for the user's (in this case,
    # the simulation) convenience. Feel free to use it to pass number of nodes,
    # number of ranks per note, time limit per simulation etc.
    sim_specs = {
        # Function whose output is being minimized. The parallel WarpX run is
        # launched from run_WarpX.
        'sim_f': run_simulation,
        # Name of input for sim_f, that LibEnsemble is allowed to modify.
        # May be a 1D array.
        'in': ['x'],
        'out': [ ('f', float) ] \
            # f is the single float output that LibEnsemble minimizes.
            + analyzed_params \
            # input parameters
            + [(name, float, (1,)) for name in var_params.keys()],
        'user': {
            'var_params': list(var_params.keys()),
            'analysis_func': analysis_func,
            # keeps only the file name of the simulation template
            'sim_template': os.path.basename(sim_template)
        }
    }

    # If multifidelity is used, add fidelity to sim_specs 'in' and 'out'.
    if mf_params is not None:
        sim_specs['in'].append('z')
        fidel_type, fidel_len = determine_fidelity_type_and_length(mf_params)
        sim_specs['out'].append((mf_params['name'], fidel_type, fidel_len))
        sim_specs['user']['z_name'] = mf_params['name']
    elif mt_params is not None:
        sim_specs['in'].append('task')
        sim_specs['user']['extra_args'] = {}
        if 'extra_args_lofi' in mt_params:
            lofi_name = mt_params['name_lofi']
            sim_specs['user']['extra_args'][lofi_name] = mt_params['extra_args_lofi']
        if 'extra_args_hifi' in mt_params:
            hifi_name = mt_params['name_hifi']
            sim_specs['user']['extra_args'][hifi_name] = mt_params['extra_args_hifi']
    return sim_specs


def create_alloc_specs(gen_type, run_async=False):
    # Allocator function, decides what a worker should do.
    # We use a LibEnsemble allocator.
    alloc_specs = {
        'alloc_f': get_alloc_function_from_gen_type(gen_type),
        'out': [('given_back', bool)]
        }
    if gen_type in ['bo', 'random', 'normal']:
        alloc_specs['user'] = {'async_return': run_async}
    elif gen_type in ['bo_mt']:
        warnings.warn(
            "Asynchronous mode not available in multi-task optimization."
            " `run_async` parameter ignored."
        )
        alloc_specs['user'] = {'async_return': False}

    return alloc_specs


def create_gen_specs(gen_type, nworkers, var_params, mf_params=None,
                     mt_params=None, bo_backend='df', ax_client=None,
                     use_cuda=False):
    # Problem dimension. This is the number of input parameters exposed,
    # that LibEnsemble will vary in order to minimize a single output parameter.
    n = len(var_params)

    use_mf = False
    discrete_mf = False
    use_mt = False
    if gen_type == 'bo':
        # Detect if multifidelity or multi-task are used
        if mf_params is not None:
            use_mf = True
            if mf_params['discrete']:
                discrete_mf = True
        elif mt_params is not None:
            use_mt = True
        # Check backend
        if bo_backend not in ['df', 'ax']:
            raise ValueError(
                "Backend '{}' not recognized. ".format(bo_backend) +
                "Possible BO backends are 'df' (dragonfly) or 'ax' (Ax/BoTorch)"
            )

    # Here, the 'user' field is for the user's (in this case,
    # the RNG) convenience.
    gen_specs = {
        # Generator function. Will randomly generate new sim inputs 'x'.
        'gen_f': get_generator_function(
            gen_type, bo_backend, use_mf, discrete_mf, use_mt),
        # Generator input. This is a RNG, no need for inputs.
        'in': ['sim_id', 'x', 'f'],
        'persis_in': ['sim_id', 'x', 'f'],
        'out': [
            # parameters to input into the simulation.
            ('x', float, (n,)),
            ('resource_sets', int)
        ],
        'user': {
            # Total max number of sims running concurrently.
            'gen_batch_size': nworkers-1,
            # Parameter names.
            'params': [ name for name in var_params.keys() ],
            # Lower bound for the n parameters.
            'lb': np.array([ v[0] for v in var_params.values() ]),
            # Upper bound for the n parameters.
            'ub': np.array([ v[1] for v in var_params.values() ]),
            # Allow generator to run on GPU.
            'use_cuda': use_cuda
        }
    }
    if mf_params is not None:
        gen_specs['in'].append('z')
        gen_specs['persis_in'].append('z')
    elif mf_params is not None:
        gen_specs['in'].append('task')
        gen_specs['persis_in'].append('task')

    # State the generating function, its arguments, output,
    # and necessary parameters.
    if gen_type == 'bo':
        # Add Ax client
        if bo_backend == 'ax':
            gen_specs['user']['client'] = ax_client

        # If multifidelity is used, add fidelity to 'out' and multifidelity
        # parameters to 'user'.
        if use_mf:
            fidel_type, fidel_len = determine_fidelity_type_and_length(mf_params)
            gen_specs['out'].append(('z', fidel_type, fidel_len))
            gen_specs['user']['mf_params'] = mf_params

        # If multi-task is used, make user async mode is not used, add task to
        # 'out' and multi-task parameters to 'user'.
        if use_mt:
            gen_specs['out'].append(
                    ('task', str, max([len(mt_params['name_hifi']), len(mt_params['name_lofi'])]))
                    )
            gen_specs['user']['mt_params'] = mt_params

    elif gen_type == 'aposmm':
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

    return gen_specs


def create_libe_specs(sim_template, libE_specs={}):
    # Add sim_template to the list of files to be copied
    # (if not present already)
    if 'sim_dir_copy_files' not in libE_specs:
        libE_specs['sim_dir_copy_files'] = [sim_template]
    elif sim_template not in libE_specs['sim_dir_copy_files']:
        libE_specs['sim_dir_copy_files'].append(sim_template)
    # Save H to file every N simulation evaluations
    # default value, if not defined
    if 'save_every_k_sims' not in libE_specs.keys():
        libE_specs['save_every_k_sims'] = 5
    # Force libEnsemble to create a directory for each simulation
    # default value, if not defined
    if 'sim_dirs_make' not in libE_specs.keys():
        libE_specs['sim_dirs_make'] = True
    # Force central mode
    if 'dedicated_mode' not in libE_specs.keys():
        libE_specs['dedicated_mode'] = False

    return libE_specs
