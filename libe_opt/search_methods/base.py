import os

import numpy as np
from mpi4py import MPI
from libensemble.libE import libE
from libensemble.tools import save_libE_output, add_unique_random_streams
from libensemble.executors.mpi_executor import MPIExecutor
from libensemble.alloc_funcs.start_only_persistent import only_persistent_gens

from libe_opt.sim_functions import run_simulation


class SearchMethod():
    def __init__(
            self, var_names, var_lb, var_ub, sim_template, analysis_func,
            sim_number, analyzed_params=[], sim_workers=1, run_async=True,
            use_cuda=False, libE_specs={}, gen_function=None,
            history=None):
        self.var_names = var_names
        self.var_ub = var_ub
        self.var_lb = var_lb
        self.sim_template = sim_template
        self.analysis_func = analysis_func
        self.sim_number = sim_number
        self.analyzed_params = analyzed_params
        self.sim_workers=sim_workers
        self.run_async = run_async
        self.use_cuda = use_cuda
        self.libE_specs = libE_specs
        self.gen_function = gen_function
        self.history = self._load_history(history)

        self._initialize_model()
        self._create_sim_specs()
        self._create_alloc_specs()
        self._create_gen_specs()
        self._create_executor()
        self._set_default_libe_specs()

    def _initialize_model(self):
        pass

    def _create_sim_specs(self):
        self.sim_specs = {
            # Function whose output is being minimized. The parallel WarpX run is
            # launched from run_WarpX.
            'sim_f': run_simulation,
            # Name of input for sim_f, that LibEnsemble is allowed to modify.
            # May be a 1D array.
            'in': ['x'],
            'out': [ ('f', float) ] \
                # f is the single float output that LibEnsemble minimizes.
                + [(name, float, (1,)) for name in self.analyzed_params] \
                # input parameters
                + [(name, float, (1,)) for name in self.var_names],
            'user': {
                'var_params': self.var_names,
                'analysis_func': self.analysis_func,
                # keeps only the file name of the simulation template
                'sim_template': os.path.basename(self.sim_template)
            }
        }

    def _create_alloc_specs(self):
        self.alloc_specs = {
            'alloc_f': only_persistent_gens,
            'out': [('given_back', bool)],
            'user': {
                'async_return': self.run_async
            }
        }

    def _create_gen_specs(self):
        self.gen_specs = {
            # Generator function. Will randomly generate new sim inputs 'x'.
            'gen_f': self.gen_function,
            # Generator input. This is a RNG, no need for inputs.
            'in': ['sim_id', 'x', 'f'],
            'persis_in': ['sim_id', 'x', 'f'],
            'out': [
                # parameters to input into the simulation.
                ('x', float, (len(self.var_names),)),
                ('resource_sets', int)
            ],
            'user': {
                # Total max number of sims running concurrently.
                'gen_batch_size': self.sim_workers,
                # Parameter names.
                'params': self.var_names,
                # Lower bound for the n parameters.
                'lb': self.var_lb,
                # Upper bound for the n parameters.
                'ub': self.var_ub,
                # Allow generator to run on GPU.
                'use_cuda': self.use_cuda
            }
        }

    def _create_executor(self):
        exctr = MPIExecutor()
        if self.sim_template.endswith('.py'):
            exctr.register_app(full_path='simulation_script.py', calc_type='sim')

    def _set_default_libe_specs(self):
        # Add sim_template to the list of files to be copied
        # (if not present already)
        if 'sim_dir_copy_files' not in self.libE_specs:
            self.libE_specs['sim_dir_copy_files'] = [self.sim_template]
        elif self.sim_template not in self.libE_specs['sim_dir_copy_files']:
            self.libE_specs['sim_dir_copy_files'].append(self.sim_template)
        # Save H to file every N simulation evaluations
        # default value, if not defined
        if 'save_every_k_sims' not in self.libE_specs.keys():
            self.libE_specs['save_every_k_sims'] = 5
        # Force libEnsemble to create a directory for each simulation
        # default value, if not defined
        if 'sim_dirs_make' not in self.libE_specs.keys():
            self.libE_specs['sim_dirs_make'] = True
        # Force central mode
        if 'dedicated_mode' not in self.libE_specs.keys():
            self.libE_specs['dedicated_mode'] = False
        # It not using CUDA, do not allocate resources for generator.
        if not self.use_cuda:
            self.libE_specs['zero_resource_workers'] = [1]
        # If not running in parallel, set communications to `local`.
        if MPI.COMM_WORLD.Get_size() <= 1:
            self.libE_specs["nworkers"] = self.sim_workers + 1
            self.libE_specs["comms"] = 'local'

    def _load_history(self, history):
        if isinstance(history, str):
            if os.path.exists(history):
                # Load array.
                history = np.load(history)
                # Only include runs that completed
                history = history[history['returned']==True]
            else:
                raise ValueError(
                    'History file {} does not exist.'.format(history))
        assert (
            history is not None or
            not isinstance(history, np.ndarray)
        ), 'Type {} not valid for `history`'.format(type(history))
        return history

    def run(self):
        exit_criteria = {'sim_max': self.sim_number}
        persis_info = add_unique_random_streams({}, self.sim_workers + 2)
        if self.use_cuda:
            persis_info['gen_resources'] = 1
        H, persis_info, flag = libE(
            self.sim_specs,
            self.gen_specs,
            exit_criteria,
            persis_info,
            self.alloc_specs,
            self.libE_specs,
            H0=self.history
        )

        if self.libE_specs["comms"] == "local":
            is_master = True
            nworkers = self.sim_workers + 1
        else:
            is_master = (MPI.COMM_WORLD.Get_rank() == 0)
            nworkers = MPI.COMM_WORLD.Get_size() - 1

        if is_master:
            save_libE_output(H, persis_info, __file__, nworkers)
