import os

import numpy as np
from mpi4py import MPI

from libensemble.libE import libE
from libensemble.tools import save_libE_output, add_unique_random_streams
from libensemble.alloc_funcs.start_only_persistent import only_persistent_gens
from libensemble.executors.mpi_executor import MPIExecutor


class Exploration():
    def __init__(self, generator, evaluator, max_evals, sim_workers,
            run_async=True, history=None, exploration_dir_path='./exploration'):
        self.generator = generator
        self.evaluator = evaluator
        self.max_evals = max_evals
        self.sim_workers = sim_workers
        self.run_async = run_async
        self.history = self._load_history(history)
        self.exploration_dir_path = exploration_dir_path
        self._set_default_libe_specs()
        self._create_alloc_specs()
        self._create_executor()
        self._initialize_evaluator()

    def run(self):
        exit_criteria = {'sim_max': self.max_evals}
        persis_info = add_unique_random_streams({}, self.sim_workers + 2)
        if self.generator.use_cuda:
            persis_info['gen_resources'] = 1
        gen_specs = self.generator.get_gen_specs(self.sim_workers)
        sim_specs = self.evaluator.get_sim_specs(self.generator.variables, self.generator.objectives)
        if self.history is not None:
            self.generator.incorporate_history(self.history)
        history, persis_info, flag = libE(
            sim_specs,
            gen_specs,
            exit_criteria,
            persis_info,
            self.alloc_specs,
            self.libE_specs,
            H0=self.history
        )

        # Update history.
        self.history = history

        if self.libE_specs["comms"] == "local":
            is_master = True
            nworkers = self.sim_workers + 1
        else:
            is_master = (MPI.COMM_WORLD.Get_rank() == 0)
            nworkers = MPI.COMM_WORLD.Get_size() - 1

        if is_master:
            save_libE_output(history, persis_info, __file__, nworkers)

    def _create_executor(self):
        self.executor = MPIExecutor()

    def _initialize_evaluator(self):
        self.evaluator.initialize()

    def _load_history(self, history):
        if isinstance(history, str):
            if os.path.exists(history):
                # Load array.
                history = np.load(history)
                # Only include runs that completed
                history = history[history['sim_ended']]
            else:
                raise ValueError(
                    'History file {} does not exist.'.format(history))
        assert history is None or isinstance(history, np.ndarray), (
            'Type {} not valid for `history`'.format(type(history))
        )
        return history

    def _set_default_libe_specs(self):
        libE_specs = {}
        # Save H to file every N simulation evaluations
        # default value, if not defined
        libE_specs['save_every_k_sims'] = 5
        # Force central mode
        libE_specs['dedicated_mode'] = False
        # It not using CUDA, do not allocate resources for generator.
        # If not running in parallel, set communications to `local`.
        if MPI.COMM_WORLD.Get_size() <= 1:
            libE_specs["nworkers"] = self.sim_workers + 1
            libE_specs["comms"] = 'local'
        # Set exploration directory path.
        libE_specs['ensemble_dir_path'] = self.exploration_dir_path

        # get specs from generator and evaluator
        gen_libE_specs = self.generator.get_libe_specs()
        ev_libE_specs = self.evaluator.get_libe_specs()
        self.libE_specs = {**gen_libE_specs, **ev_libE_specs, **libE_specs}

    def _create_alloc_specs(self):
        self.alloc_specs = {
            'alloc_f': only_persistent_gens,
            'out': [('given_back', bool)],
            'user': {
                'async_return': self.run_async
            }
        }
