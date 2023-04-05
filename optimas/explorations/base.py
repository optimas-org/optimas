"""Contains the definition of the base Exploration class."""

import os
from typing import Optional, Union

import numpy as np
from mpi4py import MPI

from libensemble.libE import libE
from libensemble.tools import save_libE_output, add_unique_random_streams
from libensemble.alloc_funcs.start_only_persistent import only_persistent_gens
from libensemble.executors.mpi_executor import MPIExecutor

from optimas.generators.base import Generator
from optimas.evaluators.base import Evaluator


class Exploration():
    """Base class in charge of launching an exploration (i.e., an optimization
    or parameter scan).

    Parameters
    ----------
    generator : Generator
        The generator used to suggest new Trials.
    evaluator : Evaluator
        The evaluator that will execute the Trials.
    max_evals : int
        Maximum number of trials that will be evaluated in the exploration.
    sim_workers : int
        Number of parallel workers performing simulations.
    run_async : bool, optional
        Whether the evaluators should be performed asynchronously (i.e.,
        without waiting for all workers to finish before staring a new
        evaluation). By default, True.
    history : str, optional
        Path to a history file of a past exploration from which to restart
        the new one. By default, None.
    exploration_dir_path : str, optional.
        Path to the exploration directory. By default, ``'./exploration'``.
    """
    def __init__(
        self,
        generator: Generator,
        evaluator: Evaluator,
        max_evals: int,
        sim_workers: int,
        run_async: Optional[bool] = True,
        history: Optional[str] = None,
        exploration_dir_path: Optional[str] = './exploration'
    ) -> None:
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

    def run(self) -> None:
        """Run the exploration."""
        # Set exit criteria to maximum number of evaluations.
        exit_criteria = {'sim_max': self.max_evals}

        # Create persis_info.
        persis_info = add_unique_random_streams({}, self.sim_workers + 2)

        # If specified, allocate dedicated resources for the generator.
        if self.generator.dedicated_resources:
            persis_info['gen_resources'] = 1

        # Get gen_specs and sim_specs.
        gen_specs = self.generator.get_gen_specs(self.sim_workers)
        sim_specs = self.evaluator.get_sim_specs(
            self.generator.varying_parameters,
            self.generator.objectives,
            self.generator.analyzed_parameters
        )

        # If provided, incorporate history into generator.
        if self.history is not None:
            self.generator.incorporate_history(self.history)

        # Launch exploration with libEnsemble.
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

        # Determine if current rank is master.
        if self.libE_specs["comms"] == "local":
            is_master = True
            nworkers = self.sim_workers + 1
        else:
            is_master = (MPI.COMM_WORLD.Get_rank() == 0)
            nworkers = MPI.COMM_WORLD.Get_size() - 1

        # Save history.
        if is_master:
            save_libE_output(history, persis_info, __file__, nworkers)

    def _create_executor(self) -> None:
        """Create libEnsemble executor."""
        self.executor = MPIExecutor()

    def _initialize_evaluator(self) -> None:
        """Initialize exploration evaluator."""
        self.evaluator.initialize()

    def _load_history(self, history: Union[str, np.ndarray, None]) -> None:
        """Load history file."""
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

    def _set_default_libe_specs(self) -> None:
        """Set default exploration libe_specs."""
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

    def _create_alloc_specs(self) -> None:
        """Create exploration alloc_specs."""
        self.alloc_specs = {
            'alloc_f': only_persistent_gens,
            'out': [('given_back', bool)],
            'user': {
                'async_return': self.run_async
            }
        }
