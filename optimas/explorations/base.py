"""Contains the definition of the base Exploration class."""

import os
import glob
from typing import Optional, Union

import numpy as np

from libensemble.libE import libE
from libensemble.tools import add_unique_random_streams
from libensemble.alloc_funcs.start_only_persistent import only_persistent_gens
from libensemble.executors.mpi_executor import MPIExecutor

from optimas.generators.base import Generator
from optimas.evaluators.base import Evaluator
from optimas.utils.logger import get_logger


logger = get_logger(__name__)


class Exploration:
    """Class for launching an exploration.

    Depending on the generator, the exploration can be an optimization, a
    parameter scan, etc.

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
    history_save_period : int, optional
        Periodicity, in number of evaluated Trials, with which to save the
        history file to disk. By default equals to ``sim_workers``.
    exploration_dir_path : str, optional.
        Path to the exploration directory. By default, ``'./exploration'``.
    resume : bool, optional
        Whether the exploration should resume from a previous run in the same
        `exploration_dir_path`. If `True`, the exploration will continue from
        the last evaluation of the previous run until the total number of
        evaluations (including those of the previous run) reaches `max_evals`.
        There is no need to provide the `history` path (it will be ignored).
        If `False` (default value), the exploration will raise an error if
        the `exploration_dir_path` already exists.
    libe_comms :  {'local', 'mpi'}, optional.
        The communication mode for libEnseble. Determines whether to use
        Python ``multiprocessing`` (local mode) or MPI for the communication
        between the manager and workers. If running in ``'mpi'`` mode, the
        Optimas script should be launched with ``mpirun`` or equivalent, for
        example, ``mpirun -np N python myscript.py``. This will launch one
        manager and ``N-1`` simulation workers. In this case, the
        ``sim_workers`` parameter is ignored. By default, ``'local'`` mode
        is used.

    """

    def __init__(
        self,
        generator: Generator,
        evaluator: Evaluator,
        max_evals: int,
        sim_workers: int,
        run_async: Optional[bool] = True,
        history: Optional[str] = None,
        history_save_period: Optional[int] = None,
        exploration_dir_path: Optional[str] = "./exploration",
        resume: Optional[bool] = False,
        libe_comms: Optional[str] = "local",
    ) -> None:
        self.generator = generator
        self.evaluator = evaluator
        self.max_evals = max_evals
        self.sim_workers = sim_workers
        self.run_async = run_async
        if history_save_period is None:
            self.history_save_period = sim_workers
        else:
            self.history_save_period = history_save_period
        self.exploration_dir_path = exploration_dir_path
        self.libe_comms = libe_comms
        self._n_evals = 0
        self._resume = resume
        self._history_file_name = "exploration_history_after_evaluation_{}"
        self._load_history(history, resume)
        self._create_alloc_specs()
        self._create_executor()
        self._initialize_evaluator()
        self._set_default_libe_specs()

    def run(self, n_evals: Optional[int] = None) -> None:
        """Run the exploration.

        Parameters
        ----------
        n_evals : int, optional
            Number of evaluations to run. If not given, the exploration will
            run until the number of evaluations reaches `max_evals`.

        """
        # Set exit criteria to maximum number of evaluations.
        remaining_evals = self.max_evals - self._n_evals
        if remaining_evals < 1:
            raise ValueError(
                "The maximum number or evaluations has been reached."
            )
        if n_evals is None:
            sim_max = remaining_evals
        else:
            sim_max = min(n_evals, remaining_evals)
        exit_criteria = {"sim_max": sim_max}

        # Get initial number of generator trials.
        n_trials_initial = self.generator.n_trials

        # Create persis_info.
        persis_info = add_unique_random_streams({}, self.sim_workers + 2)

        # If specified, allocate dedicated resources for the generator.
        if self.generator.dedicated_resources and self.generator.use_cuda:
            persis_info["gen_resources"] = 1
            persis_info["gen_use_gpus"] = True
        else:
            self.libE_specs["zero_resource_workers"] = [1]

        if self._n_evals > 0:
            self.libE_specs["reuse_output_dir"] = True

        # Get gen_specs and sim_specs.
        run_params = self.evaluator.get_run_params()
        gen_specs = self.generator.get_gen_specs(
            self.sim_workers, run_params, sim_max
        )
        sim_specs = self.evaluator.get_sim_specs(
            self.generator.varying_parameters,
            self.generator.objectives,
            self.generator.analyzed_parameters,
        )

        # Launch exploration with libEnsemble.
        history, persis_info, flag = libE(
            sim_specs,
            gen_specs,
            exit_criteria,
            persis_info,
            self.alloc_specs,
            self.libE_specs,
            H0=self.history,
        )

        # Update history.
        self.history = history

        # Update generator with the one received from libE.
        self.generator._update(persis_info[1]["generator"])

        # Update number of evaluation in this exploration.
        n_trials_final = self.generator.n_trials
        self._n_evals += n_trials_final - n_trials_initial

        # Determine if current rank is master.
        if self.libE_specs["comms"] == "local":
            is_master = True
        else:
            from mpi4py import MPI

            is_master = MPI.COMM_WORLD.Get_rank() == 0

        # Save history.
        if is_master:
            self._save_history()

    def _create_executor(self) -> None:
        """Create libEnsemble executor."""
        self.executor = MPIExecutor()

    def _initialize_evaluator(self) -> None:
        """Initialize exploration evaluator."""
        self.evaluator.initialize()

    def _load_history(
        self,
        history: Union[str, np.ndarray, None],
        resume: Optional[bool] = False,
    ) -> None:
        """Load history file."""
        # To resume an exploration, get history file from previous run.
        if resume:
            if history is not None:
                logger.info(
                    "The `history` argument is ignored when `resume=True`. "
                    "The exploration will resume using the most recent "
                    "history file."
                )
            history = self._get_most_recent_history_file_path()
            if history is None:
                raise ValueError(
                    "Previous history file not found. "
                    "Cannot resume exploration."
                )
        # Read file.
        if isinstance(history, str):
            if os.path.exists(history):
                # Load array.
                history = np.load(history)
                # Only include runs that completed
                history = history[history["sim_ended"]]
            else:
                raise ValueError(
                    "History file {} does not exist.".format(history)
                )
        assert history is None or isinstance(
            history, np.ndarray
        ), "Type {} not valid for `history`".format(type(history))
        # Incorporate history into generator.
        if history is not None:
            self.generator.incorporate_history(history)
        # When resuming an exploration, update evaluations counter.
        if resume:
            self._n_evals = history.size
        self.history = history

    def _save_history(self):
        """Save history array to file."""
        filename = self._history_file_name.format(self._n_evals)
        exploration_dir_path = os.path.abspath(self.exploration_dir_path)
        file_path = os.path.join(exploration_dir_path, filename)
        if not os.path.isfile(filename):
            old_files = os.path.join(
                exploration_dir_path, self._history_file_name.format("*")
            )
            for old_file in glob.glob(old_files):
                os.remove(old_file)
            np.save(file_path, self.history)

    def _get_most_recent_history_file_path(self):
        """Get path of most recently saved history file."""
        old_exploration_history_files = glob.glob(
            os.path.join(
                os.path.abspath(self.exploration_dir_path),
                self._history_file_name.format("*"),
            )
        )
        old_libe_history_files = glob.glob(
            os.path.join(
                os.path.abspath(self.exploration_dir_path),
                "libE_history_{}".format("*"),
            )
        )
        old_files = old_exploration_history_files + old_libe_history_files
        if old_files:
            file_evals = [int(file.split("_")[-1][:-4]) for file in old_files]
            i_max_evals = np.argmax(np.array(file_evals))
            return old_files[i_max_evals]

    def _set_default_libe_specs(self) -> None:
        """Set default exploration libe_specs."""
        libE_specs = {}
        # Save H to file every N simulation evaluations
        # default value, if not defined
        libE_specs["save_every_k_sims"] = self.history_save_period
        # Force central mode
        libE_specs["dedicated_mode"] = False
        # Set communications and corresponding number of workers.
        libE_specs["comms"] = self.libe_comms
        if self.libe_comms == "local":
            libE_specs["nworkers"] = self.sim_workers + 1
        elif self.libe_comms == "mpi":
            # Warn user if openmpi is being used.
            # When running with MPI communications, openmpi cannot be used as
            # it does not support nesting MPI.
            # MPI is only imported here to avoid issues with openmpi when
            # running with local communications.
            from mpi4py import MPI

            if "openmpi" in MPI.Get_library_version().lower():
                raise RuntimeError(
                    "Running with mpi communications is not supported with "
                    "openMPI. Please use MPICH (linux and macOS) or MSMPI "
                    "(Windows) instead."
                )
        else:
            raise ValueError(
                "Communication mode '{}'".format(self.libe_comms)
                + " not recognized. Possible values are 'local' or 'mpi'."
            )
        # Set exploration directory path.
        libE_specs["ensemble_dir_path"] = "evaluations"
        libE_specs["use_workflow_dir"] = True
        libE_specs["workflow_dir_path"] = self.exploration_dir_path

        # Ensure evaluations of last batch are sent back to the generator.
        libE_specs["final_gen_send"] = True

        # Save history file on completion and without date information in the
        # name, so that it can be overwritten in subsequent calls to `run` or
        # when resuming an exploration.
        libE_specs["save_H_on_completion"] = True
        libE_specs["save_H_with_date"] = False
        libE_specs["H_file_prefix"] = "exploration_history"

        # get specs from generator and evaluator
        gen_libE_specs = self.generator.get_libe_specs()
        ev_libE_specs = self.evaluator.get_libe_specs()
        self.libE_specs = {**gen_libE_specs, **ev_libE_specs, **libE_specs}

    def _create_alloc_specs(self) -> None:
        """Create exploration alloc_specs."""
        self.alloc_specs = {
            "alloc_f": only_persistent_gens,
            "out": [("given_back", bool)],
            "user": {"async_return": self.run_async},
        }
