"""Contains the definition of the base Exploration class."""

import os
import glob
import json
import time
from typing import Optional, Union, Dict, List, Literal

import numpy as np
import pandas as pd

from libensemble.libE import libE
from libensemble.history import History
from libensemble.tools import add_unique_random_streams
from libensemble.alloc_funcs.start_only_persistent import only_persistent_gens
from libensemble.executors.mpi_executor import MPIExecutor

from optimas.core.trial import TrialStatus
from optimas.generators.base import Generator
from optimas.evaluators.base import Evaluator
from optimas.evaluators.function_evaluator import FunctionEvaluator
from optimas.utils.logger import get_logger
from optimas.utils.other import convert_to_dataframe


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
    max_evals : int, optional
        Maximum number of trials that will be evaluated in the exploration.
        If not given, the exploration can run indefinitely.
    sim_workers : int, optional
        Number of parallel workers performing simulations. By default, 1.
    run_async : bool, optional
        Whether the evaluations should be performed asynchronously (i.e.,
        without waiting for all workers to finish before staring a new
        evaluation). This is useful when the completion time of the
        evaluations is not uniform. By default, False.
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
    libe_comms :  {'local', 'threads', 'mpi'}, optional.
        The communication mode for libEnseble. Determines whether to use
        Python ``multiprocessing`` (local), ``threading`` (threads)
        or MPI for the communication between the manager and workers.
        The ``'threads'`` mode is only recommended when running in a
        Jupyter notebook if the default 'local' mode has issues (this
        can happen especially on Windows and Mac, which use multiprocessing
        ``spawn``). ``'threads'`` only supports ``FunctionEvaluator``s.
        If running in ``'mpi'`` mode, the Optimas script should be launched
        with ``mpirun`` or equivalent, for example,
        ``mpirun -np N python myscript.py``. This will launch one
        manager and ``N-1`` simulation workers. In this case, the
        ``sim_workers`` parameter is ignored. By default, ``'local'`` mode
        is used.

    """

    def __init__(
        self,
        generator: Generator,
        evaluator: Evaluator,
        max_evals: Optional[int] = np.inf,
        sim_workers: Optional[int] = 1,
        run_async: Optional[bool] = False,
        history: Optional[str] = None,
        history_save_period: Optional[int] = None,
        exploration_dir_path: Optional[str] = "./exploration",
        resume: Optional[bool] = False,
        libe_comms: Optional[Literal["local", "threads", "mpi"]] = "local",
    ) -> None:
        # For backward compatibility, check for old threading name.
        if libe_comms == "local_threading":
            libe_comms = "threads"
        if libe_comms == "threads" and not isinstance(
            evaluator, FunctionEvaluator
        ):
            raise ValueError(
                "'threads' mode is only supported when using a "
                "`FunctionEvaluator`. Use 'local' mode instead."
            )
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
        self._history_file_prefix = "exploration_history"
        self._create_alloc_specs()
        self._create_executor()
        self._initialize_evaluator()
        self._set_default_libe_specs()
        self._libe_history = self._create_libe_history()
        self._load_history(history, resume)

    @property
    def history(self) -> pd.DataFrame:
        """Get the exploration history."""
        history = convert_to_dataframe(self._libe_history.H)
        ordered_columns = ["trial_index", "trial_status"]
        ordered_columns += [p.name for p in self.generator.varying_parameters]
        ordered_columns += [p.name for p in self.generator.objectives]
        ordered_columns += [p.name for p in self.generator.analyzed_parameters]
        ordered_columns += sorted(
            [n for n in history if n not in ordered_columns]
        )
        return history[ordered_columns]

    def run(self, n_evals: Optional[int] = None) -> None:
        """Run the exploration.

        Parameters
        ----------
        n_evals : int, optional
            Number of evaluations to run. If not given, the exploration will
            run until the number of evaluations reaches `max_evals`.

        """
        # Store current working directory. It has been observed that sometimes
        # (especially when using `local_threading`) the working directory
        # is changed to the exploration directory after the call to `libE`.
        # As a workaround, the cwd is stored and then set again at the end of
        # `run`.
        cwd = os.getcwd()

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
        exit_criteria = {}
        if np.isfinite(sim_max):
            exit_criteria["sim_max"] = sim_max

        # Get initial number of generator trials.
        n_evals_initial = self.generator.n_evaluated_trials

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

        # Save exploration parameters to json file.
        self._save_exploration_parameters()

        # Launch exploration with libEnsemble.
        history, persis_info, flag = libE(
            sim_specs,
            gen_specs,
            exit_criteria,
            persis_info,
            self.alloc_specs,
            self.libE_specs,
            H0=self._libe_history.H,
        )

        # Update history.
        self._libe_history.H = history

        # Update generator with the one received from libE.
        self.generator._update(persis_info[1]["generator"])

        # Update number of evaluation in this exploration.
        n_evals_final = self.generator.n_evaluated_trials
        self._n_evals += n_evals_final - n_evals_initial

        # Reset `cwd` to initial value before `libE` was called.
        os.chdir(cwd)

    def attach_trials(
        self,
        trial_data: Union[Dict, List[Dict], np.ndarray, pd.DataFrame],
        ignore_unrecognized_parameters: Optional[bool] = False,
    ) -> None:
        """Attach trials for future evaluation.

        Use this method to manually suggest a set of trials to the exploration.
        The given trials will be the first ones to be evaluated the next time
        that `run` is called.

        The given data should contain all necessary fields to create the trials
        (i.e., the values of the `VaryingParameters`).
        The method accepts this data as a list, dictionary, pandas DataFrame or
        numpy structured array (see example below).

        Parameters
        ----------
        trial_data : dict, list, NDArray or DataFrame
            The data containing the trial parameters.
        ignore_unrecognized_parameters : bool, optional
            Whether to ignore unrecognized parameters in the given data. By
            default, if the data contains more fields than the
            `VaryingParameters`, `AnalyzedParameters` and `Objectives` of the
            exploration, a `ValueError` is raised, since this might indicate
            a problem in the data. If set to `True`, the error will be ignored.

        Examples
        --------
        >>> import pandas as pd
        >>> from optimas.explorations import Exploration
        >>> from optimas.generators import RandomSamplingGenerator
        >>> from optimas.evaluators import FunctionEvaluator
        >>> from optimas.core import VaryingParameter, Objective
        >>> params = [VaryingParameter(f"x{i}", -5, 5) for i in range(2)]
        >>> objs = [Objective("f")]
        >>> def eval_func(input_params, output_params):
        ...     # Placeholder evaluator
        ...     output_params["f"] = 1.
        >>> ev = FunctionEvaluator(function=eval_func)
        >>> gen = RandomSamplingGenerator(
        ...     varying_parameters=params,
        ...     objectives=objs
        ... )
        >>> exploration = Exploration(
        ...     generator=gen,
        ...     evaluator=ev,
        ...     max_evals=100,
        ...     sim_workers=2
        ... )

        **Attach trials as list**

        >>> exploration.attach_trials(
        ...     [
        ...         {"x0": 1., "x1": 1.},
        ...         {"x0": 4., "x1": 3.},
        ...         {"x0": 1., "x1": 5.}
        ...     ]
        ... )

        **Attach trials as dictionary**

        >>> exploration.attach_trials(
        ...     {
        ...         "x0": [1., 4., 1.],
        ...         "x1": [1., 3., 5.]
        ...     },
        ... )

        **Attach trials as pandas dataframe**

        >>> df = pd.DataFrame({"x0": [1., 4., 1.], "x1": [1., 3., 1.]})
        >>> exploration.attach_trials(df)
        """
        trial_data = convert_to_dataframe(trial_data)
        self.generator.attach_trials(
            trial_data,
            ignore_unrecognized_parameters=ignore_unrecognized_parameters,
        )

    def evaluate_trials(
        self,
        trial_data: Union[Dict, List[Dict], np.ndarray, pd.DataFrame],
        ignore_unrecognized_parameters: Optional[bool] = False,
    ) -> None:
        """Attach and evaluate trials.

        Use this method to manually suggest a set of trials to the exploration
        and evaluate them immediately.

        The given data should contain all necessary fields to create the trials
        (i.e., the values of the `VaryingParameters`).
        The method accepts this data as a list, dictionary, pandas DataFrame or
        numpy structured array (see example in
        :meth:`.Exploration.attach_trials`).

        Parameters
        ----------
        trial_data : dict, list, NDArray or DataFrame
            The data containing the trial parameters.
        ignore_unrecognized_parameters : bool, optional
            Whether to ignore unrecognized parameters in the given data. By
            default, if the data contains more fields than the
            `VaryingParameters`, `AnalyzedParameters` and `Objectives` of the
            exploration, a `ValueError` is raised, since this might indicate
            a problem in the data. If set to `True`, the error will be ignored.
        """
        trial_data = convert_to_dataframe(trial_data)
        self.attach_trials(
            trial_data,
            ignore_unrecognized_parameters=ignore_unrecognized_parameters,
        )
        self.run(n_evals=len(trial_data))

    def attach_evaluations(
        self,
        evaluation_data: Union[Dict, List[Dict], np.ndarray, pd.DataFrame],
        ignore_unrecognized_parameters: Optional[bool] = False,
    ) -> None:
        """Attach evaluations from external source.

        Use this method to manually attach a set of evaluations to the
        exploration. These could be evaluations that were carried out in
        a previous exploration or from any other source. The data from the
        evaluations will be given to (and used by) the generator. The
        attached evaluations are not counted when determining if `max_evals`
        has been reached.

        The given data should contain all necessary fields that define an
        evaluation (i.e., the values of the `VaryingParameters`,
        `AnalyzedParameters` and `Objectives`).
        The method accepts this data as a list, dictionary, pandas DataFrame or
        numpy structured array (see example in
        :meth:`.Exploration.attach_trials`).

        Parameters
        ----------
        trial_data : dict, list, NDArray or DataFrame
            The data containing the trial parameters.
        ignore_unrecognized_parameters : bool, optional
            Whether to ignore unrecognized parameters in the given data. By
            default, if the data contains more fields than the
            `VaryingParameters`, `AnalyzedParameters` and `Objectives` of the
            exploration, a `ValueError` is raised, since this might indicate
            a problem in the data. If set to `True`, the error will be ignored.
        """
        evaluation_data = convert_to_dataframe(evaluation_data)

        # Determine if evaluations come from past history array and, if so,
        # keep only those that finished.
        is_history = "sim_ended" in evaluation_data
        if is_history:
            evaluation_data = evaluation_data[evaluation_data["sim_ended"]]

        n_evals = len(evaluation_data)
        if n_evals == 0:
            return

        # Increase length of history and get a view of the added rows.
        self._libe_history.grow_H(n_evals)
        history_new = self._libe_history.H[-n_evals:]

        fields = evaluation_data.columns.values.tolist()

        # Check if the given evaluations are missing required fields.
        all_params = (
            self.generator.varying_parameters
            + self.generator.objectives
            + self.generator.analyzed_parameters
        )
        missing_fields = [p.name for p in all_params if p.name not in fields]
        if missing_fields:
            raise ValueError(
                "Could not attach evaluations from given data because the "
                f"required fields {missing_fields} are missing."
            )

        # Check if the given evaluations have more fields than required.
        history_fields = history_new.dtype.names
        extra_fields = [f for f in fields if f not in history_fields]
        if extra_fields and not ignore_unrecognized_parameters:
            raise ValueError(
                f"The given data contains the fields {extra_fields}, which "
                "are unknown to the generator. If this is expected, ignore "
                "them by setting `ignore_unrecognized_parameters=True`."
            )

        # Fill in new rows.
        for field in fields:
            if field in history_new.dtype.names:
                history_new[field] = evaluation_data[field]

        if not is_history:
            current_time = time.time()
            history_new["gen_started_time"] = current_time
            history_new["gen_ended_time"] = current_time
            history_new["gen_informed_time"] = current_time
            history_new["sim_started_time"] = current_time
            history_new["sim_ended_time"] = current_time
            history_new["gen_informed"] = True
            history_new["sim_started"] = True
            history_new["sim_ended"] = True
            history_new["trial_index"] = np.arange(
                self.generator._trial_count,
                self.generator._trial_count + n_evals,
                dtype=int,
            )
        if "trial_status" not in fields:
            history_new["trial_status"] = TrialStatus.COMPLETED.name

        # Incorporate new history into generator.
        self.generator.incorporate_history(history_new)

    def mark_evaluation_as_failed(self, trial_index):
        """Mark an already evaluated trial as failed.

        Parameters
        ----------
        trial_index : int
            The index of the trial.
        """
        self.generator.mark_trial_as_failed(trial_index)
        i = np.where(self._libe_history.H["trial_index"] == trial_index)[0][0]
        self._libe_history.H[i]["trial_status"] = TrialStatus.FAILED.name

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
        # Incorporate history into exploration.
        if history is not None:
            self.attach_evaluations(history)
            # When resuming an exploration, update evaluations counter.
            if resume:
                self._n_evals = history.size

    def _create_libe_history(self) -> History:
        """Initialize an empty libEnsemble history."""
        run_params = self.evaluator.get_run_params()
        gen_specs = self.generator.get_gen_specs(
            self.sim_workers, run_params, None
        )
        sim_specs = self.evaluator.get_sim_specs(
            self.generator.varying_parameters,
            self.generator.objectives,
            self.generator.analyzed_parameters,
        )
        libe_history = History(
            self.alloc_specs,
            sim_specs,
            gen_specs,
            exit_criteria={"sim_max": 0},
            H0=[],
        )
        return libe_history

    def _get_most_recent_history_file_path(self):
        """Get path of most recently saved history file."""
        # Sort files by date and get the most recent one.
        # In principle there should be only one file, but just in case.
        exp_path = os.path.abspath(self.exploration_dir_path)
        history_files = glob.glob(
            os.path.join(exp_path, self._history_file_prefix + "*")
        )
        history_files.sort(
            key=lambda f: os.path.getmtime(os.path.join(exp_path, f))
        )
        if history_files:
            return history_files[-1]

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
        if self.libe_comms in ["local", "threads"]:
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
                + " not recognized. Possible values are 'local', "
                + "'threads' or 'mpi'."
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
        libE_specs["H_file_prefix"] = self._history_file_prefix

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

    def _save_exploration_parameters(self):
        """Save exploration parameters to a JSON file."""
        params = {}
        for i, param in enumerate(self.generator.varying_parameters):
            params[f"varying_parameter_{i}"] = {
                "type": "VaryingParameter",
                "value": param.model_dump_json(),
            }
        for i, param in enumerate(self.generator.objectives):
            params[f"objective_{i}"] = {
                "type": "Objective",
                "value": param.model_dump_json(),
            }
        for i, param in enumerate(self.generator.analyzed_parameters):
            params[f"analyzed_parameter_{i}"] = {
                "type": "Parameter",
                "value": param.model_dump_json(),
            }
        main_dir = os.path.abspath(self.exploration_dir_path)
        if not os.path.isdir(main_dir):
            os.makedirs(main_dir)
        file_path = os.path.join(main_dir, "exploration_parameters.json")
        with open(file_path, "w") as file:
            file.write(json.dumps(params))
