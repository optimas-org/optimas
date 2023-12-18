"""Contains the definition of the ExplorationDiagnostics class."""
import os
from warnings import warn
import pathlib
import json
from typing import Optional, List, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd
import matplotlib.pyplot as plt

from optimas.core import VaryingParameter, Objective, Parameter


class ExplorationDiagnostics:
    """Utilities for analyzing the output of an exploration.

    Parameters
    ----------
    path : str
        Path to the exploration directory or to an
        individual `.npy` history file.
    relative_start_time : bool, optional
        Whether the time diagnostics should be relative to the start time
        of the exploration. By default, True.
    remove_unfinished_evaluations : bool, optional
        Whether the data from unfinished evaluations (e.g., due to failed
        evaluation) should be removed from the diagnostics. By default, True.

    """

    def __init__(
        self,
        path: str,
        relative_start_time: Optional[bool] = True,
        remove_unfinished_evaluations: Optional[bool] = True,
    ) -> None:
        # Find the `npy` file that contains the results
        if os.path.isdir(path):
            # Get history files sorted by creation date.
            output_files = [
                filename
                for filename in os.listdir(path)
                if "_history_" in filename and filename.endswith(".npy")
            ]
            output_files.sort(
                key=lambda f: os.path.getmtime(os.path.join(path, f))
            )
            if len(output_files) == 0:
                raise RuntimeError(
                    "The specified path does not contain any history file."
                )
            elif len(output_files) > 1:
                warn(
                    "The specified path contains multiple history files. "
                    "The most recent one will be used."
                )
            output_file = output_files[-1]
            params_file = os.path.join(path, "exploration_parameters.json")
        elif path.endswith(".npy"):
            output_file = path
            params_file = os.path.join(
                pathlib.Path(path).parent, "exploration_parameters.json"
            )
        else:
            raise RuntimeError(
                "The path should either point to a folder or a `.npy` file."
            )

        # Load the file as a pandas DataFrame
        history = np.load(os.path.join(path, output_file))
        d = {label: history[label].flatten() for label in history.dtype.names}
        self._history = pd.DataFrame(d)

        # Only keep the simulations that finished properly
        if remove_unfinished_evaluations:
            self._history = self._history[self._history.sim_ended]

        # Make the time relative to the start of the simulation
        if relative_start_time:
            start_time = self._history["gen_started_time"].min()
            self._history["sim_started_time"] -= start_time
            self._history["sim_ended_time"] -= start_time
            self._history["gen_started_time"] -= start_time
            self._history["gen_ended_time"] -= start_time
            self._history["gen_informed_time"] -= start_time

        # Read varying parameters, objectives, etc.
        self._read_exploration_parameters(params_file)

        # Rearrange history dataframe.
        self._rearrange_dataframe_columns()

    def _read_exploration_parameters(self, params_file: str) -> None:
        """Read exploration parameters from json file."""
        self._varying_parameters = {}
        self._analyzed_parameters = {}
        self._objectives = {}

        with open(params_file) as f:
            d = json.load(f)
        for _, param in d.items():
            if param["type"] == "VaryingParameter":
                p = VaryingParameter.parse_raw(param["value"])
                self._varying_parameters[p.name] = p
            elif param["type"] == "Objective":
                p = Objective.parse_raw(param["value"])
                self._objectives[p.name] = p
            elif param["type"] == "Parameter":
                p = Parameter.parse_raw(param["value"])
                self._analyzed_parameters[p.name] = p

    def _rearrange_dataframe_columns(self) -> None:
        """Rearrange dataframe columns.

        This is needed to have a consistent and more convenient output
        when printing or viewing the dataframe because the order of the
        numpy history file is different from run to run.
        """
        ordered_columns = ["trial_index"]
        ordered_columns += self._varying_parameters.keys()
        ordered_columns += self._objectives.keys()
        ordered_columns += self._analyzed_parameters.keys()
        ordered_columns += [
            "sim_id",
            "sim_worker",
            "sim_started_time",
            "sim_ended_time",
            "sim_started",
            "sim_ended",
            "gen_worker",
            "gen_started_time",
            "gen_ended_time",
            "gen_informed_time",
            "gen_informed",
            "cancel_requested",
            "kill_sent",
            "given_back",
            "num_procs",
            "num_gpus",
        ]
        ordered_columns += [
            c for c in self._history if c not in ordered_columns
        ]
        self._history = self._history[ordered_columns]

    @property
    def history(self) -> pd.DataFrame:
        """Return a pandas DataFrame with the exploration history."""
        return self._history

    @property
    def varying_parameters(self) -> List[VaryingParameter]:
        """Get the varying parameters of the exploration."""
        return list(self._varying_parameters.values())

    @property
    def analyzed_parameters(self) -> List[Parameter]:
        """Get the analyzed parameters of the exploration."""
        return list(self._analyzed_parameters.values())

    @property
    def objectives(self) -> List[Objective]:
        """Get the objectives of the exploration."""
        return list(self._objectives.values())

    def plot_objective(
        self,
        objective: Optional[str] = None,
        fidelity_parameter: Optional[str] = None,
        show_trace: Optional[bool] = False,
    ) -> None:
        """Plot the values that where reached during the optimization.

        Parameters
        ----------
        objective : str, optional
            Name of the objective to plot. If `None`, the first objective of
            the exploration is shown.
        fidelity_parameter: str, optional
            Name of the fidelity parameter. If given, the different fidelity
            will be plotted in different colors.
        show_trace : bool, optional
            Whether to show the cumulative maximum or minimum of the objective.

        """
        if fidelity_parameter is not None:
            fidelity = self._history[fidelity_parameter]
        else:
            fidelity = None
        if objective is None:
            objective = list(self._objectives.keys())[0]
        _, ax = plt.subplots()
        ax.scatter(
            self._history.sim_ended_time, self._history[objective], c=fidelity
        )
        ax.set_ylabel(objective)
        ax.set_xlabel("Time (s)")

        if show_trace:
            t_trace, obj_trace = self.get_objective_trace(
                objective, fidelity_parameter
            )
            ax.step(t_trace, obj_trace, where='post')

    def get_objective_trace(
        self,
        objective: Optional[str] = None,
        fidelity_parameter: Optional[str] = None,
        min_fidelity: Optional[float] = None,
        t_array: Optional[npt.NDArray] = None,
    ) -> Tuple[npt.NDArray, npt.NDArray]:
        """Get the cumulative maximum or minimum of the objective.

        Parameters
        ----------
        objective : str, optional
            Name of the objective to plot. If `None`, the first objective of
            the exploration is shown.
        fidelity_parameter: str, optional
            Name of the fidelity parameter. If `fidelity_parameter`
            and `min_fidelity` are set, only the runs with fidelity
            above `min_fidelity` are considered.
        fidelity_min: float, optional
            Minimum fidelity above which points are considered
        t_array: 1D numpy array, optional
            Array with time values. If provided, the trace is interpolated
            to the times in the array.

        Returns
        -------
        time : 1D numpy array
        objective_trace : 1D numpy array
        """
        if objective is None:
            objective = self.objectives[0]
        elif isinstance(objective, str):
            objective = self._objectives[objective]
        if fidelity_parameter is not None:
            assert min_fidelity is not None
            df = self._history[
                self._history[fidelity_parameter] >= min_fidelity
            ]
        else:
            df = self._history.copy()

        df = df.sort_values("sim_ended_time")
        t = df.sim_ended_time.values
        if objective.minimize:
            obj_trace = df[objective.name].cummin().values
        else:
            obj_trace = df[objective.name].cummax().values

        if t_array is not None:
            # Interpolate the trace curve on t_array
            N_interp = len(t_array)
            N_ref = len(t)
            obj_trace_array = np.zeros_like(t_array)
            i_ref = 0
            for i_interp in range(N_interp):
                while i_ref < N_ref - 1 and t[i_ref + 1] < t_array[i_interp]:
                    i_ref += 1
                obj_trace_array[i_interp] = obj_trace[i_ref]
        else:
            t_array = t
            obj_trace_array = obj_trace

        return t_array, obj_trace_array

    def plot_worker_timeline(
        self, fidelity_parameter: Optional[str] = None
    ) -> None:
        """Plot the timeline of worker utilization.

        Parameters
        ----------
        fidelity_parameter: string or None
            Name of the fidelity parameter. If given, the different fidelity
            will be plotted in different colors.
        """
        df = self._history
        if fidelity_parameter is not None:
            min_fidelity = df[fidelity_parameter].min()
            max_fidelity = df[fidelity_parameter].max()

        _, ax = plt.subplots()
        for i in range(len(df)):
            start = df["sim_started_time"].iloc[i]
            duration = df["sim_ended_time"].iloc[i] - start
            if fidelity_parameter is not None:
                fidelity = df[fidelity_parameter].iloc[i]
                color = plt.cm.viridis(
                    (fidelity - min_fidelity) / (max_fidelity - min_fidelity)
                )
            else:
                color = "tab:blue"
            ax.barh(
                [str(df["sim_worker"].iloc[i])],
                [duration],
                left=[start],
                color=color,
                edgecolor="k",
                linewidth=1,
            )

        ax.set_ylabel("Worker")
        ax.set_xlabel("Time (s)")
