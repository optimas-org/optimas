"""Contains the definition of the ExplorationDiagnostics class."""

import os
import shutil
from warnings import warn
import pathlib
import json
from typing import Optional, List, Dict, Tuple, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec, SubplotSpec

from optimas.core import VaryingParameter, Objective, Parameter
from optimas.generators.base import Generator
from optimas.evaluators.base import Evaluator
from optimas.explorations import Exploration
from optimas.utils.other import get_df_with_selection


class ExplorationDiagnostics:
    """Utilities for analyzing the output of an exploration.

    Parameters
    ----------
    source : str
        Path to the exploration directory or to an
        individual `.npy` history file, or an ``Exploration`` instance.
    """

    def __init__(self, source: Union[str, Exploration]) -> None:
        if isinstance(source, str):
            path = source
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
                exploration_dir_path = path
                output_file = os.path.join(path, output_files[-1])
                params_file = os.path.join(path, "exploration_parameters.json")
            elif path.endswith(".npy"):
                exploration_dir_path = pathlib.Path(path).parent
                output_file = path
                params_file = os.path.join(
                    exploration_dir_path, "exploration_parameters.json"
                )
            else:
                raise RuntimeError(
                    "The path should either point to a folder or a `.npy` file."
                )
            exploration = self._create_exploration(
                exploration_dir_path, params_file, output_file
            )
        elif isinstance(source, Exploration):
            exploration = source
        else:
            ValueError(
                "The source of the exploration diagnostics should be a `path` "
                f"or an `Exploration`, not of type {type(source)}."
            )
        self._exploration = exploration
        self._create_sim_dir_paths()

    def _create_exploration(
        self, exploration_dir_path: str, params_file: str, history_path: str
    ) -> Exploration:
        """Create exploration from saved files."""
        # Create exploration parameters.
        varying_parameters = []
        analyzed_parameters = []
        objectives = []
        with open(params_file) as f:
            d = json.load(f)
        for _, param in d.items():
            if param["type"] == "VaryingParameter":
                p = VaryingParameter.model_validate_json(param["value"])
                varying_parameters.append(p)
            elif param["type"] == "Objective":
                p = Objective.model_validate_json(param["value"])
                objectives.append(p)
            elif param["type"] == "Parameter":
                p = Parameter.model_validate_json(param["value"])
                analyzed_parameters.append(p)

        # Create exploration using dummy generator and evaluator.
        return Exploration(
            generator=Generator(
                varying_parameters=varying_parameters,
                objectives=objectives,
                analyzed_parameters=analyzed_parameters,
            ),
            evaluator=Evaluator(sim_function=None),
            history=history_path,
            exploration_dir_path=exploration_dir_path,
        )

    def _create_sim_dir_paths(self):
        """Create a dict with the path to the sim dir of each evaluation."""
        self._sim_dir_paths = {}
        ensemble_dir_path = os.path.join(
            self._exploration.exploration_dir_path, "evaluations"
        )
        if os.path.isdir(ensemble_dir_path):
            sim_dirs = os.listdir(ensemble_dir_path)
            for sim_dir in sim_dirs:
                if sim_dir.startswith("sim"):
                    try:
                        trial_index = int(sim_dir[3:])
                        self._sim_dir_paths[trial_index] = os.path.join(
                            ensemble_dir_path, sim_dir
                        )
                    except ValueError:
                        # Handle case in which conversion to an integer fails.
                        # This can happen if there is a folder that starts with
                        # "sim" but does not continue with a number. This is a
                        # folder that might have been created by the user.
                        pass

    @property
    def exploration_dir_path(self) -> str:
        """Get the exploration dir path."""
        return self._exploration.exploration_dir_path

    @property
    def history(self) -> pd.DataFrame:
        """Return a pandas DataFrame with the exploration history."""
        return self._exploration.history

    @property
    def varying_parameters(self) -> List[VaryingParameter]:
        """Get the varying parameters of the exploration."""
        return self._exploration.generator.varying_parameters

    @property
    def analyzed_parameters(self) -> List[Parameter]:
        """Get the analyzed parameters of the exploration."""
        return self._exploration.generator.analyzed_parameters

    @property
    def objectives(self) -> List[Objective]:
        """Get the objectives of the exploration."""
        return self._exploration.generator.objectives

    def _get_varying_parameter(self, name: str) -> VaryingParameter:
        """Get varying parameter by name."""
        for par in self.varying_parameters:
            if name == par.name:
                return par
        raise ValueError(
            f"Varying parameter {name} not found. "
            "Available varying parameters are: "
            f"{[par.name for par in self.varying_parameters]}."
        )

    def _get_analyzed_parameter(self, name: str) -> Parameter:
        """Get analyzed parameter by name."""
        for par in self.analyzed_parameters:
            if name == par.name:
                return par
        raise ValueError(
            f"Analyzed parameter {name} not found. "
            "Available analyzed parameters are: "
            f"{[par.name for par in self.analyzed_parameters]}."
        )

    def _get_objective(self, name: str) -> Objective:
        """Get objective by name."""
        for par in self.objectives:
            if name == par.name:
                return par
        raise ValueError(
            f"Objective {name} not found. "
            "Available objectives are: "
            f"{[par.name for par in self.objectives]}."
        )

    def plot_objective(
        self,
        objective: Optional[Union[str, Objective]] = None,
        fidelity_parameter: Optional[str] = None,
        show_trace: Optional[bool] = False,
        use_time_axis: Optional[bool] = False,
        relative_start_time: Optional[bool] = True,
        subplot_spec: Optional[SubplotSpec] = None,
        **figure_kw,
    ) -> None:
        """Plot the values that where reached during the optimization.

        Parameters
        ----------
        objective : str, optional
            Objective, or name of the objective to plot. If `None`, the first
            objective of the exploration is shown.
        fidelity_parameter: str, optional
            Name of the fidelity parameter. If given, the different fidelity
            will be plotted in different colors.
        show_trace : bool, optional
            Whether to show the cumulative maximum or minimum of the objective.
        use_time_axis : bool, optional
            Whether the x axis should be the time at which the evaluations
            were completed, instead of the evaluation index.
        relative_start_time : bool, optional
            Whether the time axis should be relative to the start time
            of the exploration. By default, True.
        subplot_spec: SubplotSpec, optional
            The location of the plot in the GridSpec of an existing figure.
            If not given, a new figure will be created.
        **figure_kw
            Additional keyword arguments to pass to `pyplot.figure`. Only used
            if no ``subplot_spec`` is given.
        """
        if fidelity_parameter is not None:
            fidelity = self.history[fidelity_parameter]
        else:
            fidelity = None
        if objective is None:
            objective = self.objectives[0]
        elif isinstance(objective, str):
            objective = self._get_objective(objective)
        history = self.history
        history = history[history.sim_ended]
        if use_time_axis:
            x = history.sim_ended_time
            xlabel = "Time (s)"
            if relative_start_time:
                x = x - history["gen_started_time"].min()
        else:
            x = history.trial_index
            xlabel = "Number of evaluations"

        if subplot_spec is None:
            fig = plt.figure(**figure_kw)
            gs = GridSpec(1, 1)
        else:
            fig = plt.gcf()
            gs = GridSpecFromSubplotSpec(1, 1, subplot_spec)
        ax = fig.add_subplot(gs[0])
        ax.scatter(x, history[objective.name], c=fidelity)
        ax.set_ylabel(objective.name)
        ax.set_xlabel(xlabel)

        if show_trace:
            t_trace, obj_trace = self.get_objective_trace(
                objective,
                fidelity_parameter,
                use_time_axis=use_time_axis,
                relative_start_time=relative_start_time,
            )
            ax.step(t_trace, obj_trace, where="post")

    def plot_pareto_front(
        self,
        objectives: Optional[List[Union[str, Objective]]] = None,
        show_best_evaluation_indices: Optional[bool] = False,
        show_legend: Optional[bool] = False,
        subplot_spec: Optional[SubplotSpec] = None,
        **figure_kw,
    ) -> None:
        """Plot Pareto front of two optimization objectives.

        Parameters
        ----------
        objectives : list of str or Objective, optional
            A list with two objectives to plot. Only needed when the
            optimization had more than two objectives. By default ``None``.
        show_best_evaluation_indices : bool, optional
            Whether to show the indices of the best evaluations. By default
            ``False``.
        show_legend : bool, optional
            Whether to show the legend.
        subplot_spec: SubplotSpec, optional
            The location of the plot in the GridSpec of an existing figure.
            If not given, a new figure will be created.
        **figure_kw
            Additional keyword arguments to pass to `pyplot.figure`. Only used
            if no ``subplot_spec`` is given.
        """
        objectives = self._check_pareto_objectives(objectives)
        pareto_evals = self.get_pareto_front_evaluations(objectives)
        x_data = self.history[objectives[0].name].to_numpy()
        y_data = self.history[objectives[1].name].to_numpy()
        x_pareto = pareto_evals[objectives[0].name].to_numpy()
        y_pareto = pareto_evals[objectives[1].name].to_numpy()

        # Create axes
        if subplot_spec is None:
            fig = plt.figure(**figure_kw)
            gs = GridSpec(1, 1)
        else:
            fig = plt.gcf()
            gs = GridSpecFromSubplotSpec(1, 1, subplot_spec)
        axes = fig.add_subplot(gs[0])
        # Plot all evaluations
        axes.scatter(
            x_data, y_data, s=5, lw=0.0, alpha=0.5, label="All evaluations"
        )
        axes.set(xlabel=objectives[0].name, ylabel=objectives[1].name)

        # Plot best evaluations
        axes.scatter(
            x_pareto,
            y_pareto,
            s=15,
            ec="k",
            fc="tab:blue",
            lw=0.5,
            zorder=2,
            label="Best evaluations",
        )

        # Plot pareto front
        axes.step(
            x_pareto,
            y_pareto,
            c="k",
            lw=1,
            where="pre" if objectives[1].minimize else "post",
            zorder=1,
            label="Pareto front",
        )
        if show_legend:
            axes.legend(frameon=False, fontsize="x-small")

        # Add evaluation indices to plot.
        if show_best_evaluation_indices:
            sim_id_pareto = self.history["sim_id"].to_numpy()[
                pareto_evals.index
            ]
            for i, id in enumerate(sim_id_pareto):
                axes.annotate(
                    str(id),
                    (x_pareto[i], y_pareto[i]),
                    (2, -2),
                    fontsize="xx-small",
                    va="top",
                    textcoords="offset points",
                )

    def get_best_evaluation(
        self, objective: Optional[Union[str, Objective]] = None
    ) -> pd.DataFrame:
        """Get the evaluation with the best objective value.

        Parameters
        ----------
        objective : str or Objective, optional
            Objective to consider for determining the best evaluation. Only.
            needed if there is more than one objective. By default ``None``.
        """
        if objective is None:
            objective = self.objectives[0]
        elif isinstance(objective, str):
            objective = self._get_objective(objective)
        history = self.history[self.history.sim_ended]
        if objective.minimize:
            i_best = np.argmin(history[objective.name])
        else:
            i_best = np.argmax(history[objective.name])
        return history.iloc[[i_best]]

    def get_pareto_front_evaluations(
        self,
        objectives: Optional[List[Union[str, Objective]]] = None,
    ) -> pd.DataFrame:
        """Get data of evaluations in the Pareto front.

        This function is currently limited to the Pareto front of two
        objectives.

        Parameters
        ----------
        objectives : list of str or Objective, optional
            A list with two objectives to plot. Only needed when the
            optimization had more than two objectives. By default ``None``.
        """
        objectives = self._check_pareto_objectives(objectives)
        x_data = self.history[objectives[0].name].to_numpy()
        y_data = self.history[objectives[1].name].to_numpy()
        x_minimize = objectives[0].minimize
        y_minimize = objectives[1].minimize
        i_sort = np.argsort(x_data)
        if not x_minimize:
            i_sort = i_sort[::-1]  # Sort in descending order
        if y_minimize:
            y_cum = np.minimum.accumulate(y_data[i_sort])
        else:
            y_cum = np.maximum.accumulate(y_data[i_sort])
        _, i_pareto = np.unique(y_cum, return_index=True)
        return self.history.iloc[i_sort[i_pareto]]

    def get_objective_trace(
        self,
        objective: Optional[Union[str, Objective]] = None,
        fidelity_parameter: Optional[str] = None,
        min_fidelity: Optional[float] = None,
        use_time_axis: Optional[bool] = False,
        t_array: Optional[npt.NDArray] = None,
        relative_start_time: Optional[bool] = True,
    ) -> Tuple[npt.NDArray, npt.NDArray]:
        """Get the cumulative maximum or minimum of the objective.

        Parameters
        ----------
        objective : str, optional
            Objective, or name of the objective to plot. If `None`, the first
            objective of the exploration is shown.
        fidelity_parameter: str, optional
            Name of the fidelity parameter. If `fidelity_parameter`
            and `min_fidelity` are set, only the runs with fidelity
            above `min_fidelity` are considered.
        fidelity_min: float, optional
            Minimum fidelity above which points are considered
        use_time_axis : bool, optional
            Whether the x axis should be the time at which the evaluations
            were completed, instead of the evaluation index.
        t_array: 1D numpy array, optional
            Array with time values. If provided, the trace is interpolated
            to the times in the array. Requires ``use_time_axis=True``.
        relative_start_time : bool, optional
            Whether the time axis should be relative to the start time
            of the exploration. By default, True.

        Returns
        -------
        time : 1D numpy array
        objective_trace : 1D numpy array
        """
        if objective is None:
            objective = self.objectives[0]
        elif isinstance(objective, str):
            objective = self._get_objective(objective)
        if fidelity_parameter is not None:
            assert min_fidelity is not None
            df = self.history[self.history[fidelity_parameter] >= min_fidelity]
        else:
            df = self.history.copy()
        df = df[df.sim_ended]

        if use_time_axis:
            df = df.sort_values("sim_ended_time")
            x = df.sim_ended_time
            if relative_start_time:
                x = x - df["gen_started_time"].min()
            x = x.values
        else:
            x = df.trial_index.values

        if objective.minimize:
            obj_trace = df[objective.name].cummin().values
        else:
            obj_trace = df[objective.name].cummax().values

        if t_array is not None:
            # Interpolate the trace curve on t_array
            N_interp = len(t_array)
            N_ref = len(x)
            obj_trace_array = np.zeros_like(t_array)
            i_ref = 0
            for i_interp in range(N_interp):
                while i_ref < N_ref - 1 and x[i_ref + 1] < t_array[i_interp]:
                    i_ref += 1
                obj_trace_array[i_interp] = obj_trace[i_ref]
            x = t_array
            obj_trace = obj_trace_array

        return x, obj_trace

    def get_evaluation_dir_path(self, trial_index: int) -> str:
        """Get the path to the directory of the given evaluation.

        Parameters
        ----------
        trial_index : int
            Index of an evaluated trial.
        """
        try:
            return self._sim_dir_paths[trial_index]
        except KeyError:
            raise ValueError(
                f"Could not find evaluation directory of trial {trial_index}."
                "This directory is only created when using a "
                "`TemplateEvaluator`."
            )

    def get_best_evaluation_dir_path(
        self, objective: Optional[Union[str, Objective]] = None
    ) -> str:
        """Get the path to the directory of the best evaluation.

        Parameters
        ----------
        objective : str or Objective, optional
            Objective to consider for determining the best evaluation. Only.
            needed if there is more than one objective. By default ``None``.
        """
        best_ev = self.get_best_evaluation(objective)
        return self.get_evaluation_dir_path(best_ev["trial_index"].item())

    def delete_evaluation_dir(self, trial_index: int) -> None:
        """Delete the directory with the output of the given evaluation.

        Parameters
        ----------
        trial_index : int
            Index of an evaluated trial.
        """
        ev_dir_path = self.get_evaluation_dir_path(trial_index)
        shutil.rmtree(ev_dir_path)
        del self._sim_dir_paths[trial_index]

    def plot_worker_timeline(
        self,
        fidelity_parameter: Optional[str] = None,
        relative_start_time: Optional[bool] = True,
        subplot_spec: Optional[SubplotSpec] = None,
        **figure_kw,
    ) -> None:
        """Plot the timeline of worker utilization.

        Parameters
        ----------
        fidelity_parameter: string or None
            Name of the fidelity parameter. If given, the different fidelity
            will be plotted in different colors.
        relative_start_time : bool, optional
            Whether the time axis should be relative to the start time
            of the exploration. By default, True.
        subplot_spec: SubplotSpec, optional
            The location of the plot in the GridSpec of an existing figure.
            If not given, a new figure will be created.
        **figure_kw
            Additional keyword arguments to pass to `pyplot.figure`. Only used
            if no ``subplot_spec`` is given.
        """
        df = self.history
        df = df[df.sim_id >= 0]

        if fidelity_parameter is not None:
            min_fidelity = df[fidelity_parameter].min()
            max_fidelity = df[fidelity_parameter].max()

        sim_started_time = df["sim_started_time"]
        sim_ended_time = df["sim_ended_time"]
        if relative_start_time:
            sim_started_time = sim_started_time - df["gen_started_time"].min()
            sim_ended_time = sim_ended_time - df["gen_started_time"].min()

        if subplot_spec is None:
            fig = plt.figure(**figure_kw)
            gs = GridSpec(1, 1)
        else:
            fig = plt.gcf()
            gs = GridSpecFromSubplotSpec(1, 1, subplot_spec)
        ax = fig.add_subplot(gs[0])

        for i in range(len(df)):
            start = sim_started_time.iloc[i]
            duration = sim_ended_time.iloc[i] - start
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

    def plot_history(
        self,
        parnames: Optional[list] = None,
        xname: Optional[str] = None,
        select: Optional[Dict] = None,
        sort: Optional[Dict] = None,
        top: Optional[Dict] = None,
        show_top_evaluation_indices: Optional[bool] = False,
        show_legend: Optional[bool] = False,
        subplot_spec: Optional[SubplotSpec] = None,
        **figure_kw,
    ) -> None:
        """Print selected parameters versus evaluation index.

        Parameters
        ----------
        parnames: list of strings, optional
            List with the names of the parameters to show.
        xname: string, optional
            Name of the parameter to plot in the x axis.
            By default is the index of the history DataFrame.
        select: dict, optional
            Contains a set of rules to filter the dataframe, e.g.
            'f' : [None, -10.] (get data with f < -10)
        sort: dict, optional
            A dict containing as keys the names of the parameres to sort by
            and, as values, a Bool indicating if ordering ascendingly (True)
            or descendingly (False)
            e.g. {'f': False} sort simulations according to f descendingly.
        top: int, optional
            Highight the 'top' evaluations of every objective.
        show_top_evaluation_indices : bool, optional
            Whether to show the indices of the top evaluations.
        show_legend : bool, optional
            Whether to show the legend.
        subplot_spec: SubplotSpec, optional
            The location of the plot in the GridSpec of an existing figure.
            If not given, a new figure will be created.
        **figure_kw
            Additional keyword arguments to pass to `pyplot.figure`. Only used
            if no ``subplot_spec`` is given.
        """
        # Copy the history DataFrame
        df = self.history.copy()

        # Get lists of variable names
        objective_names = [obj.name for obj in self.objectives]
        varpar_names = [var.name for var in self.varying_parameters]

        # Order list of simulations and re-index
        if sort is not None:
            df = df.sort_values(
                by=list(sort.keys()), ascending=tuple(sort.values())
            ).reset_index(drop=True)

        # Define the quantity to plot in the x axis
        if xname is not None:
            xvalues = df[xname]
        else:
            xvalues = df.index

        # Apply selection to the history DataFrame
        if select is not None:
            df_select = get_df_with_selection(df, select)
        else:
            df_select = None

        # Select top cases in each objective in separate DataFrames
        # stored in a dictionary with the objective name as key
        if top is not None:
            df_top = {}
            for obj_name in objective_names:
                obj = self._get_objective(obj_name)
                ascending = obj.minimize
                index_list = list(
                    df.sort_values(by=obj_name, ascending=ascending).index
                )
                df_top[obj_name] = df.loc[index_list[:top]]
        else:
            df_top = None

        # Default list of parameters to show
        if parnames is None:
            parnames = objective_names.copy()
            parnames.extend(varpar_names)

        # Make figure
        nplots = len(parnames)
        if subplot_spec is None:
            fig = plt.figure(**figure_kw)
            gs = GridSpec(nplots, 2, width_ratios=[0.8, 0.2], wspace=0.05)
        else:
            fig = plt.gcf()
            gs = GridSpecFromSubplotSpec(
                nplots, 2, subplot_spec, width_ratios=[0.8, 0.2], wspace=0.05
            )

        # Actual plotting
        ax_histy_list = []
        histy_list = []
        for i in range(nplots):
            # Draw scatter plot
            ax_scatter = fig.add_subplot(gs[i, 0])
            ax_scatter.grid(color="lightgray", linestyle="dotted")
            yvalues = df[parnames[i]]
            ax_scatter.plot(xvalues, yvalues, "o")

            # Draw selection
            if df_select is not None:
                xvalues_cut = df_select.index
                if xname is not None:
                    xvalues_cut = df_select[xname]
                yvalues_cut = df_select[parnames[i]]
                ax_scatter.plot(xvalues_cut, yvalues_cut, "o", label="select")

            # Draw top evaluations
            if df_top is not None:
                for key, df_t in df_top.items():
                    if xname is not None:
                        xvalues_top = df_t[xname]
                    else:
                        xvalues_top = df_t.index
                    yvalues_top = df_t[parnames[i]]
                    label = "top %i" % top
                    ax_scatter.plot(xvalues_top, yvalues_top, "o", label=label)

                    # Add evaluation indices to plot
                    if show_top_evaluation_indices:
                        sim_id_top = df_t["sim_id"]
                        obj = self._get_objective(key)
                        if obj.minimize:
                            va = "bottom"
                            xytext = (2, 2)
                        else:
                            va = "top"
                            xytext = (2, -2)
                        for x, y, id in zip(
                            xvalues_top, yvalues_top, sim_id_top
                        ):
                            ax_scatter.annotate(
                                str(id),
                                (x, y),
                                xytext,
                                fontsize="xx-small",
                                va=va,
                                textcoords="offset points",
                            )

            # Draw the trace only for `objective` parameters
            if (
                (parnames[i] in objective_names)
                and (not sort)
                and (xname is None)
            ):
                obj = self._get_objective(parnames[i])
                if obj.minimize:
                    cum = df[parnames[i]].cummin().values
                else:
                    cum = df[parnames[i]].cummax().values
                ax_scatter.step(
                    xvalues, cum, zorder=-1, where="post", c="black"
                )

            # Draw projected histogram
            ax_histy = fig.add_subplot(gs[i, 1])
            ax_histy.grid(color="lightgray", linestyle="dotted")
            ymin, ymax = ax_scatter.get_ylim()
            nbins = 25
            binwidth = (ymax - ymin) / nbins
            bins = np.arange(ymin, ymax + binwidth, binwidth)
            histy, *_ = ax_histy.hist(
                yvalues,
                bins=bins,
                weights=100.0 * np.ones(len(yvalues)) / len(yvalues),
                orientation="horizontal",
            )

            # Draw selection
            if df_select is not None:
                yvalues_cut = df_select[parnames[i]]
                ax_histy.hist(
                    yvalues_cut,
                    bins=bins,
                    weights=100.0 * np.ones(len(yvalues_cut)) / len(yvalues),
                    orientation="horizontal",
                    label="selection",
                )

            # Draw top evaluations
            if df_top is not None:
                for key, df_t in df_top.items():
                    yvalues_top = df_t[parnames[i]]
                    label = "top %i" % top
                    ax_histy.hist(
                        yvalues_top,
                        bins=bins,
                        weights=100.0
                        * np.ones(len(yvalues_top))
                        / len(yvalues),
                        orientation="horizontal",
                        label=label,
                    )

            ax_histy.set_ylim(ax_scatter.get_ylim())

            # Tuning axes and labels
            ax_scatter.set_title(
                parnames[i].replace("_", " "),
                fontdict={"fontsize": "x-small"},
                loc="right",
                pad=2,
            )

            if i != nplots - 1:
                ax_scatter.tick_params(labelbottom=False)
                ax_histy.tick_params(labelbottom=False, labelleft=False)
            else:
                ax_scatter.set_xlabel("Evaluation number")
                if xname is not None:
                    ax_scatter.set_xlabel(xname.replace("_", " "))
                ax_histy.set_xlabel("%")
                ax_histy.tick_params(labelbottom=True, labelleft=False)
                if show_legend:
                    ax_histy.legend(fontsize="xx-small")

            # Make loist of histograms and axes for further manipulation
            # outside the loop
            ax_histy_list.append(ax_histy)
            histy_list.append(histy)

        # Set the range of the histograms axes
        histmax = 1.1 * max([h.max() for h in histy_list])
        for i, ax_h in enumerate(ax_histy_list):
            ax_h.set_xlim(-1, histmax)

    def _check_pareto_objectives(
        self,
        objectives: Optional[List[Union[str, Objective]]] = None,
    ) -> List[Objective]:
        """Check the objectives given to the Pareto methods."""
        if len(self.objectives) < 2:
            raise ValueError(
                "Cannot get Pareto front because only a single objective "
                "is available."
            )
        if objectives is None:
            if len(self.objectives) == 2:
                objectives = self.objectives
            else:
                raise ValueError(
                    f"There are {len(self.objectives)} available. "
                    "Please specify 2 objectives from which to get the "
                    "Pareto front."
                )
        else:
            if len(objectives) != 2:
                raise ValueError(
                    f"Two objectives are required ({len(objectives)} given)."
                )
            for i, objective in enumerate(objectives):
                if isinstance(objective, str):
                    objectives[i] = self._get_objective(objective)
        return objectives

    def print_evaluation(self, trial_index: int) -> None:
        """Print the parameters of the given evaluation.

        Parameters
        ----------
        trial_index : int
            Index of an evaluated trial.
        """
        h = self.history.loc[trial_index]
        print("Evaluation %i: " % (trial_index))
        print("%20s = %s" % ("sim_id", h["sim_id"]))

        try:
            sim_path = self.get_evaluation_dir_path(trial_index)
            print(
                "%20s = %s"
                % ("dir_path", self.get_evaluation_dir_path(trial_index))
            )
        except ValueError:
            sim_path = None
            print("%20s = None" % ("dir_path"))

        print("objectives:")
        objective_names = [obj.name for obj in self.objectives]
        for name in objective_names:
            print("%20s = %10.5f" % (name, h[name]))

        print("varying parameters:")
        varpar_names = [var.name for var in self.varying_parameters]
        for name in varpar_names:
            print("%20s = %10.5f" % (name, h[name]))

        if len(self.analyzed_parameters) > 0:
            anapar_names = [var.name for var in self.analyzed_parameters]
            print("analyzed parameters:")
            for name in anapar_names:
                print("%20s = %10.5f" % (name, h[name]))

        print()

    def print_best_evaluations(
        self,
        top: Optional[int] = 3,
        objective: Optional[Union[str, Objective]] = None,
    ) -> None:
        """Print top evaluations according to the given objective.

        Parameters
        ----------
        top : int, optional
            Number of top evaluations to consider (3 by default).
            e.g. top = 3 means that the three best evaluations will be shown.
        objective : str, optional
            Objective, or name of the objective to plot. If `None`, the first
            objective of the exploration is shown.
        """
        if objective is None:
            objective = self.objectives[0]
        if isinstance(objective, str):
            objective = self._get_objective(objective)
        top_indices = list(
            self.history.sort_values(
                by=objective.name, ascending=objective.minimize
            ).index
        )[:top]
        objective_names = [obj.name for obj in self.objectives]
        varpar_names = [var.name for var in self.varying_parameters]
        anapar_names = [var.name for var in self.analyzed_parameters]
        print(
            "Top %i evaluations in metric %s (minimize = %s): "
            % (top, objective.name, objective.minimize),
            top_indices,
        )
        print()
        print(
            self.history.loc[top_indices][
                objective_names + varpar_names + anapar_names
            ]
        )
