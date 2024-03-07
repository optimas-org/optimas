"""Contains the definition of the ExplorationDiagnostics class."""

from typing import Optional, Union, List, Tuple, Dict, Any, Literal

import numpy as np
from numpy.typing import NDArray
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec, SubplotSpec
from matplotlib.axes import Axes

# Ax utilities for model building
try:
    from ax.service.ax_client import AxClient
    from ax.modelbridge.generation_strategy import (
        GenerationStep,
        GenerationStrategy,
    )
    from ax.modelbridge.registry import Models
    from ax.modelbridge.torch import TorchModelBridge
    from ax.core.observation import ObservationFeatures
    from ax.service.utils.instantiation import ObjectiveProperties

    ax_installed = True
except ImportError:
    ax_installed = False

from optimas.core import VaryingParameter, Objective
from optimas.utils.other import convert_to_dataframe


class AxModelManager:
    """Class for building and exploring GP models using an ``AxClient``.

    Parameters
    ----------
    source : AxClient, str or DataFrame
        Source data for the model. It can be either an existing ``AxClient``
        with a GP model, a string with the path to a ``json`` file with a
        serialized ``AxClient``, or a pandas ``DataFrame``.
        When using a ``DataFrame``, a list of objectives and varying parameters
        should also be provided.
    objectives : list of `Objective`, optional
        Only needed if ``source`` is a pandas ``DataFrame``. List of
        objectives for which a GP model should be built. The names and data of
        these objectives must be contained in the source ``DataFrame``.
    varying_parameters : list of `VaryingParameter`, optional
        Only needed if ``source`` is a pandas ``DataFrame``. List of
        parameters that were varied to scan the value of the objectives.
        The names and data of these parameters must be contained in the
        source ``DataFrame``.
    """

    def __init__(
        self,
        source: Union[AxClient, str, pd.DataFrame],
        varying_parameters: Optional[List[VaryingParameter]] = None,
        objectives: Optional[List[Objective]] = None,
    ) -> None:
        if not ax_installed:
            raise ImportError(
                "`AxModelManager` requires Ax to be installed. "
                "You can do so by running `pip install ax-platform`."
            )
        if isinstance(source, AxClient):
            self.ax_client = source
        elif isinstance(source, str):
            self.ax_client = AxClient.load_from_json_file(filepath=source)
        elif isinstance(source, pd.DataFrame):
            self.ax_client = self._build_ax_client_from_dataframe(
                source, varying_parameters, objectives
            )
        else:
            raise ValueError(
                f"Wrong source type: {type(source)}. "
                "The source must be an `AxClient`, a path to an AxClient json "
                "file, or a pandas `DataFrame`."
            )
        self.ax_client.fit_model()

    @property
    def _model(self) -> TorchModelBridge:
        """Get the model from the AxClient instance."""
        return self.ax_client.generation_strategy.model

    def _build_ax_client_from_dataframe(
        self,
        df: pd.DataFrame,
        varying_parameters: List[VaryingParameter],
        objectives: List[Objective],
    ) -> AxClient:
        """Initialize the AxClient and the model using the given data.

        Parameters
        ----------
        df : DataFrame
            The source pandas ``DataFrame``.
        objectives : list of `Objective`.
            List of objectives for which a GP model should be built.
        varying_parameters : list of `VaryingParameter`.
            List of parameters that were varied to scan the value of the
            objectives.
        """
        # Define parameters for AxClient
        axparameters = []
        for par in varying_parameters:
            # Determine parameter type.
            value_dtype = np.dtype(par.dtype)
            if value_dtype.kind == "f":
                value_type = "float"
            elif value_dtype.kind == "i":
                value_type = "int"
            else:
                raise ValueError(
                    "Ax range parameter can only be of type 'float'ot 'int', "
                    "not {var.dtype}."
                )
            # Create parameter dict and append to list.
            axparameters.append(
                {
                    "name": par.name,
                    "type": "range",
                    "bounds": [par.lower_bound, par.upper_bound],
                    "is_fidelity": par.is_fidelity,
                    "target_value": par.fidelity_target_value,
                    "value_type": value_type,
                }
            )

        # Define objectives for AxClient
        axobjectives = {
            obj.name: ObjectiveProperties(minimize=obj.minimize)
            for obj in objectives
        }

        # Create Ax client.
        # We need to explicitly define a generation strategy because otherwise
        # a random sampling step will be set up by Ax, and this step does not
        # allow calling `model.predict`. Using MOO for multiobjective is
        # needed because otherwise calls to `get_pareto_optimal_parameters`
        # would fail.
        model = Models.GPEI if len(objectives) == 1 else Models.MOO
        gs = GenerationStrategy([GenerationStep(model=model, num_trials=-1)])
        ax_client = AxClient(generation_strategy=gs, verbose_logging=False)
        ax_client.create_experiment(
            parameters=axparameters, objectives=axobjectives
        )

        # Add trials from DataFrame
        for _, row in df.iterrows():
            params = {vp.name: row[vp.name] for vp in varying_parameters}
            _, trial_id = ax_client.attach_trial(params)
            data = {obj.name: (row[obj.name], np.nan) for obj in objectives}
            ax_client.complete_trial(trial_id, raw_data=data)
        return ax_client

    def _get_best_point(self, metric_name: Optional[str] = None) -> Dict:
        """Get the best point with the best predicted model value.

        Parameters
        ----------
        metric_name: str, optional.
            Name of the metric to evaluate.
            If not specified, it will take the first first objective in
            ``self.ax_client``.

        Returns
        -------
        best_point : dict
            A dictionary with the parameters of the best point.
        """
        _, best_point = self.get_best_evaluation(
            metric_name=metric_name, use_model_predictions=True
        )
        return best_point

    def _get_mid_point(self) -> Dict:
        """Get the middle point of the space of parameters.

        Returns
        -------
        mid_point : dict
            A dictionary with the parameters of the mid point.
        """
        mid_point = {}
        for key, par in self.ax_client.experiment.parameters.items():
            mid_point[key] = 0.5 * (par.lower + par.upper)

        return mid_point

    def _get_arm_index(
        self,
        arm_name: str,
    ) -> int:
        """Get the index of the arm by its name.

        Parameters
        ----------
        arm_name : str
            Name of the arm. If not given, the best arm is selected.

        Returns
        -------
        index : int
            Trial index of the arm.
        """
        df = self.ax_client.get_trials_data_frame()
        index = df.loc[df["arm_name"] == arm_name, "trial_index"].iloc[0]
        return index

    def evaluate_model(
        self,
        sample: Union[pd.DataFrame, Dict, NDArray] = None,
        metric_name: Optional[str] = None,
        fixed_parameters: Optional[Dict] = None,
    ) -> Tuple[NDArray]:
        """Evaluate the model over the specified sample.

        Parameters
        ----------
        sample : DataFrame, dict of NDArray or NDArray
            containing the data sample where to evaluate the model.
            If numpy array, it must contain the values of all the model
            parameters.
            If DataFrame or dict, it can contain only those parameters to vary.
            The rest of parameters would be set to the model best point,
            unless they are further specified using ``fixed_parameters``.
        metric_name : str, optional.
            Name of the metric to evaluate.
            If not specified, it will take the first first objective in
            ``self.ax_client``.
        fixed_parameters : dict, optional.
            A dictionary with structure ``{param_name: param_val}`` with the
            values of the parameters to be fixed in the evaluation. If a given
            parameter also exists in the ``sample``, the values in the
            ``sample`` will be overwritten by the fixed value.

        Returns
        -------
        NDArray, NDArray
            Two numpy arrays containing the mean of the model
            and the standard error of the mean (sem), respectively.
        """
        if metric_name is None:
            metric_name = self.ax_client.objective_names[0]
        else:
            metric_names = list(self.ax_client.experiment.metrics.keys())
            if metric_name not in metric_names:
                raise ValueError(
                    f"Metric name {metric_name} does not match any of the "
                    f"metrics. Available metrics are: {metric_names}."
                )

        parnames = list(self.ax_client.experiment.parameters.keys())

        sample = convert_to_dataframe(sample)

        if fixed_parameters is not None:
            for key, val in fixed_parameters.items():
                sample[key] = val

        # check if labels of the dataframe match the parnames
        for name in parnames:
            if name not in sample.columns.values:
                raise ValueError(f"Data for {name} is missing in the sample.")
        # make list of `ObservationFeatures`
        obsf_list = []
        for i in range(sample.shape[0]):
            parameters = {}
            for name in parnames:
                parameters[name] = sample[name].iloc[i]
            obsf_list.append(ObservationFeatures(parameters=parameters))

        mu, cov = self._model.predict(obsf_list)
        m_array = np.asarray(mu[metric_name])
        sem_array = np.sqrt(cov[metric_name][metric_name])
        return m_array, sem_array

    def get_best_evaluation(
        self,
        metric_name: Optional[str] = None,
        use_model_predictions: Optional[bool] = True,
    ) -> Tuple[int, Dict]:
        """Get the best scoring point in the sample.

        Parameters
        ----------
        metric_name : str, optional.
            Name of the metric to evaluate.
            If not specified, it will take the first first objective in
            ``self.ax_client``.
        use_model_predictions : bool, optional.
            Whether to extract the best point using model predictions
            or directly observed values.

        Returns
        -------
        int, dict
            The index of the best evaluation and a dictionary with its
            parameters.
        """
        # metric name
        if metric_name is None:
            metric_name = self.ax_client.objective_names[0]

        # get optimum
        if len(self.ax_client.objective_names) > 1:
            minimize = None
            for obj in self.ax_client.objective.objectives:
                if metric_name == obj.metric_names[0]:
                    minimize = obj.minimize
                    break
            pp = self.ax_client.get_pareto_optimal_parameters(
                use_model_predictions=use_model_predictions
            )
            obj_vals, param_vals, trial_indices = [], [], []
            for index, (vals, (objs, covs)) in pp.items():
                trial_indices.append(index)
                param_vals.append(vals)
                obj_vals.append(objs[metric_name])
            i_best = np.argmin(obj_vals) if minimize else np.argmax(obj_vals)
            best_point = param_vals[i_best]
            index = trial_indices[i_best]
        else:
            if use_model_predictions is True:
                best_arm, _ = self._model.model_best_point()
                best_point = best_arm.parameters
                index = self._get_arm_index(best_arm.name)
            else:
                # AxClient.get_best_parameters seems to always return the best
                # point from the observed values, independently of the value
                # of `use_model_predictions`.
                index, best_point, _ = self.ax_client.get_best_trial(
                    use_model_predictions=use_model_predictions
                )

        return index, best_point

    def plot_contour(
        self,
        param_x: Optional[str] = None,
        param_y: Optional[str] = None,
        metric_name: Optional[str] = None,
        slice_values: Optional[Union[Dict, Literal["best", "mid"]]] = "mid",
        n_points: Optional[int] = 200,
        range_x: Optional[List[float]] = None,
        range_y: Optional[List[float]] = None,
        mode: Optional[Literal["mean", "sem", "both"]] = "mean",
        show_trials: Optional[bool] = True,
        show_contour: Optional[bool] = True,
        show_contour_labels: Optional[bool] = False,
        subplot_spec: Optional[SubplotSpec] = None,
        gridspec_kw: Optional[Dict[str, Any]] = None,
        pcolormesh_kw: Optional[Dict[str, Any]] = None,
        **figure_kw,
    ) -> Tuple[Figure, Union[Axes, List[Axes]]]:
        """Plot a 2D slice of the surrogate model.

        Parameters
        ----------
        param_x : str
            Name of the parameter to plot in x axis.
        param_y : str
            Name of the parameter to plot in y axis.
        metric_name : str, optional.
            Name of the metric to plot.
            If not specified, it will take the first objective in
            ``self.ax_client``.
        slice_values : dict or str, optional.
            The values along which to slice the model, if the model has more
            than two dimensions. Possible values are: ``"best"`` (slice along
            the best predicted point), ``"mid"`` (slice along the middle
            point of the varying parameters), or a dictionary with structure
            ``{param_name: param_val}`` that contains the slice values of each
            parameter. By default, ``"mid"``.
        n_points : int, optional
            Number of points in each axis. By default, ``200``.
        range_x, range_y : list of float, optional
            Range of each axis. It not given, the lower and upper boundary
            of each parameter will be used.
        mode : str, optional.
            Whether to plot the ``"mean"`` of the model, the standard error of
            the mean ``"sem"``, or ``"both"``. By default, ``"mean"``.
        show_trials : bool
            Whether to show the trials used to build the model. By default,
            ``True``.
        show_contour : bool
            Whether to show the contour lines. By default, ``True``.
        show_contour_labels : bool
            Whether to add labels to the contour lines. By default, ``False``.
        subplot_spec : SubplotSpec, optional
            A matplotlib ``SubplotSpec`` in which to draw the axis.
        gridspec_kw : dict, optional
            Dict with keywords passed to the ``GridSpec``.
        pcolormesh_kw : dict, optional
            Dict with keywords passed to ``ax.pcolormesh``.
        **figure_kw
            Additional keyword arguments to pass to ``pyplot.figure``.
            Only used if no ``subplot_spec`` is given.

        Returns
        -------
        Figure, Axes or list of Axes
            A matplotlib figure and either a single ``Axes`` or a list of
            ``Axes`` if ``mode="both"``.
        """
        # get experiment info
        experiment = self.ax_client.experiment
        parnames = list(experiment.parameters.keys())

        if len(parnames) < 2:
            raise RuntimeError(
                "Insufficient number of parameters in data for this plot "
                "(minimum 2)."
            )

        # select the input variables
        if param_x is None:
            param_x = parnames[0]
        if param_y is None:
            param_y = parnames[1]

        # metric name
        if metric_name is None:
            metric_name = self.ax_client.objective_names[0]

        # set the plotting range
        if range_x is None:
            range_x = [None, None]
        if range_y is None:
            range_y = [None, None]
        if range_x[0] is None:
            range_x[0] = experiment.parameters[param_x].lower
        if range_x[1] is None:
            range_x[1] = experiment.parameters[param_x].upper
        if range_y[0] is None:
            range_y[0] = experiment.parameters[param_y].lower
        if range_y[1] is None:
            range_y[1] = experiment.parameters[param_y].upper

        # get grid sample of points where to evalutate the model
        xaxis = np.linspace(range_x[0], range_x[1], n_points)
        yaxis = np.linspace(range_y[0], range_y[1], n_points)
        X, Y = np.meshgrid(xaxis, yaxis)
        sample = {param_x: X.flatten(), param_y: Y.flatten()}

        if slice_values == "mid":
            # Get mid point
            slice_values = self._get_mid_point()
        elif slice_values == "best":
            # get best point
            slice_values = self._get_best_point(metric_name=metric_name)

        fixed_parameters = {}
        for name, val in slice_values.items():
            if name not in [param_x, param_y]:
                fixed_parameters[name] = slice_values[name]

        # evaluate the model
        f_plt, sd_plt = self.evaluate_model(
            sample=sample,
            metric_name=metric_name,
            fixed_parameters=fixed_parameters,
        )

        # select quantities to plot and set the labels
        f_plots = []
        labels = []
        if mode in ["mean", "both"]:
            f_plots.append(f_plt.reshape(X.shape))
            labels.append(metric_name + ", mean")
        if mode in ["sem", "both"]:
            f_plots.append(sd_plt.reshape(X.shape))
            labels.append(metric_name + ", sem")

        # create figure
        nplots = len(f_plots)
        gridspec_kw = dict(gridspec_kw or {})
        if subplot_spec is None:
            fig = plt.figure(**figure_kw)
            gs = GridSpec(1, nplots, **gridspec_kw)
        else:
            fig = plt.gcf()
            gs = GridSpecFromSubplotSpec(1, nplots, subplot_spec, **gridspec_kw)

        # draw plots
        trials = self.ax_client.get_trials_data_frame()
        axs = []
        for i, f in enumerate(f_plots):
            ax = plt.subplot(gs[i])
            # colormesh
            pcolormesh_kw = dict(pcolormesh_kw or {})
            im = ax.pcolormesh(xaxis, yaxis, f, shading="auto", **pcolormesh_kw)
            cbar = plt.colorbar(im, ax=ax, location="top")
            cbar.set_label(labels[i])
            ax.set(xlabel=param_x, ylabel=param_y)
            # contour
            if show_contour:
                cset = ax.contour(
                    X,
                    Y,
                    f,
                    levels=20,
                    linewidths=0.5,
                    colors="black",
                    linestyles="solid",
                )
                if show_contour_labels:
                    ax.clabel(
                        cset, inline=True, fmt="%1.1f", fontsize="xx-small"
                    )
            if show_trials:
                ax.scatter(
                    trials[param_x], trials[param_y], s=8, c="black", marker="o"
                )
            ax.set_xlim(range_x)
            ax.set_ylim(range_y)
            axs.append(ax)

        if nplots == 1:
            return fig, axs[0]
        else:
            return fig, axs

    def plot_slice(
        self,
        param_name: Optional[str] = None,
        metric_name: Optional[str] = None,
        slice_values: Optional[Union[Dict, Literal["best", "mid"]]] = "mid",
        n_points: Optional[int] = 200,
        range: Optional[List[float]] = None,
        show_legend: Optional[bool] = False,
        subplot_spec: Optional[SubplotSpec] = None,
        gridspec_kw: Optional[Dict[str, Any]] = None,
        plot_kw: Optional[Dict[str, Any]] = None,
        **figure_kw,
    ) -> Tuple[Figure, Axes]:
        """Plot a 1D slice of the surrogate model.

        Parameters
        ----------
        param_name : str
            Name of the parameter to plot in x axis. If not given, the first
            varying parameter will be used.
        metric_name : str, optional.
            Name of the metric to plot.
            If not specified, it will take the first objective in
            ``self.ax_client``.
        slice_values : dict or str, optional.
            The values along which to slice the model, if the model has more
            than one dimensions. Possible values are: ``"best"`` (slice along
            the best predicted point), ``"mid"`` (slice along the middle
            point of the varying parameters), or a dictionary with structure
            ``{param_name: param_val}`` that contains the slice values of each
            parameter. By default, ``"mid"``.
        n_points : int, optional
            Number of points along the x axis. By default, ``200``.
        range : list of float, optional
            Range of the x axis. It not given, the lower and upper boundary
            of the x parameter will be used.
        show_legend : bool
            Whether to show a legend with the fixed slice values. By default,
            ``False``.
        subplot_spec : SubplotSpec, optional
            A matplotlib ``SubplotSpec`` in which to draw the axis.
        gridspec_kw : dict, optional
            Dict with keywords passed to the ``GridSpec``.
        plot_kw : dict, optional
            Dict with keywords passed to ``ax.plot``.
        **figure_kw
            Additional keyword arguments to pass to ``pyplot.figure``. Only
            used if no ``subplot_spec`` is given.

        Returns
        -------
        Figure, Axes
        """
        # get experiment info
        experiment = self.ax_client.experiment
        parnames = list(experiment.parameters.keys())

        # select the input variables
        if param_name is None:
            param_name = parnames[0]

        # metric name
        if metric_name is None:
            metric_name = self.ax_client.objective_names[0]

        # set the plotting range
        if range is None:
            range = [None, None]
        if range[0] is None:
            range[0] = experiment.parameters[param_name].lower
        if range[1] is None:
            range[1] = experiment.parameters[param_name].upper

        # get sample of points where to evalutate the model
        sample = {param_name: np.linspace(range[0], range[1], n_points)}

        if slice_values == "mid":
            # Get mid point
            slice_values = self._get_mid_point()
        elif slice_values == "best":
            # get best point
            slice_values = self._get_best_point(metric_name=metric_name)

        fixed_parameters = {}
        for name, val in slice_values.items():
            if name not in [param_name]:
                fixed_parameters[name] = slice_values[name]

        # evaluate the model
        mean, sem = self.evaluate_model(
            sample=sample,
            metric_name=metric_name,
            fixed_parameters=fixed_parameters,
        )

        # create figure
        gridspec_kw = dict(gridspec_kw or {})
        if subplot_spec is None:
            fig = plt.figure(**figure_kw)
            gs = GridSpec(1, 1, **gridspec_kw)
        else:
            fig = plt.gcf()
            gs = GridSpecFromSubplotSpec(1, 1, subplot_spec, **gridspec_kw)

        # Make plot
        plot_kw = dict(plot_kw or {})
        label = ""
        for par, val in fixed_parameters.items():
            if label:
                label += ", "
            label += f"{par} = {val}"
        ax = fig.add_subplot(gs[0])
        ax.plot(sample[param_name], mean, label=label, **plot_kw)
        ax.fill_between(
            x=sample[param_name],
            y1=mean - sem,
            y2=mean + sem,
            color="lightgray",
            alpha=0.5,
        )
        ax.set_xlabel(param_name)
        ax.set_ylabel(metric_name)
        if show_legend:
            ax.legend(frameon=False)

        return fig, ax
