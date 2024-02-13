"""Contains the definition of the ExplorationDiagnostics class."""

from typing import Optional, Union, List, Tuple, Dict, Any, Literal
import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame
from copy import deepcopy
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec, SubplotSpec
from matplotlib.axes import Axes

# Ax utilities for model building
from ax.service.ax_client import AxClient
from ax.modelbridge.generation_strategy import (
    GenerationStep,
    GenerationStrategy,
)
from ax.modelbridge.registry import Models
from ax.modelbridge.torch import TorchModelBridge
from ax.core.observation import ObservationFeatures


class AxModelManager(object):
    """Utilities for building and exploring surrogate models using ``Ax``.

    Parameters
    ----------
    source: AxClient, str or DataFrame
        Source data for the model.
        If ``DataFrame``, the model has to be build using ``build_model``.
        If ``AxClient``, it uses the data in there to build a model.
        If ``str``, it should be the path to an ``AxClient`` json file.
    """

    def __init__(self, source: Union[AxClient, str, DataFrame]) -> None:
        if isinstance(source, AxClient):
            self.ax_client = source
        elif isinstance(source, str):
            self.ax_client = AxClient.load_from_json_file(filepath=source)
        elif isinstance(source, DataFrame):
            self.df = source
            self.ax_client = None
        else:
            raise RuntimeError("Wrong source.")

        if self.ax_client:
            self.ax_client.fit_model()

    @property
    def model(self) -> TorchModelBridge:
        """Get the model from the AxClient instance."""
        return self.ax_client.generation_strategy.model

    def build_model(
        self, objname: str, parnames: List[str], minimize: Optional[bool] = True
    ) -> None:
        """Initialize the AxClient and the model using the given data.

        Parameters
        ----------
        parnames: list of string
            List with the names of the parameters of the model.
        objname: string
            Name of the objective.
        minimize: bool
            Whether to minimize or maximize the objective.
            Only relevant to establish the best point.
        """
        parameters = [
            {
                "name": p_name,
                "type": "range",
                "bounds": [self.df[p_name].min(), self.df[p_name].max()],
                "value_type": "float",
            }
            for p_name in parnames
        ]

        # create Ax client
        gs = GenerationStrategy(
            [GenerationStep(model=Models.GPEI, num_trials=-1)]
        )
        self.ax_client = AxClient(generation_strategy=gs, verbose_logging=False)
        self.ax_client.create_experiment(
            name="optimas_data",
            parameters=parameters,
            objective_name=objname,
            minimize=minimize,
        )

        # adds data
        metric_name = list(self.ax_client.experiment.metrics.keys())[0]
        for index, row in self.df.iterrows():
            params = {p_name: row[p_name] for p_name in parnames}
            _, trial_id = self.ax_client.attach_trial(params)
            self.ax_client.complete_trial(
                trial_id, {metric_name: (row[metric_name], np.nan)}
            )

        # fit GP model
        self.ax_client.fit_model()

    def evaluate_model(
        self,
        sample: Union[DataFrame, Dict, NDArray] = None,
        metric_name: Optional[str] = None,
        p0: Optional[Dict] = None,
    ) -> Tuple[NDArray]:
        """Evaluate the model over the specified sample.

        Parameter:
        ----------
        sample: DataFrame, dict of arrays or numpy array,
            containing the data sample where to evaluate the model.
            If numpy array, it must contain the values of all the model parameres.
            If DataFrame or dict, it can contain only those parameters to vary.
            The rest of parameters would be set to the model best point,
            unless they are further specified using ``p0``.
        metric_name: str, optional.
            Name of the metric to evaluate.
            If not specified, it will take the first first objective in ``self.ax_client``.
        p0: dictionary, optional.
            Particular values of parameters to be fixed for the evaluation.

        Returns
        -------
        m_array, sem_array : Two numpy arrays containing the mean of the model
            and the standard error of the mean (sem), respectively.
        """
        if self.model is None:
            raise RuntimeError("Model not present. Run ``build_model`` first.")

        if metric_name is None:
            metric_name = self.ax_client.objective_names[0]
        else:
            metric_names = list(self.ax_client.experiment.metrics.keys())
            if metric_name not in metric_names:
                raise RuntimeError(
                    f"Metric name {metric_name} does not match any of the metrics. "
                    f"Available metrics are: {metric_names}."
                )

        # get optimum
        try:
            objectives = self.ax_client.objective.objectives
            minimize = None
            for obj in objectives:
                if metric_name == obj.metric_names[0]:
                    minimize = obj.minimize
                    break
            pp = self.ax_client.get_pareto_optimal_parameters()
            obj_vals = [
                objs[metric_name] for i, (vals, (objs, covs)) in pp.items()
            ]
            param_vals = [vals for i, (vals, (objs, covs)) in pp.items()]
            if minimize:
                best_obj_i = np.argmin(obj_vals)
            else:
                best_obj_i = np.argmax(obj_vals)
            best_pars = param_vals[best_obj_i]
        except AttributeError:
            best_pars = self.ax_client.get_best_parameters()[0]

        parameters = best_pars
        parnames = list(parameters.keys())

        # user specific point
        if p0 is not None:
            for key in p0.keys():
                if key in parameters.keys():
                    parameters[key] = p0[key]

        # make list of `ObservationFeatures`
        obsf_list = []
        obsf_0 = ObservationFeatures(parameters=parameters)
        if isinstance(sample, np.ndarray):
            # check the shape of the array
            if sample.shape[1] != len(parnames):
                raise RuntimeError(
                    "Second dimension of the sample array should match "
                    "the number of parameters of the model."
                )
            for i in range(sample.shape[0]):
                predf = deepcopy(obsf_0)
                for j, parname in enumerate(parameters.keys()):
                    predf.parameters[parname] = sample[i][j]
                obsf_list.append(predf)
        elif isinstance(sample, DataFrame):
            # check if labels of the dataframe match the parnames
            for col in sample.columns:
                if col not in parnames:
                    raise RuntimeError(
                        "Column %s does not match any of the parameter names"
                        % col
                    )
            for i in range(sample.shape[0]):
                predf = deepcopy(obsf_0)
                for col in sample.columns:
                    predf.parameters[col] = sample[col].iloc[i]
                obsf_list.append(predf)
        elif isinstance(sample, dict):
            # check if the keys of the dictionary match the parnames
            for key in sample.keys():
                if key not in parnames:
                    raise RuntimeError(
                        "Key %s does not match any of the parameter names" % col
                    )
            element = sample[list(sample.keys())[0]]
            if hasattr(element, "__len__"):
                for i in range(len(element)):
                    predf = deepcopy(obsf_0)
                    for key in sample.keys():
                        predf.parameters[key] = sample[key][i]
                    obsf_list.append(predf)
            else:
                predf = deepcopy(obsf_0)
                for key in sample.keys():
                    predf.parameters[key] = sample[key]
                obsf_list.append(predf)

        elif sample is None:
            predf = deepcopy(obsf_0)
            obsf_list.append(predf)
        else:
            raise RuntimeError("Wrong data type")

        mu, cov = self.model.predict(obsf_list)
        m_array = np.asarray(mu[metric_name])
        sem_array = np.sqrt(cov[metric_name][metric_name])
        return m_array, sem_array

    def plot_model(
        self,
        xname: Optional[str] = None,
        yname: Optional[str] = None,
        mname: Optional[str] = None,
        p0: Optional[Dict] = None,
        npoints: Optional[int] = 200,
        xrange: Optional[List[float]] = None,
        yrange: Optional[List[float]] = None,
        mode: Optional[Literal["mean", "sem", "both"]] = "mean",
        clabel: Optional[bool] = False,
        subplot_spec: Optional[SubplotSpec] = None,
        gridspec_kw: Dict[str, Any] = None,
        pcolormesh_kw: Dict[str, Any] = None,
        **figure_kw,
    ) -> Union[Axes, List[Axes]]:
        """Plot model in the two selected variables, while others are fixed to the optimum.

        Parameter:
        ----------
        xname: string
            Name of the variable to plot in x axis.
        yname: string
            Name of the variable to plot in y axis.
        mname: string, optional.
            Name of the metric to plot.
            If not specified, it will take the first objective in ``self.ax_client``.
        p0: dictionary
            Particular values of parameters to be fixed for the evaluation over the sample.
        npoints: int, optional
            Number of points in each axis.
        mode: string, optional.
            ``mean`` plots the model mean, ``sem`` the standard error of the mean,
            ``both`` plots both.
        clabel: bool
            when true labels are shown along the contour lines.
        gridspec_kw : dict, optional
            Dict with keywords passed to the `GridSpec`.
        pcolormesh_kw : dict, optional
            Dict with keywords passed to `pcolormesh`.
        **figure_kw
            Additional keyword arguments to pass to `pyplot.figure`. Only used
            if no ``subplot_spec`` is given.

        Returns
        -------
        `~.axes.Axes` or array of Axes
            Either a single `~matplotlib.axes.Axes` object or a list of Axes
            objects if more than one subplot was created.
        """
        if self.ax_client is None:
            raise RuntimeError("AxClient not present. Run `build_model` first.")

        if self.model is None:
            raise RuntimeError("Model not present. Run `build_model` first.")

        # get experiment info
        experiment = self.ax_client.experiment
        parnames = list(experiment.parameters.keys())
        # minimize = experiment.optimization_config.objective.minimize

        if len(parnames) < 2:
            raise RuntimeError(
                "Insufficient number of parameters in data for this plot "
                "(minimum 2)."
            )

        # select the input variables
        if xname is None:
            xname = parnames[0]
        if yname is None:
            yname = parnames[1]

        # set the plotting range
        if xrange is None:
            xrange = [None, None]
        if yrange is None:
            yrange = [None, None]
        if xrange[0] is None:
            xrange[0] = experiment.parameters[xname].lower
        if xrange[1] is None:
            xrange[1] = experiment.parameters[xname].upper
        if yrange[0] is None:
            yrange[0] = experiment.parameters[yname].lower
        if yrange[1] is None:
            yrange[1] = experiment.parameters[yname].upper

        # get grid sample of points where to evalutate the model
        xaxis = np.linspace(xrange[0], xrange[1], npoints)
        yaxis = np.linspace(yrange[0], yrange[1], npoints)
        X, Y = np.meshgrid(xaxis, yaxis)
        xarray = X.flatten()
        yarray = Y.flatten()
        sample = DataFrame({xname: xarray, yname: yarray})

        # metric name
        if mname is None:
            mname = self.ax_client.objective_names[0]

        # evaluate the model
        f_plt, sd_plt = self.evaluate_model(
            sample=sample,
            metric_name=mname,
            p0=p0,
        )

        # get numpy arrays with experiment parameters
        xtrials = np.zeros(experiment.num_trials)
        ytrials = np.zeros(experiment.num_trials)
        for i in range(experiment.num_trials):
            xtrials[i] = experiment.trials[i].arm.parameters[xname]
            ytrials[i] = experiment.trials[i].arm.parameters[yname]

        # select quantities to plot and set the labels
        f_plots = []
        labels = []
        if mode in ["mean", "both"]:
            f_plots.append(f_plt)
            labels.append(mname + ", mean")
        if mode in ["sem", "both"]:
            f_plots.append(sd_plt)
            labels.append(mname + ", sem")

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
        axs = []
        for i, f in enumerate(f_plots):
            ax = plt.subplot(gs[i])
            # colormesh
            pcolormesh_kw = dict(pcolormesh_kw or {})
            im = ax.pcolormesh(
                xaxis,
                yaxis,
                f.reshape(X.shape),
                shading="auto",
                **pcolormesh_kw,
            )
            cbar = plt.colorbar(im, ax=ax, location="top")
            cbar.set_label(labels[i])
            ax.set(xlabel=xname, ylabel=yname)
            # contour lines
            cset = ax.contour(
                X,
                Y,
                f.reshape(X.shape),
                levels=20,
                linewidths=0.5,
                colors="black",
                linestyles="solid",
            )
            if clabel:
                plt.clabel(cset, inline=True, fmt="%1.1f", fontsize="xx-small")
            # draw trials
            ax.scatter(xtrials, ytrials, s=2, c="black", marker="o")
            ax.set_xlim(xrange)
            ax.set_ylim(yrange)
            axs.append(ax)

        if nplots == 1:
            return axs[0]
        else:
            return axs

    def get_arm_index(
        self,
        arm_name: Optional[str] = None,
    ) -> int:
        """Get the index of the arm by its name.

        Parameters
        ----------
        arm_name: string, optional.
            Name of the arm. If not given, the best arm is selected.

        Returns
        -------
        index: int
            Trial index of the arm.
        """
        if arm_name is None:
            best_arm, best_point_predictions = self.model.model_best_point()
            arm_name = best_arm.name

        df = self.ax_client.experiment.fetch_data().df
        index = df.loc[df["arm_name"] == arm_name, "trial_index"].iloc[0]
        return index
