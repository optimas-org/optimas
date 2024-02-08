"""Contains the definition of the Ax generator that uses a custom AxClient."""

from typing import List, Optional

from ax.service.ax_client import AxClient
from ax.core.objective import MultiObjective

from optimas.core import Objective, VaryingParameter, Parameter
from .base import AxServiceGenerator


class AxClientGenerator(AxServiceGenerator):
    """Bayesian optimization generator with a user-defined ``AxClient``.

    This generator allows the user to provide a custom ``AxClient``,
    allowing for maximum control of the optimization.

    For this generator there is no need to provide the list of
    ``varying_parameters`` or ``objectives``. The generator will obtain
    these parameters directly from the ``AxClient``.

    Parameters
    ----------
    ax_client : AxClient
        The Ax client from which the trials will be generated.
    analyzed_parameters : list of Parameter, optional
        List of parameters to analyze at each trial, but which are not
        optimization objectives. By default ``None``.
    abandon_failed_trials : bool, optional
        Whether failed trials should be abandoned (i.e., not suggested again).
        By default, ``True``.
    gpu_id : int, optional
        The ID of the GPU in which to run the generator. By default, ``0``.
        This parameter will only have an effect if any ``GenerationStep`` in
        the ``AxClient`` uses a GPU.
    dedicated_resources : bool, optional
        Whether to allocated dedicated resources (e.g., the GPU) for the
        generator. These resources will not be available to the
        simulation workers. By default, ``False``.
        This parameter will only have an effect if any ``GenerationStep`` in
        the ``AxClient`` uses a GPU.
    save_model : bool, optional
        Whether to save the optimization model (in this case, the Ax client) to
        disk. By default ``True``.
    model_save_period : int, optional
        Periodicity, in number of evaluated Trials, with which to save the
        model to disk. By default, ``5``.
    model_history_dir : str, optional
        Name of the directory in which the model will be saved. By default,
        ``'model_history'``.

    Notes
    -----
    If the ``AxClient`` contains ``outcome_constraints``, these will appear in
    the ``optimas`` log as optimization objectives. They are still being
    correctly used as constraints by the ``AxClient``, and the optimization
    will work as expected. This is only an issue on ``optimas``, which fails to
    properly recognize them because optimization constraints have not yet been
    implemented.

    """

    def __init__(
        self,
        ax_client: AxClient,
        analyzed_parameters: Optional[List[Parameter]] = None,
        abandon_failed_trials: Optional[bool] = True,
        gpu_id: Optional[int] = 0,
        dedicated_resources: Optional[bool] = False,
        save_model: Optional[bool] = True,
        model_save_period: Optional[int] = 5,
        model_history_dir: Optional[str] = "model_history",
    ):
        varying_parameters = self._get_varying_parameters(ax_client)
        objectives = self._get_objectives(ax_client)
        analyzed_parameters = self._add_constraints_to_analyzed_parameters(
            analyzed_parameters, ax_client
        )
        use_cuda = self._use_cuda(ax_client)
        self._ax_client = ax_client
        super().__init__(
            varying_parameters=varying_parameters,
            objectives=objectives,
            analyzed_parameters=analyzed_parameters,
            enforce_n_init=True,
            abandon_failed_trials=abandon_failed_trials,
            use_cuda=use_cuda,
            gpu_id=gpu_id,
            dedicated_resources=dedicated_resources,
            save_model=save_model,
            model_save_period=model_save_period,
            model_history_dir=model_history_dir,
        )

    def _get_varying_parameters(self, ax_client: AxClient):
        """Obtain the list of varying parameters from the AxClient."""
        varying_parameters = []
        for _, p in ax_client.experiment.search_space.parameters.items():
            vp = VaryingParameter(
                name=p.name,
                lower_bound=p.lower,
                upper_bound=p.upper,
                is_fidelity=p.is_fidelity,
                fidelity_target_value=p.target_value,
                dtype=p.python_type,
            )
            varying_parameters.append(vp)
        return varying_parameters

    def _get_objectives(self, ax_client: AxClient):
        """Obtain the list of objectives from the AxClient."""
        objectives = []
        ax_objective = ax_client.experiment.optimization_config.objective
        if isinstance(ax_objective, MultiObjective):
            ax_objectives = ax_objective.objectives
        else:
            ax_objectives = [ax_objective]
        for ax_obj in ax_objectives:
            obj = Objective(
                name=ax_obj.metric_names[0], minimize=ax_obj.minimize
            )
            objectives.append(obj)
        return objectives

    def _add_constraints_to_analyzed_parameters(
        self, analyzed_parameters: List[Parameter], ax_client: AxClient
    ):
        """Add outcome constraints to the list of analyzed parameters.

        This is currently needed because optimas does not yet have a
        proper definition of constraints. The constraints will be correctly
        handled and given to the AxClient, but will appear as analyzed
        parameters in the optimization log.
        """
        ax_config = ax_client.experiment.optimization_config
        if ax_config.outcome_constraints and analyzed_parameters is None:
            analyzed_parameters = []
        for constraint in ax_config.outcome_constraints:
            analyzed_parameters.append(Parameter(name=constraint.metric.name))
        return analyzed_parameters

    def _create_ax_client(self) -> AxClient:
        """Override the base function to simply return the given."""
        return self._ax_client

    def _use_cuda(self, ax_client: AxClient):
        """Determine whether the AxClient uses CUDA."""
        for step in ax_client.generation_strategy._steps:
            if "torch_device" in step.model_kwargs:
                if step.model_kwargs["torch_device"] == "cuda":
                    return True
        return False
