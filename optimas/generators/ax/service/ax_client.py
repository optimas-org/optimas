"""Contains the definition of the Ax generator that uses a custom AxClient."""

from typing import List, Optional

from ax.service.ax_client import AxClient
from ax.core.objective import MultiObjective
from ax.core.types import ComparisonOp

from optimas.core import Parameter
from gest_api.vocs import VOCS
from .base import AxServiceGenerator


class AxClientGenerator(AxServiceGenerator):
    """Bayesian optimization generator with a user-defined ``AxClient``.

    This generator allows the user to provide a custom ``AxClient``,
    allowing for maximum control of the optimization.

    For this generator there is no need to provide the ``vocs``. The
    generator builds a VOCS (variables, objectives, constraints, and
    observables) directly from the ``AxClient``.

    Parameters
    ----------
    ax_client : AxClient
        The Ax client from which the trials will be generated.
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
    Outcome constraints are passed into VOCS as constraints and are correctly
    used by the ``AxClient``. The ``optimas`` log/display does not yet show
    constraints separately; constraint metrics may appear as extra columns.

    """

    def __init__(
        self,
        ax_client: AxClient,
        abandon_failed_trials: Optional[bool] = True,
        gpu_id: Optional[int] = 0,
        dedicated_resources: Optional[bool] = False,
        save_model: Optional[bool] = True,
        model_save_period: Optional[int] = 5,
        model_history_dir: Optional[str] = "model_history",
    ):
        # Create VOCS object from AxClient data
        vocs = self._create_vocs_from_ax_client(ax_client)
        use_cuda = self._use_cuda(ax_client)
        self._ax_client = ax_client

        super().__init__(
            vocs=vocs,
            abandon_failed_trials=abandon_failed_trials,
            use_cuda=use_cuda,
            gpu_id=gpu_id,
            dedicated_resources=dedicated_resources,
            save_model=save_model,
            model_save_period=model_save_period,
            model_history_dir=model_history_dir,
        )

    def _create_vocs_from_ax_client(self, ax_client: AxClient) -> VOCS:
        """Create a VOCS object from the AxClient data."""
        # Extract variables from search space
        variables = {}
        for _, p in ax_client.experiment.search_space.parameters.items():
            variables[p.name] = [p.lower, p.upper]

        # Extract objectives from optimization config
        objectives = {}
        ax_objective = ax_client.experiment.optimization_config.objective
        if isinstance(ax_objective, MultiObjective):
            ax_objectives = ax_objective.objectives
        else:
            ax_objectives = [ax_objective]

        for ax_obj in ax_objectives:
            obj_type = "MINIMIZE" if ax_obj.minimize else "MAXIMIZE"
            objectives[ax_obj.metric_names[0]] = obj_type

        # Extract constraints from outcome constraints (if any)
        constraints = {}
        ax_config = ax_client.experiment.optimization_config
        if ax_config.outcome_constraints:
            for constraint in ax_config.outcome_constraints:
                name = constraint.metric.name
                if constraint.op == ComparisonOp.LEQ:
                    constraints[name] = ["LESS_THAN", constraint.bound]
                elif constraint.op == ComparisonOp.GEQ:
                    constraints[name] = ["GREATER_THAN", constraint.bound]

        return VOCS(
            variables=variables,
            objectives=objectives,
            constraints=constraints,
        )

    def _convert_vocs_constraints_to_outcome_constraints(
        self,
    ) -> tuple[List[str], List[Parameter]]:
        """Override to skip conversion since AxClient already has constraints."""
        constraint_parameters = []
        if hasattr(self._vocs, "constraints") and self._vocs.constraints:
            for constraint_name in self._vocs.constraints.keys():
                constraint_parameters.append(Parameter(constraint_name))
        return [], constraint_parameters

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
