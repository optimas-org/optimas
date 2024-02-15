"""Contains the definition of the multi-fidelity Ax generator."""

from typing import List, Optional, Dict

from ax.modelbridge.generation_strategy import GenerationStep
from ax.modelbridge.registry import Models

from optimas.core import Objective, VaryingParameter, Parameter
from .base import AxServiceGenerator


class AxMultiFidelityGenerator(AxServiceGenerator):
    """Multifidelity Bayesian optimization using the Ax service API.

    Parameters
    ----------
    varying_parameters : list of VaryingParameter
        List of input parameters to vary. One them should be a fidelity.
    objectives : list of Objective
        List of optimization objectives.
    analyzed_parameters : list of Parameter, optional
        List of parameters to analyze at each trial, but which are not
        optimization objectives. By default ``None``.
    outcome_constraints : list of str, optional
        List of string representation of outcome constraints (i.e., constraints
        on any of the ``analyzed_parameters``) of form
        ``"metric_name >= bound"``, like ``"m1 <= 3."``.
    n_init : int, optional
        Number of evaluations to perform during the initialization phase using
        Sobol sampling. If external data is attached to the exploration, the
        number of initialization evaluations will be reduced by the same
        amount, unless `enforce_n_init=True`. By default, ``4``.
    enforce_n_init : bool, optional
        Whether to enforce the generation of `n_init` Sobol trials, even if
        external data is supplied. By default, ``False``.
    abandon_failed_trials : bool, optional
        Whether failed trials should be abandoned (i.e., not suggested again).
        By default, ``True``.
    fit_out_of_design : bool, optional
        Whether to fit the surrogate model taking into account evaluations
        outside of the range of the varying parameters. This can be useful
        if the range of parameter has been reduced during the optimization.
        By default, False.
    fidel_cost_intercept : float, optional
        The cost intercept for the affine cost of the form
        `cost_intercept + n`, where `n` is the number of generated points.
        Used for the knowledge gradient acquisition function. By default, 1.
    use_cuda : bool, optional
        Whether to allow the generator to run on a CUDA GPU. By default
        ``False``.
    gpu_id : int, optional
        The ID of the GPU in which to run the generator. By default, ``0``.
    dedicated_resources : bool, optional
        Whether to allocated dedicated resources (e.g., the GPU) for the
        generator. These resources will not be available to the
        simulation workers. By default, ``False``.
    save_model : bool, optional
        Whether to save the optimization model (in this case, the Ax client) to
        disk. By default ``True``.
    model_save_period : int, optional
        Periodicity, in number of evaluated Trials, with which to save the
        model to disk. By default, ``5``.
    model_history_dir : str, optional
        Name of the directory in which the model will be saved. By default,
        ``'model_history'``.

    """

    def __init__(
        self,
        varying_parameters: List[VaryingParameter],
        objectives: List[Objective],
        analyzed_parameters: Optional[List[Parameter]] = None,
        outcome_constraints: Optional[List[str]] = None,
        n_init: Optional[int] = 4,
        enforce_n_init: Optional[bool] = False,
        abandon_failed_trials: Optional[bool] = True,
        fit_out_of_design: Optional[bool] = False,
        fidel_cost_intercept: Optional[float] = 1.0,
        use_cuda: Optional[bool] = False,
        gpu_id: Optional[int] = 0,
        dedicated_resources: Optional[bool] = False,
        save_model: Optional[bool] = True,
        model_save_period: Optional[int] = 5,
        model_history_dir: Optional[str] = "model_history",
    ) -> None:
        self.fidel_cost_intercept = fidel_cost_intercept
        super().__init__(
            varying_parameters=varying_parameters,
            objectives=objectives,
            analyzed_parameters=analyzed_parameters,
            outcome_constraints=outcome_constraints,
            n_init=n_init,
            enforce_n_init=enforce_n_init,
            abandon_failed_trials=abandon_failed_trials,
            fit_out_of_design=fit_out_of_design,
            use_cuda=use_cuda,
            gpu_id=gpu_id,
            dedicated_resources=dedicated_resources,
            save_model=save_model,
            model_save_period=model_save_period,
            model_history_dir=model_history_dir,
        )

    def _create_generation_steps(
        self, bo_model_kwargs: Dict
    ) -> List[GenerationStep]:
        """Create generation steps for multifidelity optimization."""
        # Add cost intercept to model kwargs.
        bo_model_kwargs["cost_intercept"] = self.fidel_cost_intercept

        # Make generation strategy.
        steps = []

        # Add Sobol initialization with `n_init` random trials.
        steps.append(
            GenerationStep(model=Models.SOBOL, num_trials=self._n_init)
        )

        # Continue indefinitely with GPKG.
        steps.append(
            GenerationStep(
                model=Models.GPKG,
                num_trials=-1,
                model_kwargs=bo_model_kwargs,
            )
        )

        return steps
