"""Contains the definition of the single-fidelity Ax generator."""

from typing import List, Optional, Dict

from ax.modelbridge.generation_strategy import GenerationStep
from ax.modelbridge.registry import Models

from optimas.core import Objective, VaryingParameter, Parameter
from .base import AxServiceGenerator


class AxSingleFidelityGenerator(AxServiceGenerator):
    """Single-fidelity Bayesian optimization using the Ax service API.

    Depending on whether a single or multiple objectives are given, the
    acquisition function will be qNEI (Noisy Expected Improvement) or qNEHVI
    (Noisy Expected Hypervolume Improvement).

    By default, the hyperparameters of the GP are optimized by maximizing the
    maximal likelihood of the data. A fully Bayesian approach using SAAS
    priors is also available, which has been shown to perform well for
    high-dimensional optimization [1]_ [2]_.

    Parameters
    ----------
    varying_parameters : list of VaryingParameter
        List of input parameters to vary.
    objectives : list of Objective
        List of optimization objectives.
    analyzed_parameters : list of Parameter, optional
        List of parameters to analyze at each trial, but which are not
        optimization objectives. By default ``None``.
    parameter_constraints : list of str, optional
        List of string representation of parameter
        constraints, such as ``"x3 >= x4"`` or ``"-x3 + 2*x4 - 3.5*x5 >= 2"``.
        For the latter constraints, any number of arguments is accepted, and
        acceptable operators are ``<=`` and ``>=``.
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
    fully_bayesian : bool, optional
        Whether to optimize the hyperparameters of the GP with a fully
        Bayesian approach (using SAAS priors) instead of maximizing
        marginal likelihood. The fully Bayesian treatment is more expensive
        (i.e., it takes longer to generate new trials) but can lead to
        improved BO performance (i.e., requiring less evaluations). This
        approach is specially well suited for high-dimensional optimization.
        By default ``False``.
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

    References
    ----------
    .. [1] D. Eriksson, M. Jankowiak. High-Dimensional Bayesian Optimization
       with Sparse Axis-Aligned Subspaces. Proceedings of the Thirty-Seventh
       Conference on Uncertainty in Artificial Intelligence, 2021.
    .. [2] https://ax.dev/tutorials/saasbo.html

    """

    def __init__(
        self,
        varying_parameters: List[VaryingParameter],
        objectives: List[Objective],
        analyzed_parameters: Optional[List[Parameter]] = None,
        parameter_constraints: Optional[List[str]] = None,
        outcome_constraints: Optional[List[str]] = None,
        n_init: Optional[int] = 4,
        enforce_n_init: Optional[bool] = False,
        abandon_failed_trials: Optional[bool] = True,
        fit_out_of_design: Optional[bool] = False,
        fully_bayesian: Optional[bool] = False,
        use_cuda: Optional[bool] = False,
        gpu_id: Optional[int] = 0,
        dedicated_resources: Optional[bool] = False,
        save_model: Optional[bool] = True,
        model_save_period: Optional[int] = 5,
        model_history_dir: Optional[str] = "model_history",
    ) -> None:
        self._fully_bayesian = fully_bayesian
        super().__init__(
            varying_parameters=varying_parameters,
            objectives=objectives,
            analyzed_parameters=analyzed_parameters,
            parameter_constraints=parameter_constraints,
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
        """Create generation steps for single-fidelity optimization."""
        # Select BO model.
        if self._fully_bayesian:
            if len(self.objectives) > 1:
                # Use a SAAS model with qNEHVI acquisition function.
                MODEL_CLASS = Models.FULLYBAYESIANMOO
            else:
                # Use a SAAS model with qNEI acquisition function.
                MODEL_CLASS = Models.FULLYBAYESIAN
            # Disable additional logs from fully Bayesian model.
            bo_model_kwargs["disable_progbar"] = True
            bo_model_kwargs["verbose"] = False
        else:
            if len(self.objectives) > 1:
                # Use a model with qNEHVI acquisition function.
                MODEL_CLASS = Models.MOO
            else:
                # Use a model with qNEI acquisition function.
                MODEL_CLASS = Models.GPEI

        # Make generation strategy.
        steps = []

        # Add Sobol initialization with `n_init` random trials.
        steps.append(
            GenerationStep(model=Models.SOBOL, num_trials=self._n_init)
        )

        # Continue indefinitely with BO.
        steps.append(
            GenerationStep(
                model=MODEL_CLASS,
                num_trials=-1,
                model_kwargs=bo_model_kwargs,
            )
        )

        return steps
