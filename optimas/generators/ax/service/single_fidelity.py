"""Contains the definition of the single-fidelity Ax generator."""

from typing import List, Optional

import torch
from ax.service.ax_client import AxClient
from ax.modelbridge.generation_strategy import (
    GenerationStep,
    GenerationStrategy,
)
from ax.modelbridge.registry import Models
from ax.service.utils.instantiation import ObjectiveProperties

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
    n_init : int, optional
        Number of evaluations to perform during the initialization phase using
        Sobol sampling. If external data is attached to the exploration, the
        number of initialization evaluations will be reduced by the same
        amount, unless `enforce_n_init=True`. By default, ``4``.
    enforce_n_init : bool, optional
        Whether to enforce the generation of `n_init` Sobol trials, even if
        external data is supplied. By default, ``False``.
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
        n_init: Optional[int] = 4,
        enforce_n_init: Optional[bool] = False,
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
            n_init=n_init,
            enforce_n_init=enforce_n_init,
            use_cuda=use_cuda,
            gpu_id=gpu_id,
            dedicated_resources=dedicated_resources,
            save_model=save_model,
            model_save_period=model_save_period,
            model_history_dir=model_history_dir,
        )

    def _create_ax_client(self):
        """Create single-fidelity Ax client."""
        # Create parameter list.
        parameters = list()
        for var in self._varying_parameters:
            parameters.append(
                {
                    "name": var.name,
                    "type": "range",
                    "bounds": [var.lower_bound, var.upper_bound],
                    # Suppresses warning when the type is not given explicitly
                    "value_type": var.dtype.__name__,
                }
            )

        # Select BO model.
        model_kwargs = {
            "torch_dtype": torch.double,
            "torch_device": torch.device(self.torch_device),
        }
        if self._fully_bayesian:
            if len(self.objectives) > 1:
                # Use a SAAS model with qNEHVI acquisition function.
                MODEL_CLASS = Models.FULLYBAYESIANMOO
            else:
                # Use a SAAS model with qNEI acquisition function.
                MODEL_CLASS = Models.FULLYBAYESIAN
            # Disable additional logs from fully Bayesian model.
            model_kwargs["disable_progbar"] = True
            model_kwargs["verbose"] = False
        else:
            if len(self.objectives) > 1:
                # Use a model with qNEHVI acquisition function.
                MODEL_CLASS = Models.MOO
            else:
                # Use a model with qNEI acquisition function.
                MODEL_CLASS = Models.GPEI

        # Make generation strategy:
        steps = []

        # If there is no past history,
        # adds Sobol initialization with `n_init` random trials:
        # if self.history is None:
        steps.append(
            GenerationStep(model=Models.SOBOL, num_trials=self._n_init)
        )

        # continue indefinitely with BO.
        steps.append(
            GenerationStep(
                model=MODEL_CLASS,
                num_trials=-1,
                model_kwargs=model_kwargs,
            )
        )

        gs = GenerationStrategy(steps)

        ax_objectives = {}
        for obj in self.objectives:
            ax_objectives[obj.name] = ObjectiveProperties(minimize=obj.minimize)

        # Create client and experiment.
        ax_client = AxClient(generation_strategy=gs, verbose_logging=False)
        ax_client.create_experiment(
            parameters=parameters, objectives=ax_objectives
        )

        return ax_client
