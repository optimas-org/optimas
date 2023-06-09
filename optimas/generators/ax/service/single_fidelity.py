"""Contains the definition of the single-fidelity Ax generator."""

from typing import List, Optional

import torch
from ax.service.ax_client import AxClient
from ax.modelbridge.generation_strategy import (
    GenerationStep, GenerationStrategy)
from ax.modelbridge.registry import Models
from ax.service.utils.instantiation import ObjectiveProperties

from optimas.core import Objective, VaryingParameter, Parameter
from .base import AxServiceGenerator


class AxSingleFidelityGenerator(AxServiceGenerator):
    """Generator for performing single-fidelity Bayesian optimization using the
    Ax service API.

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
        Sobol sampling. By default, ``4``.
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
        n_init: Optional[int] = 4,
        use_cuda: Optional[bool] = False,
        gpu_id: Optional[int] = 0,
        dedicated_resources: Optional[bool] = False,
        save_model: Optional[bool] = True,
        model_save_period: Optional[int] = 5,
        model_history_dir: Optional[str] = 'model_history',
    ) -> None:
        super().__init__(
            varying_parameters=varying_parameters,
            objectives=objectives,
            analyzed_parameters=analyzed_parameters,
            n_init=n_init,
            use_cuda=use_cuda,
            gpu_id=gpu_id,
            dedicated_resources=dedicated_resources,
            save_model=save_model,
            model_save_period=model_save_period,
            model_history_dir=model_history_dir
        )

    def _create_ax_client(self):
        """Create single-fidelity Ax client."""
        # Create parameter list.
        parameters = list()
        for var in self._varying_parameters:
            parameters.append(
                {
                    'name': var.name,
                    'type': 'range',
                    'bounds': [var.lower_bound, var.upper_bound],
                    # Suppresses warning when the type is not given explicitly
                    'value_type': var.dtype.__name__
                }
            )

        # Make generation strategy:
        steps = []

        # If there is no past history,
        # adds Sobol initialization with `n_init` random trials:
        # if self.history is None:
        steps.append(
            GenerationStep(
                model=Models.SOBOL,
                num_trials=self._n_init
            )
        )

        # continue indefinitely with GPEI.
        steps.append(
            GenerationStep(
                model=Models.GPEI,
                num_trials=-1,
                model_kwargs={
                    'torch_dtype': torch.double,
                    'torch_device': torch.device(self.torch_device)
                }
            )
        )

        gs = GenerationStrategy(steps)

        ax_objectives = {}
        for obj in self.objectives:
            ax_objectives[obj.name] = ObjectiveProperties(
                minimize=obj.minimize)

        # Create client and experiment.
        ax_client = AxClient(generation_strategy=gs, verbose_logging=False)
        ax_client.create_experiment(
            parameters=parameters,
            objectives=ax_objectives
        )

        return ax_client
