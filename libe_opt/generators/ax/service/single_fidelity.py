import torch

from ax.service.ax_client import AxClient
from ax.modelbridge.generation_strategy import (
    GenerationStep, GenerationStrategy)
from ax.modelbridge.registry import Models
from ax.service.utils.instantiation import ObjectiveProperties

from .base import AxServiceGenerator


class AxSingleFidelityGenerator(AxServiceGenerator):

    def __init__(self, varying_parameters, objectives=None, n_init=4,
                 use_cuda=False):
        super().__init__(varying_parameters, objectives, n_init,
                         use_cuda=use_cuda)

    def _create_ax_client(self):
        # Create parameter list.
        parameters = list()
        for var in self._varying_parameters:
            parameters.append(
                {
                    'name': var.name,
                    'type': 'range',
                    'bounds': [var.lower_bound, var.upper_bound]
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

        self._ax_client = ax_client
