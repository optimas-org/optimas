import torch

from ax.service.ax_client import AxClient
from ax.modelbridge.generation_strategy import (
    GenerationStep, GenerationStrategy)
from ax.modelbridge.registry import Models
from ax.service.utils.instantiation import ObjectiveProperties

from .base import AxServiceGenerator


class AxMultiFidelityGenerator(AxServiceGenerator):
    def __init__(self, varying_parameters, objectives=None,
                 analyzed_parameters=None, n_init=4, fidel_cost_intercept=1.,
                 use_cuda=False, gpu_id=0, dedicated_resources=False,
                 save_model=True, model_save_period=5,
                 model_history_dir='model_history'):
        self.fidel_cost_intercept = fidel_cost_intercept
        super().__init__(varying_parameters,
                         objectives,
                         analyzed_parameters=analyzed_parameters,
                         n_init=n_init,
                         use_cuda=use_cuda,
                         gpu_id=gpu_id,
                         dedicated_resources=dedicated_resources,
                         save_model=save_model,
                         model_save_period=model_save_period,
                         model_history_dir=model_history_dir)

    def _create_ax_client(self):
        # Create parameter list.
        parameters = list()
        for var in self._varying_parameters:
            parameters.append(
                {
                    'name': var.name,
                    'type': 'range',
                    'bounds': [var.lower_bound, var.upper_bound],
                    'is_fidelity': var.is_fidelity,
                    'target_value': var.fidelity_target_value
                }
            )

        # Make generation strategy:
        steps = []

        # If there is no past history,
        # adds Sobol initialization with `batch_size` random trials:
        # if self.history is None:
        steps.append(
            GenerationStep(
                model=Models.SOBOL,
                num_trials=self._n_init
            )
        )

        # continue indefinitely with GPKG.
        steps.append(
                GenerationStep(
                    model=Models.GPKG,
                    num_trials=-1,
                    model_kwargs={
                        'cost_intercept': self.fidel_cost_intercept,
                        'torch_dtype': torch.double,
                        'torch_device': torch.device(self.torch_device)
                    }
                )
            )

        gs = GenerationStrategy(steps)

        ax_objs = {}
        for obj in self.objectives:
            ax_objs[obj.name] = ObjectiveProperties(minimize=obj.minimize)

        # Create client and experiment.
        ax_client = AxClient(generation_strategy=gs, verbose_logging=False)
        ax_client.create_experiment(
            parameters=parameters,
            objectives=ax_objs
        )

        self._ax_client = ax_client
