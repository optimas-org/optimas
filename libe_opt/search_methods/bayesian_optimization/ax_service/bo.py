import torch
from ax.service.ax_client import AxClient
from ax.modelbridge.generation_strategy import (
    GenerationStep, GenerationStrategy)
from ax.modelbridge.registry import Models

from .base import AxOptimizer


class BayesianOptimization(AxOptimizer):
    def __init__(
            self, var_names, var_lb, var_ub, sim_template, analysis_func,
            sim_number, analyzed_params=[], sim_workers=1, run_async=True,
            use_cuda=False, libE_specs={}, history=None, executable=None,
            sim_files=[]):
        super().__init__(
            var_names=var_names,
            var_lb=var_lb,
            var_ub=var_ub,
            sim_template=sim_template,
            analysis_func=analysis_func,
            sim_number=sim_number,
            analyzed_params=analyzed_params,
            sim_workers=sim_workers,
            run_async=run_async,
            use_cuda=use_cuda,
            libE_specs=libE_specs,
            history=history,
            executable=executable,
            sim_files=sim_files
        )

    def _create_ax_client(self):
        # Create parameter list.
        # The use of `.item()` converts from numpy types to native Python
        # types. This is needed becase Ax seems to support only native types.
        parameters = list()
        for name, lb, ub in zip(self.var_names, self.var_lb, self.var_ub):
            parameters.append(
                {
                    'name': name,
                    'type': 'range',
                    'bounds': [lb.item(), ub.item()]
                }
            )

        # Batch size
        batch_size = self.sim_workers

        # Make generation strategy:
        steps = []

        # If there is no past history,
        # adds Sobol initialization with `batch_size` random trials:
        if self.history is None:
            steps.append(
                GenerationStep(
                    model=Models.SOBOL,
                    num_trials=batch_size
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

        # Create client and experiment.
        ax_client = AxClient(generation_strategy=gs)
        ax_client.create_experiment(
            parameters=parameters,
            objective_name="f",
            minimize=True
        )

        return ax_client
