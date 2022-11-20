import torch
from ax.service.ax_client import AxClient
from ax.modelbridge.generation_strategy import (
    GenerationStep, GenerationStrategy)
from ax.modelbridge.registry import Models

from .base import AxOptimizer


class MultifidelityBayesianOptimization(AxOptimizer):
    def __init__(
            self, var_names, var_lb, var_ub, sim_template, analysis_func,
            sim_number, mf_params, analyzed_params=[], sim_workers=1,
            run_async=True, use_cuda=False, libE_specs={}, history=None):
        self.mf_params = mf_params
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
            history=history
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

        parameters.append(
            {
                'name':self. mf_params['name'],
                'type': 'range',
                'bounds': self.mf_params['range'],
                'is_fidelity': True,
                'target_value': self.mf_params['range'][-1]
            }
        )

        # Batch size
        batch_size = self.sim_workers

        # Make generation strategy:
        steps = []

        # If there is no past history,
        # adds Sobol initialization with `batch_size` random trials:
        if self.history is None:
            steps.append(GenerationStep(model=Models.SOBOL, num_trials=batch_size))

        # continue indefinitely with GPKG.
        steps.append(
                GenerationStep(
                    model=Models.GPKG,
                    num_trials=-1,
                    model_kwargs={
                        'cost_intercept': self.mf_params['cost_intercept'],
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
