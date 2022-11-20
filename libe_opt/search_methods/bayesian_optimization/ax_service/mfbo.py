import torch
from ax.service.ax_client import AxClient
from ax.modelbridge.generation_strategy import (
    GenerationStep, GenerationStrategy)
from ax.modelbridge.registry import Models

from .base import AxOptimizer


class MultifidelityBayesianOptimization(AxOptimizer):
    def __init__(
            self, var_names, var_lb, var_ub, sim_template, analysis_func,
            sim_number, fidel_name, fidel_lb, fidel_ub,
            fidel_cost_intercept=1.0, analyzed_params=[], sim_workers=1,
            run_async=True, use_cuda=False, libE_specs={}, history=None):
        self.fidel_name = fidel_name
        self.fidel_lb = fidel_lb
        self.fidel_ub = fidel_ub
        self.fidel_cost_intercept = fidel_cost_intercept
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
                'name':self.fidel_name,
                'type': 'range',
                'bounds': [self.fidel_lb, self.fidel_ub],
                'is_fidelity': True,
                'target_value': self.fidel_ub
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
                        'cost_intercept': self.fidel_cost_intercept,
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

    def _create_gen_specs(self):
        super()._create_gen_specs()
        self.gen_specs['in'].append('z')
        self.gen_specs['persis_in'].append('z')
        fidel_type, fidel_len = self._determine_fidelity_type_and_length()
        self.gen_specs['out'].append(('z', fidel_type, fidel_len))
        self.gen_specs['user']['mf_params'] = {'name': self.fidel_name}

    def _determine_fidelity_type_and_length(self):
        """
        Determine the type of the fidelity (i.e. float, int, str...) and, if it
        is a string, also its length.
        """
        # Check that all fidelities in 'range' are of the same type.
        assert (
            type(self.fidel_ub) == type(self.fidel_lb)
        ), "The lower and upper bounds of the fidelity are of different types."

        fidel_type = type(self.fidel_ub)
        fidel_len = None
        # If fidelities are strings, determine the lenght of the longest one
        # so that it can be fully stored in a numpy array.
        if fidel_type == str:
            fidel_len = max(len(self.fidel_lb), len(self.fidel_ub))
        return fidel_type, fidel_len
