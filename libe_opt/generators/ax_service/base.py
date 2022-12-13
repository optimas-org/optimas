import torch

from ax.service.ax_client import AxClient
from ax.modelbridge.generation_strategy import (
    GenerationStep, GenerationStrategy)
from ax.modelbridge.registry import Models
from ax.service.utils.instantiation import ObjectiveProperties

from libe_opt.generators.base import Generator


class AxServiceGenerator(Generator):
    def __init__(self, variables, objectives=None, n_init=4):
        super().__init__(variables, objectives)
        self.n_init = n_init
        self._determine_torch_device()
        self._create_ax_client()

    def _ask(self, trials):
        for trial in trials:
            parameters, trial_id = self.ax_client.get_next_trial()
            trial.variable_values = [parameters.get(var.name) for var in self.variables]
            trial.ax_trial_id = trial_id
            # trials.append(
            #     Trial(
            #         variables=self.variables,
            #         values=[parameters.get(var.name) for var in self.variables]
            #     )
            # )
        return trials

    def _tell(self, trials):
        for trial in trials:
            objective_eval = {}
            for oe in trial.objective_evaluations:
                objective_eval[oe.objective.name] = (oe.value, oe.sem)
            self.ax_client.complete_trial(
                        trial_index=trial.ax_trial_id,
                        raw_data=objective_eval
                    )

    def _create_ax_client(self):
        # Create parameter list.
        parameters = list()
        for var in self.variables:
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
                num_trials=self.n_init
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
            ax_objectives[obj.name] = ObjectiveProperties(minimize=obj.minimize)

        # Create client and experiment.
        ax_client = AxClient(generation_strategy=gs)
        ax_client.create_experiment(
            parameters=parameters,
            objectives=ax_objectives
        )

        self.ax_client = ax_client

    def _determine_torch_device(self):
        # If CUDA is available, run BO loop on the GPU.
        if self.use_cuda and torch.cuda.is_available():
            self.torch_device = 'cuda'
        else:
            self.torch_device = 'cpu'
