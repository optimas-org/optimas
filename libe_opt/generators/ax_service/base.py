import torch

from ax.service.ax_client import AxClient
from ax.modelbridge.generation_strategy import (
    GenerationStep, GenerationStrategy)
from ax.modelbridge.registry import Models
from ax.service.utils.instantiation import ObjectiveProperties

from libe_opt.generators.base import Generator


class AxServiceGenerator(Generator):
    def __init__(self, variables, objectives=None, n_init=4, use_cuda=False):
        super().__init__(variables, objectives, use_cuda=use_cuda)
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
            try:
                self.ax_client.complete_trial(
                            trial_index=trial.ax_trial_id,
                            raw_data=objective_eval
                        )
            except AttributeError:
                params = {}
                for var, value in zip(trial.variables, trial.variable_values):
                    params[var.name] = value
                _, trial_id = self.ax_client.attach_trial(params)
                self.ax_client.complete_trial(trial_id, objective_eval)

    def _create_ax_client(self):
        raise NotImplementedError

    def _determine_torch_device(self):
        # If CUDA is available, run BO loop on the GPU.
        if self.use_cuda and torch.cuda.is_available():
            self.torch_device = 'cuda'
        else:
            self.torch_device = 'cpu'
