from libe_opt.sim_functions import run_function
from .base import Evaluator


class FunctionEvaluator(Evaluator):
    def __init__(self, function, analyzed_params=None, n_gpus=1):
        super().__init__(
            sim_function=run_function,
            analyzed_params=analyzed_params,
            n_gpus=n_gpus)
        self.function = function

    def get_sim_specs(self, variables, objectives):
        sim_specs = super().get_sim_specs(variables, objectives)
        sim_specs['user']['evaluation_func'] = self.function
        return sim_specs
