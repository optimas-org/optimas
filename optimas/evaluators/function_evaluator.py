from libe_opt.sim_functions import run_function
from .base import Evaluator


class FunctionEvaluator(Evaluator):
    def __init__(self, function, n_gpus=1):
        super().__init__(
            sim_function=run_function,
            n_gpus=n_gpus)
        self.function = function

    def get_sim_specs(self, varying_parameters, objectives,
                      analyzed_parameters):
        sim_specs = super().get_sim_specs(varying_parameters, objectives,
                                          analyzed_parameters)
        sim_specs['user']['evaluation_func'] = self.function
        return sim_specs
