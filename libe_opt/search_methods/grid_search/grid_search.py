import numpy as np

from libe_opt.search_methods.base import SearchMethod
from .gen_function import persistent_regular_grid_search


class GridSearch(SearchMethod):
    def __init__(
            self, var_names, var_lb, var_ub, var_steps, sim_template,
            analysis_func, analyzed_params=[], sim_workers=1,
            run_async=True, libE_specs={}):
        self.var_steps = var_steps
        super().__init__(
            var_names=var_names,
            var_lb=var_lb,
            var_ub=var_ub,
            sim_template=sim_template,
            analysis_func=analysis_func,
            sim_number=np.prod(var_steps),
            analyzed_params=analyzed_params,
            sim_workers=sim_workers,
            run_async=run_async,
            use_cuda=False,
            libE_specs=libE_specs,
            gen_function=persistent_regular_grid_search
        )

    def _create_gen_specs(self):
        super()._create_gen_specs()
        self.gen_specs['user']['n_steps'] = self.var_steps
