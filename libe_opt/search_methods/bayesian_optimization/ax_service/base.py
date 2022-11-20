import torch

from libe_opt.search_methods.base import SearchMethod
from .gen_functions import persistent_ax_client

class AxOptimizer(SearchMethod):
    def __init__(
            self, var_names, var_lb, var_ub, sim_template, analysis_func,
            sim_number, analyzed_params=[], sim_workers=1, run_async=True,
            use_cuda=False, libE_specs={}, past_history=None, ax_client=None):
        self.ax_client = ax_client
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
            past_history=past_history,
            gen_function=persistent_ax_client
        )

    def _initialize_model(self):
        self._determine_torch_device()
        if self.ax_client is None:
            self.ax_client = self._create_ax_client()

    def _create_gen_specs(self):
        super()._create_gen_specs()
        self.gen_specs['user']['client'] = self.ax_client

    def _create_ax_client(self):
        raise NotImplementedError

    def _determine_torch_device(self):
        # If CUDA is available, run BO loop on the GPU.
        if self.use_cuda and torch.cuda.is_available():
            self.torch_device = 'cuda'
        else:
            self.torch_device = 'cpu'
