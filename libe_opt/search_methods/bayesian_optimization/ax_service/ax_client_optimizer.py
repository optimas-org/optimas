from .base import AxOptimizer
from .gen_functions import persistent_ax_client


class AxClientOptimizer(AxOptimizer):
    def __init__(
            self, ax_client, sim_template, analysis_func,
            sim_number, analyzed_params=[], sim_workers=1, run_async=True,
            use_cuda=False, libE_specs={}, history=None, executable=None,
            sim_files=[]):
        var_names, var_lb, var_ub = self._get_var_params(ax_client)
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
            gen_function=persistent_ax_client,
            ax_client=ax_client,
            executable=executable,
            sim_files=sim_files
        )

    def _get_var_params(self, ax_client):
        var_names = []
        var_lb = []
        var_ub = []
        parameters = ax_client.experiment.parameters
        for key, value in parameters.items():
            var_names.append(key)
            var_lb.append(value.range[0])
            var_ub.append(value.range[1])
        return var_names, var_lb, var_ub
