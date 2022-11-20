from ax.core.multi_type_experiment import MultiTypeExperiment
from ax.core.parameter import RangeParameter, ParameterType
from ax.core.search_space import SearchSpace
from ax.core.optimization_config import OptimizationConfig
from ax.core.objective import Objective
from ax.runners import SyntheticRunner

from libe_opt.search_methods.base import SearchMethod
from .ax_metric import AxMetric
from .gen_functions import persistent_gp_mt_ax_gen_f


class MultitaskBayesianOptimization(SearchMethod):
    def __init__(
            self, var_names, var_lb, var_ub, sim_template, analysis_func,
            name_hifi, name_lofi, n_init_hifi, n_init_lofi,
            n_opt_hifi, n_opt_lofi, extra_args_hifi=None, extra_args_lofi=None,
            analyzed_params=[], sim_workers=1, use_cuda=False, libE_specs={},
            history=None, executable=None, sim_files=[]):
        self.name_hifi = name_hifi
        self.name_lofi = name_lofi
        self.n_init_hifi = n_init_hifi
        self.n_init_lofi = n_init_lofi
        self.n_opt_hifi = n_opt_hifi
        self.n_opt_lofi = n_opt_lofi
        self.extra_args_hifi = extra_args_hifi
        self.extra_args_lofi = extra_args_lofi
        super().__init__(
            var_names=var_names,
            var_lb=var_lb,
            var_ub=var_ub,
            sim_template=sim_template,
            analysis_func=analysis_func,
            sim_number=n_init_hifi + n_init_lofi + n_opt_hifi + n_opt_lofi,
            analyzed_params=analyzed_params,
            sim_workers=sim_workers,
            run_async=False,
            use_cuda=use_cuda,
            libE_specs=libE_specs,
            history=history,
            gen_function=persistent_gp_mt_ax_gen_f,
            executable=executable,
            sim_files=sim_files
        )

    def _initialize_model(self):
        self._create_experiment()

    def _create_experiment(self):

        # Get task names.
        hifi_task = self.name_hifi
        lofi_task = self.name_lofi

        # Create search space.
        parameters = []
        for i, (ub, lb) in enumerate(zip(self.var_ub, self.var_lb)):
            parameters.append(
                RangeParameter(
                    name='x{}'.format(i),
                    parameter_type=ParameterType.FLOAT,
                    lower=float(lb),
                    upper=float(ub))
            )
        search_space=SearchSpace(parameters=parameters)

        # Create metrics.
        hifi_objective = AxMetric(
            name='hifi_metric',
            lower_is_better=True
        )
        lofi_objective = AxMetric(
            name='lofi_metric',
            lower_is_better=True
        )

        # Create optimization config.
        opt_config = OptimizationConfig(
            objective=Objective(hifi_objective, minimize=True))

        # Create experiment.
        experiment = MultiTypeExperiment(
                name="mt_exp",
                search_space=search_space,
                default_trial_type=hifi_task,
                default_runner=SyntheticRunner(),
                optimization_config=opt_config,
            )

        experiment.add_trial_type(lofi_task, SyntheticRunner())
        experiment.add_tracking_metric(
            metric=lofi_objective,
            trial_type=lofi_task,
            canonical_name='hifi_metric')

        # TODO: Implement reading past history (by reading saved experiment or
        # libEnsemble hystory file).

        self.experiment = experiment

    def _create_sim_specs(self):
        super()._create_sim_specs()
        self.sim_specs['in'].append('task')
        self.sim_specs['user']['extra_args'] = {}
        if self.extra_args_lofi is not None:
            self.sim_specs['user']['extra_args'][self.name_lofi] = self.extra_args_lofi
        if self.extra_args_hifi is not None:
            self.sim_specs['user']['extra_args'][self.name_hifi] = self.extra_args_hifi

    def _create_gen_specs(self):
        super()._create_gen_specs()
        self.gen_specs['out'].append(
                ('task', str, max([len(self.name_hifi), len(self.name_lofi)]))
                )
        self.gen_specs['user']['mt_params'] = {
            'name_hifi': self.name_hifi,
            'name_lofi': self.name_lofi,
            'n_init_hifi': self.n_init_hifi,
            'n_init_lofi': self.n_init_lofi,
            'n_opt_hifi': self.n_opt_hifi,
            'n_opt_lofi': self.n_opt_lofi,
        }
        self.gen_specs['user']['experiment'] = self.experiment
