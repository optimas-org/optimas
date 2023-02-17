import os

from libensemble.executors.executor import Executor

from optimas.sim_functions import run_template_simulation
from .base import Evaluator


class TemplateEvaluator(Evaluator):
    def __init__(
            self, sim_template, analysis_func,
            executable=None, sim_files=None, n_gpus=1, n_proc=1):
        super().__init__(
            sim_function=run_template_simulation,
            n_gpus=n_gpus)
        self.sim_template = sim_template
        self.analysis_func = analysis_func
        self.executable = executable
        self.sim_files = [] if sim_files is None else sim_files
        self.n_proc = n_proc
        self._app_name = 'sim'

    def get_sim_specs(self, varying_parameters, objectives,
                      analyzed_parameters):
        sim_specs = super().get_sim_specs(varying_parameters, objectives,
                                          analyzed_parameters)
        sim_specs['user']['analysis_func'] = self.analysis_func
        sim_specs['user']['sim_template'] = os.path.basename(self.sim_template)
        sim_specs['user']['app_name'] = self._app_name
        sim_specs['user']['n_proc'] = self.n_proc
        return sim_specs

    def set_app_name(self, name):
        self._app_name = name

    def _initialize(self):
        self._register_app()

    def _register_app(self):
        # Determine executable path.
        if self.sim_template.endswith('.py'):
            executable_path = 'simulation_script.py'
        else:
            # By default, if the template is not a `.py` file, we run
            # it with an executable.
            assert self.executable is not None, (
                'An executable must be provided for non-Python simulations')
            assert os.path.exists(self.executable), (
                'Executable {} does not exist.'.format(self.executable))
            executable_path = './' + self.executable
            self.sim_files.append(self.executable)

        # Register app.
        Executor.executor.register_app(
            full_path=executable_path,
            app_name=self._app_name
        )

    def get_libe_specs(self):
        libE_specs = super().get_libe_specs()
        # Add sim_template and sim_files to the list of files to be copied
        libE_specs['sim_dir_copy_files'] = [self.sim_template] + self.sim_files
        # Force libEnsemble to create a directory for each simulation
        # default value, if not defined
        libE_specs['sim_dirs_make'] = True
        return libE_specs
