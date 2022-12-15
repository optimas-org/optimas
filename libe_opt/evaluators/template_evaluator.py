import os

from libe_opt.sim_functions import run_template_simulation


class TemplateEvaluator():
    def __init__(
            self, sim_template, analysis_func, analyzed_params=None,
            executable=None, sim_files=None, n_gpus=1, n_proc=1):
        self.sim_template = os.path.basename(sim_template)
        self.analysis_func = analysis_func
        self.analyzed_params = [] if analyzed_params is None else analyzed_params
        self.executable = executable
        self.sim_files = [] if sim_files is None else sim_files
        self.sim_function = run_template_simulation
        self.n_gpus = n_gpus
        self.n_proc = n_proc
        self._app_name = 'sim'
        self._app_registered = False

    def get_sim_specs(self, variables, objectives):
        if not self._app_registered:
            raise ValueError('App must be registered before generating sim_specs')
        sim_specs = {
            # Function whose output is being minimized.
            'sim_f': self.sim_function,
            # Name of input for sim_f, that LibEnsemble is allowed to modify.
            # May be a 1D array.
            'in': [var.name for var in variables],
            'out': (
                [(obj.name, float) for obj in objectives]
                # f is the single float output that LibEnsemble minimizes.
                + [(name, float) for name in self.analyzed_params]
                # input parameters
                + [(var.name, float) for var in variables]
            ),
            'user': {
                'analysis_func': self.analysis_func,
                # keeps only the file name of the simulation template
                'sim_template': self.sim_template,
                'app_name': self._app_name,
                'n_gpus': self.n_gpus,
                'n_proc': self.n_proc
            }
        }
        return sim_specs

    def register_app(self, executor, app_name=None):
        if self._app_registered:
            print('App already registered.')
            return
        if app_name is None:
            app_name = self._app_name
        else:
            self._app_name = app_name
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
        executor.register_app(
            full_path=executable_path,
            app_name=app_name
        )
        self._app_registered = True

    def get_libe_specs(self):
        libE_specs = {}
        # Add sim_template and sim_files to the list of files to be copied
        libE_specs['sim_dir_copy_files'] = [self.sim_template] + self.sim_files
        return libE_specs
