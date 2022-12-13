import os
from libe_opt.sim_functions import run_template_simulation
from libensemble.executors.mpi_executor import MPIExecutor


class TemplateEvaluator():
    def __init__(
            self, sim_template, analysis_func, analyzed_params=[],
            executable=None, sim_files=[]):
        self.sim_template = os.path.basename(sim_template)
        self.analysis_func = analysis_func
        self.analyzed_params = analyzed_params
        self.executable = executable
        self.sim_files = sim_files
        self.sim_function = run_template_simulation
        self._create_executor()

    def get_sim_specs(self, variables, objectives):
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
                'var_params': [var.name for var in variables],
                'analysis_func': self.analysis_func,
                # keeps only the file name of the simulation template
                'sim_template': self.sim_template
            }
        }
        return sim_specs

    def _create_executor(self):
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

        # Create executor and register app.
        exctr = MPIExecutor()
        exctr.register_app(
            full_path=executable_path,
            calc_type='sim'
        )

    def get_libe_specs(self):
        libE_specs = {}
        # Add sim_template and sim_files to the list of files to be copied
        libE_specs['sim_dir_copy_files'] = [self.sim_template] + self.sim_files
        return libE_specs
