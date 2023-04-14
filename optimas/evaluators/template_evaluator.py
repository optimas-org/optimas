"""Contains the definition of the TemplateEvaluator class."""

import os
from typing import Callable, Optional, List, Dict

from libensemble.executors.executor import Executor

from optimas.sim_functions import run_template_simulation
from optimas.core import VaryingParameter, Objective, Parameter
from .base import Evaluator


class TemplateEvaluator(Evaluator):
    """Evaluator class to use when the evaluations are carried out by
    executing a templated script.

    Parameters
    ----------
    sim_template : str
        Path to the simulation template file.
    analysis_func : Callable
        Function that will analyze the simulation output to obtain the value
        of the objective(s) and other analyzed parameters.
    executable : str, optional.
        Path to the executable that will run the simulation. Only needed if
        the simulation template is not a Python script.
    sim_files : list of str, optional.
        List of files that are needed to carry out the simulation and that
        will be copied to the simulation directory.
    n_gpus : int, optional
        The number of GPUs that will be made available for each simulation. By
        default, 1.
    n_proc : int, optional
        The number of processes that will be made used for each simulation. By
        default, 1. (Currently unused)
    """
    def __init__(
        self,
        sim_template: str,
        analysis_func: Callable,
        executable: Optional[str] = None,
        sim_files: Optional[List[str]] = None,
        n_gpus: Optional[int] = 1,
        n_proc: Optional[int] = 1
    ) -> None:
        super().__init__(
            sim_function=run_template_simulation,
            n_gpus=n_gpus)
        self.sim_template = sim_template
        self.analysis_func = analysis_func
        self.executable = executable
        self.sim_files = [] if sim_files is None else sim_files
        self.n_proc = n_proc
        self._app_name = 'sim'

    @property
    def app_name(self) -> str:
        return self._app_name

    @app_name.setter
    def app_name(
        self,
        name: str
    ) -> None:
        self._app_name = name

    def get_sim_specs(
        self,
        varying_parameters: List[VaryingParameter],
        objectives: List[Objective],
        analyzed_parameters: List[Parameter],
    ) -> Dict:
        """Get a dictionary with the ``sim_specs`` as expected
        by ``libEnsemble``
        """
        # Get base sim_specs.
        sim_specs = super().get_sim_specs(varying_parameters, objectives,
                                          analyzed_parameters)
        # Add parameters specific to the template evaluator.
        sim_specs['user']['analysis_func'] = self.analysis_func
        sim_specs['user']['sim_template'] = os.path.basename(self.sim_template)
        sim_specs['user']['app_name'] = self._app_name
        sim_specs['user']['n_proc'] = self.n_proc
        return sim_specs

    def get_libe_specs(self) -> Dict:
        """Get a dictionary with the ``libE_specs`` as expected
        by ``libEnsemble``
        """
        libE_specs = super().get_libe_specs()
        # Add sim_template and sim_files to the list of files to be copied
        libE_specs['sim_dir_copy_files'] = [self.sim_template] + self.sim_files
        # Force libEnsemble to create a directory for each simulation
        # default value, if not defined
        libE_specs['sim_dirs_make'] = True
        return libE_specs

    def _initialize(self) -> None:
        self._register_app()

    def _register_app(self) -> None:
        """Register the executable as an app in the libEnsemble executor."""
        # Determine executable path.
        if self.sim_template.endswith('.py'):
            sim_script = os.path.basename(self.sim_template)
            # Strip 'template_' from name
            sim_script = sim_script[len('template_'):]
            executable_path = sim_script
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
