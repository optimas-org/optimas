"""Contains the definition of the TemplateEvaluator class."""

import os
from typing import Callable, Optional, List, Dict

from libensemble.executors.executor import Executor

from optimas.sim_functions import run_template_simulation
from optimas.core import VaryingParameter, Objective, Parameter
from .base import Evaluator


class TemplateEvaluator(Evaluator):
    """Evaluator class for executing a template script.

    Parameters
    ----------
    sim_template : str
        Path to the simulation template file.
    analysis_func : Callable, optional
        Function that will analyze the simulation output to obtain the value
        of the objective(s) and other analyzed parameters. This parameter
        is only optional if the `TemplateEvaluator` is included in a
        `ChainEvaluator`. In this case, at least one of the chained evaluators
        should have an analysis function.
    executable : str, optional.
        Path to the executable that will run the simulation. Only needed if
        the simulation template is not a Python script.
    sim_files : list of str, optional.
        List of files that are needed to carry out the simulation and that
        will be copied to the simulation directory.
    n_procs : int, optional
        The number of processes that will be used for each evaluation. By
        default, ``n_procs=1`` if ``n_gpus`` is not given. Otherwise, the
        default behavior is to match the number of processes to the number
        of GPUs, i.e., ``n_procs=n_gpus``.
    n_gpus : int, optional
        The number of GPUs that will be made available for each evaluation. By
        default, 0.
    env_script : str, optional
        The full path of a shell script to set up the environment for the
        launched simulation. This is useful when the simulation needs to run
        in a different environment than optimas. The script should start with a
        shebang.
    env_mpi : str, optional
        If the `env_script` loads an MPI different than the one in the optimas
        environment, indicate it here. Possible values are "mpich", "openmpi",
        "aprun", "srun", "jsrun", "msmpi".
    timeout : float, optional
        Time in seconds until the evaluation is forcibly terminated. By default
        ``None`` (i.e., no timeout).
    stdout : str, optional
        A standard output filename.
    stderr : str, optional
        A standard error filename.

    """

    def __init__(
        self,
        sim_template: str,
        analysis_func: Optional[Callable] = None,
        executable: Optional[str] = None,
        sim_files: Optional[List[str]] = None,
        n_procs: Optional[int] = None,
        n_gpus: Optional[int] = None,
        env_script: Optional[str] = None,
        env_mpi: Optional[str] = None,
        timeout: Optional[float] = None,
        stdout: Optional[str] = None,
        stderr: Optional[str] = None,
    ) -> None:
        super().__init__(
            sim_function=run_template_simulation, n_procs=n_procs, n_gpus=n_gpus
        )
        self.sim_template = sim_template
        self.analysis_func = analysis_func
        self.executable = executable
        self.env_script = env_script
        self.env_mpi = env_mpi
        self.timeout = timeout
        self.sim_files = [] if sim_files is None else sim_files
        self.stdout = stdout
        self.stderr = stderr
        self._app_name = "sim"

    @property
    def app_name(self) -> str:
        """Name of the libEnsemble app that executes the evaluation."""
        return self._app_name

    @app_name.setter
    def app_name(self, name: str) -> None:
        self._app_name = name

    def get_sim_specs(
        self,
        varying_parameters: List[VaryingParameter],
        objectives: List[Objective],
        analyzed_parameters: List[Parameter],
    ) -> Dict:
        """Get the `sim_specs` for `libEnsemble`."""
        # Get base sim_specs.
        sim_specs = super().get_sim_specs(
            varying_parameters, objectives, analyzed_parameters
        )
        # Add parameters specific to the template evaluator.
        sim_specs["user"]["analysis_func"] = self.analysis_func
        sim_specs["user"]["sim_template"] = os.path.basename(self.sim_template)
        sim_specs["user"]["app_name"] = self._app_name
        sim_specs["user"]["num_procs"] = self._n_procs
        sim_specs["user"]["num_gpus"] = self._n_gpus
        sim_specs["user"]["env_script"] = self.env_script
        sim_specs["user"]["env_mpi"] = self.env_mpi
        sim_specs["user"]["timeout"] = self.timeout
        sim_specs["user"]["stdout"] = self.stdout
        sim_specs["user"]["stderr"] = self.stderr
        return sim_specs

    def get_libe_specs(self) -> Dict:
        """Get the `libE_specs` for `libEnsemble`."""
        libE_specs = super().get_libe_specs()
        # Add sim_template and sim_files to the list of files to be copied.
        # Use the absolute path to the files to get around a libEnsemble bug
        # when using a workflow dir.
        sim_files = [self.sim_template] + self.sim_files
        sim_files = [os.path.abspath(file) for file in sim_files]
        libE_specs["sim_dir_copy_files"] = sim_files
        # Force libEnsemble to create a directory for each simulation
        # default value, if not defined
        libE_specs["sim_dirs_make"] = True
        return libE_specs

    def _initialize(self) -> None:
        self._register_app()

    def _register_app(self) -> None:
        """Register the executable as an app in the libEnsemble executor."""
        # Determine executable path.
        if self.sim_template.endswith(".py"):
            executable_path = os.path.basename(self.sim_template)
        else:
            # By default, if the template is not a `.py` file, we run
            # it with an executable.
            assert (
                self.executable is not None
            ), "An executable must be provided for non-Python simulations"
            assert os.path.exists(
                self.executable
            ), "Executable {} does not exist.".format(self.executable)
            executable_path = os.path.abspath(self.executable)

        # Register app.
        Executor.executor.register_app(
            full_path=executable_path, app_name=self._app_name
        )
