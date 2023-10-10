"""Contains the definition of the ChainEvaluator class."""

import copy
from typing import List, Dict


from optimas.core import VaryingParameter, Objective, Parameter
from optimas.sim_functions import run_template_simulation
from .base import Evaluator
from .template_evaluator import TemplateEvaluator


class ChainEvaluator(Evaluator):
    """Evaluator that executes a chain of `TemplateEvaluators`.

    This is useful when each evaluation consists of several steps, where each
    step is a simulation with a different simulation code. Each step
    is defined by a TemplateEvaluator and can request a different number of
    resources.

    Each evaluation with the `ChainEvaluator` gets allocated the maximum number
    of processes (`n_procs`) and GPUs (`n_gpus`) that every step might request
    (e.g., if one step requires `n_procs=20` and `n_gpus=0`, and a second step
    requires `n_procs=4` and `n_gpus=4`, each evaluation will get assigned
    `n_procs=20` and `n_gpus=4`). Then each step will only make use of the
    subset of resources it needs.

    Parameters
    ----------
    evaluators : list of TemplateEvaluators
        A list of TemplateEvaluators given in the order in which they should
        be executed.

    """

    def __init__(self, evaluators: List[TemplateEvaluator]) -> None:
        self._check_evaluators(evaluators)
        super().__init__(
            run_template_simulation,
        )
        self.evaluators = evaluators

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
        sim_specs["user"]["steps"] = []
        # Get the user specs from each step.
        for evaluator in self.evaluators:
            sim_specs["user"]["steps"].append(
                evaluator.get_sim_specs(
                    varying_parameters, objectives, analyzed_parameters
                )["user"]
            )
        return sim_specs

    def get_libe_specs(self) -> Dict:
        """Get the `libE_specs` for `libEnsemble`."""
        # Get libe_specs of each task evaluator.
        libE_specs_0 = copy.deepcopy(self.evaluators[0].get_libe_specs())
        # Combine the files to copy from all evaluators.
        for evaluator in self.evaluators[1:]:
            libE_specs_i = evaluator.get_libe_specs()
            libE_specs_0["sim_dir_copy_files"] = list(
                set(
                    libE_specs_0["sim_dir_copy_files"]
                    + libE_specs_i["sim_dir_copy_files"]
                )
            )
        # Use only the combined specs.
        return libE_specs_0

    def get_run_params(self) -> Dict:
        """Return run parameters for this evaluator."""
        num_procs = 0
        num_gpus = 0
        # Get maximum number of processes and GPUs.
        for evaluator in self.evaluators:
            ev_run_params = evaluator.get_run_params()
            num_procs = max(num_procs, ev_run_params["num_procs"])
            num_gpus = max(num_gpus, ev_run_params["num_gpus"])
        run_params = {
            "num_procs": num_procs,
            "num_gpus": num_gpus,
        }
        return run_params

    def _initialize(self) -> None:
        """Initialize the evaluator."""
        for i, evaluator in enumerate(self.evaluators):
            # Assign a different app name to each evaluator.
            evaluator.app_name = f"sim_{i}"
            evaluator.initialize()

    def _check_evaluators(self, evaluators) -> None:
        """Check the given evaluators."""
        # Check that both evaluators are of the same type.
        for evaluator in evaluators:
            assert isinstance(
                evaluator, TemplateEvaluator
            ), "Only TemplateEvaluators are supported for chain evaluation."
