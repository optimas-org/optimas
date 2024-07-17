"""Contains the definition of the MultitaskEvaluator class."""

from typing import List, Dict

from optimas.core import Task, VaryingParameter, Objective, Parameter
from .base import Evaluator
from .template_evaluator import TemplateEvaluator


class MultitaskEvaluator(Evaluator):
    """Evaluator to be used with multitask optimization.

    Each task has its own evaluator (either a ``FunctionEvaluator`` or a
    ``TemplateEvaluator``). The ``MultitaskEvaluator`` groups all tasks and
    their corresponding evaluators and generates the combined ``libe_specs``
    and ``sim_specs``.

    Parameters
    ----------
    tasks : list of Task
        List of the tasks used in the optimization.
    task_evaluators : list of Evaluator
        List with the evaluators of each task.

    """

    def __init__(
        self, tasks: List[Task], task_evaluators: List[Evaluator]
    ) -> None:
        self._check_tasks(tasks)
        self._check_evaluators(task_evaluators)
        super().__init__(sim_function=task_evaluators[0].sim_function)
        self.tasks = tasks
        self.task_evaluators = task_evaluators

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
        # Get sim_specs of each task evaluator.
        sim_specs_1 = self.task_evaluators[0].get_sim_specs(
            varying_parameters, objectives, analyzed_parameters
        )
        sim_specs_2 = self.task_evaluators[1].get_sim_specs(
            varying_parameters, objectives, analyzed_parameters
        )
        # Store user sim_specs of each task separately.
        sim_specs["user"] = {
            self.tasks[0].name: sim_specs_1["user"],
            self.tasks[1].name: sim_specs_2["user"],
        }
        # Add task name to sim_specs in.
        sim_specs["in"].append("task")
        return sim_specs

    def get_libe_specs(self) -> Dict:
        """Get the `libE_specs` for `libEnsemble`."""
        # Get libe_specs of each task evaluator.
        libE_specs_1 = self.task_evaluators[0].get_libe_specs()
        libE_specs_2 = self.task_evaluators[1].get_libe_specs()
        # Include relevant specs from the second evaluator into the first one.
        if "sim_dir_copy_files" in libE_specs_1:
            libE_specs_1["sim_dir_copy_files"] = list(
                set(
                    libE_specs_1["sim_dir_copy_files"]
                    + libE_specs_2["sim_dir_copy_files"]
                )
            )
        # Use only the combined specs.
        return libE_specs_1

    def get_run_params(self) -> Dict:
        """Return run parameters for this evaluator."""
        run_params = {}
        for task, evaluator in zip(self.tasks, self.task_evaluators):
            run_params[task.name] = evaluator.get_run_params()
        return run_params

    def _initialize(self) -> None:
        """Initialize the evaluator."""
        if isinstance(self.task_evaluators[0], TemplateEvaluator):
            for task, evaluator in zip(self.tasks, self.task_evaluators):
                evaluator.app_name = task.name
        for evaluator in self.task_evaluators:
            evaluator.initialize()

    def _check_evaluators(self, evaluators) -> None:
        """Check the given evaluators."""
        # Check that only 2 evaluators are given.
        assert len(evaluators) == 2
        # Check that both evaluators are of the same type.
        assert isinstance(evaluators[0], type(evaluators[1]))

    def _check_tasks(self, tasks) -> None:
        """Check the given tasks."""
        # Check that only 2 tasks are given.
        assert len(tasks) == 2
        # Check that the task names are different.
        assert tasks[0].name != tasks[1].name
