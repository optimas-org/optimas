from .base import Evaluator
from .template_evaluator import TemplateEvaluator


class MultitaskEvaluator(Evaluator):
    def __init__(self, tasks, task_evaluators):
        self._check_tasks(tasks)
        self._check_evaluators(task_evaluators)
        super().__init__(
            sim_function=task_evaluators[0].sim_function
        )
        self.tasks = tasks
        self.task_evaluators = task_evaluators

    def _check_evaluators(self, evaluators):
        assert len(evaluators) == 2
        assert type(evaluators[0]) is type(evaluators[1])

    def _check_tasks(self, tasks):
        assert len(tasks) == 2
        assert tasks[0].name != tasks[1].name

    def get_sim_specs(self, varying_parameters, objectives):
        sim_specs = super().get_sim_specs(varying_parameters, objectives)
        sim_specs_1 = self.task_evaluators[0].get_sim_specs(varying_parameters, objectives)
        sim_specs_2 = self.task_evaluators[1].get_sim_specs(varying_parameters, objectives)

        sim_specs['user'] = {
            self.tasks[0].name: sim_specs_1['user'],
            self.tasks[1].name: sim_specs_2['user'],
        }
        task_len = max([len(self.tasks[0].name), len(self.tasks[1].name)])
        sim_specs['in'].append('task')
        sim_specs['out'].append(('task', str, task_len))
        return sim_specs

    def get_libe_specs(self):
        libE_specs_1 = self.task_evaluators[0].get_libe_specs()
        libE_specs_2 = self.task_evaluators[1].get_libe_specs()
        if 'sim_dir_copy_files' in libE_specs_1:
            libE_specs_1['sim_dir_copy_files'] = list(
                set(libE_specs_1['sim_dir_copy_files'] +
                    libE_specs_2['sim_dir_copy_files'])
            )
        return libE_specs_1

    def _initialize(self):
        if isinstance(self.task_evaluators[0], TemplateEvaluator):
            for task, evaluator in zip(self.tasks, self.task_evaluators):
                evaluator.set_app_name(task.name)
        for evaluator in self.task_evaluators:
            evaluator.initialize()
