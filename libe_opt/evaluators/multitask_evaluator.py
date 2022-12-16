class MultitaskEvaluator():
    def __init__(self, tasks, task_evaluators):
        self._check_tasks(tasks)
        self._check_evaluators(task_evaluators)
        self.tasks = tasks
        self.task_evaluators = task_evaluators

    def _check_evaluators(self, evaluators):
        assert len(evaluators) == 2
        assert type(evaluators[0]) is type(evaluators[1])

    def _check_tasks(self, tasks):
        assert len(tasks) == 2

    def get_sim_specs(self, variables, objectives):
        sim_specs = {}
        sim_specs_1 = self.task_evaluators[0].get_sim_specs(variables, objectives)
        sim_specs_2 = self.task_evaluators[1].get_sim_specs(variables, objectives)

        sim_specs['sim_f'] = sim_specs_1['sim_f']
        sim_specs['in'] = sim_specs_1['in']
        sim_specs['out'] = sim_specs_1['out']
        sim_specs['user'] = {
            self.tasks[0].name: sim_specs_1['user'],
            self.tasks[1].name: sim_specs_2['user'],
        }
        task_len = max([len(self.tasks[0].name), len(self.tasks[1].name)])
        sim_specs['in'].append('task')
        sim_specs['out'].append(('task', str, task_len))
        return sim_specs

    def get_libe_specs(self):
        libE_specs = {}
        libe_specs_1 = self.task_evaluators[0].get_libe_specs()
        libe_specs_2 = self.task_evaluators[1].get_libe_specs()
        libE_specs['sim_dir_copy_files'] = list(
            set(libe_specs_1['sim_dir_copy_files'] +
                libe_specs_2['sim_dir_copy_files'])
        )
        return libE_specs

    def register_app(self, executor):
        for task, evaluator in zip(self.tasks, self.task_evaluators):
            evaluator.register_app(executor, app_name=task.name)
