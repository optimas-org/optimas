from libe_opt.gen_functions import persistent_generator
from libe_opt.core import Variable, Objective, Trial, Evaluation, ObjectiveEvaluation


class Generator():
    def __init__(self, variables, objectives=None, constraints=None, use_cuda=False):
        if objectives is None:
            objectives = [Objective()]
        self.variables = variables
        self.objectives = objectives
        self.constraints = constraints
        self.use_cuda = use_cuda
        self.gen_function = persistent_generator
        self._trials = []

    def ask(self, n_trials):
        trials = []
        # Initialize as many trials as requested.
        for i in range(n_trials):
            trials.append(
                Trial(
                    variables=self.variables,
                    objectives=self.objectives,
                    index=len(self._trials) + i
                )
            )
        # Ask the generator to fill them.
        trials = self._ask(trials)
        # Keep only trials that have been given data.
        trials = [trial for trial in trials if trial.variable_values]
        # Store trials.
        self._trials.extend(trials)
        return trials

    def tell(self, trials):
        for trial in trials:
            if trial not in self._trials:
                trial.index = len(self._trials)
                self._trials.append(trial)
        self._tell(trials)

    def incorporate_history(self, history):
        # Keep only evaluations where the simulation finished sucessfully.
        history = history[history['sim_ended']]
        n_sims = len(history)
        for i in range(n_sims):
            trial = Trial(
                        variables=self.variables,
                        variable_values=[
                            history[var.name][i] for var in self.variables],
                        objectives=self.objectives,
                        objective_evaluations=[
                            ObjectiveEvaluation(
                                objective=obj,
                                value=history[obj.name][i]
                            ) for obj in self.objectives
                        ]
                    )
            self.tell([trial])

    def get_gen_specs(self, sim_workers):
        gen_specs = {
            # Generator function.
            'gen_f': self.gen_function,
            # Generator input. This is a RNG, no need for inputs.
            'in': ['sim_id'],
            'persis_in': ['sim_id', 'trial_index'] + [obj.name for obj in self.objectives],
            'out': [(var.name, float) for var in self.variables] + [
                ('resource_sets', int),
                ('trial_index', int),
            ],
            'user': {
                'generator': self,
                # Total max number of sims running concurrently.
                'gen_batch_size': sim_workers,
                # Allow generator to run on GPU.
                'use_cuda': self.use_cuda
            }
        }
        return gen_specs
    
    def get_libe_specs(self):
        libE_specs = {}
        # If not using CUDA, do not allocate resources for generator.
        if not self.use_cuda:
            libE_specs['zero_resource_workers'] = [1]
        return libE_specs

    def _ask(self, trials):
        pass

    def _tell(self, val, noise=None):
        pass
