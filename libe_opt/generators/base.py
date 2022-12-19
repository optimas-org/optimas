from libe_opt.utils.logger import get_logger
from libe_opt.gen_functions import persistent_generator
from libe_opt.core import Variable, Objective, Trial, Evaluation, ObjectiveEvaluation

logger = get_logger(__name__)


class Generator():
    def __init__(self, variables, objectives=None, constraints=None,
                 use_cuda=False, custom_trial_metadata=None):
        if objectives is None:
            objectives = [Objective()]
        self.variables = variables
        self.objectives = objectives
        self.constraints = constraints
        self.custom_trial_metadata = [] if custom_trial_metadata is None else custom_trial_metadata
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
                    index=len(self._trials) + i,
                    custom_metadata=self.custom_trial_metadata
                )
            )
        # Ask the generator to fill them.
        trials = self._ask(trials)
        # Keep only trials that have been given data.
        trials = [trial for trial in trials if trial.variable_values]
        for trial in trials:
            logger.info(
                'Generated trial {} with parameters {}'.format(
                    trial.index, trial.parameters_as_dict()))
        # Store trials.
        self._trials.extend(trials)
        return trials

    def tell(self, trials):
        for trial in trials:
            if trial not in self._trials:
                trial.index = len(self._trials)
                self._trials.append(trial)
        self._tell(trials)
        for trial in trials:
            logger.info(
                'Completed trial {} with data {}'.format(
                    trial.index, trial.objectives_as_dict()))

    def incorporate_history(self, history):
        # Keep only evaluations where the simulation finished sucessfully.
        history = history[history['sim_ended']]
        n_sims = len(history)
        trials = []
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
                ],
                custom_metadata=self.custom_trial_metadata
            )
            for par in self.custom_trial_metadata:
                setattr(trial, par.name, history[par.save_name][i])
            trials.append(trial)
        self.tell(trials)

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
            ] + [(par.save_name, par.type) for par in self.custom_trial_metadata],
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
